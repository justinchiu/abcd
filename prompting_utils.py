from dataclasses import dataclass, replace
import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any, Union
from rich.progress import track
from rank_bm25 import BM25Okapi
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)

import openai
from minichain import Prompt, EmbeddingPrompt, show_log, start_chain
from minichain import TemplatePrompt as McTemplatePrompt

from inference_utils import amax, afirstmax, amono, afirstmono

EMBEDDING_MODEL = "text-embedding-ada-002"

class TemplatePrompt(McTemplatePrompt):
    def print(self, kwargs):
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = self.template  # type: ignore
        else:
            tmp = Template(self.prompt_template)
        if isinstance(kwargs, dict):
            x = tmp.render(**kwargs)
        else:
            x = tmp.render(**asdict(kwargs))
        return x

def embed(x):
    emb = openai.Embedding.create(input=x["text"], engine=EMBEDDING_MODEL)
    return {"embeddings": [
        np.array(emb['data'][i]['embedding'])
        for i in range(len(emb["data"]))
    ]}

def get_bm25(docs):
    bm25d = BM25Okapi([doc["doc"].lower().split() for doc in docs])
    bm25s = {
        doc["title"]: BM25Okapi([step.lower().split() for step in doc["steps"]])
        for doc in docs
    }
    return bm25d, bm25s

class KnnPrompt(EmbeddingPrompt):
    def find(self, out, input):
        dataset, k = self.data
        res = dataset.get_nearest_examples("embeddings", np.array(out), k)
        return {
            "query": input,
            "docs": res.examples["doc"],
            "titles": res.examples["title"],
            "steps": res.examples["steps"],
            "scores": res.scores,
        }

class StepKnnPrompt(EmbeddingPrompt):
    def find(self, out, input):
        dataset, k = self.data
        #res = dataset.get_nearest_examples("embeddings", np.array(out), k)
        embs = np.array([x["embeddings"] for x in dataset])
        scores = embs @ np.array(out)
        return {
            "query": input,
            "steps": [x["text"] for x in dataset],
            "ids": [x["id"] for x in dataset],
            "scores": scores,
        }

class DialAlignmentPrompt(TemplatePrompt):
    #template_file = "prompting/align.pmpt.tpl"
    #template_file = "prompting/zeroshotalign.pmpt.tpl"
    template_file = "prompting/original.pmpt.tpl"



    def parse(self, out: str, input) -> Any:
        # Encode the parsing logic
        jsout = json.loads(out)
        numturns = max([x["T"] for x in jsout])
        preds = np.zeros(numturns+1, dtype=int)
        for x in jsout:
            turn = x["T"]
            step = x["S"]
            #preds[turn] = step if step != "N/A" else -1
            preds[turn] = step
        return preds


class TurnAlignPrompt(TemplatePrompt):
    template_file = "prompting/turnalign.pmpt.tpl"
    def parse(self, out: str, input) -> int:
        return int(out)

class TurnStepAlignPrompt(TemplatePrompt):
    template_file = "prompting/turnstepalign.pmpt.tpl"
    def parse(self, out: str, input) -> int:
        return out.strip() == "Yes"


class AbcdTurnAlignPrompt(TurnAlignPrompt):
    template_file = "prompting/abcd-turnalign.pmpt.tpl"

class FloDialTurnAlignPrompt(TurnAlignPrompt):
    template_file = "prompting/flodial-turnalign.pmpt.tpl"


class AbcdTurnStepAlignPrompt(TurnStepAlignPrompt):
    template_file = "prompting/abcd-turnstepalign.pmpt.tpl"

class FloDialTurnStepAlignPrompt(TurnStepAlignPrompt):
    template_file = "prompting/flodial-turnstepalign.pmpt.tpl"


@dataclass
class AlignedOutput:
    title: str
    doc: str
    steps: list[str]
    doc_score: float
    alignment: Union[np.ndarray, None] = None
    step_score: Union[float, None] = None

@dataclass
class AlignOutput:
    titles: list[str]
    docs: list[str]
    steps: list[list[str]]
    doc_scores: list[float]
    alignments: Union[np.ndarray, None] = None
    step_scores: Union[np.ndarray, None] = None

    def index(self, idx):
        return AlignedOutput(
            title = self.titles[idx],
            doc = self.docs[idx],
            doc_score = self.doc_scores[idx],
            steps = self.steps[idx],
            alignment = self.alignments[idx] if self.alignments is not None else None,
            step_score = self.step_scores[idx] if self.step_scores is not None else None,
        )

class Aligner:
    def __init__(self, args, docs, doc_step_embs, backend):
        self.args = args
        self.backend = backend

        self.docs = docs
        self.doc_step_embs = doc_step_embs

        self.model = "gpt-3.5-turbo" if args.use_chat else "text-davinci-003"

        if args.stepdec == "max":
            self.stepdecision = amax
        elif args.stepdec == "firstmax":
            self.stepdecision = afirstmax
        elif args.stepdec == "mono":
            self.stepdecision = amono
        elif args.stepdec == "firstmono":
            self.stepdecision = afirstmono
        else:
            raise NotImplementedError

        # setup knn
        if args.docsel == "emb":
            self.knnprompt = KnnPrompt(backend.OpenAIEmbed(), (docs, args.k_docs))
        self.stepknnprompts = {
            doc: StepKnnPrompt(backend.OpenAIEmbed(), (embs, 1))
            for doc, embs in doc_step_embs.items()
        }

        # setup align prompt
        completion_backend = backend.OpenAIChat if args.use_chat else backend.OpenAI
        if args.stepsel == "askdial":
            self.stepprompt = DialAlignmentPrompt(completion_backend(
                model=self.model,
                max_tokens=1024,
            ))
        elif args.stepsel == "askturn":
            if args.stepprompt == "0s":
                prompt = TurnAlignPrompt
            elif args.dataset == "abcd":
                prompt = AbcdTurnAlignPrompt
            elif args.dataset == "flodial":
                prompt = FloDialTurnAlignPrompt
            self.stepprompt = prompt(completion_backend(
                model=self.model,
                max_tokens=2,
            ))
        elif args.stepsel == "askturnstep":
            if args.stepprompt == "0s":
                prompt = TurnStepAlignPrompt
            elif args.dataset == "abcd":
                prompt = AbcdTurnStepAlignPrompt
            elif args.dataset == "flodial":
                prompt = FloDialTurnStepAlignPrompt
            self.stepprompt = prompt(completion_backend(
                model=self.model,
                max_tokens=10,
            ))


        # setup lexical
        if args.docsel == "lex" or args.stepsel == "lex":
            self.bm25d, self.bm25s = get_bm25(docs)

    def select_docs(self, dial):
        # Stage 1: Align the whole dialogue to a doc
        if self.args.docsel == "emb":
            knnresult = self.knnprompt(dial)
            return AlignOutput(
                titles = knnresult["titles"],
                docs = knnresult["docs"],
                steps = knnresult["steps"],
                # aiss distance smaller = better. negate
                doc_scores = -knnresult["scores"],
            )
        elif self.args.docsel == "lex":
            lexical_scores = self.bm25d.get_scores(dial.lower().split())
            lexical_doc_idxs = np.argsort(-lexical_scores)[:self.args.k_docs].tolist()

            titles = [self.docs[x]["title"] for x in lexical_doc_idxs]
            docs = [self.docs[x]["doc"] for x in lexical_doc_idxs]
            steps = [self.docs[x]["steps"] for x in lexical_doc_idxs]
            scores = lexical_scores[lexical_doc_idxs]

            return AlignOutput(
                titles = titles,
                docs = docs,
                steps = steps,
                doc_scores = scores,
            )
        elif self.args.docsel == "model":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def select_steps(self, dial, turns, doc_out):
        # Stage 2: Given a doc, align the turns in the dial to steps.
        if self.args.stepsel == "lex":
            alignments = []
            step_scores = []
            for title, doc, steps, score in zip(
                doc_out.titles, doc_out.docs,
                doc_out.steps, doc_out.doc_scores,
            ):
                scores = np.stack([
                    self.bm25s[title].get_scores(turn.split())
                    for turn in turns
                ]) # shape = turns x steps
                preds, score = self.stepdecision(torch.tensor(scores))
                preds = np.array(preds)

                alignments.append(preds)
                step_scores.append(score)

            return replace(
                doc_out,
                alignments = np.stack(alignments),
                step_scores = np.array(step_scores),
            )
        elif self.args.stepsel == "emb":
            alignments = []
            step_scores = []
            for title, doc, steps, score in zip(
                doc_out.titles, doc_out.docs,
                doc_out.steps, doc_out.doc_scores,
            ):
                results = [self.stepknnprompts[title](turn)["scores"] for turn in turns]
                # turns x steps
                scores = np.stack(results)
                preds, score = self.stepdecision(torch.tensor(scores))

                alignments.append(preds)
                step_scores.append(score)

            return replace(
                doc_out,
                alignments = np.stack(alignments),
                step_scores = np.array(step_scores),
            )
        elif self.args.stepsel == "askdial":
            # just take the best one. no point asking for every doc.
            doc = doc_out.index(0)
            result = self.stepprompt(dict(dial=dial, doc=doc.doc))

            # singleton output
            return AlignOutput(
                titles=[doc.title],
                docs=[doc.doc],
                steps=[doc.steps],
                doc_scores=[doc.doc_score],
                alignments=result[None],
                step_scores=np.zeros(1),
            )
        elif self.args.stepsel == "askturn":
            alignments = []
            scores = []
            for title, doc, steps, score in zip(
                doc_out.titles, doc_out.docs,
                doc_out.steps, doc_out.doc_scores,
            ):
                # pre-filter using embedding
                embscores = np.stack([
                    self.stepknnprompts[title](turn)["scores"] for turn in turns
                ])
                topk = np.argsort(-embscores, -1)[:,:self.args.k_steps]
                topksteps = ["\n".join([
                    f"Step {i}: {steps[idx]}" for i, idx in enumerate(idxs)
                ]) for idxs in topk]
                align = []
                for i, (turn, topsteps) in enumerate(zip(turns, topksteps)):
                    answer = self.stepprompt(dict(turn=turn, doc=topsteps))
                    align.append(topk[i,answer])
                    print(self.stepprompt.print(dict(turn=turn, doc=topsteps)))
                    print(answer)
                    import pdb; pdb.set_trace()
                """
                alignments.append(np.array([
                    self.stepprompt(dict(turn=turn, doc=topsteps))
                    for turn, topsteps in zip(turns, topksteps)
                ]))
                """
                alignments.append(np.array(align))
                # use the same score
                scores.append(score)

            return replace(
                doc_out,
                alignments = np.stack(alignments),
                step_scores = np.array(scores),
            )
        elif self.args.stepsel == "askturnstep":
            # don't run zeroshot
            #assert self.args.stepprompt != "0s"

            alignments = []
            scores = []
            for title, doc, steps, score in zip(
                doc_out.titles, doc_out.docs,
                doc_out.steps, doc_out.doc_scores,
            ):
                # pre-filter using embedding
                embscores = np.stack([
                    self.stepknnprompts[title](turn)["scores"] for turn in turns
                ])
                topk = np.argsort(-embscores, -1)[:,:self.args.k_steps]
                topksteps = [
                    [steps[idx] for idx in idxs]
                    for idxs in topk
                ]
                align = []
                for i,(turn, topsteps) in enumerate(zip(turns, topksteps)):
                    turn2step = []
                    for step in topsteps:
                        answer = self.stepprompt(dict(turn=turn, step=step))
                        print(self.stepprompt.print(dict(turn=turn, step=step)))
                        print(answer)
                        turn2step.append(answer)
                    # either null align or align to earliest 
                    turnalign = -1 if not any(turn2step) else topk[i][turn2step].min()
                    align.append(turnalign)
                """
                align = np.array([
                    [self.stepprompt(dict(turn=turn, step=step)) for step in steps]
                    for turn in turns
                ])
                """
                alignments.append(align)
                # use the same score
                scores.append(score)

            return replace(
                doc_out,
                alignments = np.stack(alignments),
                step_scores = np.array(scores),
            )
        else:
            raise NotImplementedError

    def rerank(self, alignments):
        # Stage 3: Pick best alignment of dial turns => doc steps
        if self.args.rerank == "docscore":
            idx = np.argmax(alignments.doc_scores)
            return alignments.index(idx)
        elif self.args.rerank == "stepscore":
            idx = np.argmax(alignments.step_scores)
            return alignments.index(idx)
        elif self.args.rerank == "sum":
            raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from prompting_data import FloDial
    dataset_obj = FloDial()
    get_dataset = dataset_obj.get_docs
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    flo_docs = get_dataset()
    flo_dial, flo_labels = get_dialogues_and_labels()

    bm25d, bm25s = get_bm25(flo_docs)
    import pdb; pdb.set_trace()
