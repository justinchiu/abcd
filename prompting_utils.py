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

import openai
from minichain import Prompt, EmbeddingPrompt, TemplatePrompt, show_log, start_chain

EMBEDDING_MODEL = "text-embedding-ada-002"

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

class StepAlignmentPrompt(TemplatePrompt):
    #template_file = "prompting/align.pmpt.tpl"
    #template_file = "prompting/zeroshotalign.pmpt.tpl"
    template_file = "prompting/original.pmpt.tpl"

    def dbg_render_prompt(self, kwargs):
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

class TurnStepAlignmentPrompt(TemplatePrompt):
    template_file = "prompting/"

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
    def __init__(self, args, docs, backend):
        self.args = args
        self.docs = docs
        self.model = "gpt-3.5-turbo" if args.use_chat else "text-davinci-003"

        # setup knn
        if args.docsel == "emb":
            self.knnprompt = KnnPrompt(backend.OpenAIEmbed(), (docs, args.k_docs))

        # setup align prompt
        completion_backend = backend.OpenAIChat if args.use_chat else backend.OpenAI
        if args.stepsel == "askdoc":
            self.stepprompt = StepAlignmentPrompt(completion_backend(
                model=self.model,
                max_tokens=1024,
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
                align_preds = scores.argmax(-1)
                align_preds[1:][align_preds[1:] == align_preds[:-1]] = -1
                align_score = scores.max(-1).sum()
                alignments.append(align_preds)
                step_scores.append(align_score)

            return replace(
                doc_out,
                alignments = np.stack(alignments),
                step_scores = np.array(step_scores),
            )
            #lexical_argmax = np.argmax(lexical_align_scores)
            #lexical_docpred = lexical_alignments[lexical_argmax]
            #lexical_doc = lexical_docpreds[lexical_argmax]
        elif self.args.stepsel == "askdoc":
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
        elif self.args.stepsel == "askstep":
            import pdb; pdb.set_trace()
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
