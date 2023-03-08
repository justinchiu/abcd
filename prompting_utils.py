from dataclasses import dataclass
import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any
from rich.progress import track
from rank_bm25 import BM25Okapi

import openai
from minichain import Prompt, EmbeddingPrompt, TemplatePrompt, show_log, start_chain

EMBEDDING_MODEL = "text-embedding-ada-002"

def embed(x):
    emb = openai.Embedding.create(input=x["doc"], engine=EMBEDDING_MODEL)
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

class DocAlignmentPrompt(TemplatePrompt):
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

@dataclass
class DocSelection:
    titles: list[str]
    docs: list[str]
    steps: list[list[str]]
    scores: list[float]

@dataclass
class DocStepAlign:
    titles: list[str]
    docs: list[str]
    steps: list[list[str]]
    doc_scores: list[float]
    alignments: np.array
    align_scores: list[float]


class Aligner:
    def __init__(self, args, docs, backend):
        self.args = args
        self.model = "gpt-3.5-turbo" if args.use_chat else "text-davinci-003"

        # setup knn
        if args.doc_selection == "emb":
            self.knnprompt = KnnPrompt(backend.OpenAIEmbed(), (docs, args.k_docs))

        # setup align prompt
        completion_backend = backend.OpenAIChat if args.use_chat else backend.OpenAI
        if args.step_align == "askdoc":
            self.alignprompt = DocAlignmentPrompt(completion_backend(
                model=self.model,
                max_tokens=1024,
            ))

        # setup lexical
        if args.doc_selection == "lex" or args.step_align == "lex":
            self.bm25d, self.bm25s = get_bm25(docs)

    def select_docs(self, dial):
        # Stage 1: Align the whole dialogue to a doc
        if self.args.doc_selection == "emb":
            knnresult = self.knnprompt(dial)
            return DocSelection(
                titles = knnresult["titles"],
                docs = knnresult["docs"],
                steps = knnresult["steps"],
                scores = knnresult["scores"],
            )
        elif self.args.doc_selection == "lex":
            lexical_scores = self.bm25d.get_scores(dial.lower().split())
            lexical_doc_idxs = np.argsort(-lexical_scores)[:3].tolist()

            titles = [doc_embeddings[x]["title"] for x in lexical_doc_idxs]
            docs = [doc_embeddings[x]["doc"] for x in lexical_doc_idxs]
            steps = [doc_embeddings[x]["steps"] for x in lexical_doc_idxs]
            scores = lexical_scores[lexical_doc_idxs]

            return DocSelection(
                titles = titles,
                docs = docs,
                steps = steps,
                scores = scores,
            )
        else:
            raise NotImplementedError

    def align_dial(self, dial, turns, doc_out):
        # Stage 2: Given a doc, align the turns in the dial to steps.
        if self.args.step_align == "lex":
            alignments = []
            align_scores = []
            for title, doc, steps, score in zip(
                doc_out.titles, doc_out.docs,
                doc_out.steps, doc_out.scores,
            ):
                scores = np.stack([
                    self.bm25s[title].get_scores(turn.split())
                    for turn in turns
                ]) # shape = turns x steps
                align_preds = scores.argmax(-1)
                align_preds[1:][align_preds[1:] == align_preds[:-1]] = -1
                align_score = scores.max(-1).sum()
                alignments.append(align_preds)
                align_scores.append(align_score)

            return DocStepAlign(
                titles = doc_out.titles,
                docs = doc_out.docs,
                steps = doc_out.steps,
                doc_scores = doc_out.scores,
                alignments = np.stack(alignments),
                align_scores = align_scores,
            )
            #lexical_argmax = np.argmax(lexical_align_scores)
            #lexical_docpred = lexical_alignments[lexical_argmax]
            #lexical_doc = lexical_docpreds[lexical_argmax]
        elif self.args.step_align == "askdoc":
            result = prompt(dict(dial=dial, doc=doc))
            bi_result = np.copy(result)
            # length correction fn
            if len(bi_result) != len(true_labels):
                # need to correct length. should be rare
                if len(bi_result) > len(true_labels):
                    bi_result = bi_result[:len(true_labels)]
                elif len(bi_result) < len(true_labels):
                    new_result = np.full(true_labels.shape, -2)
                    new_result[:len(bi_result)] = bi_result
                    bi_result = new_result
            # filter out repeats fn
            bi_result[1:][bi_result[1:] == bi_result[:-1]] = -1
            import pdb; pdb.set_trace()
            return DocStepAlign()
        else:
            raise NotImplementedError

    def rerank_align(self, alignments):
        # Stage 3: Given complete alignments of dial=>doc, pick the best one
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    from prompting_data import FloDial
    dataset_obj = FloDial()
    get_dataset = dataset_obj.get_docs
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    flo_docs = get_dataset()
    flo_dial, flo_labels = get_dialogues_and_labels()

    bm25d, bm25s = get_bm25(flo_docs)
    import pdb; pdb.set_trace()
