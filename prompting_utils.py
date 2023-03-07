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

EMBEDDING_MODEL = "text-embedding-ada-002"

def embed(x):
    emb = openai.Embedding.create(input=x["doc"], engine=EMBEDDING_MODEL)
    return {"embeddings": [
        np.array(emb['data'][i]['embedding'])
        for i in range(len(emb["data"]))
    ]}

def get_bm25(docs):
    bm25d = BM25Okapi([doc["doc"].lower().split() for doc in docs])
    bm25s = [BM25Okapi([step.lower().split() for step in doc["steps"]]) for doc in docs]
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

class AlignmentPrompt(TemplatePrompt):
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

class Aligner:
    def __init__(self, args):
        self.args = args
        if args.doc_select == "",

    def select_docs(self, dial):
        # Stage 1: Align the whole dialogue to a doc
        pass

    def align_dial(self, dial, turns, doc, docsteps):
        # Stage 2: Given a doc, align the turns in the dial to steps.
        pass

    def rerank_align(self, a):
        # Stage 3: Given complete alignments of dial=>doc, pick the best one
        pass


if __name__ == "__main__":
    from prompting_data import FloDial
    dataset_obj = FloDial()
    get_dataset = dataset_obj.get_docs
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    flo_docs = get_dataset()
    flo_dial, flo_labels = get_dialogues_and_labels()

    bm25d, bm25s = get_bm25(flo_docs)
    import pdb; pdb.set_trace()
