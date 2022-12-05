import evaluate
import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi

random.seed(1234)


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)

def save_json(x, path):
    with Path(path).open("w") as f:
        return json.dump(x, f)


train_data = load_json("eba_data/abcd_train.json")
train_manual = load_json("eba_data/abcd_train_manual.json")
val_data = load_json("eba_data/abcd_val.json")
val_manual = load_json("eba_data/abcd_val_manual.json")

# flatten manual
f_s_docs = [
    (flow, subflow, " ".join(sents))
    for flow, subflow_man in val_manual.items()
    for subflow, sents in subflow_man.items()
]
docs = [
    doc for flow, subflow, doc in f_s_docs
]
fs = [
    (flow, subflow) for flow, subflow, doc in f_s_docs
]
get_index = {x: i for i, x in enumerate(fs)}

# build index
tokenized_corpus = [doc.split(" ") for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

for k in [3, 7, 11]:
    negatives = {}
    for i, (flow, subflow, doc) in enumerate(f_s_docs):
        tokenized_query = doc.split()
        scores = bm25.get_scores(tokenized_query)
        topk = set(np.argpartition(-scores, k+1)[:k+1])
        negatives[subflow] = [fs[x][1] for x in topk - {i}]

    save_json(negatives, f"eba_data/hard_negatives_k{k}.json")
