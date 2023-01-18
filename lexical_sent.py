import evaluate
import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi

from subflow_data import get_abcd_dataset

random.seed(1234)

max_length = 256
#max_length = None
K = 2

# lexical accuracy is better with lowercase
# this is the best setting for doing full-document
val_dataset, processed_docs, doc_sents, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=True, truncate_early=False
)

# build index for each doc
bm25s = [
    BM25Okapi([sent.split(" ") for sent in sents])
    for sents in doc_sents
]

with Path("data/step_annotations.json").open("r") as f:
    all_labels = json.load(f)
labels = all_labels["dev"]

accuracy = evaluate.load("accuracy")
recall = evaluate.load("accuracy")
for e in val_dataset:
    xs = e["xs"]
    str_id = str(e["ids"])
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]
    bm25 = bm25s[idx]

    if str_id not in labels:
        continue

    texts = xs.split("agent")
    for i in range(len(texts)):
        tokenized_query = " ".join(texts[:i+1]).split()[:max_length]
        #tokenized_query = texts[i].split()[:max_length]
        scores = bm25.get_scores(tokenized_query)

        topk = (-scores).argsort()[:K]

        turn_label = labels[str_id][i]
        accuracy.add(reference=turn_label, prediction=scores.argmax())
        recall.add(reference=True, prediction=(topk == turn_label).any())

print(
    "Validation document selection accuracy:",
    accuracy.compute()["accuracy"],
)
print(
    f"Validation document selection recall@{K}:",
    recall.compute()["accuracy"],
)
