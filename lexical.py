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
K = 8
K = 4
K = 16

# lexical accuracy is better with lowercase
# this is the best setting for doing full-document
val_dataset, processed_docs, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=True, truncate_early=False
)
"""
val_dataset, processed_docs, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=True, truncate_early=True
)
"""
"""
val_dataset, processed_docs, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=True, truncate_early=True
)
# val_dataset, processed_docs, subflow_map = get_abcd_dataset("dev", 0, 1, lower=True, truncate_early=True)
# first sentence of doc only does worse.
val_dataset, processed_docs, subflow_map = get_abcd_dataset(
    "dev", 0, 2, lower=True, truncate_early=True
)
# first two sentences does best
"""

# build index
tokenized_corpus = [doc.split(" ")[:max_length] for doc in processed_docs]
bm25 = BM25Okapi(tokenized_corpus)

accuracy = evaluate.load("accuracy")
contrastive_accuracy = evaluate.load("accuracy")
recall = evaluate.load("accuracy")
for e in val_dataset:
    xs = e["xs"]
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]

    # sample without replacement without idx
    r = set(range(len(subflow_map)))
    r.remove(idx)
    distractors = random.sample(list(r), 3)

    tokenized_query = xs.split()[:max_length]
    scores = bm25.get_scores(tokenized_query)
    idxs = distractors + [idx]

    topk = (-scores).argsort()[:K]

    accuracy.add(reference=idx, prediction=scores.argmax())
    contrastive_accuracy.add(reference=3, prediction=scores[idxs].argmax())
    recall.add(reference=True, prediction=(topk == idx).any())

print(
    "Validation document selection accuracy:",
    accuracy.compute()["accuracy"],
)
print(
    "Validation contrastive document selection accuracy:",
    contrastive_accuracy.compute()["accuracy"],
)
print(
    f"Validation document selection recall@{K}:",
    recall.compute()["accuracy"],
)
