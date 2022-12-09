import evaluate
import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi

from subflow_data import get_abcd_dataset

random.seed(1234)

# lexical accuracy is better with lowercase
val_dataset, processed_docs, subflow_map = get_abcd_dataset("dev", 0, 0, lower=True)

# build index
tokenized_corpus = [doc.split(" ") for doc in processed_docs]
bm25 = BM25Okapi(tokenized_corpus)

accuracy = evaluate.load("accuracy")
contrastive_accuracy = evaluate.load("accuracy")
for e in val_dataset:
    xs = e["xs"]
    subflow = e["subflows"]
    #idx = get_index[subflow]
    idx = subflow_map[subflow]

    # sample without replacement without idx
    r = set(range(len(subflow_map)))
    r.remove(idx)
    distractors = random.sample(list(r), 3)

    tokenized_query = xs.split()
    scores = bm25.get_scores(tokenized_query)
    idxs = distractors + [idx]

    accuracy.add(reference=idx, prediction=scores.argmax())
    contrastive_accuracy.add(reference=3, prediction=scores[idxs].argmax())

print(
    "Validation document selection accuracy:",
    accuracy.compute()["accuracy"],
)
print(
    "Validation contrastive document selection accuracy:",
    contrastive_accuracy.compute()["accuracy"],
)
