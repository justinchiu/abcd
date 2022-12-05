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


accuracy = evaluate.load("accuracy")
contrastive_accuracy = evaluate.load("accuracy")
for e in val_data:
    xs = e["xs"]
    flow = e["flow"]
    subflow = e["subflow"]
    idx = get_index[(flow, subflow)]

    # sample without replacement without idx
    r = set(range(len(fs)))
    r.remove(idx)
    distractors = random.sample(list(r), 3)

    predictions = []
    contrastive_predictions = []
    for x in xs:
        tokenized_query = x.split()
        scores = bm25.get_scores(tokenized_query)
        predictions.append(scores.argmax())
        idxs = distractors + [idx]
        contrastive_predictions.append(scores[idxs].argmax())
    accuracy.add_batch(references=[idx] * len(xs), predictions=predictions)
    contrastive_accuracy.add_batch(
        references=[3] * len(xs),
        predictions=contrastive_predictions,
    )
print(
    "Validation document selection accuracy:",
    accuracy.compute()["accuracy"],
)
print(
    "Validation contrastive document selection accuracy:",
    contrastive_accuracy.compute()["accuracy"],
)
