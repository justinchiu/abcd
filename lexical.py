import evaluate
import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)

train_data = load_json("eba_data/abcd_train.json")
train_manual = load_json("eba_data/abcd_train_manual.json")
val_data = load_json("eba_data/abcd_val.json")
val_manual = load_json("eba_data/abcd_val_manual.json")

# flatten manual
docs = [
    " ".join(sents)
    for flow, subflow_man in val_manual.items()
    for subflow, sents in subflow_man.items()
]
fs = [
    (flow, subflow)
    for flow, subflow_man in val_manual.items()
    for subflow, sents in subflow_man.items()
]
get_index = {x:i for i,x in enumerate(fs)}

# build index
tokenized_corpus = [doc.split(" ") for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

accuracy = evaluate.load("accuracy")
for e in val_data:
    xs = e["xs"]
    flow = e["flow"]
    subflow = e["subflow"]
    idx = get_index[(flow, subflow)]
    predictions = []
    for x in xs:
        tokenized_query = x.split()
        scores = bm25.get_scores(tokenized_query)
        predictions.append(scores.argmax())
    accuracy.add_batch(references=[idx]*len(xs), predictions=predictions)
print(
    "Validation document selection accuracy:",
    accuracy.compute()["accuracy"],
)
