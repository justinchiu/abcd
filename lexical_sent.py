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
with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]

accuracy = evaluate.load("accuracy")
agent_accuracy = evaluate.load("accuracy")
recall = evaluate.load("accuracy")
random_accuracy = evaluate.load("accuracy")
for e in val_dataset:
    xs = e["xs"]
    str_id = str(e["ids"])
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]
    bm25 = bm25s[idx]

    if str_id not in labels:
        continue

    speakers, turns = list(zip(*e["turns"]))
    for i in range(len(turns)):
        if speakers[i] == "agent":
            # POSTERIOR
            turn_idx = i+1
            # PRIOR
            # turn_idx = i
            # prefix, (29%)
            tokenized_query = " ".join(
                f"{s}: {t}"
                for s, t in zip(speakers[:turn_idx], turns[:turn_idx])
            ).split()[-max_length:]
            # only current turn, does worse (16%)
            #tokenized_query = f"{speakers[i]}: {turns[i]}"
            # TODO
            scores = bm25.get_scores(tokenized_query)

            topk = (-scores).argsort()[:K]

            turn_label = labels[str_id][i]
            #accuracy.add(reference=turn_label, prediction=scores.argmax())
            #recall.add(reference=True, prediction=(topk == turn_label).any())
            #random_accuracy.add(reference=turn_label, prediction=random.choice(range(len(doc_sents[idx]))))

            if str_id in agent_labels:
                agent_turn_label = agent_labels[str_id][i]
                if agent_turn_label != -1:
                    agent_accuracy.add(reference=agent_turn_label, prediction=scores.argmax())
                    accuracy.add(reference=turn_label, prediction=scores.argmax())
                    recall.add(reference=True, prediction=(topk == turn_label).any())
                    random_accuracy.add(reference=turn_label, prediction=random.choice(range(len(doc_sents[idx]))))

print(
    "Validation agent step selection lexical accuracy:",
    agent_accuracy.compute()["accuracy"],
)
print(
    "Validation step selection lexical accuracy:",
    accuracy.compute()["accuracy"],
)
print(
    "Validation step selection random accuracy:",
    random_accuracy.compute()["accuracy"],
)
print(
    f"Validation step selection recall@{K}:",
    recall.compute()["accuracy"],
)
