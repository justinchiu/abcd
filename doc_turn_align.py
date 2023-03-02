import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
#from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as accscore
import torch

from inference_utils import (
    first,
    monotonic_prediction,
    first_monotonic_arg_max,
)

from subflow_data import get_abcd_dataset

random.seed(1234)

max_length = 256

# lexical accuracy is better with lowercase
# this is the best setting for doing full-document
val_dataset, processed_docs, doc_sents, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=True, truncate_early=False
)

# doc index
bm25d = BM25Okapi([doc.split(" ") for doc in processed_docs])

# build index for each doc
bm25s = [
    BM25Okapi([sent.split(" ") for sent in sents])
    for sents in doc_sents
]

with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]

def get_align(
    doc_score_fn, score_fn, inference_fn,
    name,
    k=32,
):
    all_labels = []
    all_ids = []

    all_preds = []
    agent_preds = []
    doc_preds = []
    # true
    true_labels = []
    agent_true_labels = []
    true_docs = []

    for e in val_dataset:
        xs = e["xs"]
        str_id = str(e["ids"])
        subflow = e["subflows"]
        doc_idx = subflow_map[subflow]

        if str_id not in agent_labels:
            continue

        all_ids.append(str_id)

        doc_scores = doc_score_fn(xs)
        sorted_docs = np.argsort(-doc_scores)

        speakers, turns = list(zip(*e["turns"]))

        # check lexical accuracy of align/not
        labels = np.array(agent_labels[str_id])
        labels = first(labels)

        turn_preds = []
        turn_scores = []
        turn_unaries = []
        for idx in sorted_docs[:k]:
            doc_steps = doc_sents[idx]
            unary = score_fn(turns, idx, str_id)
            preds, score = inference_fn(unary)
            turn_preds.append(preds)
            turn_scores.append(score)
            turn_unaries.append(unary)
        best_doc_idx = np.argmax(turn_scores)
        best_doc = sorted_docs[best_doc_idx]
        unary = turn_unaries[best_doc_idx]

        doc_preds.append(best_doc)
        true_docs.append(doc_idx)

        agent_mask = np.array([s == "agent" for s,_ in e["turns"]])

        this_true_labels = labels[agent_mask]
        true_labels.extend(this_true_labels)

        if doc_idx == best_doc:
            agent_unary = unary[agent_mask]
            agent_argmax, agent_max = inference_fn(agent_unary)

            preds = turn_preds[best_doc_idx][agent_mask]

            all_preds.extend(preds)
            agent_preds.extend(agent_argmax)
        else:
            all_wrong = torch.full(this_true_labels.shape, -2)
            all_preds.extend(all_wrong)
            agent_preds.extend(all_wrong)


    print(f"preds {name}")
    print(accscore(true_labels, all_preds))
    print(f"agent preds {name}")
    print(accscore(true_labels, agent_preds))
    print(f"doc {name}")
    print(accscore(true_docs, doc_preds))

    """
    savepath = f"logging/oracle-sent-{name}.pt"
    torch.save(
        (all_scores, all_labels, all_ids),
        savepath,
    )
    print(f"Saved predictions to {savepath}")
    """

def lexical_score_fn(turns, idx, str_id):
    # lexical
    bm25 = bm25s[idx]
    scores = [
        bm25.get_scores(turn.split())
        for turn in turns
    ]
    return torch.tensor(np.stack(scores))

def lexical_doc_score_fn(dial):
    return bm25d.get_scores(dial.split())

def sbert_score_fn(turns, idx, str_id):
    sent_embs = docsent_embs[idx]
    turn_embs = model.encode(turns)
    return torch.tensor(np.einsum("sh,th->ts", sent_embs, turn_embs))

def model_score_fn(turns, idx, str_id):
    model_idx = model_predictions[-1].index(int(str_id))
    assert model_idx != -1
    return model_predictions[0][model_idx].T


for k in [1, 2, 3, 4, 5, 10, 25, 55]:
    print(k)
    get_align(
        lexical_doc_score_fn, lexical_score_fn, first_monotonic_arg_max,
        name=f"lexical-first-mono={k}",
        k=k,
    )
