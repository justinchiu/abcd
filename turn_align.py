import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
#from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as accscore
import torch

from inference_utils import monotonic_prediction, first_monotonic_prediction, first_argmax_prediction

from subflow_data import get_abcd_dataset

random.seed(1234)

max_length = 256

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

#model_predictions = torch.load("logging/oracle-sent-model-26-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 ip-True ds-False |step-5000.pt")
path = "logging/oracle-sent-model-28f-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 ip-False ds-False mt-True |step-2000.pt"
model_predictions = torch.load(path)

# sentence_transformers
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#embeddings = model.encode(sentences)
docsent_embs = [
    model.encode(sents)
    for sents in doc_sents
]
# /sentence_transformers

with Path("data/step_annotations.json").open("r") as f:
    all_labels = json.load(f)
contiguous_labels = all_labels["dev"]
with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]

def get_align(score_fn, name):
    all_scores = []
    all_labels = []
    all_ids = []

    all_argmax_preds = []
    all_monotonic_preds = []
    all_first_monotonic_preds = []
    all_first_argmax_preds = []

    agent_argmax_preds = []
    agent_monotonic_preds = []
    agent_first_monotonic_preds = []
    agent_first_argmax_preds = []

    # true
    true_labels = []
    agent_true_labels = []

    for e in val_dataset:
        xs = e["xs"]
        str_id = str(e["ids"])
        subflow = e["subflows"]
        # idx = get_index[subflow]
        idx = subflow_map[subflow]

        #if str_id not in labels:
        if str_id not in agent_labels:
            continue
        if int(str_id) not in model_predictions[-1]:
            continue

        speakers, turns = list(zip(*e["turns"]))

        # check lexical accuracy of align/not
        labels = np.array(agent_labels[str_id])

        doc_steps = doc_sents[idx]

        unary = score_fn(turns, idx, str_id)

        # logging
        all_ids.append(str_id)
        all_labels.append(labels)
        all_scores.append(unary)
        # /logging

        agent_mask = np.array([s == "agent" for s,_ in e["turns"]])

        agent_unary = unary[agent_mask]
        this_true_labels = labels[agent_mask]

        monotonic_preds = monotonic_prediction(agent_unary)
        argmax_preds = agent_unary.argmax(-1)
        first_monotonic_preds = first_monotonic_prediction(agent_unary)
        first_argmax_preds = first_argmax_prediction(agent_unary)

        all_argmax_preds.append(argmax_preds)
        all_monotonic_preds.append(monotonic_preds)
        all_first_monotonic_preds.append(first_monotonic_preds)
        all_first_argmax_preds.append(first_argmax_preds)

        T = unary.shape[0]
        true_labels.append(this_true_labels)

        monotonic_preds = monotonic_prediction(unary)
        argmax_preds = unary.argmax(-1)
        first_monotonic_preds = first_monotonic_prediction(unary)
        first_argmax_preds = first_argmax_prediction(unary)
        for speaker, label, argmax_pred, monotonic_pred, first_monotonic_pred, first_argmax_pred in zip(
            speakers,
            labels,
            argmax_preds,
            monotonic_preds,
            first_monotonic_preds,
            first_argmax_preds,
        ):
            if speaker == "agent":
                agent_argmax_preds.append(argmax_pred)
                agent_monotonic_preds.append(monotonic_pred)
                agent_first_monotonic_preds.append(first_monotonic_pred)
                agent_first_argmax_preds.append(first_argmax_pred)
                agent_true_labels.append(label)

    true_labels = np.array([x for xs in true_labels for x in xs])
    agent_true_labels = np.array(agent_true_labels)

    argmax_preds = np.array([x for xs in all_argmax_preds for x in xs])
    monotonic_preds = np.array([x for xs in all_monotonic_preds for x in xs])
    first_monotonic_preds = np.array([x for xs in all_first_monotonic_preds for x in xs])
    first_argmax_preds = np.array([x for xs in all_first_argmax_preds for x in xs])

    agent_argmax_preds = np.array(agent_argmax_preds)
    agent_monotonic_preds = np.array(agent_monotonic_preds)
    agent_first_monotonic_preds = np.array(agent_first_monotonic_preds)
    agent_first_argmax_preds = np.array(agent_first_argmax_preds)

    print(f"argmax {name}")
    print(accscore(true_labels, argmax_preds))
    print(f"agent argmax {name}")
    print(accscore(agent_true_labels, agent_argmax_preds))
    print(f"monotonic {name}")
    print(accscore(true_labels, monotonic_preds))
    print(f"agent monotonic {name}")
    print(accscore(agent_true_labels, agent_monotonic_preds))
    print(f"first monotonic {name}")
    print(accscore(true_labels, first_monotonic_preds))
    print(f"agent first monotonic {name}")
    print(accscore(agent_true_labels, agent_first_monotonic_preds))
    print(f"first argmax {name}")
    print(accscore(true_labels, first_argmax_preds))
    print(f"agent first argmax {name}")
    print(accscore(agent_true_labels, agent_first_argmax_preds))

    savepath = f"logging/oracle-sent-{name}.pt"
    torch.save(
        (all_scores, all_labels, all_ids),
        savepath,
    )
    print(f"Saved predictions to {savepath}")

def lexical_score_fn(turns, idx, str_id):
    # lexical
    bm25 = bm25s[idx]
    scores = [
        bm25.get_scores(turn.split())
        for turn in turns
    ]
    return torch.tensor(np.stack(scores))

def sbert_score_fn(turns, idx, str_id):
    sent_embs = docsent_embs[idx]
    turn_embs = model.encode(turns)
    return torch.tensor(np.einsum("sh,th->ts", sent_embs, turn_embs))

def model_score_fn(turns, idx, str_id):
    model_idx = model_predictions[-1].index(int(str_id))
    assert model_idx != -1
    return model_predictions[0][model_idx].T


get_align(lexical_score_fn, "lexical")
get_align(sbert_score_fn, "sbert")
get_align(model_score_fn, "model")
