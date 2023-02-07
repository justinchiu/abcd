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

# lexical
argmax_lexical_preds = []
monotonic_lexical_preds = []
first_monotonic_lexical_preds = []
first_argmax_lexical_preds = []

agent_argmax_lexical_preds = []
agent_monotonic_lexical_preds = []
agent_first_monotonic_lexical_preds = []
agent_first_argmax_lexical_preds = []

# sbert
argmax_sbert_preds = []
monotonic_sbert_preds = []
first_monotonic_sbert_preds = []
first_argmax_sbert_preds = []

agent_argmax_sbert_preds = []
agent_monotonic_sbert_preds = []
agent_first_monotonic_sbert_preds = []
agent_first_argmax_sbert_preds = []

# true
true_labels = []
agent_true_labels = []

for e in val_dataset:
    xs = e["xs"]
    str_id = str(e["ids"])
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]
    bm25 = bm25s[idx]

    #if str_id not in labels:
    if str_id not in agent_labels:
        continue

    speakers, turns = list(zip(*e["turns"]))

    # check lexical accuracy of align/not
    labels = np.array(agent_labels[str_id])

    true_labels.append(labels)

    doc_steps = doc_sents[idx]
    scores = [
        bm25.get_scores(turn.split())
        for turn in turns
    ]

    # convert to torch-struct
    unary = torch.tensor(np.stack(scores))

    monotonic_preds = monotonic_prediction(unary)
    argmax_preds = [sy.argmax() for sy in scores]
    first_monotonic_preds = first_monotonic_prediction(unary)
    first_argmax_preds = first_argmax_prediction(unary)

    argmax_lexical_preds.append(argmax_preds)
    monotonic_lexical_preds.append(monotonic_preds)
    first_monotonic_lexical_preds.append(first_monotonic_preds)
    for speaker, label, argmax_pred, monotonic_pred, first_monotonic_pred, first_argmax_pred in zip(
        speakers,
        labels,
        argmax_preds,
        monotonic_preds,
        first_monotonic_preds,
        first_argmax_preds,
    ):
        if speaker == "agent":
            agent_argmax_lexical_preds.append(argmax_pred)
            agent_monotonic_lexical_preds.append(monotonic_pred)
            agent_first_monotonic_lexical_preds.append(first_monotonic_pred)
            agent_first_argmax_lexical_preds.append(first_argmax_pred)
            agent_true_labels.append(label)

    # SBERT
    sent_embs = docsent_embs[idx]
    turn_embs = model.encode(turns)
    turn2sent_scores = np.einsum("sh,th->ts", sent_embs, turn_embs)
    argmax_preds = turn2sent_scores.argmax(-1)
    monotonic_preds = monotonic_prediction(torch.tensor(turn2sent_scores))
    first_monotonic_preds = first_monotonic_prediction(torch.tensor(turn2sent_scores))
    first_argmax_preds = first_argmax_prediction(torch.tensor(turn2sent_scores))

    argmax_sbert_preds.append(argmax_preds)
    monotonic_sbert_preds.append(monotonic_preds)
    first_monotonic_sbert_preds.append(first_monotonic_preds)
    first_argmax_sbert_preds.append(first_argmax_preds)
    for speaker, label, argmax_pred, monotonic_pred, first_monotonic_pred, first_argmax_pred in zip(
        speakers,
        labels,
        argmax_preds,
        monotonic_preds,
        first_monotonic_preds,
        first_argmax_preds,
    ):
        if speaker == "agent":
            agent_argmax_sbert_preds.append(argmax_pred)
            agent_monotonic_sbert_preds.append(monotonic_pred)
            agent_first_monotonic_sbert_preds.append(first_monotonic_pred)
            agent_first_argmax_sbert_preds.append(first_argmax_pred)
    # /SBERT

# flatten non-agent preds
argmax_lexical_preds = np.array([x for xs in argmax_lexical_preds for x in xs])
monotonic_lexical_preds = np.array([x for xs in monotonic_lexical_preds for x in xs])
first_monotonic_lexical_preds = np.array([x for xs in first_monotonic_lexical_preds for x in xs])
first_argmax_lexical_preds = np.array([x for xs in first_argmax_lexical_preds for x in xs])
agent_argmax_lexical_preds = np.array(agent_argmax_lexical_preds)
agent_monotonic_lexical_preds = np.array(agent_monotonic_lexical_preds)
agent_first_monotonic_lexical_preds = np.array(agent_first_monotonic_lexical_preds)
agent_first_argmax_lexical_preds = np.array(agent_first_argmax_lexical_preds)

true_labels = np.array([x for xs in true_labels for x in xs])
agent_true_labels = np.array(agent_true_labels)

argmax_sbert_preds = np.array([x for xs in argmax_sbert_preds for x in xs])
monotonic_sbert_preds = np.array([x for xs in monotonic_sbert_preds for x in xs])
first_monotonic_sbert_preds = np.array([x for xs in first_monotonic_sbert_preds for x in xs])
first_argmax_sbert_preds = np.array([x for xs in first_argmax_sbert_preds for x in xs])
agent_argmax_sbert_preds = np.array(agent_argmax_sbert_preds)
agent_monotonic_sbert_preds = np.array(agent_monotonic_sbert_preds)
agent_first_monotonic_sbert_preds = np.array(agent_first_monotonic_sbert_preds)
agent_first_argmax_sbert_preds = np.array(agent_first_argmax_sbert_preds)

print("argmax lexical")
print(accscore(true_labels, argmax_lexical_preds))
print("agent argmax lexical")
print(accscore(agent_true_labels, agent_argmax_lexical_preds))
print("monotonic lexical")
print(accscore(true_labels, monotonic_lexical_preds))
print("agent monotonic lexical")
print(accscore(agent_true_labels, agent_monotonic_lexical_preds))
print("first monotonic lexical")
print(accscore(true_labels, first_monotonic_lexical_preds))
print("agent first monotonic lexical")
print(accscore(agent_true_labels, agent_first_monotonic_lexical_preds))
print("agent first argmax lexical")
print(accscore(agent_true_labels, agent_first_argmax_lexical_preds))

print("argmax sbert")
print(accscore(true_labels, argmax_sbert_preds))
print("agent argmax sbert")
print(accscore(agent_true_labels, agent_argmax_sbert_preds))
print("monotonic sbert")
print(accscore(true_labels, monotonic_sbert_preds))
print("agent monotonic sbert")
print(accscore(agent_true_labels, agent_monotonic_sbert_preds))
print("first monotonic sbert")
print(accscore(true_labels, first_monotonic_sbert_preds))
print("agent first monotonic sbert")
print(accscore(agent_true_labels, agent_first_monotonic_sbert_preds))
print("agent first argmax sbert")
print(accscore(agent_true_labels, agent_first_argmax_sbert_preds))
