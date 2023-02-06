import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.metrics import precision_recall_fscore_support as prfs

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
labels = all_labels["dev"]
with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]


lexical_preds = []
agent_lexical_preds = []
sbert_preds = []
agent_sbert_preds = []

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
    if str_id not in labels or str_id not in agent_labels:
        continue

    speakers, turns = list(zip(*e["turns"]))

    # check lexical accuracy of align/not
    alabels = agent_labels[str_id]
    true_labels.append(np.array(alabels) != -1)

    doc_steps = doc_sents[idx]
    scores = [
        bm25.get_scores(turn.split())
        for turn in turns
    ]
    #preds = [sy.max() > 1 for sy in scores]
    preds = [sy.max() > 1.5 for sy in scores]

    lexical_preds.append(preds)
    for speaker, alabel, pred in zip(speakers, alabels, preds):
        if speaker == "agent":
            agent_lexical_preds.append(pred)
            agent_true_labels.append(alabel != -1)

    # SBERT
    sent_embs = docsent_embs[idx]
    turn_embs = model.encode(turns)
    turn2sent_scores = np.einsum("sh,th->ts", sent_embs, turn_embs)
    #sbert_pred = (turn2sent_scores > 0.25).any(-1)
    sbert_pred = (turn2sent_scores > 0.18).any(-1)

    sbert_preds.append(sbert_pred)
    for speaker, alabel, pred in zip(speakers, alabels, sbert_pred):
        if speaker == "agent":
            agent_sbert_preds.append(pred)
    # /SBERT

# flatten non-agent preds
lexical_preds = np.array([x for xs in lexical_preds for x in xs])
true_labels = np.array([x for xs in true_labels for x in xs])
agent_true_labels = np.array(agent_true_labels)
agent_lexical_preds = np.array(agent_lexical_preds)

sbert_preds = np.array([x for xs in sbert_preds for x in xs])
agent_sbert_preds = np.array(agent_sbert_preds)

print("lexical")
print(prfs(true_labels, lexical_preds))
print(f"num positive pred: {lexical_preds.sum()} / {len(lexical_preds)}")
print("agent lexical")
print(prfs(agent_true_labels, agent_lexical_preds))
print(f"num positive pred: {agent_lexical_preds.sum()} / {len(agent_lexical_preds)}")

print("sbert")
print(prfs(true_labels, sbert_preds))
print(f"num positive pred: {sbert_preds.sum()} / {len(sbert_preds)}")
print("agent sbert")
print(prfs(agent_true_labels, agent_sbert_preds))
print(f"num positive pred: {agent_sbert_preds.sum()} / {len(agent_sbert_preds)}")

"""
# SBERT
print(
    f"Validation sbert binary accuracy:",
    sbert_binary_accuracy.compute()["accuracy"],
)
print(
    f"Validation sbert agent binary accuracy:",
    sbert_agent_binary_accuracy.compute()["accuracy"],
)
# /SBERT
"""

