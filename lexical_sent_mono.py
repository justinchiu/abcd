import json
import numpy as np
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
#from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as accscore

import torch
from torch_struct import LinearChainCRF

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
nonmono_lexical_preds = []
agent_lexical_preds = []
agent_nonmono_lexical_preds = []

sbert_preds = []
nonmono_sbert_preds = []
agent_sbert_preds = []
agent_nonmono_sbert_preds = []

true_labels = []
nonmono_true_labels = []
agent_true_labels = []
agent_nonmono_true_labels = []
nonnull_true_labels = []

def monotonic_prediction(unary):
    T, Z = unary.shape
    potentials = unary[:,:,None].repeat(1,1,Z)
    # only one starting state
    potentials[0,:,1:] = float("-inf")
    # monotonicity constraint
    transition = torch.tril(torch.ones(Z,Z))
    log_transition = transition.log()
    full_potentials = potentials + log_transition
    crf = LinearChainCRF(full_potentials[None])
    binary_argmax = crf.argmax.detach()
    return binary_argmax.nonzero()[:,2]

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
    clabels = np.array(labels[str_id])
    alabels = np.array(agent_labels[str_id])

    true_labels.append(clabels)
    nonmono_true_labels.append(alabels)

    doc_steps = doc_sents[idx]
    scores = [
        bm25.get_scores(turn.split())
        for turn in turns
    ]

    # convert to torch-struct
    unary = torch.tensor(np.stack(scores))
    preds = monotonic_prediction(unary)

    #preds = [sy.max() > 1 for sy in scores]
    #preds = [sy.max() > 1.5 for sy in scores]
    nonmono_filter_preds = [sy.max() > 1.5 for sy in scores]
    nonmono_preds = [sy.argmax() for sy in scores]

    lexical_preds.append(preds)
    for speaker, clabel, alabel, pred, nmpred in zip(speakers, clabels, alabels, preds, nonmono_preds):
        if speaker == "agent":
            agent_lexical_preds.append(pred)
            agent_nonmono_lexical_preds.append(nmpred)
            agent_true_labels.append(clabel)
            agent_nonmono_true_labels.append(alabel)
            import pdb; pdb.set_trace()

    # SBERT
    sent_embs = docsent_embs[idx]
    turn_embs = model.encode(turns)
    turn2sent_scores = np.einsum("sh,th->ts", sent_embs, turn_embs)
    nonmono_sb_preds = turn2sent_scores.argmax(-1)
    sb_preds = monotonic_prediction(torch.tensor(turn2sent_scores))

    sbert_preds.append(sb_preds)
    nonmono_sbert_preds.append(nonmono_sb_preds)
    for speaker, clabel, alabel, pred, nmpred in zip(speakers, clabels, alabels, sb_preds, nonmono_sb_preds):
        if speaker == "agent":
            agent_sbert_preds.append(pred)
            agent_nonmono_sbert_preds.append(nmpred)
    # /SBERT

# flatten non-agent preds
lexical_preds = np.array([x for xs in lexical_preds for x in xs])
true_labels = np.array([x for xs in true_labels for x in xs])
nonmono_true_labels = np.array([x for xs in nonmono_true_labels for x in xs])
agent_true_labels = np.array(agent_true_labels)
agent_lexical_preds = np.array(agent_lexical_preds)

sbert_preds = np.array([x for xs in sbert_preds for x in xs])
agent_sbert_preds = np.array(agent_sbert_preds)

print("lexical")
print(accscore(true_labels, lexical_preds))
print("agent lexical")
print(accscore(agent_true_labels, agent_lexical_preds))

print("sbert")
print(accscore(true_labels, sbert_preds))
print("agent sbert")
print(accscore(agent_true_labels, agent_sbert_preds))

