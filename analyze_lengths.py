import evaluate
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import matplotlib.pyplot as plt
import seaborn as sns

from subflow_data import get_abcd_dataset

random.seed(1234)

max_length = 256
#max_length = None
K = 2

# lexical accuracy is better with lowercase
# this is the best setting for doing full-document
train_dataset, processed_docs, doc_sents, subflow_map = get_abcd_dataset(
    "train", 0, 0, lower=False, truncate_early=False
)
val_dataset, processed_docs, doc_sents, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=False, truncate_early=False
)

with Path("data/step_annotations.json").open("r") as f:
    all_labels = json.load(f)
labels = all_labels["dev"]
with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

agent_lens = []
turn_lens = []
num_turns = []
num_agent_turns = []
for e in train_dataset:
    xs = e["xs"]
    str_id = str(e["ids"])
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]

    speakers, turns = list(zip(*e["turns"]))

    for s,t in zip(speakers, turns):
        string = f"{s}: {t}"
        tokstring = tokenizer.tokenize(string)
        turn_lens.append(len(tokstring))
        if s == "agent":
            agent_lens.append(len(tokstring))

    num_turns.append(len(speakers))
    num_agent_turns.append(len([x for x in speakers if x == "agent"]))

hists = [agent_lens, turn_lens, num_turns, num_agent_turns]
names = ["agent_lens", "turn_lens", "num_turns", "num_agent_turns"]

print(f"{np.min(agent_lens)} {np.mean(agent_lens)} {np.max(agent_lens)}")
print(f"{np.min(turn_lens)} {np.mean(turn_lens)} {np.max(turn_lens)}")
print(f"{np.min(num_agent_turns)} {np.mean(num_agent_turns)} {np.max(num_agent_turns)}")
print(f"{np.min(num_turns)} {np.mean(num_turns)} {np.max(num_turns)}")

for name, data in zip(names, hists):
    sns.histplot(data)
    plt.savefig(f"plots/{name}.png")
    plt.close("all")


