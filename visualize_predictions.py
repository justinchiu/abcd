import json
import numpy as np
import random
from pathlib import Path

import streamlit as st

from subflow_data import convert_manual, get_abcd_dataset

random.seed(1234)

"""
# this data is too processed. use the raw data instead
val_dataset, processed_docs, val_doc_sents, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=False, truncate_early=False
)
test_dataset, processed_docs, test_doc_sents, subflow_map = get_abcd_dataset(
    "test", 0, 0, lower=False, truncate_early=False
)
"""
data_dir = Path("data")
with (data_dir / "abcd_v1.2.json").open("r") as f:
    raw_data = json.load(f)
with (data_dir / "guidelines.json").open("r") as f:
    manual = json.load(f)
with (data_dir / "ontology.json").open("r") as f:
    ontology = json.load(f)

# annotations file. load up here and just write to it on save
datafile = data_dir / "step_annotations.json"
all_labels = None
if datafile.exists():
    print("Loading datafile")
    with datafile.open("r") as f:
        all_labels = json.load(f)
else:
    all_labels = {"dev": {}, "test": {}}

path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
#path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-512 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
preds, labels, ids = torch.load(path)

guidelines, subflow_map = convert_manual(ontology, manual, False)

example_idx = st.number_input("Example number", min_value=0, max_value=len(ids), value=0)

import pdb; pdb.set_trace()
example = conversations[example_idx]

# viz example
id = str(example["convo_id"])
dialogue = example["original"]
flow = example["scenario"]["flow"]
subflow = example["scenario"]["subflow"]
document_sents = guidelines[subflow_map[subflow]]

st.write(f"# Conversation id: {id}")

if id in all_labels[split]:
    st.write("## Alignments already processed")
    st.write("## Dialogue")
    for t, ((speaker, turn), step) in enumerate(zip(dialogue, all_labels[split][id])):
        st.write(f"(turn {t}, step {step}) {speaker}: {turn}")

else:
    st.write("Unannotated")
    with st.form("alignment"):
        st.write("## Dialogue")

        turn_alignments = []
        for t, (speaker, turn) in enumerate(dialogue):
            st.write(f"(turn {t}) {speaker}: {turn}")

with st.sidebar:
    st.write("## Document steps")
    st.write(f"### Subflow {subflow} ({flow})")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
