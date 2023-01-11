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


guidelines, subflow_map = convert_manual(ontology, manual, False)

split = st.selectbox("Data split", ["dev", "test"])
conversations = raw_data[split]
example_idx = st.number_input("Example number", min_value=0, max_value=len(conversations), value=0)
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
    if st.button("Reset alignments?"):
        del all_labels[split][id]
        with datafile.open("w") as f:
            json.dump(all_labels, f)

else:
    with st.form("alignment"):
        st.write("## Dialogue")

        turn_alignments = []
        for t, (speaker, turn) in enumerate(dialogue):
            st.write(f"(turn {t}) {speaker}: {turn}")
            z = st.radio(f"Document step for turn {t}", options=range(len(document_sents)), horizontal=True)
            turn_alignments.append(z)

        submitted = st.form_submit_button("Submit alignment")
        if submitted:
            st.write("Submitted! Save to DB here")
            st.write("Saved alignments")
            st.write(turn_alignments)

            all_labels[split][id] = turn_alignments
            with datafile.open("w") as f:
                json.dump(all_labels, f)


with st.sidebar:
    st.write("## Document steps")
    st.write(f"### Subflow {subflow} ({flow})")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
