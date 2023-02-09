import json
import numpy as np
import random
from pathlib import Path

import torch

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

agent_datafile = data_dir / "agent_step_annotations.json"
agent_labels = None
if datafile.exists():
    print("Loading datafile")
    with agent_datafile.open("r") as f:
        agent_labels = json.load(f)
else:
    agent_labels = {"dev": {}, "test": {}}

#path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
#path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-512 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
path = "logging/oracle-sent-model-28f-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 ip-False ds-False mt-True |step-250.pt"
preds, labels, ids = torch.load(path)

guidelines, subflow_map = convert_manual(ontology, manual, False)

example_num = st.number_input("Example number", min_value=0, max_value=len(ids), value=0)

split = "dev"
id = ids[example_num]
conversations = raw_data[split]
example = [x for x in conversations if x["convo_id"] == id][0]

# viz example
id = str(example["convo_id"])
dialogue = example["original"]
flow = example["scenario"]["flow"]
subflow = example["scenario"]["subflow"]
document_sents = guidelines[subflow_map[subflow]]

st.write(f"# Conversation id: {id}")

print(preds[example_num].argmax(0))
print(labels[example_num])

if id in all_labels[split] and id in agent_labels[split]:
    st.write("## Dialogue")
    for t, ((speaker, turn), step, agent_step) in enumerate(zip(dialogue, all_labels[split][id], agent_labels[split][id])):
        blackstring = f"(turn {t}, step {step}, astep {agent_step}, pred {preds[example_num][:,t].argmax(0)}) {speaker}: {turn}"
        colorstring = f"<p style='color:Blue'>(turn {t}, step {step}, astep {agent_step}, pred {preds[example_num][:,t].argmax(0)}) {speaker}: {turn}</p>"
        string = blackstring if agent_step == -1 or speaker != "agent" else colorstring
        st.markdown(string, unsafe_allow_html=True)

else:
    st.write("Unannotated and therefore no labels")
    st.write("May not be annotated in all or agent labels")

with st.sidebar:
    st.write("## Document steps")
    st.write(f"### Subflow {subflow} ({flow})")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
