import json
import numpy as np
import random
from pathlib import Path

import torch

import streamlit as st

from subflow_data import convert_manual, get_abcd_dataset
from inference_utils import first_monotonic_prediction

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
#path = "logging/oracle-sent-model-28f-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 ip-False ds-False mt-True |step-250.pt"
path = "logging/oracle-sent-model-213-bart-base lr-1e-05 bs-16 dt-0 ds-0 ml-512 s-subflow sk-0 ss-250 sp-0 ip-False ds-False mt-True dta-True |step-4000.pt"
preds, labels, ids = torch.load(path)

id2pred = {id: pred for id, pred in zip(ids, preds)}

lexicalpath = "logging/oracle-sent-lexical.pt"
lexicalpreds, lexlabels, lexids = torch.load(lexicalpath)
lexids = [int(id) for id in lexids]

lexid2pred = {id: pred for id, pred in zip(lexids, lexicalpreds)}


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


if id in agent_labels[split]:
    print("true")
    print(agent_labels[split][id])

    modelunary = id2pred[int(id)].T
    modelargmax = modelunary.argmax(1) 
    print("model argmax")
    print(modelargmax)
    modelpreds = first_monotonic_prediction(modelunary)
    print("model monotonic")
    print(modelpreds)

    lexunary = lexid2pred[int(id)]
    lexargmax = lexunary.argmax(1) 
    print("lex argmax")
    print(lexargmax)
    lexpreds = first_monotonic_prediction(lexunary)
    print("lex monotonic")
    print(lexpreds)

    st.write("## Dialogue")
    for t, ((speaker, turn), agent_step) in enumerate(zip(dialogue, agent_labels[split][id])):
        blackstring = f"(turn {t}, step {agent_step}, model pred {modelpreds[t]}, lexpred {lexpreds[t]}) {speaker}: {turn}"
        colorstring = f"<p style='color:Blue'>{blackstring}</p>"
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
