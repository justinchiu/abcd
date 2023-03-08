import json
import numpy as np
import random
from pathlib import Path

import torch

import streamlit as st

from subflow_data import convert_manual, get_abcd_dataset
from inference_utils import first_monotonic_prediction

random.seed(1234)

data_dir = Path("data")
with (data_dir / "abcd_v1.2.json").open("r") as f:
    raw_data = json.load(f)
with (data_dir / "guidelines.json").open("r") as f:
    manual = json.load(f)
with (data_dir / "ontology.json").open("r") as f:
    ontology = json.load(f)

guidelines, subflow_map = convert_manual(ontology, manual, False)

split = "train"
conversations = raw_data[split]

example_num = st.number_input("Example number", min_value=0, max_value=len(conversations), value=0)
example = conversations[example_num]

# viz example
id = str(example["convo_id"])
dialogue = example["original"]
flow = example["scenario"]["flow"]
subflow = example["scenario"]["subflow"]
document_sents = guidelines[subflow_map[subflow]]

st.write(f"# Conversation id: {id}")


st.write("## Dialogue")
for t, (speaker, turn) in enumerate(dialogue):
    string = f"(turn {t}) {speaker}: {turn}"
    st.markdown(string, unsafe_allow_html=True)


with st.sidebar:
    st.write("## Document steps")
    st.write(f"### Subflow {subflow} ({flow})")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
