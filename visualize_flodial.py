import json
import numpy as np
import random
from pathlib import Path

import torch

import streamlit as st

from prompting_data import FloDial



random.seed(1234)

dataobj = FloDial()
docs = dataobj.get_docs()
dials, labels = dataobj.get_dialogues_and_labels(split="train")


example_num = st.number_input("Example number", min_value=0, max_value=len(dials), value=0)
example = dials[example_num]

# viz example
id = str(example["id"])
dialogue = example["turns"]
title = example["doc"]
document_sents = [d["steps"] for d in docs if d["title"] == title][0]

st.write(f"# Conversation id: {id}")


st.write("## Dialogue")
for t, turn in enumerate(dialogue):
    string = f"(turn {t}) {turn}"
    st.markdown(string, unsafe_allow_html=True)


with st.sidebar:
    st.write("## Document steps")
    st.write(f"### {title}")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
