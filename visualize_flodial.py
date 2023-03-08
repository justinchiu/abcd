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
step_labels = labels[id]

st.write(f"# Conversation id: {id}")

st.write("## Dialogue")
for t, (turn, step) in enumerate(zip(dialogue, step_labels)):
    st.write(f"(turn {t}, STEP {step}) {document_sents[step]}" if step != -1 else "no step")
    st.write(f"(turn {t}, step {step}) {turn}")
    st.write()


with st.sidebar:
    st.write("## Document steps")
    st.write(f"### {title}")
    for i, sent in enumerate(document_sents):
        st.write(f"{i}: {sent}")
