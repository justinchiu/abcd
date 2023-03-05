import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any
from rich.progress import track


EMBEDDING_MODEL = "text-embedding-ada-002"


def get_guidelines(guidelines):
    docs = []
    for flow, subflow_dict in guidelines.items():
        for subflow, content in subflow_dict["subflows"].items():
            actions = content["actions"]
            strings = [content["instructions"][0]]
            for step in actions:
                stepstring = step["text"] + " ".join(step["subtext"])
                strings.append(stepstring)
            strings.append(content["instructions"][1])
            numbered_steps = [
                f"Step {i}: {x}"
                for i, x in enumerate(strings)
            ]
            docs.append({
                "doc": "\n".join(numbered_steps),
                "title": subflow,
            })
    return docs

def get_dataset():
    with Path("data/guidelines.json").open("r") as f:
        guidelines = json.load(f)
        docs = get_guidelines(guidelines)
        return Dataset.from_list(docs)

def get_dialogue(dial):
    return "\n".join([
        f"Turn {i} {speaker}: {turn}"
        for i, (speaker, turn) in enumerate(dial)
    ])

def get_speakers(dial):
    return [speaker for speaker, turn in dial]


def get_dialogues_and_labels():
    with Path("data/agent_step_annotations.json").open("r") as f:
        agent_labels = json.load(f)["dev"]
    with Path("data/abcd_v1.2.json").open("r") as f:
        data = json.load(f)["dev"]
    return [
        {
            "id": str(x["convo_id"]),
            "dialogue": get_dialogue(x["original"]),
            "doc": subflow_map[x["scenario"]["subflow"]],
            "speakers": get_speakers(x["original"]),
        }
        for x in data
        if str(x["convo_id"]) in agent_labels
    ], agent_labels

def embed(x):
    emb = openai.Embedding.create(input=x["doc"], engine=EMBEDDING_MODEL)
    return {"embeddings": [
        np.array(emb['data'][i]['embedding'])
        for i in range(len(emb["data"]))
    ]}
