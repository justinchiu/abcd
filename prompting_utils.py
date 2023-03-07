import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any
from rich.progress import track

import openai

from utils.manual_map import subflow_map


EMBEDDING_MODEL = "text-embedding-ada-002"

class Abcd:
    def get_guidelines(self, guidelines):
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
                    "steps": strings,
                })
        return docs

    def get_dataset(self):
        with Path("data/guidelines.json").open("r") as f:
            guidelines = json.load(f)
            docs = self.get_guidelines(guidelines)
            return Dataset.from_list(docs)

    def get_dialogue(self, dial):
        return "\n".join([
            f"Turn {i} {speaker}: {turn}"
            for i, (speaker, turn) in enumerate(dial)
        ])

    def get_speakers(self, dial):
        return [speaker for speaker, turn in dial]

    def get_dialogues_and_labels(self):
        with Path("data/agent_step_annotations.json").open("r") as f:
            agent_labels = json.load(f)["dev"]
        with Path("data/abcd_v1.2.json").open("r") as f:
            data = json.load(f)["dev"]
        return [
            {
                "id": str(x["convo_id"]),
                "dialogue": self.get_dialogue(x["original"]),
                "doc": subflow_map[x["scenario"]["subflow"]],
                "speakers": [speaker for speaker, turn in x["original"]],
                "turns": [f"{speaker}: {turn}" for speaker, turn in x["original"]],
            }
            for x in data
            if str(x["convo_id"]) in agent_labels
        ], agent_labels


class FloDial:
    def get_dataset(self):
        dir = Path("FloDial-dataset/knowledge-sources")
        docs = []
        for path in dir.iterdir():
            with path.open("r") as f:
                doc = json.load(f)
                # WARNING: MIGHT BE NON-CONTIGUOUS FLOWS
                # eg they are MISSING numbers...
                flow = [doc["problem_description"]] + [
                    #doc["nodes"][key]["label"]
                    doc["nodes"][key]["utterance"]
                    for key in sorted(doc["nodes"], key=lambda x: int(x))
                ]
                faqs = [f"{x['q']} {x['a']}" for x in doc["supporting_faqs"]]
                numbered_steps = [
                    f"Step {i}: {x}"
                    for i, x in enumerate(flow)
                ]
                docs.append({
                    "doc": "\n".join(numbered_steps),
                    "title": doc["name"],
                    "steps": flow,
                    "nodes": flow,
                    #"edges": ,
                    "faqs": faqs, # noise
                })
        return Dataset.from_list(docs)

    def get_dialogue(self, dial):
        return "\n".join([
            f"Turn {i} {turn['speaker']}: {turn['utterance']}"
            for i, turn in enumerate(dial)
        ])

    def get_label(self, turn):
        key = "grounded_doc_id"
        if key not in turn:
            return -1

        value = turn[key]
        source, idx = value.split("-")
        return int(idx) if source == "chart" else -1


    def get_dialogues_and_labels(self):
        with Path("FloDial-dataset/dialogs/s-flo.json").open("r") as f:
            data_split = json.load(f)
        with Path("FloDial-dataset/dialogs/dialogs.json").open("r") as f:
            dialogs = json.load(f)

        def add_key(d, k, v):
            d[k] = v
            return d

        train_dialogs = [add_key(dialogs[i], "id", i) for i in data_split["trn"]]
        valid_dialogs = [add_key(dialogs[i], "id", i) for i in data_split["val"]]

        dial_key = "utterences"

        agent_labels = {
            id: [self.get_label(turn) for turn in dial[dial_key]]
            for id, dial in dialogs.items()
        }

        return [
            {
                "id": str(dial["id"]),
                "dialogue": self.get_dialogue(dial[dial_key]),
                "doc": dial["flowchart"],
                "speakers": [turn["speaker"] for turn in dial[dial_key]],
                "turns": [f"{turn['speaker']}: {turn['utterance']}" for turn in dial[dial_key]],
            }
            for dial in valid_dialogs
        ], agent_labels


def embed(x):
    emb = openai.Embedding.create(input=x["doc"], engine=EMBEDDING_MODEL)
    return {"embeddings": [
        np.array(emb['data'][i]['embedding'])
        for i in range(len(emb["data"]))
    ]}




if __name__ == "__main__":
    dataset_obj = Abcd()
    get_dataset = dataset_obj.get_dataset
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    abcd_docs = get_dataset()
    abcd_dial, abcd_labels = get_dialogues_and_labels()

    dataset_obj = FloDial()
    get_dataset = dataset_obj.get_dataset
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    flo_docs = get_dataset()
    flo_dial, flo_labels = get_dialogues_and_labels()

