from typing import List
from collections import defaultdict, Counter
import json
from pathlib import Path
import torch

from datasets import Dataset

from utils.manual_map import flow_map, subflow_map

Sentence = str


class SubflowAbcdDataset(torch.utils.data.Dataset):
    def __init__(self, xs, docs, doc_labels, doc_negatives, ids, subflows):
        self.xs = xs
        self.docs = docs
        self.doc_labels = doc_labels
        self.doc_negatives = doc_negatives
        self.ids = ids
        self.subflows = subflows

    def __getitem__(self, idx):
        item = dict()
        item["x"] = self.xs[idx]
        item["docs"] = self.docs
        item["doc_label"] = self.doc_labels[idx]
        item["doc_negatives"] = self.doc_negatives[idx]
        item["id"] = self.ids[idx]
        item["subflow"] = self.subflows[idx]

        return item

    def __len__(self):
        return len(self.x)


def truncate_left(seq, maxlen):
    seqlen = len(seq)
    return seq[seqlen - maxlen :] if seqlen > maxlen else seq


def maybe_lower(x, lower):
    return x.lower() if lower else x


def get_subflow_sentences(manual, flow, subflow, lower):
    flow_manual = manual[flow_map[flow]]
    flow_description = flow_manual["description"]

    subflows = flow_manual["subflows"]
    subflow_manual = subflows[subflow_map[subflow]]
    subflow_instructions = subflow_manual["instructions"]
    subflow_actions = subflow_manual["actions"]

    if len(subflow_instructions) != 2:
        import pdb

        pdb.set_trace()

    # use mask token for subflow embedding?
    sentences = [f"{subflow} {maybe_lower(subflow_instructions[0], lower)}"]
    for action in subflow_actions:
        # main subflow instructions
        senttype = maybe_lower(action["type"], lower)
        button = maybe_lower(action["button"], lower)
        text = maybe_lower(action["text"], lower)
        # concatenate subtext into step
        # there are some issues with this. for example in recover username,
        # this will look really weird.
        subtext = [maybe_lower(x, lower) for x in action["subtext"]]
        alltext = " ".join([text] + subtext)
        sentence = (
            f"{senttype}: {alltext}"
            if button == "N/A" or button == "n/a"
            else f"{senttype} - {button}: {alltext}"
        )
        sentences.append(sentence)

    sentences.append(maybe_lower(subflow_instructions[1], lower))

    return sentences


def convert_manual(ontology, manual, lower):
    # get all flow, subflows
    manual_sents = defaultdict(dict)
    for flow, subflows in ontology["intents"]["subflows"].items():
        for subflow in subflows:
            manual_sents[subflow] = get_subflow_sentences(manual, flow, subflow, lower)
    docs = [sents for subflow, sents in manual_sents.items()]
    subflow_map = {
        subflow: idx for idx, (subflow, sents) in enumerate(manual_sents.items())
    }
    return docs, subflow_map


def get_abcd_dataset(
    split,
    num_dialogue_turns,
    num_doc_sents,
    lower=False,
):
    print(f"prepare abcd {split}")

    data_dir = Path("data")
    with (data_dir / "abcd_v1.2.json").open("r") as f:
        raw_data = json.load(f)
    with (data_dir / "guidelines.json").open("r") as f:
        manual = json.load(f)
    with (data_dir / "ontology.json").open("r") as f:
        ontology = json.load(f)
    with open(f"eba_data/hard_negatives_k3.json", "r") as fin:
        negatives = json.load(fin)

    docs, subflow_map = convert_manual(ontology, manual, lower)

    # truncate docs
    if num_doc_sents > 0:
        docs = [doc[:num_doc_sents] for doc in docs]
    docs = [" ".join(doc) for doc in docs]

    xs, doc_labels, doc_negatives = [], [], []
    ids, flows, subflows = [], [], []
    for conversation in raw_data[split]:
        id = conversation["convo_id"]

        # get docs
        flow = conversation["scenario"]["flow"]
        subflow = conversation["scenario"]["subflow"]
        subflow_negatives = negatives[subflow]

        dialogue = [
            f"{speaker}: {maybe_lower(utt, lower)}"
            for speaker, utt in conversation["original"]
        ]
        # perform truncation
        if num_dialogue_turns > 0:
            dialogue = dialogue[:num_dialogue_turns]
        dialogue = " ".join(dialogue)

        xs.append(dialogue)
        doc_labels.append(subflow_map[subflow])
        doc_negatives.append([subflow_map[neg] for neg in negatives[subflow]])

        # non tensorizable
        ids.append(id)
        flows.append(flow)
        subflows.append(subflow)

    dataset = Dataset.from_dict(
        dict(
            xs=xs,
            doc_labels=doc_labels,
            doc_negatives=doc_negatives,
            ids=ids,
            subflows=subflows,
        )
    )
    return dataset, docs, subflow_map
    import pdb

    pdb.set_trace()
    return SubflowAbcdDataset(xs, docs, doc_labels, doc_negatives, ids, subflows)
