# Similar to subflow_data.py,
# but assuming all turns are modeled independently
# this is much slower than batching by conversation, due to
# redundant computation on the encoder side (steps).

from typing import List
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import numpy as np

import pdb

from datasets import Dataset

import torch
from torch.utils.data import DataLoader

from utils.manual_map import flow_map, subflow_map


start_customer_token = "custom"
customer_token = "Ġcustomer"
start_agent_token = "agent"
agent_token = "Ġagent"
action_token = "Ġaction"


Sentence = str


speaker_map = {
    "agent": 0,
    "customer": 1,
    "action": 2,
}

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
    truncate_early=False,
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

    doc_sents, subflow_map = convert_manual(ontology, manual, lower)

    # truncate docs
    if num_doc_sents > 0:
        docs = [sents[:num_doc_sents] for sents in doc_sents]
    docs = [" ".join(sents) for sents in doc_sents]

    turns = []
    turn_nums = []
    speakers = []
    doc_labels, doc_negatives = [], []
    ids, flows, subflows = [], [], []
    for conversation in raw_data[split]:
        id = conversation["convo_id"]

        # get docs
        flow = conversation["scenario"]["flow"]
        subflow = conversation["scenario"]["subflow"]
        subflow_negatives = negatives[subflow]

        for turn_num, (speaker, turn) in enumerate(conversation["original"]):
            turns.append(f"{speaker}: {maybe_lower(turn, lower)}")
            turn_nums.append(turn_num)
            speakers.append(speaker_map[speaker])
            doc_labels.append(subflow_map[subflow])
            doc_negatives.append([subflow_map[neg] for neg in negatives[subflow]])
            ids.append(id)
            flows.append(flow)
            subflows.append(subflow)

    dataset = Dataset.from_dict(
        dict(
            turns=turns,
            turn_nums=turn_nums,
            speakers=speakers,
            doc_labels=doc_labels,
            doc_negatives=doc_negatives,
            ids=ids,
            flows=flows,
            subflows=subflows,
        )
    )
    return dataset, docs, doc_sents, subflow_map


def prepare_dataloader(tokenizer, args, device, subsample="subflow", k=1):
    train_dataset, docs, doc_sents, subflow_map = get_abcd_dataset(
        "train",
        args.num_dialogue_turns,
        args.num_doc_sents,
        truncate_early=args.truncate_early,
    )
    valid_dataset, _, _, _ = get_abcd_dataset(
        "dev",
        args.num_dialogue_turns,
        args.num_doc_sents,
        truncate_early=args.truncate_early,
    )

    num_docs = len(docs)

    padding_id = tokenizer.pad_token_id

    (
        start_customer_id,
        customer_id,
        start_agent_id,
        agent_id,
        action_id,
    ) = tokenizer.convert_tokens_to_ids(
        [
            start_customer_token,
            customer_token,
            start_agent_token,
            agent_token,
            action_token,
        ]
    )

    # pad doc_sents
    def pad_sents(sents, length):
        L = len(sents)
        return sents if L == length else sents + [""] * (length - L)

    doc_num_sents = [len(sents) for sents in doc_sents]
    max_num_sents = max(doc_num_sents) + 1 if args.dummy_step else max(doc_num_sents)
    padded_doc_sents = [pad_sents(sents, max_num_sents) for sents in doc_sents]

    tokenized_doc_sents = tokenizer(
        [sent for sents in padded_doc_sents for sent in sents],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_step_length,
    ).to(device)

    def convert_to_features(example_batch):
        bsz = len(example_batch["turns"])
        tokenized_turns = tokenizer(
            example_batch["turns"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        turn_ids = tokenized_turns.input_ids
        turn_mask = tokenized_turns.attention_mask

        doc_labels = example_batch["doc_labels"]
        doc_negatives = example_batch["doc_negatives"]

        # ONLY FOR ORACLE Z*
        random_negatives = []
        s = set(range(num_docs))
        for i, (label, negs) in enumerate(zip(doc_labels, doc_negatives)):
            this_s = s.difference(set(negs + [label]))
            negatives = random.sample(list(this_s), args.num_negatives)
            random_negatives.append(list(negatives))

        doc_idxs = [
            rnegs + negs + [label]
            for rnegs, negs, label in zip(random_negatives, doc_negatives, doc_labels)
        ]
        # / ORACLE Z*

        encodings = {
            "ids": example_batch["ids"],
            "doc_labels": doc_labels,
            "doc_idxs": doc_idxs, # ONLY FOR ORACLE Z*
            "turn_ids": turn_ids,
            "turn_mask": turn_mask,
            "turn_nums": example_batch["turn_nums"],
            "speakers": example_batch["speakers"],
        }
        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            "ids",
            "doc_labels",
            "doc_idxs", # ONLY FOR ORACLE Z*
            "turn_ids",
            "turn_mask",
            "turn_nums",
            "speakers",
        ]
        dataset.set_format(type="torch", columns=columns, output_all_columns=False)
        return dataset

    train = process_dataset(train_dataset)
    valid = process_dataset(valid_dataset)

    train_dataloader = DataLoader(
        train,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(device),
    )

    valid_dataloader = DataLoader(
        valid,
        batch_size=args.eval_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(device),
    )

    return (
        train_dataloader, valid_dataloader, None,
        list(range(num_docs)), tokenized_doc_sents, doc_num_sents,
    )

