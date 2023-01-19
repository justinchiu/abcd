from typing import List
from collections import defaultdict, Counter
import json
from pathlib import Path
import random

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

    xs, doc_labels, doc_negatives = [], [], []
    turns = []
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
        if truncate_early:
            # truncate right before first action
            idx_action = -1
            for idx, turn in enumerate(dialogue):
                if turn.split()[0] == "action:":
                    idx_action = idx
                    break
            dialogue = dialogue[:idx_action]
        flat_dialogue = " ".join(dialogue)

        xs.append(flat_dialogue)
        doc_labels.append(subflow_map[subflow])
        doc_negatives.append([subflow_map[neg] for neg in negatives[subflow]])

        # non tensorizable
        ids.append(id)
        flows.append(flow)
        subflows.append(subflow)
        turns.append([(speaker, maybe_lower(turn, lower)) for speaker, turn in conversation["original"]])

    dataset = Dataset.from_dict(
        dict(
            xs=xs,
            doc_labels=doc_labels,
            doc_negatives=doc_negatives,
            ids=ids,
            flows=flows,
            subflows=subflows,
            turns=turns,
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

    tokenized_docs = tokenizer(
        docs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    ).to(device)

    # pad doc_sents
    def pad_sents(sents, length):
        L = len(sents)
        return sents if L == length else sents + [""] * (length - L)

    doc_num_sents = [len(sents) for sents in doc_sents]
    max_num_sents = max(doc_num_sents)
    padded_doc_sents = [pad_sents(sents, max_num_sents) for sents in doc_sents]

    tokenized_doc_sents = tokenizer(
        [sent for sents in padded_doc_sents for sent in sents],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    ).to(device)

    def convert_to_features(example_batch):
        tokenized_x = tokenizer(
            example_batch["xs"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        x_ids = tokenized_x.input_ids
        x_mask = tokenized_x.attention_mask
        # x_ids[x_ids == padding_id] = -100
        # the -100 is for NLLCriterion, but we manually select non-padding

        # GET TURN INDICES
        # True if conversation starts with agent
        # first token is bos <s>
        agent_start = x_ids[:, 1] == start_agent_id

        # make sure all start-of-turn tokens have a trailing comma,
        # "Gagent:" and "Gcustomer:"
        # If agent does not have trailing colon, it may be due to truncation.
        # Customer is a valid word, so may not have a trailing colon,
        # eg "welcome to customer service".
        colon_id = tokenizer.convert_tokens_to_ids(":")
        is_next_token_colon = torch.zeros_like(x_ids, dtype=bool)
        is_next_token_colon[:, :-1] = x_ids[:, 1:] == colon_id

        customer_turn = (x_ids == customer_id) & is_next_token_colon
        agent_turn = (x_ids == agent_id) & is_next_token_colon
        agent_turn[:, 1] = agent_start
        customer_turn[:, 1] = ~agent_start
        action_turn = (x_ids == action_id) & is_next_token_colon
        turn_locations = customer_turn | agent_turn | action_turn

        speakers = [
            [speaker for speaker, turn in conv]
            for conv in example_batch["turns"]
        ]
        is_agent_turn = [
            [speaker == "agent" for speaker, turn in conv]
            for conv in example_batch["turns"]
        ]

        #max_turns = max([len(x) for x in is_agent_turn])
        max_turns = 128
        # CONSTANT

        def pad(xs, length, val):
            if len(xs) < length:
                return xs + [val] * (length - len(xs))
            else:
                return xs

        padded_is_agent_turn = [
            pad([speaker == "agent" for speaker, turn in conv], max_turns, False)
            for conv in example_batch["turns"]
        ]

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
            "x_ids": x_ids,
            "x_mask": x_mask,
            "ids": example_batch["ids"],
            "doc_labels": doc_labels,
            "doc_idxs": doc_idxs, # ONLY FOR ORACLE Z*
            "agent_turn_mask": agent_turn,
            "customer_turn_mask": customer_turn,
            "action_turn_mask": action_turn,
            "is_agent_turn": padded_is_agent_turn,
        }
        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            "x_ids",
            "x_mask",
            "ids",
            "doc_labels",
            "doc_idxs", # ONLY FOR ORACLE Z*
            "agent_turn_mask",
            "customer_turn_mask",
            "action_turn_mask",
            "is_agent_turn",
        ]
        dataset.set_format(type="torch", columns=columns, output_all_columns=False)
        return dataset

    train = process_dataset(train_dataset)
    valid = process_dataset(valid_dataset)

    # subsampling
    subsampled_dataloader = None
    if k > 0:
        if subsample == "flow":
            map_idxs = {
                flow: [i for i, ex in enumerate(train_dataset) if ex["flows"] == flow]
                for flow in flow_map.keys()
            }
        elif subsample == "subflow":
            map_idxs = {
                subflow: [i for i, ex in enumerate(train_dataset) if ex["subflows"] == subflow]
                for subflow in subflow_map.keys()
            }
        else:
            raise ValueError(f"subsample must be flow or subflow, instead was {subsample}")

        subsampled_list = [
            train_dataset[i]
            for idxs in map_idxs.values()
            for i in random.sample(idxs, k)
        ]
        subsampled_dataset = process_dataset(Dataset.from_list(subsampled_list))
        subsampled_dataloader = DataLoader(
            subsampled_dataset,
            batch_size=args.subsampled_batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            pin_memory_device=str(device),
        )
    # / subsampling

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
        train_dataloader, valid_dataloader, subsampled_dataloader,
        tokenized_docs, tokenized_doc_sents, doc_num_sents,
    )

