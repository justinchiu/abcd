from collections import defaultdict, Counter
from glob import glob
from itertools import combinations, product
import json
from pathlib import Path
import pickle
import random
from rich.progress import track
import os
import torch


def truncate_left(seq, maxlen):
    seqlen = len(seq)
    return seq[seqlen-maxlen:] if seqlen > maxlen else seq

# SIMPLIFIED: do not break up documents. subflow classification only
def preprocess_subflow_abcd_docs(split, tok, answ_tok):
    assert tok.unk_token == answ_tok.unk_token

    fp = Path("eba_data") / f"abcd_{split}_manual.json"
    with fp.open("r") as f:
        docs = json.load(f)

        flow_subflow_to_idx = {}
        paragraphs = []
        for flow, subflow_doc in docs.items():
            for subflow, sentences in subflow_doc.items():
                flow_subflow_to_idx[(flow, subflow)] = len(paragraphs)
                paragraph = " ".join(sentences)
                paragraphs.append(paragraph)

        # different tokenizers for document choice and answer choice
        tokenized_sents = tok(
            paragraphs,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens = False,
        )['input_ids']
        tokenized_supps = answ_tok(
            paragraphs,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens = False,
        )['input_ids']

        return tokenized_sents, tokenized_supps, flow_subflow_to_idx


def preprocess_subflow_abcd(examples, tok, answ_tok, docs, num_distractors=0):
    maxlen = tok.max_len_single_sentence
    xs = [e["x"] for e in examples]
    tokenized_x = tok(xs, truncation=True, return_attention_mask=False)["input_ids"]
    tokenized_sents = [truncate_left(x, maxlen) for x in tokenized_x]
    #tokenized_sents = [truncate_left(x, 64) for x in tokenized_x]
    answ_tokenized_x = answ_tok(xs, truncation=True, return_attention_mask=False)["input_ids"]

    doc_idxs = []

    answers = [e["y"] for e in examples]
    tokenized_y = answ_tok(answers, truncation=True, return_attention_mask=False)[
        "input_ids"
    ]

    # tokenized_sents: List[List[x <unk> d]]
    # tokenized_supps: List[List[x <unk> d]]
    tok_docs, answ_tok_docs, flow_subflow = docs

    num_docs = len(flow_subflow)

    tok_unk_idx = tok.convert_tokens_to_ids(tok.unk_token)
    tok_bos_idx = tok.convert_tokens_to_ids(tok.bos_token)
    tok_eos_idx = tok.convert_tokens_to_ids(tok.eos_token)
    answ_tok_unk_idx = answ_tok.convert_tokens_to_ids(answ_tok.unk_token)
    answ_tok_bos_idx = answ_tok.convert_tokens_to_ids(answ_tok.bos_token)
    answ_tok_eos_idx = answ_tok.convert_tokens_to_ids(answ_tok.eos_token)

    tokenized_supps = []
    labels = []
    #for x, e in track(zip(tokenized_x, examples)):
    for x, e in zip(answ_tokenized_x, examples):
        flow = e["flow"]
        subflow = e["subflow"]

        z_idx = flow_subflow[(flow, subflow)]
        z_idxs = list(range(num_docs))
        if num_distractors > 0:
            # sample distractors
            s = set(range(num_docs))
            s.remove(z_idx)
            distractors = random.sample(list(s), num_distractors)
            z_idxs = [z_idx] + distractors
            labels.append(0) # index of true document is always the first one
        else:
            # distractors = all subflows
            # append label = index in manual
            labels.append(z_idx)

        doc_idxs.append(z_idxs)

        supps = [
            x + [answ_tok_unk_idx] + answ_tok_docs[z] + [answ_tok_eos_idx]
            for z in z_idxs
        ]
        supps = [truncate_left(s, maxlen) for s in supps]
        tokenized_supps.append(supps)

    assert len(tokenized_sents) == len(tokenized_y) == len(tokenized_supps)
    return tokenized_sents, doc_idxs, tokenized_supps, tokenized_y, labels


def prepare_subflow_abcd(tokenizer, answer_tokenizer, split, path="eba_data", num_distractors=0):
    print(f"prepare abcd {split}")

    # save/load/cache docs
    fname = f"cache/abcd_subflow_new_manual_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = preprocess_subflow_abcd_docs(split, tokenizer, answer_tokenizer)
        with open(fname, "wb") as f:
            pickle.dump(docs, f)

    with open(f"{path}/abcd_{split}.json", "r") as fin:
        data = json.load(fin)
    out = []
    labels = []
    sent_labels = []
    for conversation in data:
        xs = conversation["xs"]
        ys = conversation["ys"]
        id = conversation["id"]

        # get docs
        flow = conversation["flow"]
        subflow = conversation["subflow"]

        for x, y in zip(xs, ys):
            out.append(
                {
                    "x": x,
                    "y": y,
                    "flow": flow,
                    "subflow": subflow,
                    "id": id,
                }
            )

    fname = f"cache/abcd_new_tok_{split}_factored_d{num_distractors}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            sents, doc_idxs, supps, answs, labels = pickle.load(f)
    else:
        sents, doc_idxs, supps, answs, labels = preprocess_subflow_abcd(
            out, tokenizer, answer_tokenizer, docs,
            num_distractors,
        )
        with open(fname, "wb") as f:
            pickle.dump((sents, doc_idxs, supps, answs, labels), f)
    # docs[0] = tokenized documents (not answer-tokenized)
    return (sents, docs[0], doc_idxs, supps, answs, labels)


class SubflowAbcdDataset(torch.utils.data.Dataset):
    def __init__(self, paras, docs, doc_idxs, supps, answs, labels):
        self.paras = paras
        self.docs = docs
        self.doc_idxs = doc_idxs
        self.supps = supps
        self.answs = answs
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item["paras"] = self.paras[idx]
        item["docs"] = self.docs
        item["doc_idxs"] = self.doc_idxs[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
