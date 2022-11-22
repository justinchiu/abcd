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


def len_helper(l):
    l.insert(0, 0)
    for i in range(1, len(l)):
        l[i] += l[i-1]
    return l

def preprocess_abcd_docs(split, tok, answ_tok):
    assert tok.unk_token == answ_tok.unk_token

    fp = Path("eba_data") / f"abcd_{split}_manual.json"
    with fp.open("r") as f:
        docs = json.load(f)

        sents = []
        supps = []
        lengths = []
        slengths = []
        flow_subflow = []

        for flow, subflow_doc in docs.items():
            for subflow, sentences in subflow_doc.items():
                sentences1 = [f"{answ_tok.unk_token} {a}" for a in sentences]
                sentences2 = [
                    f"{answ_tok.unk_token} {a} {tok.unk_token} {b}"
                    for a, b in zip(sentences, sentences[1:])
                ]
                sentences3 = [
                    f"{answ_tok.unk_token} {a} {tok.unk_token} {b} {tok.unk_token} {c}"
                    for a, b, c in zip(sentences, sentences[1:], sentences[2:])
                ]

                # this encoding is used for p(z|x)
                curr_sents = sentences1
                # this encoding is used for p(y|x,z)
                curr_supps = sentences1 + sentences2 + sentences3

                lengths.append(len(curr_sents))
                slengths.append(len(curr_supps))
                flow_subflow.append((flow, subflow))

                sents += curr_sents
                supps += curr_supps

        lengths = len_helper(lengths)
        slengths = len_helper(slengths)

        tokenized_sents = tok(
            sents,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens = False,
        )['input_ids']
        tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
        tokenized_supps = answ_tok(
            supps,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens = False,
        )['input_ids']
        tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]

        assert len(tokenized_sents) == len(tokenized_supps) == len(flow_subflow)

        tok_sents = defaultdict(dict)
        tok_supps = defaultdict(dict)
        for (flow, subflow), toksents, toksupps in zip(flow_subflow, tokenized_sents, tokenized_supps):
            tok_sents[flow][subflow] = toksents
            tok_supps[flow][subflow] = toksupps

        # for debugging: but roberta tokenizer doesnt seem to satisfy round-trip due to spaces around apostrophe
        if False:
            tokenized_sents_flat = [x for xs in tokenized_sents for x in xs]
            out = tok.batch_decode(tokenized_sents_flat, cleanup_tokenization_spaces=False)
            out_toks = [tok.convert_ids_to_tokens(xs) for xs in tokenized_sents_flat]
            for x,y,z in zip(out, sents, out_toks):
                if x != y:
                    print(x)
                    print(y)
                    print(z)
            assert all(x == y for x,y in zip(out, sents))

        return tok_sents, tok_supps


def preprocess_abcd(examples, tok, answ_tok, fixed, max_e, docs):
    sents = []
    supps = []
    lengths = []
    slengths = []
    tokenized_sents = []
    tokenized_supps = []
    ds = []
    num_s = []

    xs = [e["x"] for e in examples]
    tokenized_x = tok(xs, truncation=True, return_attention_mask=False)["input_ids"]

    answers = [e["y"] for e in examples]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)[
        "input_ids"
    ]

    # tokenized_sents: List[List[x <unk> d]]
    # tokenized_supps: List[List[x <unk> d]]
    raise NotImplementedError
    import pdb; pdb.set_trace()

    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s


def prepare_abcd(tokenizer, answer_tokenizer, split, fixed, max_e, path="eba_data"):
    print(f"prepare abcd {split}")

    # save/load/cache docs
    fname = f"cache/abcd_new_manual_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = preprocess_abcd_docs(split, tokenizer, answer_tokenizer)
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

    fname = f"cache/abcd_new_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_abcd(
            out, tokenizer, answer_tokenizer, fixed, max_e, docs,
        )
        with open(fname, "wb") as f:
            pickle.dump((sents, supps, answs, ds, num_s), f)
    return (sents, supps, answs, ds, num_s, sent_labels, labels)


class AbcdDataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        (
            self.sents,
            self.supps,
            self.answs,
            self.ds,
            self.num_s,
            self.sent_labels,
            self.labels,
        ) = everything

    def __getitem__(self, idx):
        item = dict()
        item["sents"] = self.sents[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["ds"] = self.ds[idx]
        item["num_s"] = self.num_s[idx]
        item["sent_labels"] = self.sent_labels[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)


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


def preprocess_subflow_abcd(examples, tok, answ_tok, docs):
    sents = []
    supps = []
    lengths = []
    slengths = []
    tokenized_sents = []
    tokenized_supps = []
    ds = []
    num_s = []

    xs = [e["x"] for e in examples]
    tokenized_x = tok(xs, truncation=True, return_attention_mask=False)["input_ids"]

    answers = [e["y"] for e in examples]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)[
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

    tokenized_sents = []
    tokenized_supps = []
    labels = []
    #for x, e in track(zip(tokenized_x, examples)):
    for x, e in zip(tokenized_x, examples):
        flow = e["flow"]
        subflow = e["subflow"]

        z_idx = flow_subflow[(flow, subflow)]
        # sample distractors
        distractors = random.sample(range(num_docs), 3)
        z_idxs = [z_idx] + distractors
        labels.append(0) # index of true document

        tokenized_sents.append([
            x + [tok_unk_idx] + tok_docs[z] + [tok_eos_idx]
            for z in z_idxs
        ])
        tokenized_supps.append([
            x + [answ_tok_unk_idx] + answ_tok_docs[z] + [answ_tok_eos_idx]
            for z in z_idxs
        ])

        # all docs
        #tokenized_sents.append([x + [tok_unk_idx] + z + [tok_eos_idx] for z in tok_docs])
        #tokenized_supps.append([x + [answ_tok_unk_idx] + z + [answ_tok_eos_idx] for z in answ_tok_docs])

    assert len(tokenized_sents) == len(tokenized_answers) == len(tokenized_supps)
    return tokenized_sents, tokenized_supps, tokenized_answers, labels


def prepare_subflow_abcd(tokenizer, answer_tokenizer, split, path="eba_data"):
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

    fname = f"cache/abcd_new_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            sents, supps, answs, labels = pickle.load(f)
    else:
        sents, supps, answs, labels = preprocess_subflow_abcd(
            out, tokenizer, answer_tokenizer, docs,
        )
        with open(fname, "wb") as f:
            pickle.dump((sents, supps, answs, labels), f)
    return (sents, supps, answs, labels)


class SubflowAbcdDataset(torch.utils.data.Dataset):
    def __init__(self, paras, supps, answs, labels):
        self.paras = paras
        self.supps = supps
        self.answs = answs
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item["paras"] = self.paras[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
