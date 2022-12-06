from collections import defaultdict, Counter
from glob import glob
from itertools import combinations, product
import json
from pathlib import Path
import pickle
import random
from rich.progress import track
import os
import numpy as np
import torch


def truncate_left(seq, maxlen):
    seqlen = len(seq)
    return seq[seqlen - maxlen :] if seqlen > maxlen else seq


# SIMPLIFIED: do not break up documents. subflow classification only
def preprocess_subflow_abcd_docs(split, tok, answ_tok, sep_token="</s>"):
    assert tok.unk_token == answ_tok.unk_token

    fp = Path("eba_data") / f"abcd_{split}_manual.json"
    with fp.open("r") as f:
        docs = json.load(f)

        flow_subflow_to_idx = {}
        subflow_to_idx = {}
        subflows = []
        subflow_first_sentences = []
        for flow, subflow_doc in docs.items():
            for subflow, sentences in subflow_doc.items():
                flow_subflow_to_idx[(flow, subflow)] = len(subflows)
                subflow_to_idx[subflow] = len(subflows)
                # subflow = f" {sep_token} ".join(sentences)
                subflow = " ".join(sentences)
                subflows.append(subflow)
                subflow_first_sentences.append(sentences[0])

        # different tokenizers for document choice and answer choice
        tokenized_subflows = tok(
            subflows,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        answer_tokenized_subflows = answ_tok(
            subflows,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]

        tokenized_subflow_first_sentences = tok(
            subflow_first_sentences,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        answer_tokenized_subflow_first_sentences = tok(
            subflow_first_sentences,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]

        return (
            tokenized_subflows,
            answer_tokenized_subflows,
            tokenized_subflow_first_sentences,
            answer_tokenized_subflow_first_sentences,
            flow_subflow_to_idx,
            subflow_to_idx,
        )


def preprocess_subflow_abcd(examples, tok, answ_tok, subflow_map, sep=" "):
    maxlen = tok.max_len_single_sentence
    xs = [e["x"] for e in examples]
    tokenized_x = tok(xs, truncation=True, return_attention_mask=False)["input_ids"]
    tokenized_x = [truncate_left(x, maxlen) for x in tokenized_x]
    # tokenized_x = [truncate_left(x, 64) for x in tokenized_x]
    answer_tokenized_x = answ_tok(xs, truncation=True, return_attention_mask=False)[
        "input_ids"
    ]
    answer_tokenized_x = [truncate_left(x, maxlen) for x in answer_tokenized_x]

    answers = [e["y"] for e in examples]
    tokenized_y = answ_tok(answers, truncation=True, return_attention_mask=False)[
        "input_ids"
    ]

    num_docs = len(subflow_map)

    doc_labels = [subflow_map[e["subflow"]] for e in examples]
    doc_negatives = [[subflow_map[s] for s in e["subflow_negatives"]] for e in examples]

    assert len(tokenized_x) == len(answer_tokenized_x) == len(tokenized_y)
    return tokenized_x, answer_tokenized_x, doc_labels, doc_negatives, tokenized_y


def preprocess_sents_abcd(data, tokenizer):
    x_to_sent_idxs = []
    sents = []
    tokenized_sents = []
    for conversation in data:
        xs = conversation["xs"]
        ys = conversation["ys"]
        id = conversation["id"]

        start_idx = len(sents)

        raw_ys = np.array(conversation["raw_ys"])
        this_sents = conversation["sents"]

        sents.extend(this_sents)

        this_tokenized_sents = tokenizer(
            this_sents,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        tokenized_sents.extend(this_tokenized_sents)

        idxs = np.arange(len(this_sents)) + start_idx

        is_turns = raw_ys != None
        x_idx = 0
        for i, is_turn in enumerate(is_turns):
            if is_turn:
                x_to_sent_idxs.append(idxs[: i + 1].tolist())
                # string = " </s> ".join([sents[i] for i in idxs[:i+1]])
                string = " ".join([sents[i] for i in idxs[: i + 1]])
                # print(string == xs[x_idx])
                # print(string)
                # print(xs[x_idx])
                assert string == xs[x_idx]
                x_idx += 1

    return x_to_sent_idxs, tokenized_sents


def encode_sents_abcd(sents, tokenizer, encoder):
    device = encoder.device
    bsz = 64
    sentence_vectors = []
    for idx in track(range(0, len(sents), bsz), description="Encode sents"):
        input = tokenizer.pad(
            [{"input_ids": x} for x in sents[idx : idx + bsz]],
            return_tensors="pt",
        ).to(device)
        output = encoder(**input)
        sentence_vectors.extend(output.pooler_output.cpu().numpy())
    return sentence_vectors


def encode_xs_abcd(xs, tokenizer, encoder):
    device = encoder.device
    bsz = 64
    contextual_sentence_vectors = []
    for idx in track(range(0, len(xs), bsz), description="Encode x"):
        input = tokenizer.pad(
            [{"input_ids": x} for x in xs[idx : idx + bsz]],
            return_tensors="pt",
        ).to(device)
        output = encoder(**input)

        # sepmask should be true at <s> or </s>
        sepmask = input.input_ids == 2
        sepmask[:, 0] = True  # first element is <s>
        # there will be an "extra" vector at the end of the sequence
        # might as well use it
        for i in range(sepmask.shape[0]):
            hs = output.last_hidden_state[i][sepmask[i]].cpu().numpy()
            contextual_sentence_vectors.append(hs)
    return contextual_sentence_vectors


def encode_docs_abcd(docs, tokenizer, encoder):
    device = encoder.device
    bsz = 64
    doc_sentence_vectors = []
    for idx in track(range(0, len(docs), bsz), description="Encode docs"):
        input = tokenizer.pad(
            [{"input_ids": x} for x in docs[idx : idx + bsz]],
            return_tensors="pt",
        ).to(device)
        output = encoder(**input)

        # sepmask should be true at <s> or </s>
        sepmask = input.input_ids == 2
        sepmask[:, 0] = True  # first element is <s>
        # there will be an "extra" vector at the end of the sequence
        # might as well use it
        for i in range(sepmask.shape[0]):
            hs = output.last_hidden_state[i][sepmask[i]].cpu().numpy()
            doc_sentence_vectors.append(hs)
    return doc_sentence_vectors


def prepare_subflow_abcd(
    tokenizer,
    answer_tokenizer,
    split,
    path="eba_data",
    encoder=None,
):
    print(f"prepare abcd {split}")

    # save/load/cache docs
    fname = f"cache/abcd_efficient_subflow_manual_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            (
                docs,
                answer_docs,
                doc_first_sentences,
                answer_doc_first_sentences,
                flow_subflow_map,
                subflow_map,
            ) = pickle.load(f)
    else:
        docs_tuple = preprocess_subflow_abcd_docs(split, tokenizer, answer_tokenizer)
        with open(fname, "wb") as f:
            pickle.dump(docs_tuple, f)
        (
            docs,
            answer_docs,
            doc_first_sentences,
            answer_doc_first_sentences,
            flow_subflow_map,
            subflow_map,
        ) = docs_tuple

    with open(f"{path}/hard_negatives_k3.json", "r") as fin:
        negatives = json.load(fin)

    with open(f"{path}/abcd_{split}.json", "r") as fin:
        data = json.load(fin)

    examples = []
    for conversation in data:
        xs = conversation["xs"]
        ys = conversation["ys"]
        id = conversation["id"]

        # get docs
        flow = conversation["flow"]
        subflow = conversation["subflow"]
        subflow_negatives = negatives[subflow]
        # import pdb; pdb.set_trace()

        for turn, (x, y) in enumerate(zip(xs, ys)):
            examples.append(
                {
                    "x": x,
                    "y": y,
                    "flow": flow,
                    "subflow": subflow,
                    "id": id,
                    "turn": turn,
                    "subflow_negatives": subflow_negatives,
                }
            )

    fname = f"cache/abcd_efficient_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            x, answer_x, doc_labels, doc_negatives, y = pickle.load(f)
    else:
        x, answer_x, doc_labels, doc_negatives, y = preprocess_subflow_abcd(
            examples,
            tokenizer,
            answer_tokenizer,
            subflow_map,
        )
        with open(fname, "wb") as f:
            pickle.dump((x, answer_x, doc_labels, doc_negatives, y), f)
    # docs[0] = tokenized documents (not answer-tokenized)

    # map x to sent idxs for efficient encoding
    fname = f"cache/abcd_efficient_x_sents_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            x_to_sent_idxs, sents = pickle.load(f)
    else:
        x_to_sent_idxs, sents = preprocess_sents_abcd(
            data,
            tokenizer,
        )
        with open(fname, "wb") as f:
            pickle.dump((x_to_sent_idxs, sents), f)

    """
    # pre-encode everything with roberta
    fname = f"cache/abcd_efficient_enc_sents_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            enc_sents = pickle.load(f)
    else:
        with torch.no_grad():
            enc_sents = encode_sents_abcd(sents, tokenizer, encoder)
            with open(fname, "wb") as f:
                pickle.dump(enc_sents, f)

    fname = f"cache/abcd_efficient_enc_x_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            enc_x = pickle.load(f)
    else:
        with torch.no_grad():
            enc_x = encode_xs_abcd(x, tokenizer, encoder)
            with open(fname, "wb") as f:
                pickle.dump(enc_x, f)

    fname = f"cache/abcd_efficient_enc_docs_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            enc_docs = pickle.load(f)
    else:
        with torch.no_grad():
            enc_docs = encode_docs_abcd(docs, tokenizer, encoder)
            with open(fname, "wb") as f:
                pickle.dump(enc_docs, f)
    """
    enc_sents, enc_x, enc_docs = None, None, None

    return (
        x,
        answer_x,
        docs,
        answer_docs,
        doc_first_sentences,
        answer_doc_first_sentences,
        doc_labels,
        doc_negatives,
        y,
        x_to_sent_idxs,
        enc_sents,
        enc_x,
        enc_docs,
    )


class SubflowAbcdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        answer_x,
        docs,
        answer_docs,
        doc_first_sentences,
        answer_doc_first_sentences,
        doc_labels,
        doc_negatives,
        answers,
        x_to_sent_idxs,
        enc_sents,
        enc_x,
        enc_docs,
    ):
        self.x = x
        self.answer_x = answer_x
        self.docs = docs
        self.answer_docs = answer_docs
        self.doc_first_sentences = doc_first_sentences
        self.answer_doc_first_sentences = answer_doc_first_sentences
        self.doc_labels = doc_labels
        self.doc_negatives = doc_negatives
        self.answers = answers

        # pre-computed sentence embeddings
        self.x_to_sent_idxs = x_to_sent_idxs
        self.enc_sents = enc_sents
        self.enc_x = enc_x
        self.enc_docs = enc_docs

    def __getitem__(self, idx):
        item = dict()
        item["x"] = self.x[idx]
        item["answer_x"] = self.answer_x[idx]

        item["docs"] = self.docs
        item["answer_docs"] = self.answer_docs

        item["doc_first_sentences"] = self.doc_first_sentences
        item["answer_doc_first_sentences"] = self.answer_doc_first_sentences

        item["doc_label"] = self.doc_labels[idx]
        item["doc_negatives"] = self.doc_negatives[idx]

        item["answer"] = self.answers[idx]

        if self.enc_x is not None:
            item["enc_x"] = self.enc_x[idx]
            item["enc_docs"] = self.enc_docs

            item["sent_idxs"] = self.x_to_sent_idxs[idx]
            item["enc_sents"] = np.concatenate(
                [self.enc_sents[i][None] for i in self.x_to_sent_idxs[idx]],
                0,
            )

        return item

    def __len__(self):
        return len(self.doc_labels)
