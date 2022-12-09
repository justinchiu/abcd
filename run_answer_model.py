import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
import numpy as np
import random
import wandb

import pdb

from datasets import load_metric
from torch.utils.data.sampler import BatchSampler, RandomSampler


# from dataset import prepare_simplified, SimplifiedHotpotQADataset
# from eba_subflow_dataset import prepare_subflow_abcd, SubflowAbcdDataset
# from eba_subflow_factored_dataset import prepare_subflow_abcd, SubflowAbcdDataset
from subflow_data import get_abcd_dataset, SubflowAbcdDataset
from rich.progress import track
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import set_seed
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn

from eba_utils import prepare_optim_and_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 555
set_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--num_dialogue_turns", default=0, type=int)
    parser.add_argument("--num_doc_sents", default=0, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_negatives", default=0, type=int)
    parser.add_argument("--num_hard_negatives", default=0, type=int)
    parser.add_argument(
        "--batch_size", "-b", default=1, type=int, help="batch size per gpu."
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="eval batch size per gpu."
    )
    parser.add_argument(
        "--eval_steps",
        default=5000,
        type=int,
        help="number of steps between each evaluation.",
    )
    parser.add_argument(
        "--full_eval_steps",
        default=50000,
        type=int,
        help="number of steps between each FULL/expensive evaluation.",
    )
    parser.add_argument(
        "--epoch",
        "-epoch",
        default=5,
        type=int,
        help="The number of epochs for fine-tuning.",
    )
    parser.add_argument(
        "--model_dir",
        default="roberta-large",
        type=str,
        help="The directory where the pretrained model will be loaded.",
    )
    parser.add_argument(
        "--answer_model_dir",
        default="facebook/bart-base",
        type=str,
        help="The directory where the pretrained model will be loaded.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--output_model_dir",
        default="./saved_models",
        type=str,
        help="The directory where the pretrained model will be saved.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio in the lr scheduler.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()
    return args


def prepare_dataloader(tokenizer, args):
    train_dataset, docs, subflow_map = get_abcd_dataset(
        tokenizer, "train", args.num_dialogue_turns, args.num_doc_sents
    )
    valid_dataset, _, _ = get_abcd_dataset(
        tokenizer, "dev", args.num_dialogue_turns, args.num_doc_sents
    )

    num_docs = len(docs)

    padding_token = tokenizer.pad_token_id

    tokenized_docs = tokenizer(
        docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(device)

    def convert_to_features(example_batch):
        tokenized_x = tokenizer(
            example_batch["xs"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        x_ids = tokenized_x.input_ids
        x_mask = tokenized_x.attention_mask
        x_ids[x_ids == padding_token] = -100

        doc_labels = example_batch["doc_labels"]
        doc_negatives = example_batch["doc_negatives"]

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

        encodings = {
            "x_ids": x_ids,
            "x_mask": x_mask,
            "ids": example_batch["ids"],
            "doc_idxs": doc_idxs,
            "doc_labels": doc_labels,
        }
        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            "x_ids",
            "x_mask",
            "doc_idxs",
            "doc_labels",
        ]
        dataset.set_format(type="torch", columns=columns, output_all_columns=False)
        return dataset

    train = process_dataset(train_dataset)
    valid = process_dataset(valid_dataset)

    # is this a bug in BatchSampler?
    def collate_fn(batch):
        return {k: v.to(device) for k, v in  batch[0].items()}

    #sampler = BatchSampler(
    #    RandomSampler(train), batch_size=args.batch_size, drop_last=False
    #)
    train_dataloader = DataLoader(
        train,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        pin_memory_device=str(device),
    )

    #sampler = BatchSampler(valid, batch_size=args.eval_batch_size, drop_last=False)
    valid_dataloader = DataLoader(
        valid,
        batch_size=args.eval_batch_size,
        drop_last=False,
        shuffle = False,
        pin_memory=True,
        pin_memory_device=str(device),
    )
    # valid_dataloader = DataLoader(valid, sampler=sampler)

    #for batch in train_dataloader:
    #    print(batch)
    #    pdb.set_trace()

    return train_dataloader, valid_dataloader, tokenized_docs


def run_model(batch, docs, model):
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)
    z_idxs = batch["doc_idxs"].to(device)

    bsz, x_len = x_ids.shape

    z_ids = docs.input_ids
    z_mask = docs.attention_mask

    bsz, num_z = z_idxs.shape
    total_num_z, z_len = z_ids.shape

    z = z_ids[z_idxs]
    mask = z_mask[z_idxs]

    labels = x_ids[:,None].expand(bsz,num_z,x_len).contiguous().view(bsz*num_z, x_len)

    out = model(
        input_ids = z.view(bsz*num_z, z_len),
        attention_mask = mask.view(bsz*num_z, z_len),
        labels = labels,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_z, T
    )
    tok_loss[~x_mask.bool()[:,None].expand(bsz, num_z, T)] = 0
    log_py_z = tok_loss.sum(-1)
    neg_log_py = -log_py_z.logsumexp(-1).mean()
    return neg_log_py, log_py_z

def evaluate(steps, args, model, dataloader, docs, split):
    y_nll = 0
    num_examples = 0
    acc_metric = load_metric("accuracy")
    contrastive_acc_metric = load_metric("accuracy")

    if args.save_results and split == "Valid":
        con_preds = []
        con_golds = []
        con_docs = []
        doc_preds = []
        doc_golds = []


    num_docs = docs.input_ids.shape[0]
    z_idxs = torch.arange(num_docs, device=device, dtype=torch.int64)
    #for step, batch in enumerate(dataloader):
    for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        doc_idxs = batch["doc_idxs"].to(device)
        bsz, num_z = doc_idxs.shape

        batch["doc_idxs"] = z_idxs[None].expand(bsz, num_docs)
        loss, log_py_z = run_model(batch, docs, model)
        y_nll += loss * bsz
        num_examples += bsz

        z_hat = log_py_z.argmax(-1)
        contrastive_scores = log_py_z[torch.arange(bsz)[:,None], doc_idxs]
        z_hat_contrastive = contrastive_scores.argmax(-1)

        acc_metric.add_batch(
            predictions=z_hat,
            references=batch["doc_labels"],
        )
        contrastive_acc_metric.add_batch(
            predictions=z_hat_contrastive,
            references=[num_z-1]*bsz,
        )

    avg_loss = y_nll.item() / num_examples
    z_acc = acc_metric.compute()
    con_acc = contrastive_acc_metric.compute()

    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer NLL": avg_loss,
                f"{split} Subflow Acc": z_acc,
                f"{split} Contrastive Subflow Acc": z_con_acc,
            }
        )

    return avg_loss


def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    model_name = args.model_dir.split("/")[-1]
    run_name = (
        f"answer-model-{model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
        f"k-{args.num_negatives} "
        f"hn-{args.num_hard_negatives}"
    )
    args.run_name = run_name

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    # answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

    train_dataloader, eval_dataloader, docs = prepare_dataloader(
        answer_tokenizer,
        args,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler([answer_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name, project="abcd_unsup_subflow", tags=["abcd"])
        wandb.config.lr = args.learning_rate
        wandb.watch(answer_model)

    #best_valid = float("-inf")
    best_valid = float("inf")
    answer_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if (
                completed_steps % args.eval_steps == 0
                and completed_steps > 0
                and step % args.gradient_accumulation_steps == 0
            ):
                answer_model.eval()
                with torch.no_grad():
                    valid_loss = evaluate(
                        steps=completed_steps,
                        args=args,
                        model=answer_model,
                        docs=docs,
                        dataloader=eval_dataloader,
                        split="Valid",
                    )
                if valid_loss < best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        answer_model.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-answer"
                        )
                answer_model.train()
            loss, _ = run_model(batch, docs, answer_model)
            loss.backward()
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if not args.nolog:
                    wandb.log({"step": completed_steps, "Train Loss": loss.item()})


if __name__ == "__main__":
    main()
