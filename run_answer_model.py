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
    )
    doc_ids = tokenized_docs.input_ids
    doc_mask = tokenized_docs.attention_mask

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
        }
        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            "x_ids",
            "x_mask",
            "doc_idxs",
        ]
        dataset.set_format(type="torch", columns=columns, output_all_columns=False)
        return dataset

    train = process_dataset(train_dataset)
    valid = process_dataset(valid_dataset)

    def collate_fn(batch):
        return batch[0]

    sampler = BatchSampler(
        RandomSampler(train), batch_size=args.batch_size, drop_last=False
    )
    train_dataloader = DataLoader(train, sampler=sampler, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train, sampler=sampler)

    sampler = BatchSampler(valid, batch_size=args.eval_batch_size, drop_last=False)
    valid_dataloader = DataLoader(valid, sampler=sampler, collate_fn=collate_fn)
    # valid_dataloader = DataLoader(valid, sampler=sampler)

    for batch in train_dataloader:
        print(batch)
        pdb.set_trace()
    return train_dataloader, valid_dataloader


def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    return model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=answs,
    )


def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    model_name = args.model_dir.split("/")[-1]
    run_name = (
        f"answer-model-{model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"k-{args.num_negatives} "
        f"hn-{args.num_hard_negatives}"
    )
    args.run_name = run_name

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    # answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

    train_dataloader, eval_dataloader = prepare_dataloader(
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
        wandb.watch(all_layers[0])
        wandb.watch(answer_model)

    best_valid = float("-inf")
    all_layers[0].train()
    answer_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if (
                completed_steps % args.eval_steps == 0
                and completed_steps > 0
                and step % args.gradient_accumulation_steps == 0
            ):
                all_layers[0].eval()
                answer_model.eval()
                with torch.no_grad():
                    valid_acc = evaluate(
                        completed_steps,
                        args,
                        all_layers,
                        answer_model,
                        tokenizer,
                        answer_tokenizer,
                        eval_dataloader,
                        "Valid",
                    )
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        all_layers[0].save_pretrained(
                            f"{args.output_model_dir}/{run_name}"
                        )
                        torch.save(
                            all_layers[1:],
                            f"{args.output_model_dir}/{run_name}-others.pt",
                        )
                        answer_model.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-answer"
                        )
                all_layers[0].train()
                answer_model.train()
            _, _, loss, _ = run_model(
                batch,
                all_layers,
                answer_model,
                tokenizer,
                answer_tokenizer,
                reg_coeff=args.reg_coeff,
                t=args.sentence_threshold,
                max_p=args.max_p,
                num_z=args.topk_doc,
                use_first_sentence=args.use_first_sentence,
            )
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
