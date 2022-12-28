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
import sys

import pdb

from datasets import load_metric
from torch.utils.data.sampler import BatchSampler, RandomSampler


# from dataset import prepare_simplified, SimplifiedHotpotQADataset
# from eba_subflow_dataset import prepare_subflow_abcd, SubflowAbcdDataset
# from eba_subflow_factored_dataset import prepare_subflow_abcd, SubflowAbcdDataset
from subflow_data import get_abcd_dataset
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


start_customer_token = "custom"
customer_token = "Ġcustomer"
start_agent_token = "agent"
agent_token = "Ġagent"
action_token = "Ġaction"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="1222", help="date")

    parser.add_argument("--interact_data", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--no_save_results", action="store_true")

    parser.add_argument("--num_dialogue_turns", default=0, type=int)
    parser.add_argument("--num_doc_sents", default=0, type=int)

    parser.add_argument("--max_length", default=512, type=int)

    parser.add_argument(
        "--truncate_early",
        action="store_true",
        help="truncate conversations right before first agent action. only allowed during evaluation, since it hurts during training.",
    )

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
        "train",
        args.num_dialogue_turns,
        args.num_doc_sents,
        truncate_early=args.truncate_early,
    )
    valid_dataset, _, _ = get_abcd_dataset(
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
        x_ids[x_ids == padding_id] = -100

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

        doc_labels = example_batch["doc_labels"]

        encodings = {
            "x_ids": x_ids,
            "x_mask": x_mask,
            "doc_labels": doc_labels,
            "agent_turn_mask": agent_turn,
            "customer_turn_mask": customer_turn,
            "action_turn_mask": action_turn,
        }
        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            "x_ids",
            "x_mask",
            "ids",
            "doc_labels",
            "agent_turn_mask",
            "customer_turn_mask",
            "action_turn_mask",
        ]
        dataset.set_format(type="torch", columns=columns, output_all_columns=False)
        return dataset

    train = process_dataset(train_dataset)
    valid = process_dataset(valid_dataset)

    if args.interact_data:
        raise NotImplementedError
        con_docs, doc_preds, doc_golds = torch.load(
            "logging/answer-model-roberta-large lr-2e-05 bs-8 "
            "dt-0 ds-0 ml-256 k-3 hn-0|step-10000.pt"
        )
        con_docs = torch.cat(con_docs, 0)
        doc_preds = torch.cat(doc_preds, 0)
        doc_golds = torch.cat(doc_golds, 0)

        cum_log_py_z = doc_preds.cumsum(-1)
        z_hat = cum_log_py_z.argmax(1)
        contrastive_scores = cum_log_py_z.gather(1, con_docs[:, :, None])[:, :, 0]
        z_hat_contrastive = contrastive_scores.argmax(1)

        idxs = random.choices(range(len(valid)), k=20)
        idxs = range(len(valid))
        for idx in idxs:
            x = valid["xs"][idx]
            tok_x = tokenizer.tokenize(x)
            label = doc_golds[idx].item()
            string = []
            for tok, doc in zip(tok_x[:64], z_hat[idx]):
                string.append(tok.replace("Ġ", ""))
                string.append(str(doc.item()))
            print(" ".join(string))
            print(idx)
            print("Last doc prediction")
            print(docs[doc.item()][:64])
            print(f"Gold doc: {doc_golds[idx].item()}")
            print(docs[label][:64])
            if doc.item() != label:
                pdb.set_trace()

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

    return train_dataloader, valid_dataloader, tokenized_docs


def run_model(batch, docs, model):
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)

    bsz, x_len = x_ids.shape

    z_ids = docs.input_ids
    z_mask = docs.attention_mask

    num_z, z_len = z_ids.shape

    # bsz x num_z x x_len

    labels = (
        x_ids[:, None].repeat(1, num_z, 1).view(bsz * num_z, x_len)
    )

    out = model(
        input_ids=z_ids[None].repeat(bsz, 1, 1).view(bsz * num_z, z_len),
        attention_mask=z_mask[None].repeat(bsz, 1, 1).view(bsz * num_z, z_len),
        labels=labels,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_z, T
    )
    tok_loss[~x_mask.bool()[:, None].expand(bsz, num_z, T)] = 0
    log_py_z = tok_loss.sum(-1)
    neg_log_py = -log_py_z.logsumexp(-1).mean()
    return neg_log_py, tok_loss


def evaluate(steps, args, model, dataloader, docs, split):
    y_nll = 0
    num_examples = 0
    acc_metric = load_metric("accuracy")
    first_action_acc_metric = load_metric("accuracy")

    if not args.no_save_results and split == "Valid":
        con_docs = []
        doc_preds = []
        doc_golds = []

    num_docs = docs.input_ids.shape[0]
    # for step, batch in enumerate(dataloader):
    for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["x_ids"].shape[0]
        num_z = num_docs

        loss, log_py_z = run_model(batch, docs, model)

        y_nll += loss * bsz
        num_examples += bsz

        # log_py_z: bsz x num_z x time
        cum_log_py_z = log_py_z.cumsum(-1)
        z_hat = cum_log_py_z.argmax(1)

        # index into start of agent turns
        agent_mask = batch["agent_turn_mask"]
        agent_z_hat = z_hat[agent_mask]

        action_mask = batch["action_turn_mask"]
        first_action_mask = action_mask.cumsum(1).cumsum(1) == 1
        # take the prediction at last token if no action seen
        first_action_mask[action_mask.sum(1) == 0, -1] = True
        action_z_hat = z_hat[first_action_mask]

        num_agent_turns = agent_mask.sum(-1).tolist()
        doc_labels = []
        for label, num in zip(batch["doc_labels"].tolist(), num_agent_turns):
            doc_labels += [label] * num

        acc_metric.add_batch(
            predictions=agent_z_hat,
            references=doc_labels,
        )
        first_action_acc_metric.add_batch(
            predictions=action_z_hat,
            references=batch["doc_labels"],
        )

        if not args.no_save_results and split == "Valid":
            doc_preds.append(log_py_z.cpu())
            doc_golds.append(batch["doc_labels"].cpu())

    avg_loss = y_nll.item() / num_examples
    z_acc = acc_metric.compute()
    z_first_action_acc = first_action_acc_metric.compute()

    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer NLL": avg_loss,
                f"{split} Subflow Acc": z_acc,
                f"{split} Subflow First action Acc": z_first_action_acc,
            }
        )
    if not args.no_save_results and split == "Valid":
        torch.save(
            (
                doc_preds,
                doc_golds,
            ),
            f"logging/{args.run_name}|step-{steps}.pt",
        )

    print("average loss")
    print(avg_loss)
    print("z acc")
    print(z_acc)
    print("z first action acc")
    print(z_first_action_acc)

    return avg_loss


def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    model_name = args.answer_model_dir.split("/")[-1]
    run_name = (
        f"all-answer-model-{args.prefix}-{model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
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

    if not args.nolog:
        wandb.init(name=run_name, project="abcd_unsup_subflow", tags=["abcd"])
        wandb.config.lr = args.learning_rate
        wandb.watch(answer_model)

    if args.eval_only:
        args.no_save_model = True
        args.nolog = True
        completed_steps = -1

        savepath = f"{args.output_model_dir}/{run_name}-answer"
        answer_model = AutoModelForSeq2SeqLM.from_pretrained(savepath)
        answer_model.to(device)

        with torch.no_grad():
            valid_loss = evaluate(
                steps=completed_steps,
                args=args,
                model=answer_model,
                docs=docs,
                dataloader=eval_dataloader,
                split="Valid",
            )
        sys.exit()

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    # best_valid = float("-inf")
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
                    best_valid = valid_loss
                    if not args.no_save_model:
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