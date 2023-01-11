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

from subflow_data import prepare_dataloader
from rich.progress import track
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import set_seed
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn

from eba_utils import prepare_optim_and_scheduler, prepare_optim_and_scheduler2
from subflow_args import get_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 555
set_seed(seed)

torch.autograd.set_detect_anomaly(True)

start_customer_token = "custom"
customer_token = "Ġcustomer"
start_agent_token = "agent"
agent_token = "Ġagent"
action_token = "Ġaction"


def run_model(batch, docs, doc_sents, doc_num_sents, model):
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)

    bsz, x_len = x_ids.shape

    z_ids = docs.input_ids
    z_mask = docs.attention_mask

    doc_labels = batch["doc_labels"]

    num_docs, doc_len = z_ids.shape

    z_len = doc_sents.input_ids.shape[-1]
    sent_ids = doc_sents.input_ids.view(num_docs, -1, z_len)
    sent_mask = doc_sents.attention_mask.view(num_docs, -1, z_len)

    num_z = sent_ids.shape[1]

    # bsz x num_z x x_len

    labels = (
        x_ids[:, None].repeat(1, num_z, 1).view(bsz * num_z, x_len)
    )

    out = model(
        input_ids=sent_ids[doc_labels].view(bsz * num_z, z_len),
        attention_mask=sent_mask[doc_labels].view(bsz * num_z, z_len),
        labels=labels,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_z, T
    )
    #tok_loss[~x_mask.bool()[:, None].expand(bsz, num_z, T)] = 0
    tok_loss = tok_loss.masked_fill(~x_mask.bool()[:, None].expand(bsz, num_z, T), 0)

    turn_mask = batch["agent_turn_mask"] | batch["customer_turn_mask"] | batch["action_turn_mask"]
    turn_mask = turn_mask[:,None,:].repeat(1, num_z, 1)
    turn_numbers = turn_mask.cumsum(-1)

    loss_buffer = torch.zeros_like(tok_loss)
    loss_out = torch.scatter_add(loss_buffer, -1, turn_numbers.to(device), tok_loss)

    turn_logprobs = loss_out.logsumexp(1)

    turn_mask = torch.arange(x_len) <= turn_numbers[:,0,-1,None]

    #turn_logprobs[~turn_mask.to(device)] = 0
    #conversation_logprob = turn_logprobs.sum(-1)
    conversation_logprob = turn_logprobs.masked_fill(~turn_mask.to(device), 0).sum(-1)
    neg_log_py = -conversation_logprob.mean()

    #import pdb; pdb.set_trace()

    return neg_log_py, tok_loss


def evaluate(steps, args, model, dataloader, docs, doc_sents, doc_num_sents, split):
    y_nll = 0
    num_examples = 0
    acc_metric = load_metric("accuracy")

    if not args.no_save_results and split == "Valid":
        sent_preds = []
        sent_golds = []

    num_docs = docs.input_ids.shape[0]
    # for step, batch in enumerate(dataloader):
    for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["x_ids"].shape[0]
        num_z = num_docs

        loss, log_py_z = run_model(batch, docs, doc_sents, doc_num_sents, model)

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
        f"oracle-sent-model-{args.prefix}-{model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
        f"s-{args.subsample} "
        f"sk-{args.subsample_k} "
        f"ss-{args.subsample_steps} "
        f"sp-{args.subsample_passes} "
    )
    args.run_name = run_name

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    #answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model_dir = "saved_models/ws-encoder-answer-model-14-s-roberta-base-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 k-16 tz-False s-subflow sk-3 ss-250 sp-4 -answer"
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
    answer_model = answer_model.to(device)

    (
        train_dataloader, eval_dataloader, subsample_dataloader,
        docs, doc_sents, doc_num_sents,
    ) = prepare_dataloader(
        answer_tokenizer,
        args,
        device,
        subsample=args.subsample,
        k=args.subsample_k,
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
                doc_sents=doc_sents,
                doc_num_sents=doc_num_sents,
                dataloader=eval_dataloader,
                split="Valid",
            )
        sys.exit()

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    # subsample
    s_completed_steps = 0
    s_optim, s_lr_scheduler = prepare_optim_and_scheduler2([answer_model], args,
        args.max_train_steps // args.subsample_steps * args.subsample_passes)
    # / subsample

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
                        doc_sents=doc_sents,
                        doc_num_sents=doc_num_sents,
                        dataloader=eval_dataloader,
                        split="Valid",
                    )
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    if not args.nolog:
                        wandb.run.summary["best_valid_loss"] = valid_loss
                        wandb.run.summary["best_valid_step"] = completed_steps
                    if not args.no_save_model:
                        answer_model.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-answer"
                        )
                answer_model.train()
            loss, _ = run_model(batch, docs, doc_sents, doc_num_sents, answer_model)
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
