# run model p(turn | step, doc*) w/o dial history
 
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
from pathlib import Path
import json

from datasets import load_metric
from torch.utils.data.sampler import BatchSampler, RandomSampler

from turn_data import prepare_dataloader
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
from inference_utils import (
    monotonic_partition,
    monotonic_prediction,
    first_monotonic_prediction,
    first_argmax_prediction,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 555
set_seed(seed)

#torch.autograd.set_detect_anomaly(True)

start_customer_token = "custom"
customer_token = "Ġcustomer"
start_agent_token = "agent"
agent_token = "Ġagent"
action_token = "Ġaction"


def run_model(
    batch, docs, doc_sents, doc_num_sents, model,
    allow_dummy_step=False,
    monotonic=False,
    decoder_turn_attention=False,
):
    x_ids = batch["turn_ids"].to(device)
    x_mask = batch["turn_mask"].to(device)

    bsz, x_len = x_ids.shape

    #z_ids = docs.input_ids
    #z_mask = docs.attention_mask

    doc_labels = batch["doc_labels"]

    num_docs = len(docs)

    z_len = doc_sents.input_ids.shape[-1]
    sent_ids = doc_sents.input_ids.view(num_docs, -1, z_len)
    sent_mask = doc_sents.attention_mask.view(num_docs, -1, z_len)

    num_z = sent_ids.shape[1]

    # bsz x num_z x x_len
    labels = (
        x_ids[:, None].repeat(1, num_z, 1).view(bsz * num_z, x_len)
    )

    input_ids = sent_ids[doc_labels].view(bsz * num_z, z_len)
    inputs_embeds = model.model.decoder.embed_tokens(input_ids) * model.model.decoder.embed_scale
    attention_mask = sent_mask[doc_labels].view(bsz * num_z, z_len)

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

    # padding steps will only have <bos> <eos>, so mask will only have two elements.
    padding_z = sent_mask[doc_labels].sum(-1) <= 2
    if allow_dummy_step:
        # allow use of last doc step == no conditioning
        padding_z[:,-1] = False
    log_p_z = torch.zeros(bsz, num_z, device=device)
    #log_p_z[padding_z] = -1e5
    log_p_z[padding_z] = float("-inf")
    log_p_z = log_p_z.log_softmax(-1)

    log_p_turn_z = tok_loss.sum(-1) + log_p_z
    logprob_turn = log_p_turn_z.logsumexp(1)

    #turn_mask = torch.arange(x_len) <= turn_numbers[:,0,-1,None]
    #conversation_logprob = turn_logprobs.masked_fill(~turn_mask.to(device), 0).sum(-1)
    neg_log_py = -logprob_turn.mean()

    return neg_log_py, tok_loss, log_p_turn_z


def evaluate(steps, args, model, dataloader, docs, doc_sents, doc_num_sents, split):
    with Path("data/step_annotations.json").open("r") as f:
        all_labels = json.load(f)
    with Path("data/agent_step_annotations.json").open("r") as f:
        agent_all_labels = json.load(f)
    labels = all_labels["dev"] if split == "Valid" else all_labels["test"]
    agent_labels = agent_all_labels["dev"] if split == "Valid" else agent_all_labels["test"]

    y_nll = 0
    num_turns = 0
    dials = set()
    scores = {}

    monotonic_acc = load_metric("accuracy")
    first_monotonic_acc = load_metric("accuracy")
    argmax_acc = load_metric("accuracy")
    first_argmax_acc = load_metric("accuracy")

    if not args.no_save_results and split == "Valid":
        sent_preds = []
        sent_golds = []
        sent_ids = []
        agent_sent_preds = []
        agent_sent_golds = []
        agent_sent_ids = []
        agent_sent_filter = []

    num_docs = len(docs)
    for step, batch in enumerate(dataloader):
    #for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["turn_ids"].shape[0]
        num_z = num_docs

        loss, log_py_z, log_pturn_z = run_model(
            batch, docs, doc_sents, doc_num_sents, model,
            allow_dummy_step=args.dummy_step,
            decoder_turn_attention=args.decoder_turn_attention,
        )
        log_pturn_z = log_pturn_z.cpu().numpy()

        y_nll += loss.item() * bsz
        num_turns += bsz

        ids = batch["ids"].tolist()
        for id in ids: dials.add(id)
        # keep track of uniq dials

        # aggregate scores
        for i in range(bsz):
            id = ids[i]
            id_str = str(id)

            # check against labels with sparse annotations
            # but only the turns that are on
            if id_str in agent_labels:
                logp = log_pturn_z[i]
                turn_num = int(batch["turn_nums"][i].item())
                speaker = int(batch["speakers"][i].item())
                if id not in scores:
                    scores[id] = {}
                scores[id][turn_num] = (speaker, logp)

    for id, turn_map in scores.items():
        id_str = str(id)
        z_labels = torch.tensor(agent_labels[id_str])
        num_turns = len(z_labels)
        unary = torch.tensor(np.stack([turn_map[x][1] for x in range(num_turns)]))
        monotonic_preds = monotonic_prediction(unary)
        argmax_preds = unary.argmax(-1)
        first_monotonic_preds = first_monotonic_prediction(unary)
        first_argmax_preds = first_argmax_prediction(unary)

        speakers = torch.tensor(np.stack([turn_map[x][0] for x in range(num_turns)]))

        agent_turn_mask = speakers == 0
        monotonic_acc.add_batch(
            predictions=monotonic_preds[agent_turn_mask],
            references=z_labels[agent_turn_mask],
        )
        first_monotonic_acc.add_batch(
            predictions=first_monotonic_preds[agent_turn_mask],
            references=z_labels[agent_turn_mask],
        )
        argmax_acc.add_batch(
            predictions=argmax_preds[agent_turn_mask],
            references=z_labels[agent_turn_mask],
        )
        first_argmax_acc.add_batch(
            predictions=first_argmax_preds[agent_turn_mask],
            references=z_labels[agent_turn_mask],
        )
        # /decision rules


        if not args.no_save_results and split == "Valid":
            agent_sent_preds.append(unary)
            agent_sent_golds.append(z_labels)
            agent_sent_ids.append(id)


    avg_loss = y_nll.item() / num_examples

    # decision rules
    monotonic_acc = monotonic_acc.compute()
    first_monotonic_acc = first_monotonic_acc.compute()
    argmax_acc = argmax_acc.compute()
    first_argmax_acc = first_argmax_acc.compute()
    # /decision rules

    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer NLL": avg_loss,
                f"{split} Monotonic Step Acc": monotonic_acc,
                f"{split} First monotonic Step Acc": first_monotonic_acc,
                f"{split} Argmax Step Acc": argmax_acc,
                f"{split} First argmax Step Acc": first_argmax_acc,
            }
        )
    if not args.no_save_results and split == "Valid":
        torch.save(
            (
                agent_sent_preds,
                agent_sent_golds,
                agent_sent_ids,
            ),
            f"logging/{args.run_name}|step-{steps}.agent.pt",
        )

    print("average loss")
    print(avg_loss)
    print("monotonic acc")
    print(monotonic_acc)
    print("first monotonic acc")
    print(first_monotonic_acc)
    print("argmax acc")
    print(argmax_acc)
    print("first argmax acc")
    print(first_argmax_acc)

    return avg_loss




def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    model_name = args.answer_model_dir.split("/")[-1]
    run_name = (
        f"oracle-turn-model-{args.prefix}-{model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
        f"s-{args.subsample} "
        f"sk-{args.subsample_k} "
        f"ss-{args.subsample_steps} "
        f"sp-{args.subsample_passes} "
        f"ip-{args.init_from_previous} "
        f"ds-{args.dummy_step} "
        f"mt-{args.monotonic_train} "
        f"dta-{args.decoder_turn_attention} "
    )
    args.run_name = run_name

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    #answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model_dir = args.answer_model_dir
    if args.init_from_previous:
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
            loss, _, _ = run_model(
                batch, docs, doc_sents, doc_num_sents, answer_model,
                allow_dummy_step=args.dummy_step,
                monotonic=args.monotonic_train,
                decoder_turn_attention=args.decoder_turn_attention,
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