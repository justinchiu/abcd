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
from pathlib import Path
import json

import pdb

from evaluate import load
from torch.utils.data.sampler import BatchSampler, RandomSampler


from subflow_data import prepare_dataloader
from rich.progress import track
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import set_seed

import torch
from torch.optim import AdamW
from torch import nn

from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical

from eba_utils import prepare_optim_and_scheduler, prepare_optim_and_scheduler2
from subflow_args import get_args
from model_utils import q_doc, subsample_docs, score_step_aligned_turns
from inference_utils import (
    first, first_monotonic_prediction, batch_monotonic_arg_max,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 555
set_seed(seed)


start_customer_token = "custom"
customer_token = "Ġcustomer"
start_agent_token = "agent"
agent_token = "Ġagent"
action_token = "Ġaction"



def run_model(
    batch, docs, doc_sents, doc_num_sents,
    encoder, model,
    num_docs=4,
    monotonic=False,
    decoder_turn_attention=False,
):
    # set up variables
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)
    doc_labels = batch["doc_labels"].to(device)

    doc_ids = docs.input_ids
    doc_mask = docs.attention_mask

    total_num_docs, doc_len = doc_ids.shape
    bsz, x_len = x_ids.shape

    sent_len = doc_sents.input_ids.shape[-1]
    num_sents = doc_sents.input_ids.shape[0] // total_num_docs

    sent_ids = doc_sents.input_ids.view(total_num_docs, -1, sent_len)
    sent_mask = doc_sents.attention_mask.view(total_num_docs, -1, sent_len)

    # encoder and log q(z|x)
    with torch.no_grad():
        logits_qdoc_x, neg_log_qdoc = q_doc(encoder, x_ids, x_mask, doc_ids, doc_mask, doc_labels)

    # answer log p(y|x,z)

    # subsample docs
    (
        sampled_doc_idxs,
        sampled_logits_qdoc_x,
        sampled_doc_ids, sampled_doc_mask,
        sampled_sent_ids, sampled_sent_mask,
    ) = subsample_docs(
        num_docs, total_num_docs,
        logits_qdoc_x, doc_ids, doc_mask,
        sent_ids, sent_mask,
    )
    # sampled_doc_ids.shape =  bsz x num_docs x doc_len
    # sampled_sent_ids.shape =  bsz x num_docs x x num_sents x doc_len

    labels = (x_ids[:, None, None]
        .repeat(1, num_docs, num_sents, 1)
        .view(bsz * num_docs * num_sents, x_len)
    )

    # attention mask hacking
    inputs_embeds = model.model.decoder.embed_tokens(sent_ids) * model.model.decoder.embed_scale
    decoder_causal_attention_mask = model.model.decoder._prepare_decoder_attention_mask(
        None, labels.shape, inputs_embeds, 0,
    )
    # decoder_causal_attention_mask: bsz x 1 x tgt_seq_len x src_seq_len
     
    is_turn_mask = batch["agent_turn_mask"] | batch["customer_turn_mask"] | batch["action_turn_mask"]
    is_turn_mask = is_turn_mask[:,None,:].repeat(1, num_docs * num_sents, 1)
    turn_numbers = is_turn_mask.cumsum(-1) - 1
    turn_numbers[:,:,0] = 0
    turn_numbers = turn_numbers.view(bsz, num_docs, num_sents, x_len)
    # input        = <s> agent:
    # turn_numbers = 0   1 ...
    # the <s> = 0 is a waste of a turn number, so fold it into first turn

    tn = turn_numbers.view(bsz * num_docs * num_sents, x_len).to(device)
    same_turn_mask= tn[:,:,None] == tn[:,None]
    minval = torch.finfo(decoder_causal_attention_mask.dtype).min
    decoder_turn_attention_mask = torch.full(
        decoder_causal_attention_mask.shape, minval, device=decoder_causal_attention_mask.device)
    decoder_turn_attention_mask = decoder_turn_attention_mask.masked_fill(same_turn_mask[:,None], 0)
    # decoder_turn_attention_mask: bsz x 1 x tgt_seq_len x src_seq_len

    decoder_attention_mask = (
        decoder_causal_attention_mask + decoder_turn_attention_mask
        if decoder_turn_attention else None
    )

    out = model(
        input_ids=sampled_sent_ids.view(bsz * num_docs * num_sents, sent_len),
        attention_mask=sampled_sent_mask.view(bsz * num_docs * num_sents, sent_len),
        labels=labels,
        decoder_expanded_attention_mask = decoder_attention_mask,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_docs, num_sents, T,
    )
    tok_loss = tok_loss.masked_fill(
        ~x_mask.bool()[:, None, None].expand(bsz, num_docs, num_sents, T),
        0,
    )
    neg_log_py, log_py_doc, log_py_doc_step = score_step_aligned_turns(
        tok_loss, turn_numbers,
        sampled_sent_mask, doc_labels, device, monotonic,
    )

    # loss = logsumexp_z log p(y|z) - KL[p(z|x) || q(z|x)]
    # approximate the latter with self-normalized importance sampling
    #approx_log_pz_x = log_py_z.log_softmax(-1)
    approx_log_pdoc_x = log_py_doc.log_softmax(-1)
    approx_log_qdoc_x = sampled_logits_qdoc_x.log_softmax(-1)
    p_q_kl = kl_divergence(
        Categorical(logits=approx_log_pdoc_x),
        Categorical(logits=approx_log_qdoc_x),
    ).mean()

    loss = neg_log_py + p_q_kl

    return loss, logits_qdoc_x, log_py_doc_step, sampled_doc_idxs


def evaluate(
    steps, args, encoder, model, dataloader,
    docs, doc_sents, doc_num_sents, split,
):
    with Path("data/agent_step_annotations.json").open("r") as f:
        agent_all_labels = json.load(f)
    agent_labels = agent_all_labels["dev"] if split == "Valid" else agent_all_labels["test"]

    y_nll = 0
    num_examples = 0

    doc_acc_metric = load("accuracy")
    q_doc_acc_metric = load("accuracy")

    first_monotonic_acc_metric = load("accuracy")

    if not args.no_save_results and split == "Valid":
        step_preds = []
        doc_scores = []
        doc_idxs = []
        golds = []
        dialids = []

    #num_docs = docs.input_ids.shape[0]
    num_docs = args.num_z_samples
    z_idxs = torch.arange(num_docs, device=device, dtype=torch.int64)
    for step, batch in enumerate(dataloader):
    #for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["x_ids"].shape[0]

        loss, log_qdoc_x, log_pturn_doc_step, sampled_doc_idxs = run_model(
            batch, docs, doc_sents, doc_num_sents, encoder, model,
            num_docs=num_docs,
            monotonic=args.monotonic_train,
            decoder_turn_attention=args.decoder_turn_attention,
        )

        # compute q accuracy
        q_doc_acc_metric.add_batch(
            predictions=log_qdoc_x.argmax(-1),
            references=batch["doc_labels"],
        )

        y_nll += loss * bsz
        num_examples += bsz

        max_turns = batch["turn_ids"].shape[1]
        ids = batch["ids"].tolist()
        batch_z_labels = []
        batch_z_hat = []
        for i, id in enumerate(ids):
            id_str = str(id)

            # check against labels with sparse annotations
            # but only the turns that are on
            if id_str in agent_labels:
                z_labels = torch.tensor(agent_labels[id_str])
                z_labels = first(z_labels)
                #num_turns = min(len(z_labels), max_turns)
                num_turns = len(z_labels)

                logp = log_pturn_doc_step[i,:,:,:num_turns]
                preds, scores = batch_monotonic_arg_max(logp.permute(0,2,1))
                fpreds = [first(x.cpu().numpy()) for x in preds]

                best_pred_idx = scores.argmax().item()
                best_pred = fpreds[best_pred_idx]

                # compute doc accuracy
                doc_pred = sampled_doc_idxs[i][best_pred_idx].item()
                doc_label = batch["doc_labels"][i].item()
                doc_acc_metric.add(
                    prediction=doc_pred,
                    reference=doc_label,
                )
                
                # compute step accuracy
                agent_turn_mask = batch["is_agent_turn"][i,:num_turns]
                first_monotonic_acc_metric.add_batch(
                    predictions=best_pred[agent_turn_mask],
                    references=z_labels[agent_turn_mask]
                        if doc_pred == doc_label
                        else [-2 for x in agent_turn_mask if x],
                        # all incorrect
                )

                if not args.no_save_results and split == "Valid":
                    step_preds.append(fpreds)
                    doc_scores.append(scores)
                    doc_idxs.append(sampled_doc_idxs)
                    golds.append(z_labels)
                    dialids.append(id)


    avg_loss = y_nll.item() / num_examples
    doc_acc = doc_acc_metric.compute()
    q_doc_acc = q_doc_acc_metric.compute()
    first_monotonic_acc = first_monotonic_acc_metric.compute()

    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer NLL": avg_loss,
                f"{split} Doc Acc": doc_acc,
                f"{split} Q Doc Acc": q_doc_acc,
                f"{split} First monotonic Step Acc": first_monotonic_acc,
            }
        )
    if not args.no_save_results and split == "Valid":
        torch.save(
            (
                step_preds,
                doc_scores,
                doc_idxs,
                golds,
                dialids,
            ),
            f"logging/{args.run_name}|step-{steps}.pt",
        )

    print("average loss")
    print(avg_loss)
    print("doc acc")
    print(doc_acc)
    print("q doc acc")
    print(q_doc_acc)
    print("first monotonic acc")
    print(first_monotonic_acc)

    return avg_loss


def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    encoder_name = args.model_dir.split("/")[-1]
    answer_model_name = args.answer_model_dir.split("/")[-1]
    run_name = (
        f"doc-step-{args.prefix}-{encoder_name}-{answer_model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
        f"k-{args.num_z_samples} "
        f"ip-{args.init_from_previous} "
        f"mt-{args.monotonic_train} "
        f"dta-{args.decoder_turn_attention} "
    )
    args.run_name = run_name
    print(run_name)

    model_dir = args.model_dir
    if args.init_from_previous:
        model_dir = "saved_models/ws-encoder-answer-model-14-s-roberta-base-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 k-16 tz-False s-subflow sk-3 ss-250 sp-4 -encoder"
    encoder = AutoModel.from_pretrained(model_dir)
    encoder = encoder.to(device)

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    # answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
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

    if not args.nolog:
        wandb.init(name=run_name, project="abcd_unsup_subflow", tags=["abcd"])
        wandb.config.lr = args.learning_rate
        wandb.watch(encoder)
        wandb.watch(answer_model)

    if args.eval_only:
        args.no_save_model = True
        args.nolog = True
        completed_steps = -1

        # load models
        savepath = f"{args.output_model_dir}/{run_name}-encoder"
        encoder = AutoModelForSeq2SeqLM.from_pretrained(savepath)
        encoder = encoder.to(device)

        savepath = f"{args.output_model_dir}/{run_name}-answer"
        answer_model = AutoModelForSeq2SeqLM.from_pretrained(savepath)
        answer_model.to(device)

        with torch.no_grad():
            valid_loss = evaluate(
                steps=completed_steps,
                args=args,
                encoder=encoder,
                model=answer_model,
                docs=docs,
                dataloader=eval_dataloader,
                split="Valid",
            )
        sys.exit()

    # full training loop
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler([encoder, answer_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    best_valid = float("inf")
    completed_steps = 0
    encoder.train()
    answer_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if (
                completed_steps % args.eval_steps == 0
                and completed_steps > 0
                and step % args.gradient_accumulation_steps == 0
            ):
                encoder.eval()
                answer_model.eval()
                with torch.no_grad():
                    valid_loss = evaluate(
                        steps=completed_steps,
                        args=args,
                        encoder=encoder,
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
                        encoder.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-encoder"
                        )
                        answer_model.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-answer"
                        )
                encoder.train()
                answer_model.train()
            loss, _, _, _ = run_model(
                batch, docs, doc_sents, doc_num_sents,
                encoder, answer_model,
                num_docs=args.num_z_samples,
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
