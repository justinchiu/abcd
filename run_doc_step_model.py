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
from model_utils import q_doc

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

    total_num_docs, doc_len = z_ids.shape
    bsz, x_len = x_ids.shape

    sent_len = doc_sents.input_ids.shape[-1]
    sent_ids = doc_sents.input_ids.view(total_num_docs, -1, z_len)
    sent_mask = doc_sents.attention_mask.view(num_docs, -1, z_len)

    num_sents = sent_ids.shape[1]

    # encoder and log q(z|x)
    log_qdoc_x, neg_log_qdoc = q_doc(encoder, x_ids, x_mask, doc_ids, doc_mask, doc_labels)

    # answer log p(y|x,z)

    # subsample docs
    if num_z < total_num_docs:
        topk_doc = logits_qdoc_x.topk(num_docs, -1)
        sampled_logits_qdoc_x, doc_idxs = topk_doc
        # TODO: add sample without replacement
        if supervised or true_z:
            doc_idxs = torch.tensor([
                idxs[:-1] + [doc_labels[i]] if doc_labels[i] not in idxs else idxs
                for i, idxs in enumerate(doc_idxs.tolist())
            ], dtype=torch.int64, device=doc_idxs.device)
            sampled_logits_qdoc_x = log_qdoc_x[torch.arange(bsz)[:,None], doc_idxs]
        docs = doc_ids[doc_idxs]
        mask = doc_mask[doc_idxs]
    else:
        sampled_logits_qdoc_x = logits_qdoc_x
        docs = doc_ids[None].repeat(bsz, 1, 1)
        mask = doc_mask[None].repeat(bsz, 1, 1)

    labels = x_ids[:, None].repeat(1, num_docs, 1).view(bsz * num_docs, x_len)

    out = model(
        input_ids=z.view(bsz * num_docs, doc_len),
        attention_mask=mask.view(bsz * num_docs, doc_len),
        labels=labels,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_z, T
    )
    #tok_loss[~x_mask.bool()[:, None].expand(bsz, num_z, T)] = 0
    tok_loss = tok_loss.masked_fill(~x_mask.bool()[:, None].expand(bsz, num_z, T), 0)
    log_py_z = tok_loss.sum(-1)
    log_py = log_py_z.logsumexp(-1)
    neg_log_py = -log_py.mean()

    # loss = logsumexp_z log p(y|z) - KL[p(z|x) || q(z|x)]
    # approximate the latter with self-normalized importance sampling
    #approx_log_pz_x = log_py_z.log_softmax(-1)
    approx_log_pz_x = log_py_z.log_softmax(-1)
    approx_log_qz_x = sampled_logits_qz_x.log_softmax(-1)
    p_q_kl = kl_divergence(
        Categorical(logits=approx_log_pz_x),
        Categorical(logits=approx_log_qz_x)
    ).mean()

    if not supervised:
        loss = neg_log_py + p_q_kl
    else:
        correct_z_mask = z_idxs == z_labels[:,None]
        loss = neg_log_qz - log_py_z.log_softmax(-1)[correct_z_mask].mean()

    return loss, log_qz_x, tok_loss


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
    doc_first_action_acc_metric = load("accuracy")
    q_acc_metric = load("accuracy")
    step_acc_metric = load("eccuracy")

    if not args.no_save_results and split == "Valid":
        q_out = []
        p_out = []
        doc_golds = []

    num_docs = docs.input_ids.shape[0]
    z_idxs = torch.arange(num_docs, device=device, dtype=torch.int64)
    #for step, batch in enumerate(dataloader):
    for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["x_ids"].shape[0]

        loss, log_qdoc_x, log_py_z, log_pturn_step_doc = run_model(
            batch, docs, doc_sents, doc_num_sents, encoder, model,
            num_docs=num_docs,
            monotonic=args.monotonic_train,
            decoder_turn_attention=args.decoder_turn_attention,
        )

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
        # hack for recall @ topk
        q_acc_metric.add_batch(
            predictions=(
                log_qz_x.topk(args.num_z_samples, dim=-1).indices == batch["doc_labels"][:,None].to(device)
            ).sum(-1) > 0,
            references=[1 for _ in range(bsz)],
        )

        if not args.no_save_results and split == "Valid":
            q_out.append(log_qz_x.cpu())
            p_out.append(log_py_z.cpu())
            doc_golds.append(batch["doc_labels"].cpu())

    avg_loss = y_nll.item() / num_examples
    z_acc = acc_metric.compute()
    z_first_action_acc = first_action_acc_metric.compute()
    q_acc = q_acc_metric.compute()

    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer NLL": avg_loss,
                f"{split} Subflow Acc": z_acc,
                f"{split} Subflow First action Acc": z_first_action_acc,
                f"{split} Q(z|x) Recall @ {args.num_z_samples}": q_acc,
            }
        )
    if not args.no_save_results and split == "Valid":
        torch.save(
            (
                q_out,
                p_out,
                doc_golds,
            ),
            f"logging/{args.run_name}|step-{steps}.pt",
        )

    print("avg los")
    print(avg_loss)
    print("z acc")
    print(z_acc)
    print("z first action acc")
    print(z_first_action_acc)
    print(f"q(z|x) recall@{args.num_z_samples}")
    print(q_acc)

    return avg_loss


def main():
    args = get_args()
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)

    encoder_name = args.model_dir.split("/")[-1]
    answer_model_name = args.answer_model_dir.split("/")[-1]
    run_name = (
        f"doc-step-model-{args.prefix}-{encoder_name}-{answer_model_name} "
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

    encoder = AutoModel.from_pretrained(args.model_dir)
    encoder = encoder.to(device)

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    # answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
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

    # subsample
    s_completed_steps = 0
    s_optim, s_lr_scheduler = prepare_optim_and_scheduler2([answer_model], args,
        args.max_train_steps // args.subsample_steps * args.subsample_passes)
    # / subsample

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
            loss, _, _ = run_model(
                batch, docs, doc_sents, doc_num_sents,
                encoder, answer_model,
                num_docs=args.num_z_samples,
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
