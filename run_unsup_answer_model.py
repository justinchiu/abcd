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

from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical

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
    parser.add_argument("--prefix", default="1220", help="date")

    parser.add_argument("--interact_data", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--true_z", action="store_true")
    parser.add_argument("--kl_weight", default=1.0, type=float)

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

    parser.add_argument("--num_z_samples", default=4, type=int)
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
        "--supervised_examples",
        default=0,
        type=int,
        help="number of fully supervised training examples for q(z|x) and p(x|z). "
        "this should be as small as possible",
    )
    parser.add_argument(
        "--q_warmup_steps",
        default=0,
        type=int,
        help="number of q warmup steps",
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
        default="roberta-base",
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
        # x_ids[x_ids == padding_id] = -100
        # the -100 is for NLLCriterion, but we manually select non-padding

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

        if False:
            # DBG
            next_token_id = x_ids[:, 1:]
            not_agent_token = (x_ids == agent_id) & ~is_next_token_colon
            not_customer_token = (x_ids == customer_id) & ~is_next_token_colon

            print(not_agent_token.nonzero())
            # check x_ids for this
            # / DBG

        doc_labels = example_batch["doc_labels"]
        doc_negatives = example_batch["doc_negatives"]

        encodings = {
            "x_ids": x_ids,
            "x_mask": x_mask,
            "ids": example_batch["ids"],
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


def run_model(batch, docs, encoder, model, num_z=4, supervised=False, true_z=False, kl_weight=1.0):
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)
    z_labels = batch["doc_labels"].to(device)

    z_ids = docs.input_ids
    z_mask = docs.attention_mask

    total_num_z, z_len = z_ids.shape
    bsz, x_len = x_ids.shape

    # encoder and log q(z|x)
    x_out = encoder(input_ids=x_ids, attention_mask=x_mask)
    z_out = encoder(input_ids=z_ids, attention_mask=z_mask)

    score_z_x = torch.einsum("xh,zh->xz", x_out.pooler_output, z_out.pooler_output)
    log_qz_x = (score_z_x / 32).log_softmax(-1)

    neg_log_qz = -log_qz_x[torch.arange(bsz), z_labels].mean()

    # answer log p(y|x,z)

    # subsample docs
    if num_z < total_num_z:
        topk_z = log_qz_x.topk(num_z, -1)
        sampled_log_qz_x, z_idxs = topk_z
        # TODO: add sample without replacement
        if supervised or true_z:
            z_idxs = torch.tensor([
                idxs[:-1] + [z_labels[i]] if z_labels[i] not in idxs else idxs
                for i, idxs in enumerate(z_idxs.tolist())
            ], dtype=torch.int64, device=z_idxs.device)
            sampled_log_qz_x = log_qz_x[torch.arange(bsz)[:,None], z_idxs]
        z = z_ids[z_idxs]
        mask = z_mask[z_idxs]
    else:
        sampled_log_qz_x = log_qz_x
        z = z_ids[None].repeat(bsz, 1, 1)
        mask = z_mask[None].repeat(bsz, 1, 1)

    labels = x_ids[:, None].repeat(1, num_z, 1).view(bsz * num_z, x_len)

    out = model(
        input_ids=z.view(bsz * num_z, z_len),
        attention_mask=mask.view(bsz * num_z, z_len),
        labels=labels,
    )
    logits = out.logits.log_softmax(-1)
    N, T, V = logits.shape
    tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
        bsz, num_z, T
    )
    tok_loss[~x_mask.bool()[:, None].expand(bsz, num_z, T)] = 0
    log_py_z = tok_loss.sum(-1)
    log_py = log_py_z.logsumexp(-1)
    neg_log_py = -log_py.mean()

    R = log_py_z.detach()
    baseline = (R[:,None] - R[:,:,None]).sum(-1) / (num_z-1)

    reconstruction = (sampled_log_qz_x.exp() * (log_py_z - baseline)).sum(-1)
    # KL has better scaling than entropy, since subtracts uniform entropy
    posterior_prior_kl = kl_divergence(
        Categorical(logits=log_qz_x), Categorical(logits=torch.zeros_like(log_qz_x))
    )
    # posterior_prior_kl = Categorical(logits=log_qz_x).entropy()
    # posterior_prior_kl = 0
    neg_elbo = -(reconstruction - kl_weight * posterior_prior_kl).mean()

    if not supervised:
        loss = neg_elbo
    else:
        correct_z_mask = z_idxs == z_labels[:,None]
        loss = neg_log_qz - log_py_z.log_softmax(-1)[correct_z_mask].mean()

    return loss, log_qz_x, tok_loss

def run_q_only(batch, docs, encoder, model, kl_weight=1.0):
    x_ids = batch["x_ids"].to(device)
    x_mask = batch["x_mask"].to(device)
    z_labels = batch["doc_labels"].to(device)

    z_ids = docs.input_ids
    z_mask = docs.attention_mask

    total_num_z, z_len = z_ids.shape
    bsz, x_len = x_ids.shape
    num_z = total_num_z

    # encoder and log q(z|x)
    x_out = encoder(input_ids=x_ids, attention_mask=x_mask)
    z_out = encoder(input_ids=z_ids, attention_mask=z_mask)

    score_z_x = torch.einsum("xh,zh->xz", x_out.pooler_output, z_out.pooler_output)
    log_qz_x = (score_z_x / 32).log_softmax(-1)

    neg_log_qz = -log_qz_x[torch.arange(bsz), z_labels].mean()

    # answer log p(y|x,z)

    # subsample docs
    sampled_log_qz_x = log_qz_x
    z = z_ids[None].repeat(bsz, 1, 1)
    mask = z_mask[None].repeat(bsz, 1, 1)

    labels = x_ids[:, None].repeat(1, num_z, 1).view(bsz * num_z, x_len)

    with torch.no_grad():
        out = model(
            input_ids=z.view(bsz * num_z, z_len),
            attention_mask=mask.view(bsz * num_z, z_len),
            labels=labels,
        )
        logits = out.logits.log_softmax(-1)
        N, T, V = logits.shape
        tok_loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
            bsz, num_z, T
        )
        tok_loss[~x_mask.bool()[:, None].expand(bsz, num_z, T)] = 0
    log_py_z = tok_loss.sum(-1)
    log_py = log_py_z.logsumexp(-1)
    neg_log_py = -log_py.mean()

    reconstruction = (sampled_log_qz_x.exp() * log_py_z).sum(-1)
    # KL has better scaling than entropy, since subtracts uniform entropy
    posterior_prior_kl = kl_divergence(
        Categorical(logits=log_qz_x), Categorical(logits=torch.zeros_like(log_qz_x))
    )
    # posterior_prior_kl = Categorical(logits=log_qz_x).entropy()
    # posterior_prior_kl = 0
    neg_elbo = -(reconstruction - kl_weight * posterior_prior_kl).mean()

    loss = neg_elbo

    return loss, log_qz_x, tok_loss


def evaluate(steps, args, encoder, model, dataloader, docs, split):
    y_nll = 0
    num_examples = 0

    # evaluate encoder and intent model later

    # answer model
    acc_metric = load("accuracy")
    first_action_acc_metric = load("accuracy")
    q_acc_metric = load("accuracy")

    if not args.no_save_results and split == "Valid":
        q_out = []
        p_out = []
        doc_golds = []

    num_docs = docs.input_ids.shape[0]
    z_idxs = torch.arange(num_docs, device=device, dtype=torch.int64)
    #for step, batch in enumerate(dataloader):
    for step, batch in track(enumerate(dataloader), total=len(dataloader)):
        bsz = batch["x_ids"].shape[0]

        loss, log_qz_x, log_py_z = run_model(
            batch, docs, encoder, model, num_z=num_docs
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
        f"encoder-answer-model-{args.prefix}-{encoder_name}-{answer_model_name} "
        f"lr-{args.learning_rate} "
        f"bs-{args.batch_size*args.gradient_accumulation_steps} "
        f"dt-{args.num_dialogue_turns} "
        f"ds-{args.num_doc_sents} "
        f"ml-{args.max_length} "
        f"k-{args.num_z_samples} "
        f"se-{args.supervised_examples} "
        f"qw-{args.q_warmup_steps} "
        f"tz-{args.true_z} "
        f"kl-{args.kl_weight} "
    )
    args.run_name = run_name
    print(run_name)

    encoder = AutoModel.from_pretrained(args.model_dir)
    encoder = encoder.to(device)

    # answer_model_dir = args.answer_model_dir if not load_answer else load_answer
    # answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_dir)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

    train_dataloader, eval_dataloader, docs = prepare_dataloader(
        answer_tokenizer,
        args,
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

    completed_steps = 0
    if args.q_warmup_steps > 0:
        nolog = args.nolog
        args.nolog = True
        print("WARMING UP Q")
        # q initial training loop
        args.max_train_steps = args.q_warmup_steps
        optim, lr_scheduler = prepare_optim_and_scheduler([encoder], args)

        progress_bar = tqdm(range(args.q_warmup_steps))
        best_valid = float("inf")
        encoder.train()
        answer_model.eval()
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
                    if not args.no_save_model:
                        encoder.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-encoder"
                        )
                encoder.train()
            loss, log_qz_x, log_py_z = run_q_only(
                batch, docs, encoder, answer_model,
                kl_weight=args.kl_weight,
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
            if completed_steps > args.q_warmup_steps:
                break
        # reset logging
        args.nolog = nolog

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
                        dataloader=eval_dataloader,
                        split="Valid",
                    )
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    if not args.no_save_model:
                        encoder.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-encoder"
                        )
                        answer_model.save_pretrained(
                            f"{args.output_model_dir}/{run_name}-answer"
                        )
                encoder.train()
                answer_model.train()
            loss, log_qz_x, log_py_z = run_model(
                batch, docs, encoder, answer_model, num_z=args.num_z_samples,
                supervised=completed_steps * args.batch_size < args.supervised_examples,
                true_z=args.true_z,
                kl_weight=args.kl_weight,
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
