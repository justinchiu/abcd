from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
import numpy as np
import random

# from dataset import prepare_simplified, SimplifiedHotpotQADataset
# from eba_subflow_dataset import prepare_subflow_abcd, SubflowAbcdDataset
# from eba_subflow_factored_dataset import prepare_subflow_abcd, SubflowAbcdDataset
from eba_efficient_subflow_dataset import prepare_subflow_abcd, SubflowAbcdDataset
from datasets import load_metric
from rich.progress import track
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import wandb
from eba_utils import get_args, load_hotpotqa, mean_pooling, padding, normalize_answer
from eba_utils import prepare_linear, prepare_optim_and_scheduler, padding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)


def pad_and_mask(xs):
    lengths = [x.shape[0] for x in xs]
    bsz = len(lengths)
    dim = xs[0].shape[1]
    maxlen = max(lengths)
    padded = np.zeros((bsz, maxlen, dim), dtype=np.float32)
    mask = np.zeros((bsz, maxlen), dtype=bool)
    for i, xlen in enumerate(lengths):
        mask[i, :xlen] = 1
        padded[i, :xlen] = xs[i]
    return padded, mask


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    sep_idx: int = 2
    num_distractors: int = 0

    def __call__(self, features):
        # features[i].keys() == x, answer_x, docs, answer_docs, doc_label, answer
        batch_size = len(features)

        raw_xs = [feature.pop("x") for feature in features]
        lengths = [len(i) for i in raw_xs]
        # paras = [p for ps in paras for p in ps]
        xs = [{"input_ids": x} for x in raw_xs]

        batch = self.tokenizer.pad(
            xs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        answer_name = "answer"
        raw_answers = [feature.pop(answer_name) for feature in features]
        answer_docs_name = "answer_docs"
        answer_docs = [feature.pop(answer_docs_name) for feature in features][0]
        labels_name = "doc_label"
        labels = [feature.pop(labels_name) for feature in features]
        docs_name = "docs"
        docs = [feature.pop(docs_name) for feature in features][0]

        doc_idxs = []
        if self.num_distractors > 0:
            num_docs = len(docs)
            for label in labels:
                s = set(range(num_docs))
                s.remove(label)
                distractors = random.sample(list(s), self.num_distractors)
                z_idxs = [label] + distractors
                doc_idxs.append(z_idxs)

        doc_lengths = [len(x) for x in docs]
        batch_docs = self.tokenizer.pad(
            [{"input_ids": x} for x in docs],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        """
        # TOKENIZATION IS IDENTICAL
        batch_answer_docs = self.tokenizer.pad(
            [{"input_ids": x} for x in answer_docs],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        assert (batch_docs == batch_answer_docs).all()
        """

        # Add back labels for un-encoded inputs
        batch["raw_xs"] = raw_xs
        batch["lengths"] = lengths
        batch["answers"] = raw_answers
        batch["labels"] = labels

        batch["docs"] = docs
        batch["doc_lengths"] = doc_lengths
        batch["doc_idxs"] = torch.tensor(doc_idxs, dtype=torch.long)

        batch["doc_input_ids"] = batch_docs.input_ids
        batch["doc_attention_mask"] = batch_docs.attention_mask

        # encoded inputs
        enc_x = [feature.pop("enc_x") for feature in features]
        enc_sents = [feature.pop("enc_sents") for feature in features]
        enc_docs = [feature.pop("enc_docs") for feature in features][0]

        # perform padding of encoded inputs and get attention_masks
        enc_x_padded, enc_x_mask = pad_and_mask(enc_x)
        enc_docs_padded, enc_docs_mask = pad_and_mask(enc_docs)
        enc_sents_padded, enc_sents_mask = pad_and_mask(enc_sents)

        batch["enc_x_emb"] = torch.tensor(enc_x_padded)
        batch["enc_x_mask"] = torch.tensor(enc_x_mask)
        batch["enc_docs_emb"] = torch.tensor(enc_docs_padded)
        batch["enc_docs_mask"] = torch.tensor(enc_docs_mask)
        batch["enc_sents_emb"] = torch.tensor(enc_sents_padded)
        batch["enc_sents_mask"] = torch.tensor(enc_sents_mask)

        # raw x and docs to concat for answer later
        batch["enc_x"] = enc_x
        batch["enc_docs"] = enc_docs

        return batch


def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    return [model, linear]


def prepare_dataloader(tok, answer_tok, args, encoder):
    (
        x,
        ax,
        docs,
        adocs,
        doc_labels,
        answers,
        x_to_sent_idxs,
        enc_sents,
        enc_x,
        enc_docs,
    ) = prepare_subflow_abcd(
        tok,
        answer_tok,
        "train",
        encoder=encoder,
    )
    (
        tx,
        tax,
        tdocs,
        tadocs,
        tdoc_labels,
        tanswers,
        tx_to_sent_idxs,
        tenc_sents,
        tenc_x,
        tenc_docs,
    ) = prepare_subflow_abcd(
        tok,
        answer_tok,
        "val",
        encoder=encoder,
    )

    train_dataset = SubflowAbcdDataset(
        x,
        ax,
        docs,
        adocs,
        doc_labels,
        answers,
        x_to_sent_idxs,
        enc_sents,
        enc_x,
        enc_docs,
    )
    eval_dataset = SubflowAbcdDataset(
        tx,
        tax,
        tdocs,
        tadocs,
        tdoc_labels,
        tanswers,
        tx_to_sent_idxs,
        tenc_sents,
        tenc_x,
        tenc_docs,
    )

    data_collator = DataCollatorForMultipleChoice(
        tok, padding="longest", max_length=512,
        num_distractors = args.num_distractors,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )

    return train_dataloader, eval_dataloader


def run_lm(model, batch, bs, train=True, z_outputs=None):
    model, linear = model

    x = batch["input_ids"]
    x_attention_mask = batch["attention_mask"]

    # contrastive
    z = batch.doc_input_ids[batch.doc_idxs]
    z_attention_mask = batch.doc_attention_mask[batch.doc_idxs]

    bsz = x.shape[0]
    ndocs = z.shape[1] if train else 55

    x_outputs = model(
        input_ids=x,
        attention_mask=x_attention_mask,
    )
    x_pooled_output = x_outputs[1]

    if z_outputs is None:
        z_outputs = model(
            input_ids=z.view(bsz * ndocs, -1),
            attention_mask=z_attention_mask.view(bsz * ndocs, -1),
        )
    z_pooled_output = (
        z_outputs[1].view(bsz, ndocs, 1024)
        if train
        else z_outputs.pooler_output.view(1, ndocs, 1024)
    )

    """
    # try using pre-computed embeddings
    x = batch.enc_x_emb
    x_mask = batch.enc_x_mask
    z = batch.enc_docs_emb
    z_mask = batch.enc_docs_mask
    # worried about scaling.
    # maybe divide x and z by sqrt(1024)?
    x_outputs = model(
        inputs_embeds=x,
        attention_mask=x_mask,
    )
    x_pooled_output = x_outputs[1]
    if z_outputs is None:
        z_outputs = model(
            inputs_embeds=z,
            attention_mask=z_mask,
        )
    z_pooled_output = z_outputs.pooler_output
    """

    logits = torch.einsum(
        "bh,bdh->bd",
        x_pooled_output,
        z_pooled_output,
    )
    """
    if train:
        dropout = nn.Dropout(model.config.hidden_dropout_prob)
        pooled_output = dropout(pooled_output)
    logits = linear(pooled_output).view(bsz, ndocs)
    """
    if train:
        return logits.log_softmax(-1), None
    else:
        return logits, z_outputs


# why isnt this done in the data collator???
def pad_answers(tokenizer, xs, docs, raw_answers):
    # concatenate xs and docs into contexts
    # add eos at the end
    contexts = [[x + d + [2] for d in ds] for x, ds in zip(xs, docs)]
    lens = [len(c) for c in contexts]
    contexts = [c for cs in contexts for c in cs]

    # truncate from left
    maxlen = 1024
    contexts = [c if len(c) < maxlen else c[-maxlen:] for c in contexts]

    contexts = [{"input_ids": c} for c in contexts]
    out = tokenizer.pad(
        contexts,
        padding="longest",
        return_tensors="pt",
    )
    raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
    raw_answers = [a for ans in raw_answers for a in ans]
    raw_answers = [{"input_ids": a} for a in raw_answers]
    answers_out = tokenizer.pad(
        raw_answers,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    return (
        out["input_ids"].to(device),
        out["attention_mask"].to(device),
        answers_out.input_ids,
        answers_out.attention_mask.bool(),
    )


def cat_pad_answers(tokenizer, batch, doc_idxs):
    doc_idxs = doc_idxs.cpu().numpy()
    enc_x = batch.enc_x
    enc_docs = batch.enc_docs

    # flattened. bsz is outer dimension
    x_and_z = []
    for x, idxs in zip(enc_x, doc_idxs):
        for idx in idxs:
            z = enc_docs[idx]
            x_and_z.append(np.concatenate((x, z), 0))
    xz_emb, xz_mask = pad_and_mask(x_and_z)

    raw_answers = batch.answers
    lens = [doc_idxs.shape[1]] * doc_idxs.shape[0]
    raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
    raw_answers = [a for ans in raw_answers for a in ans]
    raw_answers = [{"input_ids": a} for a in raw_answers]
    answers_out = tokenizer.pad(
        raw_answers,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    return (
        torch.tensor(xz_emb).to(device),
        torch.tensor(xz_mask).to(device),
        answers_out.input_ids,
        answers_out.attention_mask.bool(),
    )


def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    answs[answs == model.config.pad_token_id] = -100
    if train:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=answs,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            num_beams=beam,
            min_length=1,
            max_length=20,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if beam > 1:
            outputs = (outputs.sequences, outputs.sequences_scores)
        else:
            raise NotImplementedError
            # scores = model(input_ids=input_ids, attention_mask=attn_mask, labels=outputs).loss
    return outputs


def run_model(
    batch,
    layers,
    answer_model,
    tokenizer,
    answer_tokenizer,
    max_p,
    reg_coeff,
    t,
    beam=2,
    train=True,
    num_z=4,
    z_outputs=None,
):
    for key in batch:
        # if key == "input_ids" or key == "attention_mask":
        if key in [
            "input_ids",
            "attention_mask",
            "doc_idxs",
            "doc_input_ids",
            "doc_attention_mask",
            "enc_x_emb",
            "enc_x_mask",
            "enc_docs_emb",
            "enc_docs_mask",
            "enc_sents_emb",
            "enc_sents_mask",
        ]:
            # doc_input_ids, doc_attention_mask should be cached
            batch[key] = batch[key].to(device)
    bs = len(batch.answers)
    z_size = len(batch.docs)
    p_z, z_outputs = run_lm(layers, batch, bs, train=train, z_outputs=z_outputs)
    if train:
        answer_in, answer_attn, labels, labels_mask = pad_answers(
            answer_tokenizer,
            batch.raw_xs,
            [[batch.docs[i] for i in doc_idxs] for doc_idxs in batch.doc_idxs],
            batch["answers"],
        )
        # top_z = p_z.topk(num_z, -1)

        # answer_in, answer_attn, labels, labels_mask = cat_pad_answers(
        #    answer_tokenizer, batch, top_z.indices,
        # )
        in_len = len(answer_in)
        answ_out = run_answer_model(
            answer_model,
            answer_in,
            answer_attn,
            labels,
            answer_tokenizer,
            beam=beam,
            train=train,
        )
        assert bs == 1
        logits = answ_out.logits.log_softmax(-1)
        # ASSUME N = total num documents during training
        N, T, V = logits.shape
        loss = logits[torch.arange(N)[:, None], torch.arange(T), labels].view(
            bs, num_z, -1
        )
        loss[~labels_mask.view(bs, num_z, -1)] = 0
        loss = -(loss.sum(-1) + p_z).logsumexp(-1).mean()
    else:
        # pick out argmax contexts
        # assuming pouts: bs * z_size
        assert z_size == 55
        idxs = p_z.view(bs, z_size).argmax(-1).view(-1, 1)
        answer_in, answer_attn, labels, labels_mask = pad_answers(
            answer_tokenizer,
            batch.raw_xs,
            [[batch.docs[i] for i in doc_idxs] for doc_idxs in idxs],
            batch["answers"],
        )
        in_len = len(answer_in)
        # only run answer prediction for argmax context
        answ_out = run_answer_model(
            answer_model,
            answer_in,
            answer_attn,
            labels,
            answer_tokenizer,
            beam=beam,
            train=train,
        )
        loss = 0.0
    return answ_out, p_z, loss, z_outputs


def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    m = nn.LogSoftmax(dim=-1)
    exact_match = load_metric("exact_match")
    prior_metric = load_metric("accuracy")
    contrastive_prior_metric = load_metric("accuracy")

    prior_ents = []
    pos_ents = []
    if args.save_results and split == "Valid":
        con_preds = []
        con_golds = []
        con_docs = []
        doc_preds = []
        doc_golds = []
        answer_preds = []
        answer_golds = []

    z_outputs = None
    # run evaluation
    # for step, eval_batch in enumerate(dataloader):
    for step, eval_batch in track(enumerate(dataloader), total=len(dataloader)):
        bs = len(eval_batch.answers)
        n_docs = len(eval_batch.docs)

        if z_outputs is None:
            # precompute z_outputs
            z = eval_batch.doc_input_ids.to(device)
            z_attention_mask = eval_batch.doc_attention_mask.to(device)

            z_outputs = layers[0](
                input_ids=z,
                attention_mask=z_attention_mask,
            )

        # ANSWER EVAL Y|X
        gold = answ_tok.batch_decode(eval_batch.answers, skip_special_tokens=True)
        # ONLY DOING TOP-1
        eval_outs, para_preds, _, _ = run_model(
            eval_batch,
            layers,
            answ_model,
            tok,
            answ_tok,
            max_p=True,
            reg_coeff=args.reg_coeff,
            t=args.sentence_threshold,
            train=False,
            beam=args.beam,
            z_outputs=z_outputs,
        )

        eval_outs, scores = eval_outs
        # eval_outs: bs x max_toks=20
        # scores: bs
        preds = [tok.decode(out, skip_special_tokens=True) for out in eval_outs]
        gold = [normalize_answer(s) for s in gold]
        preds = [normalize_answer(s) for s in preds]
        if args.save_results and split == "Valid":
            answer_preds.append(preds)
            answer_golds.append(gold)
            #answer_results.append((preds, gold))
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )

        # PRIOR Z|X
        idxes = [s.argmax().item() for s in para_preds.view(bs, n_docs)]
        labels = eval_batch.labels
        prior_metric.add_batch(
            predictions=idxes,
            references=labels,
        )

        contrastive_scores = para_preds[
            torch.arange(bs)[:, None],
            eval_batch.doc_idxs,
        ]
        contrastive_preds = contrastive_scores.argmax(-1)
        contrastive_prior_metric.add_batch(
            predictions=contrastive_preds,
            references=[0] * bs,
        )

        if args.save_results and split == "Valid":
            con_preds.append(contrastive_scores)
            con_golds.append([0] * bs)
            con_docs.append(eval_batch.doc_idxs)
            doc_preds.append(para_preds)
            doc_golds.append(labels)

    y_exact_match = exact_match.compute()
    z_acc = prior_metric.compute()
    z_contrastive_acc = contrastive_prior_metric.compute()
    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer EM": y_exact_match,
                f"{split} Subflow Acc": z_acc,
                f"{split} Contrastive Subflow Acc": z_contrastive_acc,
            }
        )
    if args.save_results and split == "Valid":
        torch.save(
            (
                con_preds, con_golds, con_docs,
                doc_preds, doc_golds, answer_preds, answer_golds,
            ),
            f"logging/{args.run_name}|step-{steps}.pt"
        )
    # return z_acc["accuracy"]
    print("contrastive", z_contrastive_acc["accuracy"])
    print("full", z_acc["accuracy"])
    return y_exact_match["exact_match"]


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, truncation_side="left")
    answer_tokenizer = AutoTokenizer.from_pretrained(
        args.answer_model_dir, truncation_side="left"
    )

    model_name = args.model_dir.split("/")[-1]
    run_name = f"fact2-model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} k-{args.num_distractors} tp-{args.truncate_paragraph} beam-{args.beam} reg-{args.reg_coeff} topk-doc-{args.topk_doc}"
    args.run_name = run_name
    all_layers = prepare_model(args)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

    train_dataloader, eval_dataloader = prepare_dataloader(
        tokenizer,
        answer_tokenizer,
        args,
        encoder=all_layers[0],
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler(all_layers + [answer_model], args)

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
                        all_layers[0].save_pretrained(f"{args.output_model_dir}/{run_name}")
                        torch.save(all_layers[1:], f"{args.output_model_dir}/{run_name}-others.pt")
                        answer_model.save_pretrained(f"{args.output_model_dir}/{run_name}-answer")
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
