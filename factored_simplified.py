from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb

# from dataset import prepare_simplified, SimplifiedHotpotQADataset
# from eba_subflow_dataset import prepare_subflow_abcd, SubflowAbcdDataset
from eba_subflow_factored_dataset import prepare_subflow_abcd, SubflowAbcdDataset
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


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["paras"])
        paras = [feature.pop("paras") for feature in features]
        lengths = [len(i) for i in paras]
        # paras = [p for ps in paras for p in ps]
        paras = [{"input_ids": x} for x in paras]

        batch = self.tokenizer.pad(
            paras,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        answer_name = "answs"
        raw_answers = [feature.pop(answer_name) for feature in features]
        context_name = "supps"
        contexts = [feature.pop(context_name) for feature in features]
        labels_name = "labels"
        labels = [feature.pop(labels_name) for feature in features]
        docs_name = "docs"
        docs = [feature.pop(docs_name) for feature in features][0]
        doc_idxs_name = "doc_idxs"
        doc_idxs = [feature.pop(doc_idxs_name) for feature in features]

        doc_lengths = [len(x) for x in docs]
        batch_docs = self.tokenizer.pad(
            [{"input_ids": x} for x in docs],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Add back labels
        batch["contexts"] = contexts
        batch["answers"] = raw_answers
        batch["lengths"] = lengths
        batch["labels"] = labels

        batch["docs"] = docs
        batch["doc_lengths"] = doc_lengths
        batch["doc_idxs"] = torch.tensor(doc_idxs, dtype=torch.long)

        batch["doc_input_ids"] = batch_docs.input_ids
        batch["doc_attention_mask"] = batch_docs.attention_mask

        return batch


def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    return [model, linear]


def prepare_dataloader(tok, answer_tok, args):
    # paras, supps, answs, ds = prepare_simplified(tok, answer_tok, "train", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    # tparas, tsupps, tansws, tds = prepare_simplified(tok, answer_tok, "validation", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    # train_dataset = SimplifiedHotpotQADataset(paras, supps, answs, ds)
    # eval_dataset = SimplifiedHotpotQADataset(tparas, tsupps, tansws, tds)
    paras, docs, doc_idxs, supps, answs, labels = prepare_subflow_abcd(
        tok, answer_tok, "train", num_distractors=args.num_distractors
    )
    tparas, tdocs, tdoc_idxs, tsupps, tansws, tlabels = prepare_subflow_abcd(
        tok, answer_tok, "val", num_distractors=args.num_distractors
    )
    train_dataset = SubflowAbcdDataset(paras, docs, doc_idxs, supps, answs, labels)
    eval_dataset = SubflowAbcdDataset(tparas, tdocs, tdoc_idxs, tsupps, tansws, tlabels)
    data_collator = DataCollatorForMultipleChoice(
        tok, padding="longest", max_length=512
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
    z = batch.doc_input_ids[batch.doc_idxs]
    z_attention_mask = batch.doc_attention_mask[batch.doc_idxs]

    bsz, ndocs = batch.doc_idxs.shape
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
    z_pooled_output = z_outputs[1].view(bsz, ndocs, -1)
    logits = torch.einsum(
        "bh,bdh->bd",
        x_pooled_output,
        z_pooled_output.view(bsz, ndocs, -1),
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


def pad_answers(tokenizer, contexts, raw_answers):
    lens = [len(c) for c in contexts]
    contexts = [c for cs in contexts for c in cs]
    contexts = [{"input_ids": c} for c in contexts]
    out = tokenizer.pad(
        contexts,
        padding="longest",
        return_tensors="pt",
    )
    raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
    raw_answers = [a for ans in raw_answers for a in ans]
    raw_answers = [{"input_ids": a} for a in raw_answers]
    answers = tokenizer.pad(
        raw_answers,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=False,
    )["input_ids"]

    return (
        out["input_ids"].to(device),
        out["attention_mask"].to(device),
        answers.to(device),
    )


def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    answs[answs == model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs)
    else:
        outputs = model.generate(
            input_ids,
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
        ]:
            # doc_input_ids, doc_attention_mask should be cached
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    n_distractors = len(batch.contexts[0])
    pouts, z_outputs = run_lm(layers, batch, bs, train=train, z_outputs=z_outputs)
    if train:
        answer_in, answer_attn, labels = pad_answers(
            answer_tokenizer, batch["contexts"], batch["answers"]
        )
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
        loss = logits[torch.arange(N)[:, None], torch.arange(T), labels]
        loss = -(loss.sum(-1) + pouts).logsumexp(-1).mean()
    else:
        # pick out argmax contexts
        # assuming pouts: bs * n_distractors
        idxs = pouts.view(bs, n_distractors).argmax(-1).tolist()
        contexts = [[c[i]] for i, c in zip(idxs, batch["contexts"])]
        answer_in, answer_attn, labels = pad_answers(
            answer_tokenizer, contexts, batch["answers"]
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
    return answ_out, pouts, loss, z_outputs


def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    m = nn.LogSoftmax(dim=-1)
    exact_match = load_metric("exact_match")
    metric = load_metric("accuracy", "multilabel")
    prior_exact_match = load_metric("exact_match")
    prior_metric = load_metric("accuracy", "multilabel")
    prior_ents = []
    pos_ents = []
    if args.save_results and split == "Valid":
        para_results = []
        answ_results = []

    z_outputs = None
    for step, eval_batch in track(enumerate(dataloader), total=len(dataloader)):
        # for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["answers"])
        n_distractors = len(eval_batch["contexts"][0])

        # ANSWER EVAL Y|X
        gold = answ_tok.batch_decode(eval_batch["answers"], skip_special_tokens=True)
        # ONLY DOING TOP-1
        eval_outs, para_preds, _, z_outputs = run_model(
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
        if args.num_distractors > 0:
            z_outputs = None
        eval_outs, scores = eval_outs
        # eval_outs: bs x max_toks=20
        # scores: bs
        preds = [tok.decode(out, skip_special_tokens=True) for out in eval_outs]
        gold = [normalize_answer(s) for s in gold]
        preds = [normalize_answer(s) for s in preds]
        if args.save_results and split == "Valid":
            answ_results.append((preds, gold))
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )

        # cant compute posterior with only argmax z
        # POSTERIOR Z|X,Y
        # para_tmp = [[s.argmax().item()] for s in scores]
        # correct z is always 0
        # labels = [[0]] * bs
        # metric.add_batch(
        #    predictions=para_tmp,
        #    references=labels,
        # )

        # PRIOR Z|X
        para_tmp = [[s.argmax().item()] for s in para_preds.view(bs, n_distractors)]
        idxes = [s.argmax().item() for s in para_preds.view(bs, n_distractors)]
        # correct z is always 0
        labels = [[0]] * bs
        prior_metric.add_batch(
            predictions=para_tmp,
            references=labels,
        )

        if args.save_results and split == "Valid":
            para_results += idxes
        """
        preds = []
        for i in range(len(idxes)):
            curr_out = eval_outs[i][idxes[i]]
            pred = tok.decode(curr_out, skip_special_tokens=True)
            preds.append(pred)
        preds = [normalize_answer(s) for s in preds]
        prior_exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        """

    y_exact_match = exact_match.compute()
    # pos_para_acc = metric.compute()
    # prior_eval_metric = prior_exact_match.compute()
    z_acc = prior_metric.compute()
    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer EM": y_exact_match,
                f"{split} Subflow Acc": z_acc,
            }
        )
    if args.save_results and split == "Valid":
        torch.save(
            (para_results, answ_results), f"logging/{args.run_name}|step-{steps}.pt"
        )
    # return y_exact_match['exact_match']
    return z_acc["accuracy"]


def evaluate_full(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    m = nn.LogSoftmax(dim=-1)
    exact_match = load_metric("exact_match")
    prior_metric = load_metric("accuracy")
    prior_ents = []
    pos_ents = []
    if args.save_results and split == "Valid":
        para_results = []
        answ_results = []

    z_outputs = None
    for step, eval_batch in track(enumerate(dataloader), total=len(dataloader)):
        # for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["answers"])
        n_distractors = len(eval_batch["contexts"][0])

        # ANSWER EVAL Y|X
        gold = answ_tok.batch_decode(eval_batch["answers"], skip_special_tokens=True)
        # ONLY DOING TOP-1
        eval_outs, para_preds, _, z_outputs = run_model(
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
        if args.num_distractors > 0:
            z_outputs = None
        eval_outs, scores = eval_outs
        # eval_outs: bs x max_toks=20
        # scores: bs
        preds = [tok.decode(out, skip_special_tokens=True) for out in eval_outs]
        gold = [normalize_answer(s) for s in gold]
        preds = [normalize_answer(s) for s in preds]
        if args.save_results and split == "Valid":
            answ_results.append((preds, gold))
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )

        # cant compute posterior with only argmax z
        # POSTERIOR Z|X,Y
        # para_tmp = [[s.argmax().item()] for s in scores]
        # correct z is always 0
        # labels = [[0]] * bs
        # metric.add_batch(
        #    predictions=para_tmp,
        #    references=labels,
        # )

        # PRIOR Z|X
        idxes = [s.argmax().item() for s in para_preds.view(bs, n_distractors)]
        # correct z is always 0
        labels = batch.labels
        prior_metric.add_batch(
            predictions=idxes,
            references=labels,
        )

        if args.save_results and split == "Valid":
            para_results += idxes
        """
        preds = []
        for i in range(len(idxes)):
            curr_out = eval_outs[i][idxes[i]]
            pred = tok.decode(curr_out, skip_special_tokens=True)
            preds.append(pred)
        preds = [normalize_answer(s) for s in preds]
        prior_exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        """

    y_exact_match = exact_match.compute()
    # pos_para_acc = metric.compute()
    # prior_eval_metric = prior_exact_match.compute()
    z_acc = prior_metric.compute()
    if not args.nolog:
        wandb.log(
            {
                "step": steps,
                f"{split} Answer EM": y_exact_match,
                f"{split} Subflow Acc": z_acc,
            }
        )
    if args.save_results and split == "Valid":
        torch.save(
            (para_results, answ_results), f"logging/{args.run_name}|step-{steps}.pt"
        )
    # return y_exact_match['exact_match']
    return z_acc["accuracy"]


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, truncation_side="left")
    answer_tokenizer = AutoTokenizer.from_pretrained(
        args.answer_model_dir, truncation_side="left"
    )
    # data = load_hotpotqa()
    train_dataloader, eval_dataloader = prepare_dataloader(
        tokenizer, answer_tokenizer, args
    )

    model_name = args.model_dir.split("/")[-1]
    run_name = f"factored-model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} k-{args.num_distractors} tp-{args.truncate_paragraph} beam-{args.beam} reg-{args.reg_coeff}"
    args.run_name = run_name
    all_layers = prepare_model(args)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

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
                        all_layers[0].save_pretrained(
                            f"{args.output_model_dir}/{run_name}"
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
