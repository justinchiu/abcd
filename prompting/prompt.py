import openai
import torch
import json
import time
import os
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score as accscore

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


from subflow_data import get_abcd_dataset

from inference_utils import (
    first,
    monotonic_prediction,
    first_monotonic_prediction,
    first_argmax_prediction,
)


openai_api_key = os.environ.get("OPENAI_API_KEY")
model_name = "text-davinci-003"

openai.api_key = openai_api_key


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def gpt(step, turn, stop_tokens=None, query_kwargs=None):
    prompt = f"in order to {step}\na customer representative said, {turn}"
    use_query_kwargs = {
        'engine': model_name,
        'max_tokens': 0,
        'temperature': 0,
        "top_p": 1,
        "n": 1,
        "logprobs": 1,
        "echo": True,
    }
    if query_kwargs is not None:
      use_query_kwargs.update(query_kwargs)

    response = completion_with_backoff(
        prompt=prompt, stop=stop_tokens, **use_query_kwargs
    )['choices'][0]#['logprobs']
    #return response
    logprobs = response["logprobs"]["token_logprobs"]
    idx = response["logprobs"]["tokens"].index("\n")
    return logprobs[idx+1:]

with Path("data/agent_step_annotations.json").open("r") as f:
    agent_all_labels = json.load(f)
agent_labels = agent_all_labels["dev"]

val_dataset, processed_docs, doc_sents, subflow_map = get_abcd_dataset(
    "dev", 0, 0, lower=False, truncate_early=False
)

def get_gpt_scores():
    gpt_scores = {}
    for e in val_dataset:
        xs = e["xs"]
        str_id = str(e["ids"])
        subflow = e["subflows"]
        # idx = get_index[subflow]
        idx = subflow_map[subflow]
        doc_steps = doc_sents[idx]

        #if str_id not in labels:
        if str_id not in agent_labels:
            continue

        speakers, turns = list(zip(*e["turns"]))

        labels = np.array(agent_labels[str_id])
        first_labels = first(labels)
        turn_scores = {}
        for i, (speaker, turn) in enumerate(zip(speakers, turns)):
            if speaker == "agent":
                step_scores = []
                for step in doc_steps:
                    logprob_turn = gpt(step, turn)
                    step_scores.append(np.sum(logprob_turn))
                turn_scores[i] = step_scores
        gpt_scores[str_id] = turn_scores
    return gpt_scores

gpt_score_path = Path("logging/gpt-turn-scores.pt")
if gpt_score_path.exists():
    print(f"Loading gpt scores from {gpt_score_path}")
    gpt_scores = torch.load(gpt_score_path)
else:
    gpt_scores = get_gpt_scores()
    print(f"Saving gpt scores to {gpt_score_path}")
    torch.save(gpt_scores, gpt_score_path)


all_true_labels = []
all_monotonic_preds = []
all_argmax_preds = []
all_first_monotonic_preds = []
all_first_argmax_preds = []
for e in val_dataset:
    xs = e["xs"]
    str_id = str(e["ids"])
    subflow = e["subflows"]
    # idx = get_index[subflow]
    idx = subflow_map[subflow]

    #if str_id not in labels:
    if str_id not in agent_labels or str_id not in gpt_scores:
        continue

    speakers, turns = list(zip(*e["turns"]))

    # check lexical accuracy of align/not
    labels = np.array(agent_labels[str_id])
    first_labels = first(labels)

    doc_steps = doc_sents[idx]

    agent_mask = np.array([s == "agent" for s,_ in e["turns"]])

    this_true_labels = first_labels[agent_mask]
    agent_unary = torch.tensor([scores for (turn, scores) in sorted(gpt_scores[str_id].items())])

    monotonic_preds = monotonic_prediction(agent_unary)
    argmax_preds = agent_unary.argmax(-1)
    first_monotonic_preds = first_monotonic_prediction(agent_unary)
    first_argmax_preds = first_argmax_prediction(agent_unary)

    all_true_labels.extend(this_true_labels)
    all_monotonic_preds.extend(monotonic_preds)
    all_argmax_preds.extend(argmax_preds)
    all_first_monotonic_preds.extend(first_monotonic_preds)
    all_first_argmax_preds.extend(first_argmax_preds)

    for i, (speaker, turn) in enumerate(zip(speakers, turns)):
        if speaker == "agent":
            step_scores = gpt_scores[str_id][i]
            prompts = []
            for step in doc_steps:
                prompt = f"in order to {step}\na customer representative said, {turn}"
                prompts.append(prompt)
            #import pdb; pdb.set_trace()


print(f"argmax")
print(accscore(all_true_labels, all_argmax_preds))
print(f"monotonic")
print(accscore(all_true_labels, all_monotonic_preds))
print(f"first monotonic")
print(accscore(all_true_labels, all_first_monotonic_preds))
print(f"first argmax")
print(accscore(all_true_labels, all_first_argmax_preds))

import pdb; pdb.set_trace()
