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
    prompt = f"in order to {step}\n#a customer representative said, {turn}"
    prompt = f"{step}\n# Agent: a customer representative said, {turn}"
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
    idx = response["logprobs"]["tokens"].index("#")
    return logprobs[idx+1:]

turn = "Agent: Ok, I see your refund is In Progress and it looks like it should be going through to completion later today or by tomorrow at the latest"
span = """Agent: Ok, I see your refund is In Progress and it looks like it should be going through to completion later today or by tomorrow at the latest
Customer: okay, thank you
Agent: you're welcome"""

step0 = """Customers want to know the status and payment method of their refund. In this case:"""
step1 = """Pull up Account: Get Full Name or Account ID for [Pull up Account]"""
step2 = """Validate Purchase: Gather details to find the appropriate order:
* Username - first
* Email Address - second
* Order ID - third
Then use a KB query to [Validate Purchase]"""
step3 = """Respond with the information to the customer in natural language
* Pick a refund status from the list of three options
> not started, in progress, complete
* Tell the customer this is their refund status
* Pick a payment method from the following three options
> online, by phone, by chat
* Tell the customer this is how they initiated that refund (i.e their payment method)"""
step4 = """Notify Internal Team: If the customer is not satisfied with your answer, and would like to change their refund
* If they do not like the status:
> Enter 'manager' into [Notify Internal Team]
> Explain that you have escalated the issue to the manager
* If they do not like the payment method:
> Enter “change method” into [Update Order]"""
step5 = """As usual, end by asking if the customer needs anything else."""


steps = [step0,step1,step2,step3,step4,step5]


for step in steps:
    print(np.sum(gpt(step, turn)))
