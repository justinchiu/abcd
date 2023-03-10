import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any
from rich.progress import track

import openai
from minichain import Prompt, EmbeddingPrompt, TemplatePrompt, show_log, start_chain

from inference_utils import first, monotonic_prediction, first_monotonic_arg_max

from prompting_args import get_args, get_logpath
from prompting_data import Abcd, FloDial
from prompting_utils import embed, Aligner


def get_guidelines(self, guidelines):
    docs = []
    for flow, subflow_dict in guidelines.items():
        for subflow, content in subflow_dict["subflows"].items():
            actions = content["actions"]
            strings = [content["instructions"][0]]
            for step in actions:
                stepstring = step["text"] + " " + " ".join(step["subtext"])
                strings.append(stepstring)
            strings.append(content["instructions"][1])
            numbered_steps = [
                f"Step {i}: {x}"
                for i, x in enumerate(strings)
            ]
            docs.append({
                "doc": "\n".join(numbered_steps),
                "text": "\n".join(numbered_steps),
                "title": subflow,
                "steps": strings,
            })
    return docs

def get_docs(self):
    with Path("data/guidelines.json").open("r") as f:
        guidelines = json.load(f)
        docs = self.get_guidelines(guidelines)
        return docs

class ExpandPrompt(Prompt[str,str]):
    def prompt(self, query: str) -> str:
        return query
    def parse(self, output, input) -> str:
        return output

def main():
    doc_obj = None
    data_path = None
    doc_obj = Abcd()
    data_path = Path("openai-data/guideline-docs.data")
    step_path = Path("openai-data/abcd-steps")

    get_docs = doc_obj.get_docs
    get_dialogues_and_labels = doc_obj.get_dialogues_and_labels

    doc_dataset = get_docs()

    dialogues, labels = get_dialogues_and_labels()

    with start_chain("promptlogs/chatgpt-chat-expansion") as backend:
        prompt = ExpandPrompt(backend.OpenAIChat(model="gpt-3.5-turbo", max_tokens=512))
        for doc in doc_dataset:
            title = doc["title"]
            for step in doc["steps"]:
                print(step)
                answer = prompt(step)
                print(answer)
                import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
