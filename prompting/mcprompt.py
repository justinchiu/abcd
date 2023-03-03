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

from utils.manual_map import subflow_map
from inference_utils import first

BATCH_SIZE = 128
EMBEDDING_MODEL = "text-embedding-ada-002"
LOG_NAME = "prompting"


def get_guidelines(guidelines):
    docs = []
    for flow, subflow_dict in guidelines.items():
        for subflow, content in subflow_dict["subflows"].items():
            actions = content["actions"]
            strings = [content["instructions"][0]]
            for step in actions:
                stepstring = step["text"] + " ".join(step["subtext"])
                strings.append(stepstring)
            strings.append(content["instructions"][1])
            numbered_steps = [
                f"Step {i}: {x}"
                for i, x in enumerate(strings)
            ]
            docs.append({
                "doc": "\n".join(numbered_steps),
                "title": subflow,
            })
    return docs

def load_or_make_dataset():
    with Path("data/guidelines.json").open("r") as f:
        guidelines = json.load(f)
        docs = get_guidelines(guidelines)
        return Dataset.from_list(docs)

def get_dialogue(dial):
    return "\n".join([
        f"Turn {i} {speaker}: {turn}"
        for i, (speaker, turn) in enumerate(dial)
    ])

def get_speakers(dial):
    return [speaker for speaker, turn in dial]


def get_dialogues_and_labels():
    with Path("data/agent_step_annotations.json").open("r") as f:
        agent_labels = json.load(f)["dev"]
    with Path("data/abcd_v1.2.json").open("r") as f:
        data = json.load(f)["dev"]
    return [
        {
            "id": str(x["convo_id"]),
            "dialogue": get_dialogue(x["original"]),
            "doc": subflow_map[x["scenario"]["subflow"]],
            "speakers": get_speakers(x["original"]),
        }
        for x in data
        if str(x["convo_id"]) in agent_labels
    ], agent_labels

def embed(x):
    emb = openai.Embedding.create(input=x["doc"], engine=EMBEDDING_MODEL)
    return {"embeddings": [np.array(emb['data'][i]['embedding'])
                           for i in range(len(emb["data"]))]}


def main():
    data_path = Path("openai-data/guideline-docs.data")
    if data_path.exists():
        doc_embeddings = datasets.load_from_disk(data_path)
    else:
        print("WARNING: rerunning embedding")
        dataset = load_or_make_dataset()
        doc_embeddings = dataset.map(embed, batch_size=BATCH_SIZE, batched=True)
        doc_embeddings.save_to_disk(data_path)
    doc_embeddings.add_faiss_index("embeddings")

    dialogues, labels = get_dialogues_and_labels()

    class KnnPrompt(EmbeddingPrompt):
        def find(self, out, input):
            res = doc_embeddings.get_nearest_examples("embeddings", np.array(out), 5)
            return {
                "query": input,
                "docs": res.examples["doc"],
                "titles": res.examples["title"],
                "scores": res.scores,
            }

    class AlignmentPrompt(Prompt[str, Any]):
        def prompt(self, input: str) -> str:
            return input

        def parse(self, out: str, input) -> Any:
            # Encode the parsing logic
            jsout = json.loads(out)
            numturns = max([x["T"] for x in jsout])
            preds = np.zeros(numturns+1, dtype=int)
            for x in jsout:
                turn = x["T"]
                step = x["S"]
                preds[turn] = step
            return preds

    def prepare_prompt(dialogue, doc):
        # Encode prompting logic. Had to pull this out from prompt class.
        return f"""Manual
{doc}
Dialogue
{dialogue}

Please align each turn in this dialogue to a step in the manual above.
Return the answer only with JSON (no text) in the format [{{"T": turn, "S": step}}].

"""

    with start_chain(LOG_NAME) as backend:
        #prompt = KnnPrompt(backend.OpenAIEmbed()).chain(AlignmentPrompt(backend.OpenAI()))
        knnprompt = KnnPrompt(backend.OpenAIEmbed())
        prompt = AlignmentPrompt(backend.OpenAI(max_tokens=512))

        doc_acc = evaluate.load("accuracy")
        step_acc = evaluate.load("accuracy")
        #for x in track(dialogues):
        for x in dialogues:
            id = x["id"]
            dial = x["dialogue"]
            true_doc = x["doc"]
            speakers = x["speakers"]

            knnresult = knnprompt(dial)

            docpreds = knnresult["titles"]
            docs = knnresult["docs"]
            scores = knnresult["scores"]

            docpred = docpreds[0]
            doc = docs[0]

            true_labels = first(np.array(labels[id], dtype=int))

            result = prompt(prepare_prompt(dial, doc))
            bi_result = np.copy(result)
            bi_result[1:][bi_result[1:] == bi_result[:-1]] = -1

            wrong_result = np.full(bi_result.shape, -2)

            steppred = bi_result if docpred == true_doc else wrong_result

            doc_acc.add(prediction=docpred == true_doc, reference=True)
            agent_mask = np.array([s == "agent" for s in speakers])
            step_acc.add_batch(
                predictions=steppred[agent_mask],
                references=true_labels[agent_mask],
            )
            import pdb; pdb.set_trace()


        docacc = doc_acc.compute()
        stepacc = step_acc.compute()
        print(docacc)
        print(stepacc)

    #show_log(LOG_NAME)

if __name__ == "__main__":
    main()
