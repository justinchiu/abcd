import evaluate
import datasets
from datasets import Dataset
import numpy as np
import torch
from pathlib import Path
import json
from typing import Any
from rich.progress import track

from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)

import openai
from minichain import Prompt, EmbeddingPrompt, TemplatePrompt, show_log, start_chain

from utils.manual_map import subflow_map
from inference_utils import first

from prompting_utils import get_dataset, embed, get_dialogues_and_labels

BATCH_SIZE = 128
EMBEDDING_MODEL = "text-embedding-ada-002"
MODEL = "gpt-3.5-turbo"
LOG_NAME = "prompting"

def main():
    data_path = Path("openai-data/guideline-docs.data")
    if data_path.exists():
        doc_embeddings = datasets.load_from_disk(data_path)
    else:
        print("WARNING: rerunning embedding")
        dataset = get_dataset()
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

    class AlignmentPrompt(TemplatePrompt):
        #template_file = "prompting/align.pmpt.tpl"
        #template_file = "prompting/zeroshotalign.pmpt.tpl"
        template_file = "prompting/original.pmpt.tpl"

        def dbg_render_prompt(self, kwargs):
            if self.template_file:
                tmp = Environment(loader=FileSystemLoader(".")).get_template(
                    name=self.template_file
                )
            elif self.template:
                tmp = self.template  # type: ignore
            else:
                tmp = Template(self.prompt_template)
            if isinstance(kwargs, dict):
                x = tmp.render(**kwargs)
            else:
                x = tmp.render(**asdict(kwargs))
            return x


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

    with start_chain(LOG_NAME) as backend:
        #prompt = KnnPrompt(backend.OpenAIEmbed()).chain(AlignmentPrompt(backend.OpenAI()))
        knnprompt = KnnPrompt(backend.OpenAIEmbed())
        prompt = AlignmentPrompt(backend.OpenAI(model=MODEL,max_tokens=1024))

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

            out = prompt.dbg_render_prompt(dict(dial=dial, doc=doc))
            print(out)
            #import pdb; pdb.set_trace()
            result = prompt(dict(dial=dial, doc=doc))
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

        docacc = doc_acc.compute()
        stepacc = step_acc.compute()
        print(docacc)
        print(stepacc)

    #show_log(LOG_NAME)

if __name__ == "__main__":
    main()
