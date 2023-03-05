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
from prompting.utils import get_dataset, embed, get_dialogues_and_labels

BATCH_SIZE = 128
LOG_NAME = "prompting2"
K = 5


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
        """
        Looks up k-nearest neighbors for query
        """
        def parse(self, out, input):
            dataset, k = self.data
            query_embedding = np.array(out)        
            res = dataset.get_nearest_examples("embeddings", query_embedding, k)
            return [
                dict(
                    doc=doc,
                    title=title,
                    dial=input,
                    score=score,
                )
                for doc, title, score in zip(
                    res.examples["doc"], res.examples["title"], res.scores,
                )
            ]

    class AlignmentPrompt(TemplatePrompt):
        # doesn't work
        #template_file = "prompting/align.pmpt.tpl"
        # works the same as original prompt but more expensive
        #template_file = "prompting/zeroshotalign.pmpt.tpl"
        # works pretty well
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
            input["alignment"] = preds
            return input

    with start_chain(LOG_NAME) as backend:
        #prompt = KnnPrompt(backend.OpenAIEmbed()).chain(AlignmentPrompt(backend.OpenAI()))
        knnprompt = KnnPrompt(backend.OpenAIEmbed(), (doc_embeddings, K))
        prompt = AlignmentPrompt(backend.OpenAI(model="text-davinci-003",max_tokens=512))

        chainprompt = knnprompt.chain(prompt.map())

        doc_acc = evaluate.load("accuracy")
        step_acc = evaluate.load("accuracy")
        #for x in track(dialogues):
        for x in dialogues:
            id = x["id"]
            dial = x["dialogue"]
            true_doc = x["doc"]
            speakers = x["speakers"]

            true_labels = first(np.array(labels[id], dtype=int))


            knnresult = knnprompt(dial)
            docpreds = knnresult["titles"]
            docs = knnresult["docs"]
            scores = knnresult["scores"]
            docpred = docpreds[0]
            doc = docs[0]
            out = prompt.dbg_render_prompt(dict(dial=dial, doc=doc))
            print(out)
            #import pdb; pdb.set_trace()
            result = prompt(dict(dial=dial, doc=doc))
            bi_result = np.copy(result)
            bi_result[1:][bi_result[1:] == bi_result[:-1]] = -1

            result = chainprompt(dial)
            import pdb; pdb.set_trace()

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
