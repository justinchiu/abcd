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

from inference_utils import first
from prompting_utils import get_dataset, embed, get_dialogues_and_labels

BATCH_SIZE = 128
LOG_NAME = "prompting2"
K = 3


def main():
    data_path = Path("openai-data/guideline-docs.data")
    if data_path.exists():
        print(f"LOADING EMBEDDING FROM {data_path}")
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
                    steps=steps,
                )
                for doc, title, steps, score in zip(
                    res.examples["doc"], res.examples["title"],
                    res.examples["steps"], res.scores,
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

    class DocsPrompt(Prompt[str, int]):
        def prompt(self, inp: str) -> str:
            "Encode prompting logic"
            return inp

        def parse(self, out: str, inp) -> int:
            # Encode the parsing logic
            return json.loads(out)["D"]

    with start_chain(LOG_NAME) as backend:
        #prompt = KnnPrompt(backend.OpenAIEmbed()).chain(AlignmentPrompt(backend.OpenAI()))
        knnprompt = KnnPrompt(backend.OpenAIEmbed(), (doc_embeddings, K))
        prompt = AlignmentPrompt(backend.OpenAI(model="text-davinci-003",max_tokens=1024))
        chainprompt = knnprompt.chain(prompt.map())
        docsprompt = DocsPrompt(backend.OpenAI(model="text-davinci-003",max_tokens=5))

        doc_acc = evaluate.load("accuracy")
        step_acc = evaluate.load("accuracy")
        for x in track(dialogues):
        #for x in dialogues:
            id = x["id"]
            dial = x["dialogue"]
            true_doc = x["doc"]
            speakers = x["speakers"]
            turns = x["turns"]

            true_labels = first(np.array(labels[id], dtype=int))

            results = chainprompt(dial)

            for i,result in enumerate(results):
                if len(result["alignment"]) != len(turns):
                    del results[i]

            alignments = np.stack([x["alignment"] for x in results])
            alignments_b = np.copy(alignments)
            alignments_b[:,1:][alignments[:,1:] == alignments[:,:-1]] = -1
            scores = [x["score"] for x in results]
            titles = [x["title"] for x in results]
            stepss = [x["steps"] for x in results]

            # doc selection prompt
            docprompt = []
            for result in results:
                steps = result["steps"]
                alignment = np.copy(result["alignment"])
                #alignment[1:][alignment[1:] == alignment[:-1]] = -1

                strbuilder = []
                for turn, stepidx in zip(turns, alignment):
                    if stepidx != -1 and stepidx < len(steps):
                        strbuilder.append(f"{steps[stepidx]} => {turn}")
                    else:
                        strbuilder.append(turn)
                docprompt.append("\n".join(strbuilder))

            docspromptstring = "\n\n".join([f"Document {i}\n{x}" for i,x in enumerate(docprompt)])
            instruction = "Which document has the best rationales? Rationale => turn. Give your answer as JSON {\"D\": number}.\n\n"
            bestidx = docsprompt(instruction + docspromptstring + "\nAnswer:")

            alignment = alignments_b[bestidx]
            docpred = titles[bestidx]
            steps = stepss[bestidx]

            wrong_result = np.full(alignment.shape, -2)

            steppred = alignment if docpred == true_doc else wrong_result

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
