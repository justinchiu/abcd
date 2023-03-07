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

from inference_utils import first, monotonic_prediction, first_monotonic_arg_max

#from prompting_utils import get_dataset, embed, get_dialogues_and_labels
from prompting_data import Abcd, FloDial
from prompting_utils import (
    embed,
    get_bm25,
)

NUM_EXAMPLES = 25
BATCH_SIZE = 128
K_DOCS = 3
EMBEDDING_MODEL = "text-embedding-ada-002"
LOG_NAME = "prompting"

USE_CHAT = False
MODEL = "gpt-3.5-turbo" if USE_CHAT else "text-davinci-003"

dataset_choices = ["abcd", "flodial", "sgd"]
DATASET = dataset_choices[1]


def main():
    dataset_obj = None
    data_path = None
    if DATASET == "abcd":
        doc_obj = Abcd()
        data_path = Path("openai-data/guideline-docs.data")
    elif DATASET == "flodial":
        doc_obj = FloDial()
        data_path = Path("openai-data/flodial-guideline-docs.data")
    else:
        raise NotImplementedError(f"Unimplemented dataset {DATASET}")

    print(f"RUNNING GPT ON DATASET {DATASET}")

    get_docs = doc_obj.get_docs
    get_dialogues_and_labels = dataset_obj.get_dialogues_and_labels

    if data_path.exists():
        print(f"Loading embedding from {data_path}")
        doc_embeddings = datasets.load_from_disk(data_path)
    else:
        print("WARNING: rerunning embedding")
        doc_dataset = get_docs()
        doc_embeddings = dataset.map(embed, batch_size=BATCH_SIZE, batched=True)
        doc_embeddings.save_to_disk(data_path)
        print(f"Saved to {data_path}")
    doc_embeddings.add_faiss_index("embeddings")

    dialogues, labels = get_dialogues_and_labels()

    # BM25
    bm25d, bm25s = get_bm25(doc_embeddings)

    class KnnPrompt(EmbeddingPrompt):
        def find(self, out, input):
            dataset, k = self.data
            res = dataset.get_nearest_examples("embeddings", np.array(out), k)
            return {
                "query": input,
                "docs": res.examples["doc"],
                "titles": res.examples["title"],
                "steps": res.examples["steps"],
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
                #preds[turn] = step if step != "N/A" else -1
                preds[turn] = step
            return preds

    with start_chain(LOG_NAME) as backend:
        knnprompt = KnnPrompt(backend.OpenAIEmbed(), (doc_embeddings, K_DOCS))
        completion_backend = backend.OpenAIChat if USE_CHAT else backend.OpenAI
        prompt = AlignmentPrompt(completion_backend(model=MODEL,max_tokens=1024))

        doc_acc = evaluate.load("accuracy")
        step_acc = evaluate.load("accuracy")
        #for x in track(dialogues):
        for x in dialogues[:NUM_EXAMPLES]:
            id = x["id"]
            dial = x["dialogue"]
            true_doc = x["doc"]
            speakers = x["speakers"]
            turns = x["turns"]
            true_labels = first(np.array(labels[id], dtype=int))
            wrong_result = np.full(true_labels.shape, -2)

            # DOCUMENT SCORING
            # ada embedding
            knnresult = knnprompt(dial)
            docpreds = knnresult["titles"]
            docs = knnresult["docs"]
            steps = knnresult["steps"]
            scores = knnresult["scores"]

            # bm25
            lexical_scores = bm25d.get_scores(dial.lower().split())
            lexical_doc_idxs = np.argsort(-lexical_scores)[:3].tolist()

            lexical_docpreds = [doc_embeddings[x]["title"] for x in lexical_doc_idxs]
            lexical_docs = [doc_embeddings[x]["doc"] for x in lexical_doc_idxs]
            lexical_steps = [doc_embeddings[x]["steps"] for x in lexical_doc_idxs]
            lexical_scores = lexical_scores[lexical_doc_idxs]

            # STEP PREDICTION
            # lexical
            lexical_alignments = []
            lexical_align_scores = []
            for idx, title, doc, steps, score in zip(
                lexical_doc_idxs, lexical_docpreds, lexical_docs, lexical_steps, lexical_scores,
            ):
                scores = np.stack([
                    bm25s[idx].get_scores(turn.split())
                    for turn in turns
                ]) # shape = turns x steps
                lexical_align_preds = scores.argmax(-1)
                lexical_align_preds[1:][lexical_align_preds[1:] == lexical_align_preds[:-1]] = -1
                lexical_align_score = scores.max(-1).sum()
                lexical_alignments.append(lexical_align_preds)
                lexical_align_scores.append(lexical_align_score)

            lexical_argmax = np.argmax(lexical_align_scores)
            lexical_docpred = lexical_alignments[lexical_argmax]
            lexical_doc = lexical_docpreds[lexical_argmax]

            """
            # gpt
            docpred = docpreds[0]
            doc = docs[0]
            out = prompt.dbg_render_prompt(dict(dial=dial, doc=doc))
            print(out)
            #import pdb; pdb.set_trace()
            result = prompt(dict(dial=dial, doc=doc))
            bi_result = np.copy(result)
            # length correction fn
            if len(bi_result) != len(true_labels):
                # need to correct length. should be rare
                if len(bi_result) > len(true_labels):
                    bi_result = bi_result[:len(true_labels)]
                elif len(bi_result) < len(true_labels):
                    new_result = np.full(true_labels.shape, -2)
                    new_result[:len(bi_result)] = bi_result
                    bi_result = new_result
            # filter out repeats fn
            bi_result[1:][bi_result[1:] == bi_result[:-1]] = -1
            steppred = bi_result if docpred == true_doc else wrong_result
            """

            # DOCUMENT+STEP SELECTION
            steppred = lexical_docpred
            docpred = lexical_doc

            doc_acc.add(prediction=docpred == true_doc, reference=True)
            agent_mask = np.array([s == "agent" for s in speakers])

            step_acc.add_batch(
                predictions=steppred[agent_mask],
                references=true_labels[agent_mask],
            )

        docacc = doc_acc.compute()
        stepacc = step_acc.compute()
        print("docacc")
        print(docacc)
        print("stepacc")
        print(stepacc)

    #show_log(LOG_NAME)

if __name__ == "__main__":
    main()
