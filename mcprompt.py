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

from prompting_args import get_args
from prompting_data import Abcd, FloDial
from prompting_utils import embed, Aligner


def main(args):
    doc_obj = None
    data_path = None
    if args.dataset == "abcd":
        doc_obj = Abcd()
        data_path = Path("openai-data/guideline-docs.data")
    elif args.dataset == "flodial":
        doc_obj = FloDial()
        data_path = Path("openai-data/flodial-guideline-docs.data")
    else:
        raise NotImplementedError(f"Unimplemented dataset {args.dataset}")

    print(f"RUNNING ON DATASET {args.dataset}")

    get_docs = doc_obj.get_docs
    get_dialogues_and_labels = doc_obj.get_dialogues_and_labels

    if data_path.exists():
        print(f"Loading embedding from {data_path}")
        doc_embeddings = datasets.load_from_disk(data_path)
    else:
        print("WARNING: rerunning embedding")
        doc_dataset = get_docs()
        doc_embeddings = dataset.map(embed, batch_size=128, batched=True)
        doc_embeddings.save_to_disk(data_path)
        print(f"Saved to {data_path}")
    doc_embeddings.add_faiss_index("embeddings")

    dialogues, labels = get_dialogues_and_labels()

    with start_chain(args.log_name) as backend:
        # initialize decision model
        aligner = Aligner(args, doc_embeddings, backend)

        doc_acc = evaluate.load("accuracy")
        step_acc = evaluate.load("accuracy")

        #for x in track(dialogues):
        for x in dialogues[:args.num_examples]:
            id = x["id"]
            dial = x["dialogue"]
            true_doc = x["doc"]
            speakers = x["speakers"]
            turns = x["turns"]
            true_labels = first(np.array(labels[id], dtype=int))
            all_wrong = np.full(true_labels.shape, -2)

            # DOCUMENT SCORING
            doc_selection = aligner.select_docs(dial)

            # STEP PREDICTION
            step_align = aligner.align_dial(dial, turns, doc_selection)

            # DOCUMENT+STEP SELECTION
            steppred, docpred = aligner.rerank_align(step_align)
            #steppred = lexical_docpred
            #docpred = lexical_doc

            steppred[1:][steppred[1:] == steppred[:-1]] = -1

            steppred = steppred if docpred == true_doc else all_wrong

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

    #show_log(args.log_name)

if __name__ == "__main__":
    args = get_args()
    main(args)
