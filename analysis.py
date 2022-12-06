import torch
import numpy as np

logfiles = [
    "logging/fact2-model-roberta-large lr-2e-05 bs-1 k-7 tp-0 beam-2 reg-0 topk-doc-4|step-5000.pt",
    "logging/fact2-model-roberta-large lr-2e-05 bs-1 k-7 tp-0 beam-2 reg-0 topk-doc-8 hn-3|step-10000.pt",
    "logging/fact2-model-roberta-large lr-2e-05 bs-1 k-7 tp-0 beam-2 reg-0 topk-doc-8 hn-3|step-55000.pt",
    "logging/simp2-model-roberta-large lr-5e-05 bs-1 k-3 tp-0 beam-2 reg-0 topk-doc-4|step-5000.pt",
    "logging/simp2-model-roberta-large lr-5e-05 bs-1 k-3 tp-0 beam-2 reg-0 topk-doc-4|step-60000.pt",
]
for logfile in logfiles:
    (
        con_preds,
        con_golds,
        con_docs,
        doc_preds,
        doc_golds,
        answer_preds,
        answer_golds,
    ) = torch.load(
        logfile,
        map_location="cpu",
    )
    print(logfile)
    print(doc_preds[0])
    print(con_preds[0])

import pdb

pdb.set_trace()
