import torch

def q_doc(encoder, x_ids, x_mask, doc_ids, doc_mask, doc_labels):
    x_out = encoder(input_ids=x_ids, attention_mask=x_mask)
    doc_out = encoder(input_ids=doc_ids, attention_mask=doc_mask)

    score_doc_x = torch.einsum("xh,zh->xz", x_out.pooler_output, doc_out.pooler_output)
    logits_qdoc_x = (score_doc_x / 32)

    log_qdoc_x = logits_qdoc_x.log_softmax(-1)
    neg_log_qdoc = -log_qdoc_x[torch.arange(bsz), doc_labels].mean()
    return log_qdoc_x, neg_log_qdoc
