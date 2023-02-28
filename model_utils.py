import torch
import numpy as np

from inference_utils import monotonic_partition


def q_doc(encoder, x_ids, x_mask, doc_ids, doc_mask, doc_labels):
    x_out = encoder(input_ids=x_ids, attention_mask=x_mask)
    doc_out = encoder(input_ids=doc_ids, attention_mask=doc_mask)

    score_doc_x = torch.einsum("xh,zh->xz", x_out.pooler_output, doc_out.pooler_output)
    logits_qdoc_x = (score_doc_x / 32)
    log_qdoc_x = logits_qdoc_x.log_softmax(-1)

    bsz = x_ids.shape[0]
    neg_log_qdoc = -log_qdoc_x[torch.arange(bsz), doc_labels].mean()
    return logits_qdoc_x, neg_log_qdoc


def subsample_docs(
    num_docs, total_num_docs,
    logits_qdoc_x, doc_ids, doc_mask,
    sent_ids, sent_mask,
):
    if num_docs < total_num_docs:
        topk_doc = logits_qdoc_x.topk(num_docs, -1)
        sampled_logits_qdoc_x, doc_idxs = topk_doc
        sampled_doc_ids = doc_ids[doc_idxs]
        sampled_doc_mask = doc_mask[doc_idxs]
        sampled_sent_ids = sent_ids[doc_idxs]
        sampled_sent_mask = sent_mask[doc_idxs]
    else:
        bsz = logits_qdoc_x.shape[0]
        sampled_logits_qdoc_x = logits_qdoc_x
        sampled_doc_ids = doc_ids[None].repeat(bsz, 1, 1)
        sampled_doc_mask = doc_mask[None].repeat(bsz, 1, 1)
        sampled_sent_ids = sent_ids[None].repeat(bsz, 1, 1, 1)
        sampled_sent_mask = sent_mask[None].repeat(bsz, 1, 1, 1)
    return (
        sampled_logits_qdoc_x,
        sampled_doc_ids, sampled_doc_mask,
        sampled_sent_ids, sampled_sent_mask,
    )


def score_step_aligned_turns(
    tok_loss, turn_numbers,
    sent_mask, doc_labels,
    device,
    monotonic=False,
):
    """
    Given tok_loss = log p(words | doc, step), get log p(turn | doc, step).
    """
    bsz, num_docs, num_steps, step_len = sent_mask.shape

    loss_buffer = torch.zeros_like(tok_loss)
    log_p_turn_given_z = torch.scatter_add(loss_buffer, -1, turn_numbers.to(device), tok_loss)

    # padding steps will only have <bos> <eos>, so mask will only have two elements.
    padding_step = sent_mask.sum(-1) <= 2
    log_p_step = torch.zeros(bsz, num_docs, num_steps, device=device)
    log_p_step[padding_step] = float("-inf")
    log_p_step = log_p_step.log_softmax(-1)

    log_p_doc = -np.log(num_docs)

    log_p_turn_step = log_p_turn_given_z + log_p_step[:,:,:,None] + log_p_doc

    if monotonic:
        #logprob_dial = monotonic_partition_old(log_p_turn_z.permute(0,2,1))
        logprob_dial_doc = monotonic_partition(
            log_p_turn_step.view(bsz*num_docs, num_steps, step_len).permute(0,2,1),
            padding_step.view(bsz*num_docs, num_steps),
        )
        logprob_dial = logprob_dial_doc.view(bsz, num_docs).logsumexp(-1)
    else:
        # logsumexp over steps, then sum over turns
        logprob_dial_doc = log_p_turn_z.logsumexp(2).sum(-1)
        logprob_dial = logprob_dial_doc.logsumexp(1)

    #turn_mask = torch.arange(x_len) <= turn_numbers[:,0,-1,None]
    #conversation_logprob = turn_logprobs.masked_fill(~turn_mask.to(device), 0).sum(-1)
    neg_log_py = -logprob_dial.mean()
    return neg_log_py, logprob_dial_doc, log_p_turn_step
