import numpy as np
import torch
from torch_struct import LinearChainCRF

def monotonic_prediction(unary):
    T, Z = unary.shape
    potentials = unary[:,:,None].repeat(1,1,Z)
    # only one starting state
    potentials[0,:,1:] = float("-inf")
    # monotonicity constraint
    transition = torch.tril(torch.ones(Z,Z))
    log_transition = transition.log()
    full_potentials = potentials + log_transition
    crf = LinearChainCRF(full_potentials[None])
    binary_argmax = crf.argmax.detach()
    return binary_argmax.nonzero()[:,2]

def first_monotonic_prediction(unary):
    monotonic_preds = monotonic_prediction(unary)
    # annotation has first prediction as -1
    return [-1] + monotonic_preds[1:].masked_fill(
        monotonic_preds[1:] <= monotonic_preds[:-1],
        -1,
    ).tolist()

def first_argmax_prediction(unary):
    preds = unary.argmax(-1).numpy()
    vals, idxs = np.unique(preds, return_index=True)
    x = np.full(preds.shape, -1)
    x[idxs] = vals
    return x
