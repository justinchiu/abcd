import numpy as np
import torch
from torch_struct import LinearChainCRF

def monotonic_prediction(unary):
    T, Z = unary.shape
    potentials = unary[:,:,None].repeat(1,1,Z)
    # only one starting state
    potentials[0,:,1:] = float("-inf")
    # monotonicity constraint
    transition = torch.tril(torch.ones(Z,Z, device=unary.device))
    log_transition = transition.log()
    full_potentials = potentials + log_transition
    crf = LinearChainCRF(full_potentials[None])
    binary_argmax = crf.argmax.detach()
    return binary_argmax.nonzero()[:,2]

def first_monotonic_prediction(unary):
    preds = monotonic_prediction(unary).cpu().numpy()
    vals, idxs = np.unique(preds, return_index=True)
    x = np.full(preds.shape, -1)
    x[idxs] = vals
    x[0] = -1
    return torch.tensor(x)
    # annotation has first prediction as -1
    return torch.tensor([-1] + monotonic_preds[1:].masked_fill(
        monotonic_preds[1:] <= monotonic_preds[:-1],
        -1,
    ).tolist())

def first_argmax_prediction(unary):
    preds = unary.argmax(-1).cpu().numpy()
    vals, idxs = np.unique(preds, return_index=True)
    x = np.full(preds.shape, -1)
    x[idxs] = vals
    return torch.tensor(x)

def most_confident_step(unary):
    import pdb; pdb.set_trace()
    preds = unary.argmax(-1).numpy()
    vals, idxs = np.unique(preds, return_index=True)
    x = np.full(preds.shape, -1)
    x[idxs] = vals
    return x

def monotonic_partition_old(unary):
    B, T, Z = unary.shape
    potentials = unary[:,:,:,None].repeat(1,1,1,Z)
    # only one starting state
    potentials[:,0,:,1:] = float("-inf")
    # monotonicity constraint
    log_transition = torch.tril(torch.ones(Z,Z, device=unary.device)).log()
    full_potentials = potentials + log_transition
    crf = LinearChainCRF(full_potentials)
    return crf.partition

def monotonic_partition(unary, state_mask):
    B, T, Z = unary.shape
    potentials = unary[:,:,:,None].repeat(1,1,1,Z)
    # only one starting state
    potentials[:,0,:,1:] = float("-inf")
    # monotonicity constraint
    log_transition = torch.tril(torch.ones(Z,Z, device=unary.device)).log()
    log_transition = log_transition[None].masked_fill(state_mask[:,:,None], -1e5).log_softmax(1)
    full_potentials = potentials + log_transition[:,None]
    crf = LinearChainCRF(full_potentials)
    return crf.partition
