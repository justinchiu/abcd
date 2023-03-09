import numpy as np
import torch
from torch_struct import LinearChainCRF

def first(preds):
    vals, idxs = np.unique(preds, return_index=True)
    x = np.full(preds.shape, -1)
    x[idxs] = vals
    return torch.tensor(x)

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
    return first(preds)

def first_argmax_prediction(unary):
    preds = unary.argmax(-1).cpu().numpy()
    return first(preds)

def most_confident_step(unary):
    import pdb; pdb.set_trace()
    return x

def monotonic_partition_nomask(unary):
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

# score as well
def monotonic_arg_max(unary):
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
    return binary_argmax.nonzero()[:,2].cpu().numpy(), crf.max.item()


def first_monotonic_arg_max(unary):
    preds, score = monotonic_arg_max(unary)
    preds = first(preds)
    return preds.numpy(), score

def batch_monotonic_arg_max(unary):
    N, T, Z = unary.shape
    potentials = unary[:,:,:,None].repeat(1,1,1,Z)
    # only one starting state
    potentials[:,0,:,1:] = float("-inf")
    # monotonicity constraint
    transition = torch.tril(torch.ones(Z,Z, device=unary.device))
    log_transition = transition.log()
    full_potentials = potentials + log_transition
    crf = LinearChainCRF(full_potentials)
    binary_argmax = crf.argmax.detach().cpu()
    return [x.nonzero()[:,2] for x in binary_argmax], crf.max.detach()

# renamed for prompting

def amax(unary):
    return unary.argmax(-1), unary.max(-1).sum().item()

def afirstmax(unary):
    return first(unary.argmax(-1).cpu().numpy()), unary.max(-1).sum().item()

def amono(unary):
    return monotonic_arg_max(unary)

def afirstmono(unary):
    return first_monotonic_arg_max(unary)
