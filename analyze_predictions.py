import torch
import pdb

path = (
    "logging/encoder-answer-model-1222-roberta-base-bart-base "
    "lr-2e-05 bs-1 dt-0 ds-0 ml-256 k-32 se-0 qw-0 tz-True "
    "|step-3250.pt"
)

out = torch.load(path)

pdb.set_trace()
