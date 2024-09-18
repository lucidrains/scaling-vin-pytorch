import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# modules and classes

class ValueIteration(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x
