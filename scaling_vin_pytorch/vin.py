import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

# modules and classes

class ValueIteration(Module):
    def __init__(
        self,
        *,
        dim,
        reward_dim,
        receptive_field = 3
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.dim = dim
        self.transition = nn.Conv2d(dim, dim, receptive_field, padding = padding, bias = False)
        self.select_action = nn.MaxPool2d(receptive_field, padding = padding)

    def forward(
        self,
        *,
        value,
        reward
    ):
        q_values = self.transition(value + reward)

        next_step_value = self.select_action(q_values)

        return next_step_value

class Planner(Module):
    def __init__(
        self,
        vi_module: ValueIteration,
        reward_dim,
        recurrent_steps,
        reward_kernel_size = 3
    ):
        super().__init__()

        self.vi_module = vi_module
        self.encode_rewards = nn.Conv2d(reward_dim, vi_module.dim, reward_kernel_size, padding = reward_kernel_size // 2, bias = False)
        self.recurrent_steps = recurrent_steps

    def forward(
        self,
        *,
        values,
        rewards,
    ):

        rewards = self.encode_rewards(rewards)

        layer_values = []

        for _ in range(self.recurrent_steps):
            values = self.vi_module(values, rewards)

            layer_values.append(values)

        return values, layer_values
