import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

from einops import rearrange, pack, unpack

# ein notation
# b- batch
# c - channels
# h - height
# w - width

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def pack_one(t, pattern):
    return pack([t], pattern)

# modules and classes

class ValueIteration(Module):
    def __init__(
        self,
        *,
        reward_dim,
        action_channels,
        logsumexp_pool = False,
        receptive_field = 3
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2
        self.reward_dim = reward_dim

        self.action_channels = action_channels
        self.transition = nn.Conv2d(reward_dim + 1, action_channels, receptive_field, padding = padding, bias = False)

        # allow for logsumexp pooling
        # https://mpflueger.github.io/assets/pdf/svin_iclr2018_v2.pdf

        self.logsumexp_pool = logsumexp_pool

    def forward(
        self,
        values,
        rewards
    ):
        rewards_and_values, _ = pack([rewards, values], 'b * h w')

        q_values = self.transition(rewards_and_values)

        # selecting the next action

        if not self.logsumexp_pool:
            next_values = q_values.amax(dim = 1)
        else:
            next_values = q_values.logsumexp(dim = 1)

        return next_values

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
        assert vi_module.reward_dim == reward_dim

        self.encode_rewards = nn.Conv2d(reward_dim, reward_dim, reward_kernel_size, padding = reward_kernel_size // 2, bias = False)
        self.recurrent_steps = recurrent_steps

    def forward(
        self,
        values,
        rewards,
    ):

        values, _ = pack_one(values, 'b * h w')
        rewards, _ = pack_one(rewards, 'b * h w')

        rewards = self.encode_rewards(rewards)

        layer_values = []

        for _ in range(self.recurrent_steps):
            values = self.vi_module(values, rewards)

            layer_values.append(values)

        return values, layer_values
