import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

from einops import rearrange, pack, unpack

# ein notation
# b- batch
# c - channels
# a - actions (channels)
# o - output (channels)
# i - input (channels)
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
    output, packed_shape = pack([t], pattern)

    def inverse_fn(packed, override_pattern = None):
        override_pattern = default(override_pattern, pattern)

        return unpack(packed, packed_shape, override_pattern)[0]

    return output, inverse_fn

# modules and classes

class ValueIteration(Module):
    def __init__(
        self,
        *,
        reward_dim,
        action_channels,
        receptive_field = 3,
        pad_value = 0.,
        softmax_transition_weight = True,
        logsumexp_pool = False,
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.reward_dim = reward_dim
        self.action_channels = action_channels

        self.transition = nn.Conv2d(reward_dim + 1, action_channels, receptive_field, padding = padding, bias = False)

        self.pad_value = pad_value
        self.padding = padding

        # allow for logsumexp pooling
        # https://mpflueger.github.io/assets/pdf/svin_iclr2018_v2.pdf

        self.logsumexp_pool = logsumexp_pool

        self.softmax_transition_weight = softmax_transition_weight

    def forward(
        self,
        values,
        rewards
    ):
        rewards_and_values, _ = pack([rewards, values], 'b * h w')

        # prepare for transition

        transition_weight = self.transition.weight

        # pad so output is same

        rewards_and_values = F.pad(rewards_and_values, (self.padding,) * 4, value = self.pad_value)

        # in this paper, they propose a softmax latent transition kernel to stabilize to high depths
        # seems like the loss of expressivity is made up for by depth

        if self.softmax_transition_weight:

            transition_weight, inverse_fn = pack_one(transition_weight, 'o *')
            transition_weight = transition_weight.softmax(dim = -1)

            transition_weight = inverse_fn(transition_weight) # (o *) -> (o i h w)

        # transition

        q_values = F.conv2d(rewards_and_values, transition_weight)

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
