from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

from einops import rearrange, einsum, pack, unpack

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
        logsumexp_pool = False,
        logsumexp_temperature = 1.,
        softmax_transition_weight = True,
        dynamic_transition_kernel = True,
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.reward_dim = reward_dim
        self.action_channels = action_channels

        reward_and_value_dim = reward_dim + 1

        conv_out = action_channels * (1 if not dynamic_transition_kernel else reward_and_value_dim * receptive_field ** 2)
        self.transition = nn.Conv2d(reward_and_value_dim, conv_out, receptive_field, padding = padding, bias = False)

        self.kernel_size = receptive_field

        self.pad_value = pad_value
        self.pad = partial(F.pad, pad = (padding,) * 4, value = pad_value)

        self.padding = padding

        # allow for logsumexp pooling
        # https://mpflueger.github.io/assets/pdf/svin_iclr2018_v2.pdf

        self.logsumexp_pool = logsumexp_pool
        self.logsumexp_temperature = logsumexp_temperature

        self.dynamic_transition_kernel = dynamic_transition_kernel
        self.softmax_transition_weight = softmax_transition_weight

    def forward(
        self,
        values,
        rewards
    ):
        pad = self.pad

        rewards_and_values, _ = pack([rewards, values], 'b * h w')

        # prepare for transition

        transition_weight = self.transition.weight

        # dynamic transition kernel - in other words, first convolution outputs the transition kernel

        if self.dynamic_transition_kernel:

            dynamic_transitions = F.conv2d(pad(rewards_and_values), transition_weight)

            # reshape the output into the next transition weight kernel

            dynamic_transitions = rearrange(dynamic_transitions, 'b (o i k1 k2) h w -> b o h w (i k1 k2)', k1 = self.kernel_size, k2 = self.kernel_size, o = self.action_channels)

            if self.softmax_transition_weight:
                # oh, the softmax latent transition was applied to the dynamic output, makes sense
                # but it should be compared to the original formulation

                dynamic_transitions = F.softmax(dynamic_transitions, dim = -1)

            # unfold the reward and values to manually do "conv" with data dependent kernel

            width = rewards_and_values.shape[-1] # for rearranging back after unfold

            unfolded_values = F.unfold(pad(rewards_and_values), self.kernel_size)
            unfolded_values = rearrange(unfolded_values, 'b i (h w) -> b i h w', w = width)

            # dynamic kernel

            q_values = einsum(unfolded_values, dynamic_transitions, 'b i h w, b o h w i -> b o h w')

        else:

            # in this paper, they propose a softmax latent transition kernel to stabilize to high depths
            # seems like the loss of expressivity is made up for by depth

            if self.softmax_transition_weight:

                transition_weight, inverse_fn = pack_one(transition_weight, 'o *')
                transition_weight = transition_weight.softmax(dim = -1)

                transition_weight = inverse_fn(transition_weight) # (o *) -> (o i h w)

            # transition

            q_values = F.conv2d(pad(rewards_and_values), transition_weight)

        # selecting the next action

        if not self.logsumexp_pool:
            next_values = q_values.amax(dim = 1)
        else:
            temp = self.logsumexp_temperature
            next_values = (q_values / temp).logsumexp(dim = 1) * temp

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

# main class

class ScalableVIN(Module):
    def __init__(
        self
    ):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        values,
        rewards
    ):
        raise NotImplementedError
