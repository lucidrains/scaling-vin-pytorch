from functools import partial

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import Attention

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack

# ein notation
# b- batch
# c - channels
# a - actions (channels)
# o - output (channels)
# i - input (channels)
# h - height
# w - width
# d - depth of value iteration networ,

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
        action_channels,
        *,
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

        self.action_channels = action_channels

        conv_out = action_channels * (1 if not dynamic_transition_kernel else receptive_field ** 2)
        self.transition = nn.Conv2d(1, conv_out, receptive_field, padding = padding, bias = False)

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

        rewards_and_values = values + rewards

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
            next_values = q_values.amax(dim = 1, keepdim = True)
        else:
            temp = self.logsumexp_temperature
            next_values = (q_values / temp).logsumexp(dim = 1, keepdim = True) * temp

        return next_values

class ValueIterationNetwork(Module):
    def __init__(
        self,
        vi_module: ValueIteration,
        depth,
    ):
        super().__init__()
        self.vi_module = vi_module
        self.depth = depth

    def forward(
        self,
        values,
        rewards,
    ):

        values, _ = pack_one(values, 'b * h w')
        rewards, _ = pack_one(rewards, 'b * h w')

        layer_values = []

        for _ in range(self.depth):
            values = self.vi_module(values, rewards)

            layer_values.append(values)

        return layer_values

# main class

class ScalableVIN(Module):
    def __init__(
        self,
        state_dim,
        reward_dim,
        num_actions,
        init_kernel_size = 7,
        depth = 100,                    # they scaled this to 5000 in the paper to solve 100x100 maze
        loss_every_num_layers = 4,      # calculating loss every so many layers in the value iteration network
        final_cropout_kernel_size = 3,
        vi_module_kwargs: dict = dict(),
        vin_kwargs: dict = dict()
    ):
        super().__init__()

        self.state_to_values = nn.Conv2d(state_dim, 1, init_kernel_size, padding = init_kernel_size // 2, bias = False)
        self.reward_mapper = nn.Conv2d(reward_dim, 1, init_kernel_size, padding = init_kernel_size // 2, bias = False)

        # value iteration network

        value_iteration_module = ValueIteration(
            num_actions,
            **vi_module_kwargs
        )

        self.planner = ValueIterationNetwork(
            value_iteration_module,
            depth = depth,
            **vin_kwargs
        )

        # losses

        self.final_cropout_kernel_size = final_cropout_kernel_size
        self.to_action_logits = nn.Linear(final_cropout_kernel_size ** 2, num_actions, bias = False)

        self.loss_every_num_layers = loss_every_num_layers

    def forward(
        self,
        state,
        reward,
        agent_positions,
        target_actions = None
    ):
        value = self.state_to_values(state)
        reward = self.reward_mapper(reward)

        layer_values = self.planner(value, reward)

        # values across all layers

        layer_values = torch.stack(layer_values)
        layer_values = torch.flip(layer_values, dims = (0,)) # so depth goes from last layer to first

        # gather all layers for calculating losses

        if self.loss_every_num_layers < 1:
            # anything less than 1, just treat as only loss on last layer
            layer_values = layer_values[:1]
        else:
            layer_values = layer_values[::self.loss_every_num_layers]

        layer_values, inverse_pack = pack_one(layer_values, '* c h w')  # pack the depth with the batch and calculate loss across all layers in one go

        # unfold the values across all layers, and select out the coordinate for the agent position

        unfolded_layer_values = F.unfold(layer_values, self.final_cropout_kernel_size, padding = self.final_cropout_kernel_size // 2)

        unfolded_layer_values = inverse_pack(unfolded_layer_values, '* a hw')

        # get only the values at the agent coordinates

        height_width_strides = tensor(state.stride()[-2:])
        agent_position_hw_index = (agent_positions * height_width_strides).sum(dim = -1)

        unfolded_layer_values = einx.get_at('d b a [hw], b -> d b a', unfolded_layer_values, agent_position_hw_index)

        # calculating action logits

        action_logits = self.to_action_logits(unfolded_layer_values)

        # return logits if no labels

        if not exists(target_actions):
            return action_logits

        num_layers_calc_loss = action_logits.shape[0]

        # else calculate the loss

        losses = F.cross_entropy(
            rearrange(action_logits, 'd b a -> b a d'),
            repeat(target_actions, 'b -> b d', d = num_layers_calc_loss),
            reduction = 'none'
        )

        losses = reduce(losses, 'b d -> b', 'sum') # sum losses across depth, could do some sort of weighting too

        return losses.mean()
