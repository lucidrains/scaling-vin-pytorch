from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, tensor, Tensor, is_tensor
from torch.utils.checkpoint import checkpoint_sequential

from x_transformers import Attention
from x_transformers.x_transformers import RotaryEmbedding

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack

# ein notation
# b- batch
# c - channels
# s - state dimension
# a - actions (channels)
# o - output (channels)
# i - input (channels)
# h - height
# w - width
# d - depth of value iteration network

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

def should_checkpoint(
    self,
    inputs: Tensor | tuple[Tensor, ...],
    check_instance_variable = 'checkpoint'
) -> bool:
    if is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training and
        any([i.requires_grad for i in inputs]) and
        (not exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )

# tensor helpers

def soft_maxpool(
    t,
    temperature = 1.,
    dim = 1,
    keepdim = True
):
    t = t / temperature
    out = t.softmax(dim = dim) * t
    out = out.sum(dim = dim, keepdim = keepdim)
    return out * temperature

# modules and classes

class ValueIteration(Module):
    def __init__(
        self,
        action_channels,
        *,
        receptive_field = 3,
        pad_value = 0.,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
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

        # allow for softmax(x) * x pooling
        # https://mpflueger.github.io/assets/pdf/svin_iclr2018_v2.pdf

        self.soft_maxpool = soft_maxpool
        self.soft_maxpool_temperature = soft_maxpool_temperature

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

        if not self.soft_maxpool:
            next_values = q_values.amax(dim = 1, keepdim = True)
        else:
            next_values = soft_maxpool(q_values,  temperature = self.soft_maxpool_temperature)

        return next_values

class ValueIterationNetwork(Module):
    def __init__(
        self,
        vi_module: ValueIteration,
        depth,
        checkpoint = True,
        checkpoint_segments = None
    ):
        super().__init__()
        self.depth = depth
        self.vi_module = vi_module

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

    def forward(
        self,
        values,
        rewards,
    ):
        depth = self.depth

        values, _ = pack_one(values, 'b * h w')
        rewards, _ = pack_one(rewards, 'b * h w')

        # checkpointable or not

        if depth > 1 and should_checkpoint(self, (values, rewards)):

            layer_values = [None] * depth

            segments = default(self.checkpoint_segments, self.depth)
            checkpoint_fn = partial(checkpoint_sequential, segments = segments, use_reentrant = False)

            def recurrent_layer(inputs):
                layer_ind, values, rewards, *layer_values = inputs

                next_values = self.vi_module(values, rewards)

                layer_values[layer_ind] = next_values

                return layer_ind + 1, next_values, rewards, *layer_values

            all_recurrent_layers = (recurrent_layer,) * self.depth

            _, _, _, *layer_values = checkpoint_fn(all_recurrent_layers, input = (0, values, rewards, *layer_values))

        else:

            layer_values = []

            for _ in range(depth):
                values = self.vi_module(values, rewards)
                layer_values.append(values)

        # return all values

        assert len(layer_values) == depth

        return layer_values

# main class

class ScalableVIN(Module):
    def __init__(
        self,
        state_dim,
        reward_dim,
        num_actions,
        dim_hidden = 150,
        init_kernel_size = 7,
        depth = 100,                    # they scaled this to 5000 in the paper to solve 100x100 maze
        loss_every_num_layers = 4,      # calculating loss every so many layers in the value iteration network
        final_cropout_kernel_size = 3,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
        vi_module_kwargs: dict = dict(),
        vin_kwargs: dict = dict(),
        attn_dim_head = 64,
        attn_kwargs: dict = dict()
    ):
        super().__init__()
        self.depth = depth

        self.reward_mapper = nn.Sequential(
            nn.Conv2d(state_dim + reward_dim, dim_hidden, 3, padding = 1),
            nn.Conv2d(dim_hidden, 1, 1, bias = False)
        )

        self.to_init_value = nn.Conv2d(1, num_actions, 3, padding = 1, bias = False)
        self.soft_maxpool = soft_maxpool
        self.soft_maxpool_temperature = soft_maxpool_temperature

        # value iteration network

        value_iteration_module = ValueIteration(
            num_actions,
            soft_maxpool = soft_maxpool,
            soft_maxpool_temperature = soft_maxpool_temperature,
            **vi_module_kwargs
        )

        self.planner = ValueIterationNetwork(
            value_iteration_module,
            depth = depth,
            **vin_kwargs
        )

        self.final_cropout_kernel_size = final_cropout_kernel_size

        final_value_dim = final_cropout_kernel_size ** 2
        final_state_dim = state_dim * final_cropout_kernel_size ** 2

        # final attention across all values

        self.attn_pool = Attention(
            dim = final_value_dim + final_state_dim,
            dim_out = final_value_dim,
            causal = True,
            dim_head = attn_dim_head,
            **attn_kwargs
        )

        self.rotary_pos_emb = RotaryEmbedding(attn_dim_head)

        # losses

        self.to_action_logits = nn.Linear(final_value_dim, num_actions, bias = False)

        self.loss_every_num_layers = loss_every_num_layers

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        state,
        reward,
        agent_positions,
        target_actions = None
    ):
        state_reward, _ = pack([state, reward], 'b * h w')

        reward = self.reward_mapper(state_reward)

        q_values = self.to_init_value(reward)

        if self.soft_maxpool:
            value = soft_maxpool(q_values, temperature = self.soft_maxpool_temperature)
        else:
            value = q_values.amax(dim = 1, keepdim = True)

        layer_values = self.planner(value, reward)

        # values across all layers

        layer_values = torch.stack(layer_values)
        layer_values = torch.flip(layer_values, dims = (0,)) # so depth goes from last layer to first

        layer_values, inverse_pack = pack_one(layer_values, '* c h w')  # pack the depth with the batch and calculate loss across all layers in one go

        # unfold the values across all layers, and select out the coordinate for the agent position

        unfolded_layer_values = F.unfold(layer_values, self.final_cropout_kernel_size, padding = self.final_cropout_kernel_size // 2)

        unfolded_layer_values = inverse_pack(unfolded_layer_values, '* a hw')

        # get only the values at the agent coordinates

        height_width_strides = tensor(state.stride()[-2:])
        agent_position_hw_index = (agent_positions * height_width_strides).sum(dim = -1)

        unfolded_layer_values = einx.get_at('d b a [hw], b -> b d a', unfolded_layer_values, agent_position_hw_index)

        # concat states onto each value and do an attention across all values across all layers

        unfolded_state_values = F.unfold(state, self.final_cropout_kernel_size, padding = self.final_cropout_kernel_size // 2)
        unfolded_state_values = einx.get_at('b s [hw], b -> b s', unfolded_state_values, agent_position_hw_index)

        unfolded_state_values = repeat(unfolded_state_values, 'b s -> b d s', d = self.depth)

        state_and_all_values, _ = pack([unfolded_layer_values, unfolded_state_values], 'b d *')

        rotary_pos_emb = self.rotary_pos_emb(torch.arange(self.depth, device = self.device))

        attended = self.attn_pool(state_and_all_values, rotary_pos_emb = rotary_pos_emb)

        # add the output of the 'causal' attention (across depth) to the values

        unfolded_layer_values = unfolded_layer_values + attended

        # gather all layers for calculating losses

        if self.loss_every_num_layers < 1:
            # anything less than 1, just treat as only loss on last layer
            unfolded_layer_values = unfolded_layer_values[:, :1]
        else:
            unfolded_layer_values = unfolded_layer_values[:, ::self.loss_every_num_layers]

        num_layers_calc_loss = unfolded_layer_values.shape[1]

        # calculating action logits

        action_logits = self.to_action_logits(unfolded_layer_values)

        # return logits if no labels

        if not exists(target_actions):
            return action_logits

        # else calculate the loss

        losses = F.cross_entropy(
            rearrange(action_logits, 'b d a -> b a d'),
            repeat(target_actions, 'b -> b d', d = num_layers_calc_loss),
            reduction = 'none'
        )

        losses = reduce(losses, 'b d -> b', 'sum') # sum losses across depth, could do some sort of weighting too

        return losses.mean()
