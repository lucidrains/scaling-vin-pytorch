from functools import partial, wraps

from beartype import beartype
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, tensor, Tensor, is_tensor
from torch.utils.checkpoint import checkpoint_sequential

from x_transformers import Attention
from x_transformers.x_transformers import RotaryEmbedding

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from unfoldNd import unfoldNd

# ein notation
# b - batch
# c - channels
# s - state dimension
# v - values
# a - actions (channels)
# o - output (channels)
# i - input (channels)
# t - depth
# h - height
# w - width
# d - depth of value iteration network
# p - number of plans (different value maps transitioned on)
# n - attention sequence length

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
):
    t = t / temperature
    out = t.softmax(dim = dim) * t
    out = out.sum(dim = dim)
    return out * temperature

# modules and classes

class ActionSelector(Module):
    def __init__(
        self,
        num_plans = 1,
        soft_maxpool = False,
        soft_maxpool_temperature = 1,
    ):
        super().__init__()
        self.num_plans = num_plans

        # allow for softmax(x) * x pooling
        # https://mpflueger.github.io/assets/pdf/svin_iclr2018_v2.pdf
        self.soft_maxpool = soft_maxpool
        self.soft_maxpool_temperature = soft_maxpool_temperature

    def forward(self, q_values):

        q_values = rearrange(q_values, 'b (p a) ... -> b p a ...', p = self.num_plans)

        if not self.soft_maxpool:
            next_values = q_values.amax(dim = 2)
        else:
            next_values = soft_maxpool(q_values, temperature = self.soft_maxpool_temperature, dim = 2)

        return next_values

# value iteration modules

class ValueIteration(Module):
    def __init__(
        self,
        action_channels,
        *,
        num_plans = 1,
        receptive_field = 3,
        pad_value = 0.,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.action_channels = action_channels
        plan_actions = num_plans * action_channels
        self.num_plans = num_plans

        self.transition = nn.Conv3d(num_plans, action_channels * num_plans, receptive_field, padding = padding, bias = False)

        self.kernel_size = receptive_field

        self.pad_value = pad_value
        self.pad = partial(F.pad, pad = (padding,) * 6, value = pad_value)

        self.padding = padding

        self.action_selector = ActionSelector(
            num_plans = num_plans,
            soft_maxpool = soft_maxpool,
            soft_maxpool_temperature = soft_maxpool_temperature
        )

    def forward(
        self,
        values,
        rewards
    ):
        pad = self.pad

        rewards_and_values = values + rewards

        # prepare for transition

        transition_weight = self.transition.weight

        # transition

        q_values = F.conv3d(pad(rewards_and_values), transition_weight)

        # selecting the next action

        next_values = self.action_selector(q_values)

        return next_values

class DynamicValueIteration(Module):
    def __init__(
        self,
        action_channels,
        *,
        num_plans = 1,
        receptive_field = 3,
        pad_value = 0.,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.action_channels = action_channels
        plan_actions = num_plans * action_channels
        self.num_plans = num_plans

        self.transition = nn.Conv3d(
            num_plans,
            action_channels * (num_plans ** 2) * (receptive_field ** 3),
            receptive_field,
            padding = padding,
            bias = False
        )

        self.kernel_size = receptive_field

        self.pad_value = pad_value
        self.pad = partial(F.pad, pad = (padding,) * 6, value = pad_value)

        self.padding = padding

        self.action_selector = ActionSelector(
            num_plans = num_plans,
            soft_maxpool = soft_maxpool,
            soft_maxpool_temperature = soft_maxpool_temperature
        )

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

        dynamic_transitions = F.conv3d(pad(rewards_and_values), transition_weight)

        # reshape the output into the next transition weight kernel

        dynamic_transitions = rearrange(dynamic_transitions, 'b (o i k1 k2 k3) t h w -> b o t h w (i k1 k2 k3)', k1 = self.kernel_size, k2 = self.kernel_size, k3 = self.kernel_size, i = self.num_plans)

        # the author then uses a softmax on the dynamic transition kernel
        # this actually becomes form of attention pooling seen in other papers

        dynamic_transitions = F.softmax(dynamic_transitions, dim = -1)

        # unfold the reward and values to manually do "conv" with data dependent kernel

        height, width = rewards_and_values.shape[-2:] # for rearranging back after unfold

        unfolded_values = unfoldNd(pad(rewards_and_values), self.kernel_size)
        unfolded_values = rearrange(unfolded_values, 'b i (t h w) -> b i t h w', w = width, h = height)

        # dynamic kernel

        q_values = einsum(unfolded_values, dynamic_transitions, 'b i t h w, b o t h w i -> b o t h w')

        # selecting the next action

        next_values = self.action_selector(q_values)

        return next_values

class AttentionValueIteration(Module):
    def __init__(
        self,
        action_channels,
        *,
        dim_qk = 8,
        num_plans = 1,
        receptive_field = 3,
        pad_value = 0.,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
    ):
        super().__init__()
        assert is_odd(receptive_field)
        padding = receptive_field // 2

        self.action_channels = action_channels
        plan_actions = num_plans * action_channels
        self.num_plans = num_plans

        # queries, keys, values

        self.dim_qk = dim_qk

        self.to_qk = nn.Conv3d(num_plans, 2 * num_plans * action_channels * dim_qk, receptive_field, bias = False)
        self.to_v = nn.Conv3d(num_plans, num_plans * action_channels, receptive_field, bias = False)

        # padding related

        self.kernel_size = receptive_field

        self.pad_value = pad_value
        self.pad = partial(F.pad, pad = (padding,) * 6, value = pad_value)
        self.padding = padding

        self.action_selector = ActionSelector(
            num_plans = num_plans,
            soft_maxpool = soft_maxpool,
            soft_maxpool_temperature = soft_maxpool_temperature
        )

    def forward(
        self,
        values,
        rewards
    ):
        kernel_size, pad = self.kernel_size, self.pad

        rewards_and_values = values + rewards

        width = rewards_and_values.shape[-1] # for rearranging back after unfold

        # prepare queries, keys, values

        q, k = self.to_qk(pad(rewards_and_values)).chunk(2, dim = 1)

        # softmax the kernel for the values for stability

        value_weights = self.to_v.weight

        value_weights, inverse_pack = pack_one(value_weights, 'o i *')
        value_weights = value_weights.softmax(dim = -1)
        value_weights = inverse_pack(value_weights)

        v = F.conv3d(pad(rewards_and_values), value_weights)

        # unfold the keys and values

        k, v = map(pad, (k, v))

        k = unfoldNd(k, kernel_size)
        v = unfoldNd(v, kernel_size)

        seq_len = kernel_size ** 3

        q, inverse_pack = pack_one(q, 'b a *')
        k = rearrange(k, 'b (a n) thw -> b a n thw', n = seq_len)

        # split out the hidden dimension for the queries and keys

        q, k = map(lambda t: rearrange(t, 'b (a d) ... -> b a d ...', d = self.dim_qk), (q, k))

        v = rearrange(v, 'b (a n) thw -> b a n thw', n = seq_len)

        # perform attention

        sim = einsum(q, k, 'b a d thw, b a d n thw -> b a thw n')

        attn = sim.softmax(dim = -1)

        q_values = einsum(attn, v, 'b a thw n, b a n thw -> b a thw')

        # reshape height and width back

        q_values = inverse_pack(q_values)

        # selecting the next action

        next_values = self.action_selector(q_values)

        return next_values

# value iteration network

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
        depth = None
    ):
        depth = default(depth, self.depth)

        values, _ = pack_one(values, 'b * t h w')
        rewards, _ = pack_one(rewards, 'b * t h w')

        # checkpointable or not

        if depth > 1 and should_checkpoint(self, (values, rewards)):

            layer_values = [None] * depth

            segments = default(self.checkpoint_segments, depth)
            checkpoint_fn = partial(checkpoint_sequential, segments = segments, use_reentrant = False)

            def recurrent_layer(inputs):
                layer_ind, values, rewards, *layer_values = inputs

                next_values = self.vi_module(values, rewards)

                layer_values[layer_ind] = next_values

                return layer_ind + 1, next_values, rewards, *layer_values

            all_recurrent_layers = (recurrent_layer,) * depth

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

    @beartype
    def __init__(
        self,
        state_dim,
        reward_dim,
        num_actions,
        num_plans = 1,
        dim_hidden = 150,
        init_kernel_size = 7,
        depth = 100,                    # they scaled this to 5000 in the paper to solve 100x100 maze
        loss_every_num_layers = 4,      # calculating loss every so many layers in the value iteration network
        final_cropout_kernel_size = 3,
        soft_maxpool = False,
        soft_maxpool_temperature = 1.,
        dynamic_transition = True,
        checkpoint = True,
        vi_module_type: Literal['invariant', 'dynamic', 'attention'] = 'dynamic',
        vi_module_kwargs: dict = dict(),
        vin_kwargs: dict = dict(),
        attn_dim_head = 32,
        attn_kwargs: dict = dict()
    ):
        super().__init__()
        assert is_odd(init_kernel_size)

        self.depth = depth
        self.num_actions = num_actions
        self.num_plans = num_plans

        plan_actions = num_actions * num_plans

        self.reward_mapper = nn.Sequential(
            nn.Conv3d(state_dim + reward_dim, dim_hidden, 3, padding = 1),
            nn.Conv3d(dim_hidden, 1, 1, bias = False)
        )

        self.to_init_value = nn.Conv3d(1, plan_actions, 3, padding = 1, bias = False)

        self.action_selector = ActionSelector(
            num_plans = num_plans,
            soft_maxpool = soft_maxpool,
            soft_maxpool_temperature = soft_maxpool_temperature
        )

        # value iteration network

        if vi_module_type == 'dynamic':

            value_iteration_module = DynamicValueIteration(
                num_actions,
                num_plans = num_plans,
                soft_maxpool = soft_maxpool,
                soft_maxpool_temperature = soft_maxpool_temperature,
                **vi_module_kwargs
            )

        elif vi_module_type == 'invariant':

            value_iteration_module = ValueIteration(
                num_actions,
                num_plans = num_plans,
                soft_maxpool = soft_maxpool,
                soft_maxpool_temperature = soft_maxpool_temperature,
                **vi_module_kwargs
            )

        elif vi_module_type == 'attention':

            value_iteration_module = AttentionValueIteration(
                num_actions,
                num_plans = num_plans,
                soft_maxpool = soft_maxpool,
                soft_maxpool_temperature = soft_maxpool_temperature,
                **vi_module_kwargs
            )

        else:
            raise ValueError(f'invalid value iteration module type {vi_module_type} given')

        # the value iteration network just calls the value iteration module recurrently

        self.planner = ValueIterationNetwork(
            value_iteration_module,
            depth = depth,
            checkpoint = checkpoint,
            **vin_kwargs
        )

        self.final_cropout_kernel_size = final_cropout_kernel_size

        final_value_dim = final_cropout_kernel_size ** 3
        final_state_dim = state_dim * final_cropout_kernel_size ** 3

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

        self.to_action_logits = nn.Linear(num_plans * final_value_dim, num_actions, bias = False)

        self.loss_every_num_layers = loss_every_num_layers

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        state,
        reward,
        agent_positions,
        target_actions = None,
        depth = None
    ):
        batch, num_plans = state.shape[0], self.num_plans

        depth = default(depth, self.depth)

        state_reward, _ = pack([state, reward], 'b * t h w')

        reward = self.reward_mapper(state_reward)

        q_values = self.to_init_value(reward)

        init_value = self.action_selector(q_values)

        layer_values = self.planner(init_value, reward, depth = depth)

        # values across all layers

        layer_values = torch.stack(layer_values)
        layer_values = torch.flip(layer_values, dims = (0,)) # so depth goes from last layer to first

        layer_values, inverse_pack = pack_one(layer_values, '* c t h w')  # pack the depth with the batch and calculate loss across all layers in one go

        # unfold the values across all layers, and select out the coordinate for the agent position

        unfolded_layer_values = unfoldNd(layer_values, self.final_cropout_kernel_size, padding = self.final_cropout_kernel_size // 2)

        unfolded_layer_values = inverse_pack(unfolded_layer_values, '* a thw')

        # get only the values at the agent coordinates

        dimension_strides = tensor(state.stride()[-3:])
        agent_position_index = (agent_positions * dimension_strides).sum(dim = -1)

        unfolded_layer_values = einx.get_at('d b v [thw], b -> b d v', unfolded_layer_values, agent_position_index)

        unfolded_layer_values = rearrange(unfolded_layer_values, 'b d (p v) -> (b p) d v', p = num_plans)

        # concat states onto each value and do an attention across all values across all layers

        unfolded_state_values = unfoldNd(state, self.final_cropout_kernel_size, padding = self.final_cropout_kernel_size // 2)
        unfolded_state_values = einx.get_at('b s [thw], b -> b s', unfolded_state_values, agent_position_index)

        unfolded_state_values = repeat(unfolded_state_values, 'b s -> (b p) d s', d = depth, p = num_plans)

        state_and_all_values, _ = pack([unfolded_layer_values, unfolded_state_values], 'b d *')

        # allow each layer values to attend to all layers of the past
        # if there are multiple plans, layer values of each plan will have its own attention map

        rotary_pos_emb = self.rotary_pos_emb(torch.arange(depth, device = self.device))

        attended = self.attn_pool(state_and_all_values, rotary_pos_emb = rotary_pos_emb)

        # add the output of the 'causal' attention (across depth) to the values

        unfolded_layer_values = unfolded_layer_values + attended

        # fold num plans dimension back into action dimension

        unfolded_layer_values = rearrange(unfolded_layer_values, '(b p) d v -> b d (p v)', p = num_plans)

        # gather all layers for calculating losses

        only_last_layer = self.loss_every_num_layers < 1

        if only_last_layer:
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
