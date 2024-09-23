import pytest

import torch
from scaling_vin_pytorch import ScalableVIN

@pytest.mark.parametrize('dynamic_transition', (True, False))
def test_scaling_vin(
    dynamic_transition
):

    scalable_vin = ScalableVIN(
        state_dim = 3,
        reward_dim = 2,
        num_actions = 10,
        dynamic_transition = dynamic_transition
    )

    state = torch.randn(2, 3, 32, 32)
    reward = torch.randn(2, 2, 32, 32)

    agent_positions = torch.randint(0, 32, (2, 2))

    target_actions = torch.randint(0, 10, (2,))

    loss = scalable_vin(
        state,
        reward,
        agent_positions,
        target_actions
    )

    loss.backward()

    action_logits = scalable_vin(
        state,
        reward,
        agent_positions
    )
