import torch
import torch.nn as nn
from typing import List


class CentralizedCritic(nn.Module):
    """
    Centralized critic for CTDE framework.
    Takes full global state + all agents' actions and estimates V(s).
    Used only during TRAINING — not deployed at execution time.
    """
    def __init__(self, global_state_dim: int, n_agents: int, n_actions: int,
                 hidden_dims=(512, 256)):
        super().__init__()
        self.n_agents = n_agents
        input_dim = global_state_dim + n_agents * n_actions

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor, actions_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state:   (B, global_state_dim)
            actions_onehot: (B, n_agents * n_actions)
        Returns:
            v: (B, 1)
        """
        x = torch.cat([global_state, actions_onehot], dim=-1)
        return self.net(x)
