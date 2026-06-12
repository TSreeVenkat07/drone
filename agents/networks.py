import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with optional action masking.
    Splits Q-value into Value (V) + Advantage (A) streams.
    Action masking sets invalid action Q-values to -infinity before selection.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims=(256, 256), dropout: float = 0.1):
        super().__init__()
        self.n_actions = n_actions

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # Value stream: single scalar
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream: per-action scalar
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            obs:         (B, obs_dim) float tensor
            action_mask: (B, n_actions) bool tensor — True = valid action
        Returns:
            q_values: (B, n_actions) with invalid actions set to -1e9
        """
        feat = self.shared(obs)
        value = self.value_stream(feat)           # (B, 1)
        advantage = self.advantage_stream(feat)   # (B, n_actions)
        # Dueling combination: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # ACTION MASKING: set invalid action Q-values to -infinity
        # This mathematically prevents the agent from ever choosing a wall move
        if action_mask is not None:
            q = q.masked_fill(~action_mask, float("-1e9"))
        return q

    def get_action(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0
    ) -> int:
        """Epsilon-greedy action selection with action masking."""
        with torch.no_grad():
            if np.random.random() < epsilon:
                # Random valid action only
                if action_mask is not None:
                    valid = action_mask.squeeze().cpu().numpy().astype(bool)
                    valid_indices = np.where(valid)[0]
                    return int(np.random.choice(valid_indices))
                return int(np.random.randint(self.n_actions))
            q = self.forward(obs, action_mask)
            return int(q.argmax(dim=-1).item())


# Alias for clarity
MaskedDuelingDQN = DuelingDQN
