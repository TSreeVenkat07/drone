import torch
import numpy as np
from .networks import DuelingDQN


class UAVAgent:
    """Wraps a DuelingDQN network with epsilon-greedy and action masking."""

    def __init__(self, obs_dim: int, n_actions: int, config: dict, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.cfg = config

        hidden = tuple(config["network"]["hidden_dims"])
        dropout = config["network"]["dropout"]
        self.online_net = DuelingDQN(obs_dim, n_actions, hidden, dropout).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions, hidden, dropout).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=1e-5,
        )
        if config["training"]["fp16"] and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.epsilon = config["training"]["epsilon_start"]
        self.epsilon_end = config["training"]["epsilon_end"]
        self.epsilon_decay = config["training"]["epsilon_decay"]

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, explore: bool = True) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        if explore:
            action = self.online_net.get_action(obs_t, mask_t, self.epsilon)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            action = self.online_net.get_action(obs_t, mask_t, 0.0)
        return action

    def update_target(self, tau: float = None):
        if tau is None:
            tau = self.cfg["training"]["tau"]
        for p_online, p_target in zip(self.online_net.parameters(), self.target_net.parameters()):
            p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)

    def hard_update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
