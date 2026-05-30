import torch
import numpy as np
from .networks import DuelingDQN


class UAVAgent:
    """Wraps a DuelingDQN network with epsilon-greedy and action masking."""

    ACTION_DELTAS = [
        (-1,  0),  # N
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
        ( 0,  0),  # Hover
    ]

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

        # Dynamically load env configuration parameters for Tabu coordinate tracking
        import yaml
        try:
            with open("config/env_config.yaml") as f:
                env_cfg = yaml.safe_load(f)
            self.grid_size = env_cfg.get("grid_size", 32)
            obs_r = env_cfg.get("local_obs_radius", 5)
            therm_r = env_cfg.get("thermal_radius", 2)
            self.pos_start_idx = (2 * obs_r + 1) ** 2 + (2 * therm_r + 1) ** 2
            n_agents = env_cfg.get("n_agents", 4)
            self.step_ratio_idx = self.pos_start_idx + 2 + (n_agents - 1) * 2 + 1
        except Exception:
            self.grid_size = 32
            self.pos_start_idx = 146
            self.step_ratio_idx = 155

        self.history_len = 4
        self.pos_history = []
        self.last_step_ratio = -1.0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, explore: bool = True) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy().flatten()
        else:
            obs_np = np.array(obs).flatten()

        step_ratio = obs_np[self.step_ratio_idx]

        # Reset history on new episode (detecting step ratio jump/restart)
        if step_ratio < self.last_step_ratio or step_ratio <= 0.002:
            self.pos_history = []
        self.last_step_ratio = step_ratio

        r = int(round(obs_np[self.pos_start_idx] * self.grid_size))
        c = int(round(obs_np[self.pos_start_idx + 1] * self.grid_size))

        if explore and np.random.random() < self.epsilon:
            if isinstance(action_mask, torch.Tensor):
                mask_np = action_mask.cpu().numpy()
            else:
                mask_np = np.array(action_mask)
            valid_indices = np.where(mask_np.astype(bool))[0]
            if len(valid_indices) > 0:
                action = int(np.random.choice(valid_indices))
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                # Log actual movement to history
                dr, dc = self.ACTION_DELTAS[action]
                self.pos_history.append((r + dr, c + dc))
                if len(self.pos_history) > self.history_len:
                    self.pos_history.pop(0)
                return action

        # Q-value forward pass
        with torch.no_grad():
            q = self.online_net(obs_t, mask_t).squeeze(0)

        sorted_actions = torch.argsort(q, descending=True).cpu().numpy()

        best_action = None
        for action in sorted_actions:
            if q[action].item() < -1e8:
                continue

            dr, dc = self.ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc

            if (nr, nc) in self.pos_history:
                continue
            else:
                best_action = action
                break

        # Fallback: if all valid actions lead to visited positions, pick the highest Q-value action
        if best_action is None:
            for action in sorted_actions:
                if q[action].item() > -1e8:
                    best_action = action
                    break

        if best_action is not None:
            dr, dc = self.ACTION_DELTAS[best_action]
            self.pos_history.append((r + dr, c + dc))
            if len(self.pos_history) > self.history_len:
                self.pos_history.pop(0)
        else:
            best_action = 8

        if explore:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return int(best_action)

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
