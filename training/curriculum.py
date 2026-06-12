import os
import torch
import yaml
from typing import List, Dict, Optional


class CurriculumManager:
    """
    Mastery-based curriculum learning manager.
    Difficulty and scenario advance only when evaluation goals are met.
    """
    def __init__(self, config: dict, checkpoint_dir: str = "checkpoints"):
        self.cfg = config["curriculum"]
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.difficulties = self.cfg.get("difficulties", ["easy", "medium", "hard"])
        self.scenarios = self.cfg.get("scenarios", ["building_collapse", "wildfire", "flood", "all"])
        self.warmup_epochs = self.cfg.get("warmup_epochs", 5)

        self.current_diff_idx = 0
        self.current_scen_idx = 0
        self.override_difficulty = None

    def get_difficulty(self) -> str:
        if self.current_diff_idx >= len(self.difficulties):
            return self.difficulties[-1]
        return self.difficulties[self.current_diff_idx]

    def get_scenario(self) -> str:
        if self.current_scen_idx >= len(self.scenarios):
            return self.scenarios[-1]
        return self.scenarios[self.current_scen_idx]

    def get_scenarios_for_epoch(self) -> List[str]:
        """Returns list of scenarios to train on this epoch."""
        s = self.get_scenario()
        if s == "all":
            return ["building_collapse", "wildfire", "flood"]
        return [s]

    def advance_curriculum(self):
        """Advances difficulty. If hard is beaten, advances scenario."""
        self.current_diff_idx += 1
        if self.current_diff_idx >= len(self.difficulties):
            self.current_diff_idx = 0
            self.current_scen_idx += 1
            if self.current_scen_idx >= len(self.scenarios):
                self.current_scen_idx = len(self.scenarios) - 1 # maxed out

    def save_checkpoint(self, agents: List, critic, critic_optimizer, epoch: int, metrics: Dict, buffer=None):
        """Save all agents and critic for curriculum transfer."""
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        payloads = {
            f"agent_{i}": {
                "online": agent.online_net.state_dict(),
                "target": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "scaler": agent.scaler.state_dict() if agent.scaler is not None else None,
            }
            for i, agent in enumerate(agents)
        }
        if critic is not None:
            payloads["critic"] = {
                "net": critic.state_dict(),
                "optimizer": critic_optimizer.state_dict() if critic_optimizer is not None else None,
            }
        payloads["metrics"] = metrics
        payloads["epoch"] = epoch
        payloads["curriculum_state"] = {
            "diff_idx": self.current_diff_idx,
            "scen_idx": self.current_scen_idx
        }
        if buffer is not None:
            payloads["replay_buffer"] = buffer
        torch.save(payloads, ckpt_path)
        # Keep symlink to latest
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest):
            os.remove(latest)
        import shutil
        shutil.copy(ckpt_path, latest)
        return ckpt_path

    def load_checkpoint(self, agents: List, critic, critic_optimizer=None, path: Optional[str] = None) -> int:
        """Load checkpoint for curriculum transfer. Returns epoch number."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "latest.pt")
        if not os.path.exists(path):
            return 0
        device = agents[0].device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        for i, agent in enumerate(agents):
            key = f"agent_{i}"
            if key in ckpt:
                agent.online_net.load_state_dict(ckpt[key]["online"])
                agent.target_net.load_state_dict(ckpt[key]["target"])
                try:
                    agent.optimizer.load_state_dict(ckpt[key]["optimizer"])
                except Exception:
                    pass
                if agent.scaler is not None and ckpt[key].get("scaler") is not None:
                    try:
                        agent.scaler.load_state_dict(ckpt[key]["scaler"])
                    except Exception:
                        pass
                # Anneal epsilon from where we left off (don't reset to 1.0)
                agent.epsilon = ckpt[key].get("epsilon", agent.epsilon)
        if "critic" in ckpt and critic is not None:
            critic.load_state_dict(ckpt["critic"]["net"])
            if critic_optimizer is not None and ckpt["critic"].get("optimizer") is not None:
                try:
                    critic_optimizer.load_state_dict(ckpt["critic"]["optimizer"])
                except Exception:
                    pass
            
        cstate = ckpt.get("curriculum_state", {})
        self.current_diff_idx = cstate.get("diff_idx", 0)
        self.current_scen_idx = cstate.get("scen_idx", 0)
        
        # Migration from old epoch-based checkpoint format
        epoch_val = int(ckpt.get("epoch", 0))
        if not cstate and epoch_val >= 10:
            self.current_diff_idx = 0  # Force to easy so they can master it!
            
        if epoch_val >= 48:
            self.current_diff_idx = self.difficulties.index("hard")
            print("[CURRICULUM OVERRIDE] Enforcing HARD phase since Epoch >= 48")
            
        if self.override_difficulty is not None:
            if self.override_difficulty in self.difficulties:
                self.current_diff_idx = self.difficulties.index(self.override_difficulty)
                print(f"[CURRICULUM OVERRIDE] Forcing difficulty to: {self.override_difficulty} (index {self.current_diff_idx})")
            
        return epoch_val

    def should_load_prev(self, epoch: int) -> bool:
        return self.cfg.get("enabled", True) and epoch > 0
