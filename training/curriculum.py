import os
import torch
import yaml
from typing import List, Dict, Optional


class CurriculumManager:
    """
    Epoch-to-epoch curriculum learning manager.
    Each epoch builds on previous knowledge via checkpoint transfer.
    Difficulty and scenario are scheduled based on epoch number.
    """
    def __init__(self, config: dict, checkpoint_dir: str = "checkpoints"):
        self.cfg = config["curriculum"]
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.difficulty_schedule = self.cfg["difficulty_schedule"]
        self.scenario_schedule = self.cfg["scenario_schedule"]
        self.warmup_epochs = self.cfg["warmup_epochs"]

    def get_difficulty(self, epoch: int) -> str:
        diff = "easy"
        for entry in self.difficulty_schedule:
            if epoch >= entry["epoch"]:
                diff = entry["difficulty"]
        return diff

    def get_scenario(self, epoch: int) -> str:
        scenario = "building_collapse"
        for entry in self.scenario_schedule:
            if epoch >= entry["epoch"]:
                scenario = entry["scenario"]
        return scenario

    def get_scenarios_for_epoch(self, epoch: int) -> List[str]:
        """Returns list of scenarios to train on this epoch."""
        s = self.get_scenario(epoch)
        if s == "all":
            return ["building_collapse", "wildfire", "flood"]
        return [s]

    def save_checkpoint(self, agents: List, critic, epoch: int, metrics: Dict):
        """Save all agents and critic for curriculum transfer."""
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        payloads = {
            f"agent_{i}": {
                "online": agent.online_net.state_dict(),
                "target": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
            }
            for i, agent in enumerate(agents)
        }
        payloads["critic"] = {
            "net": critic.state_dict(),
        }
        payloads["metrics"] = metrics
        payloads["epoch"] = epoch
        torch.save(payloads, ckpt_path)
        # Keep symlink to latest
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest):
            os.remove(latest)
        import shutil
        shutil.copy(ckpt_path, latest)
        return ckpt_path

    def load_checkpoint(self, agents: List, critic, path: Optional[str] = None) -> int:
        """Load checkpoint for curriculum transfer. Returns epoch number."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "latest.pt")
        if not os.path.exists(path):
            return 0
        device = agents[0].device
        ckpt = torch.load(path, map_location=device)
        for i, agent in enumerate(agents):
            key = f"agent_{i}"
            if key in ckpt:
                agent.online_net.load_state_dict(ckpt[key]["online"])
                agent.target_net.load_state_dict(ckpt[key]["target"])
                try:
                    agent.optimizer.load_state_dict(ckpt[key]["optimizer"])
                except Exception:
                    pass
                # Anneal epsilon from where we left off (don't reset to 1.0)
                agent.epsilon = ckpt[key].get("epsilon", agent.epsilon)
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"]["net"])
        return int(ckpt.get("epoch", 0))

    def should_load_prev(self, epoch: int) -> bool:
        return self.cfg["enabled"] and epoch > 0
