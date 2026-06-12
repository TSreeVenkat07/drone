import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import yaml

from environment import DisasterEnv
from agents import UAVAgent, CentralizedCritic
from .per_buffer import PrioritizedReplayBuffer
from .curriculum import CurriculumManager
from evaluation import Evaluator
from utils.logger import TrainingLogger
from rewards.reward import Reward, REWARD_VERSION


class CTDETrainer:
    """
    Centralized Training, Decentralized Execution trainer.
    - Agents observe LOCAL state only at execution time.
    - Centralized critic uses GLOBAL state during training.
    - PER buffer stores full multi-agent transitions.
    - Curriculum manager advances difficulty epoch by epoch.
    """

    def __init__(self, env_cfg: str, agent_cfg: str, training_cfg: str,
                 reward_cfg: str = "config/reward_config.yaml"):
        with open(training_cfg) as f:
            self.tcfg = yaml.safe_load(f)
        with open(agent_cfg) as f:
            self.acfg = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.tcfg["device"] == "cuda" else "cpu"
        )
        print(f"Training on: {self.device}")

        # Build a probe env to get dimensions
        probe_env = DisasterEnv(env_cfg, reward_cfg, "building_collapse", 4, "easy")
        obs, info = probe_env.reset()
        self.n_agents = probe_env.n_agents
        self.obs_dim = probe_env.obs_dim
        self.n_actions = probe_env.N_ACTIONS
        self.global_state_dim = probe_env.global_state_dim

        # Initialize agents (shared architecture, independent weights)
        self.agents: List[UAVAgent] = [
            UAVAgent(self.obs_dim, self.n_actions, self.acfg, str(self.device))
            for _ in range(self.n_agents)
        ]

        # VDN uses Value Decomposition, so no centralized critic is needed.
        self.critic = None
        self.critic_optimizer = None

        # PER buffer
        pcfg = self.acfg["per"]
        self.buffer = PrioritizedReplayBuffer(
            pcfg["buffer_size"], pcfg["alpha"], pcfg["beta_start"],
            pcfg["beta_end"], pcfg["beta_steps"], pcfg["epsilon"]
        )

        # Curriculum
        self.curriculum = CurriculumManager(self.tcfg, self.tcfg["checkpoint_dir"])
        self.evaluator = Evaluator(env_cfg, reward_cfg, self.agents, self.device)
        self.logger = TrainingLogger(self.tcfg["log_dir"])

        self.batch_size = self.acfg["training"]["batch_size"]
        self.gamma = self.acfg["training"]["gamma"]
        self.grad_clip = self.acfg["training"]["grad_clip"]
        self.target_update_freq = self.acfg["training"]["target_update_freq"]
        self.update_count = 0

        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg

        # Initialize default shared reward function and assert version lock
        self.reward_fn = Reward(reward_cfg=reward_cfg)
        assert self.reward_fn.REWARD_VERSION == "v3_full_spectrum"

    def set_reward_fn(self, reward_fn):
        """Set a single shared reward function instance and assert version lock."""
        self.reward_fn = reward_fn
        assert self.reward_fn.REWARD_VERSION == "v3_full_spectrum"

    # ------------------------------------------------------------------ train
    def train(self, start_epoch: int = 0):
        if self.curriculum.should_load_prev(start_epoch):
            loaded_epoch = self.curriculum.load_checkpoint(self.agents, self.critic, self.critic_optimizer)
            print(f"Resumed from epoch {loaded_epoch}")
            start_epoch = loaded_epoch + 1
            # REPLAY BUFFER RULE (CHECKPOINT ONLY)
            if os.path.exists("checkpoint.pt"):
                try:
                    self.buffer = torch.load("checkpoint.pt", weights_only=False)
                    self.buffer.trim_to_newest(keep_fraction=0.50)
                    print("Replay buffer loaded from checkpoint.pt and trimmed to newest 50%.")
                except Exception as e:
                    print(f"Failed to load/trim replay buffer from checkpoint.pt: {e}")
            # Baseline evaluation on resume to populate venkateval.txt
            print("Running baseline evaluation on the resumed checkpoint...")
            try:
                difficulty = self.curriculum.get_difficulty()
                scenarios = self.curriculum.get_scenarios_for_epoch()
                eval_metrics = self.evaluator.evaluate(difficulty, scenarios)
                with open("venkateval.txt", "a") as f:
                    f.write(f"--- Baseline Eval on Resume (Epoch {loaded_epoch}) ---\n")
                    f.write(f"Epoch: {loaded_epoch} (resumed) | Scenario: {scenarios} | Difficulty: {difficulty}\n")
                    f.write(f"Cov Improv: {eval_metrics.get('coverage_improvement_pct', 0.0):+.1f}% | Latency Red: {eval_metrics.get('latency_reduction_pct', 0.0):+.1f}%\n")
                    f.write(f"MARL Cov: {eval_metrics.get('marl_coverage_pct', 0.0):.1f}% | Vic: {eval_metrics.get('avg_victims_found', 0.0):.2f}/7 | Col: {eval_metrics.get('avg_collisions', 0.0):.1f}\n\n")
                print("Baseline evaluation written to venkateval.txt.")
            except Exception as e:
                print(f"Error running baseline evaluation: {e}")

        total_epochs = self.tcfg["total_epochs"]
        eps_per_epoch = self.tcfg["episodes_per_epoch"]

        for epoch in range(start_epoch, total_epochs):
            difficulty = self.curriculum.get_difficulty()
            scenarios = self.curriculum.get_scenarios_for_epoch()
            current_scenario = self.curriculum.get_scenario()

            epoch_metrics = {
                "epoch": epoch, "difficulty": difficulty,
                "scenarios": scenarios, "episodes": [],
            }

            for ep_idx in tqdm(range(eps_per_epoch), desc=f"Epoch {epoch % 10}/10 [{difficulty}]"):
                scenario = scenarios[ep_idx % len(scenarios)]
                ep_metrics = self._run_episode(scenario, difficulty, training=True, ep_idx=ep_idx)
                epoch_metrics["episodes"].append(ep_metrics)
                with open("live_track.txt", "a") as f:
                    f.write(f"Epoch {epoch}/{total_epochs} | Ep {ep_idx}/{eps_per_epoch} | Scen: {scenario} | Cov: {ep_metrics['coverage_pct']:.1f}% | Vic: {ep_metrics['victims_found']}/7 | Col: {ep_metrics['total_collisions']} | Steps: {ep_metrics['steps']}\n")

            # Aggregate epoch stats
            agg = self._aggregate_metrics(epoch_metrics["episodes"])
            epoch_metrics.update(agg)
            self.logger.log_epoch(epoch_metrics)
            print(f"\nEpoch {epoch}: coverage={agg['avg_coverage']:.1f}% | "
                  f"victims={agg['avg_victims_found']:.2f}/7 | "
                  f"collisions={agg['avg_collisions']:.1f} | "
                  f"latency={agg['avg_detection_step']:.1f}")

            is_eval_epoch = (epoch % self.tcfg["eval_interval"] == 0)
            
            def write_to_venkateval(label, metrics):
                with open("venkateval.txt", "a") as f:
                    f.write(f"--- {label} ---\n")
                    f.write(f"Epoch: {epoch} (completed) | Scenario: {current_scenario} | Difficulty: {difficulty}\n")
                    f.write(f"Cov Improv: {metrics.get('coverage_improvement_pct', 0.0):+.1f}% | Latency Red: {metrics.get('latency_reduction_pct', 0.0):+.1f}%\n")
                    f.write(f"MARL Cov: {metrics.get('marl_coverage_pct', 0.0):.1f}% | Vic: {metrics.get('avg_victims_found', 0.0):.2f}/7 | Col: {metrics.get('avg_collisions', 0.0):.1f}\n\n")

            # Evaluation vs greedy
            if is_eval_epoch:
                try:
                    eval_metrics = self.evaluator.evaluate(difficulty, scenarios)
                    self.logger.log_eval(epoch, eval_metrics)
                    print(f"  [EVAL] vs greedy: coverage_improvement={eval_metrics.get('coverage_improvement_pct', 0.0):+.1f}% | "
                          f"latency_reduction={eval_metrics.get('latency_reduction_pct', 0.0):+.1f}%")
                    write_to_venkateval(f"Routine Eval (Epoch {epoch})", eval_metrics)
                    
                    if eval_metrics.get("all_goals_achieved", False):
                        print(f"\n🎉 MASTERY ACHIEVED: {current_scenario} ({difficulty}) 🎉")
                        print("Advancing curriculum to next difficulty!")
                        self.curriculum.advance_curriculum()
                        write_to_venkateval(f"Difficulty Mastery Complete: {current_scenario} - {difficulty}", eval_metrics)
                        
                except Exception as e:
                    print(f"\n  [EVAL ERROR] Head-to-head evaluation failed! Error: {e}")

            # STRICT SAVE FIRST (After all tests run for this epoch step)
            if epoch % self.tcfg["save_interval"] == 0 or is_eval_epoch:
                print(f"[SAVE SECURED] Saving checkpoint to disk AFTER all evaluations (Epoch {epoch})...")
                self.curriculum.save_checkpoint(self.agents, self.critic, self.critic_optimizer, epoch, agg, buffer=None)
                print("Saved agent checkpoints without replay buffer (MemoryError prevention).")

    # --------------------------------------------------------------- episode
    def _run_episode(self, scenario: str, difficulty: str, training: bool = True, ep_idx: int = 0) -> Dict:
        env = DisasterEnv(self.env_cfg, self.reward_cfg, scenario,
                          self.n_agents, difficulty)
        
        # Episode reset rule: reset collision counters and progressive penalties
        self.reward_fn.on_episode_start()
        obs_dict, info = env.reset()
        
        masks = info["action_masks"]
        done = False
        ep_rewards = np.zeros(self.n_agents)
        step = 0
        first_detection_step = None

        while not done:
            step += 1
            actions = {}
            for i in range(self.n_agents):
                key = f"agent_{i}"
                actions[key] = self.agents[i].select_action(obs_dict[key], masks[key], explore=training)

            # Capture states before step
            prev_positions = {i: tuple(env.agent_positions[i]) for i in range(self.n_agents)}
            prev_coverage_map = env.coverage_map.copy()
            prev_victim_found = env.victim_found.copy()
            prev_thermal_map = env.thermal_map.copy()

            next_obs_dict, env_rewards_dict, terminated, truncated, next_info = env.step(actions)
            done = terminated or truncated
            next_masks = next_info["action_masks"]

            # Construct state and next_state dicts for reward_fn.compute_reward
            state_dict = {
                "agent_positions": prev_positions,
                "coverage_map": prev_coverage_map,
                "victim_found": prev_victim_found,
                "obstacle_map": env.obstacle_map,
                "victim_positions": env.victim_positions,
                "thermal_map": prev_thermal_map,
                "grid_size": env.grid_size,
                "thermal_radius": env.thermal_radius,
                "step_count": step,
            }
            next_state_dict = {
                "agent_positions": {i: tuple(env.agent_positions[i]) for i in range(self.n_agents)},
                "coverage_map": env.coverage_map.copy(),
                "victim_found": env.victim_found.copy(),
                "obstacle_map": env.obstacle_map,
                "victim_positions": env.victim_positions,
                "thermal_map": env.thermal_map.copy(),
                "grid_size": env.grid_size,
                "thermal_radius": env.thermal_radius,
            }

            # Compute reward using only compute_reward(agent, state, action, next_state)
            rewards_dict = {}
            for i in range(self.n_agents):
                agent_key = f"agent_{i}"
                rewards_dict[agent_key] = self.reward_fn.compute_reward(i, state_dict, actions[agent_key], next_state_dict)

            # Debug check: print every 10 episodes on step 1
            if ep_idx % 10 == 0 and step == 1:
                print("REWARD VERSION:", self.reward_fn.REWARD_VERSION, flush=True)
                print("Sample reward:", rewards_dict["agent_0"], flush=True)

            # Track first detection
            if next_info["victims_found"] > 0 and first_detection_step is None:
                first_detection_step = step

            # Store transition in PER buffer
            global_state = env.get_global_state()
            transition = {
                "obs": [obs_dict[f"agent_{i}"] for i in range(self.n_agents)],
                "actions": [actions[f"agent_{i}"] for i in range(self.n_agents)],
                "rewards": [rewards_dict[f"agent_{i}"] for i in range(self.n_agents)],
                "next_obs": [next_obs_dict[f"agent_{i}"] for i in range(self.n_agents)],
                "done": float(done),
                "masks": [masks[f"agent_{i}"] for i in range(self.n_agents)],
                "next_masks": [next_masks[f"agent_{i}"] for i in range(self.n_agents)],
                "global_state": global_state,
            }
            self.buffer.push(transition)

            for i in range(self.n_agents):
                ep_rewards[i] += rewards_dict[f"agent_{i}"]

            if training and len(self.buffer) >= self.batch_size * 2:
                self._update_agents()

            obs_dict = next_obs_dict
            masks = next_masks

        return {
            "total_reward": float(ep_rewards.sum()),
            "victims_found": next_info["victims_found"],
            "total_collisions": next_info["total_collisions"],
            "coverage_pct": next_info["coverage_pct"],
            "steps": step,
            "detection_step": first_detection_step or step,
            "success": next_info["victims_found"] == env.n_victims,
        }

    # --------------------------------------------------------------- update
    def _update_agents(self):
        result = self.buffer.sample(self.batch_size)
        if result is None:
            return
        transitions, indices, weights = result
        weights = weights.to(self.device)

        # Unpack batch
        obs_batch = [[] for _ in range(self.n_agents)]
        act_batch = [[] for _ in range(self.n_agents)]
        rew_batch = [[] for _ in range(self.n_agents)]
        nobs_batch = [[] for _ in range(self.n_agents)]
        mask_batch = [[] for _ in range(self.n_agents)]
        nmask_batch = [[] for _ in range(self.n_agents)]
        dones, global_states = [], []

        for t in transitions:
            for i in range(self.n_agents):
                obs_batch[i].append(t["obs"][i])
                act_batch[i].append(t["actions"][i])
                rew_batch[i].append(t["rewards"][i])
                nobs_batch[i].append(t["next_obs"][i])
                mask_batch[i].append(t["masks"][i])
                nmask_batch[i].append(t["next_masks"][i])
            dones.append(t["done"])
            global_states.append(t["global_state"])

        dones_t = torch.FloatTensor(dones).to(self.device)
        gs_t = torch.FloatTensor(np.array(global_states)).to(self.device)

        current_q_list = []
        next_q_val_list = []
        rew_tot = torch.zeros(len(transitions), device=self.device)

        for i, agent in enumerate(self.agents):
            obs_t = torch.FloatTensor(np.array(obs_batch[i])).to(self.device)
            act_t = torch.LongTensor(act_batch[i]).to(self.device)
            rew_t = torch.FloatTensor(rew_batch[i]).to(self.device)
            nobs_t = torch.FloatTensor(np.array(nobs_batch[i])).to(self.device)
            mask_t = torch.BoolTensor(np.array(mask_batch[i])).to(self.device)
            nmask_t = torch.BoolTensor(np.array(nmask_batch[i])).to(self.device)

            # Double DQN target
            with torch.no_grad():
                next_q_online = agent.online_net(nobs_t, nmask_t)
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target = agent.target_net(nobs_t, nmask_t)
                next_q_val = next_q_target.gather(1, next_actions).squeeze(-1)
                next_q_val_list.append(next_q_val)

            current_q = agent.online_net(obs_t, mask_t).gather(1, act_t.unsqueeze(1)).squeeze(-1)
            current_q_list.append(current_q)
            rew_tot += rew_t

        # VDN: Sum Q-values across all agents
        current_q_tot = torch.stack(current_q_list, dim=0).sum(dim=0)
        next_q_tot = torch.stack(next_q_val_list, dim=0).sum(dim=0)
        target_q_tot = rew_tot + self.gamma * next_q_tot * (1.0 - dones_t)

        td_errors = (target_q_tot - current_q_tot).detach().cpu().numpy()
        total_td = np.abs(td_errors)
        
        # Global Loss
        loss = (weights * F.smooth_l1_loss(current_q_tot, target_q_tot, reduction="none")).mean()

        for agent in self.agents:
            agent.optimizer.zero_grad()

        if any(a.scaler is not None for a in self.agents):
            scaler = self.agents[0].scaler
            scaler.scale(loss).backward()
            for agent in self.agents:
                scaler.unscale_(agent.optimizer)
                torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), self.grad_clip)
                scaler.step(agent.optimizer)
            scaler.update()
        else:
            loss.backward()
            for agent in self.agents:
                torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), self.grad_clip)
                agent.optimizer.step()

        for agent in self.agents:
            agent.update_target()

        self.buffer.update_priorities(indices, total_td)
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for agent in self.agents:
                agent.hard_update_target()

    # ----------------------------------------------------------- aggregation
    def _aggregate_metrics(self, episodes: List[Dict]) -> Dict:
        if not episodes:
            return {}
        return {
            "avg_reward": float(np.mean([e["total_reward"] for e in episodes])),
            "avg_victims_found": float(np.mean([e["victims_found"] for e in episodes])),
            "avg_collisions": float(np.mean([e["total_collisions"] for e in episodes])),
            "avg_coverage": float(np.mean([e["coverage_pct"] for e in episodes])),
            "avg_detection_step": float(np.mean([e["detection_step"] for e in episodes])),
            "success_rate": float(np.mean([e["success"] for e in episodes])),
            "max_collisions": int(max(e["total_collisions"] for e in episodes)),
        }
