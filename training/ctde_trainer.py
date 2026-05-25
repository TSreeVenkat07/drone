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

        # Centralized critic
        self.critic = CentralizedCritic(
            self.global_state_dim, self.n_agents, self.n_actions
        ).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4)

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

    # ------------------------------------------------------------------ train
    def train(self, start_epoch: int = 0):
        if self.curriculum.should_load_prev(start_epoch):
            loaded_epoch = self.curriculum.load_checkpoint(self.agents, self.critic)
            print(f"Resumed from epoch {loaded_epoch}")
            # Baseline evaluation on resume to populate venkateval.txt
            print("Running baseline evaluation on the resumed checkpoint...")
            try:
                difficulty = self.curriculum.get_difficulty(loaded_epoch)
                scenarios = self.curriculum.get_scenarios_for_epoch(loaded_epoch)
                eval_metrics = self.evaluator.evaluate(difficulty, scenarios)
                with open("venkateval.txt", "a") as f:
                    f.write(f"--- Baseline Eval on Resume (Epoch {loaded_epoch}) ---\n")
                    f.write(f"Epoch: {loaded_epoch} (resumed) | Scenario: {scenarios} | Difficulty: {difficulty}\n")
                    f.write(f"Cov Improv: {eval_metrics.get('coverage_improvement_pct', 0.0):+.1f}% | Latency Red: {eval_metrics.get('latency_reduction_pct', 0.0):+.1f}%\n")
                    f.write(f"MARL Cov: {eval_metrics.get('avg_marl_coverage', 0.0):.1f}% | Vic: {eval_metrics.get('avg_marl_victims_found', 0.0):.2f}/7 | Col: {eval_metrics.get('avg_marl_collisions', 0.0):.1f}\n\n")
                print("Baseline evaluation written to venkateval.txt.")
            except Exception as e:
                print(f"Error running baseline evaluation: {e}")

        total_epochs = self.tcfg["total_epochs"]
        eps_per_epoch = self.tcfg["episodes_per_epoch"]

        for epoch in range(start_epoch, total_epochs):
            difficulty = self.curriculum.get_difficulty(epoch)
            scenarios = self.curriculum.get_scenarios_for_epoch(epoch)

            epoch_metrics = {
                "epoch": epoch, "difficulty": difficulty,
                "scenarios": scenarios, "episodes": [],
            }

            for ep_idx in tqdm(range(eps_per_epoch), desc=f"Epoch {epoch % 10}/10 [{difficulty}]"):
                scenario = scenarios[ep_idx % len(scenarios)]
                ep_metrics = self._run_episode(scenario, difficulty, training=True)
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
            
            # Phase completion check
            next_scenario = self.curriculum.get_scenario(epoch + 1)
            current_scenario = self.curriculum.get_scenario(epoch)
            next_difficulty = self.curriculum.get_difficulty(epoch + 1)
            current_difficulty = self.curriculum.get_difficulty(epoch)

            is_phase_complete = (next_scenario != current_scenario or epoch == total_epochs - 1)
            is_diff_complete = (next_difficulty != current_difficulty and not is_phase_complete)

            def write_to_venkateval(label, metrics):
                with open("venkateval.txt", "a") as f:
                    f.write(f"--- {label} ---\n")
                    f.write(f"Epoch: {epoch} (completed) | Scenario: {current_scenario} | Difficulty: {current_difficulty}\n")
                    f.write(f"Cov Improv: {metrics.get('coverage_improvement_pct', 0.0):+.1f}% | Latency Red: {metrics.get('latency_reduction_pct', 0.0):+.1f}%\n")
                    f.write(f"MARL Cov: {metrics.get('avg_marl_coverage', 0.0):.1f}% | Vic: {metrics.get('avg_marl_victims_found', 0.0):.2f}/7 | Col: {metrics.get('avg_marl_collisions', 0.0):.1f}\n\n")

            # Evaluation vs greedy
            if is_eval_epoch:
                try:
                    eval_metrics = self.evaluator.evaluate(difficulty, scenarios)
                    self.logger.log_eval(epoch, eval_metrics)
                    print(f"  [EVAL] vs greedy: coverage_improvement={eval_metrics.get('coverage_improvement_pct', 0.0):+.1f}% | "
                          f"latency_reduction={eval_metrics.get('latency_reduction_pct', 0.0):+.1f}%")
                    write_to_venkateval(f"Routine Eval (Epoch {epoch})", eval_metrics)
                except Exception as e:
                    print(f"\n  [EVAL ERROR] Head-to-head evaluation failed! Error: {e}")
            
            # Difficulty completion test (Every 10 epochs)
            if is_diff_complete:
                print(f"\n--- Difficulty Phase Complete: {current_scenario} ({current_difficulty}) ---")
                try:
                    diff_eval_metrics = self.evaluator.evaluate(current_difficulty, [current_scenario], n_eval_eps=10)
                    write_to_venkateval(f"Difficulty Phase Complete: {current_scenario} - {current_difficulty}", diff_eval_metrics)
                    print("--- Difficulty Evaluation Complete ---\n")
                except Exception as e:
                    print(f"Difficulty evaluation error: {e}")

            # Phase completion test (Every 30 epochs)
            if is_phase_complete:
                print(f"\n--- Environment Phase Complete: {current_scenario} ---")
                eval_scenarios = [current_scenario] if current_scenario != "all" else ["building_collapse", "wildfire", "flood"]
                print(f"Running comprehensive evaluation table testing {current_scenario} skills on {eval_scenarios}...")
                try:
                    phase_eval_metrics = self.evaluator.evaluate(current_difficulty, eval_scenarios, n_eval_eps=10)
                    write_to_venkateval(f"Environment Phase Complete: {current_scenario} (Tested on {eval_scenarios})", phase_eval_metrics)
                    print("--- Phase Evaluation Complete ---\n")
                except Exception as e:
                    print(f"Phase evaluation error: {e}")

            # STRICT SAVE FIRST (After all tests run for this epoch step)
            if epoch % self.tcfg["save_interval"] == 0 or is_eval_epoch or is_diff_complete or is_phase_complete:
                print(f"[SAVE SECURED] Saving checkpoint to disk AFTER all evaluations (Epoch {epoch})...")
                self.curriculum.save_checkpoint(self.agents, self.critic, epoch, agg)

            if is_phase_complete:
                input(f"\n[PAUSED] Training for {current_scenario} is complete! Review how it performed on the MIXED maps above. Press Enter to proceed...")

    # --------------------------------------------------------------- episode
    def _run_episode(self, scenario: str, difficulty: str, training: bool = True) -> Dict:
        env = DisasterEnv(self.env_cfg, self.reward_cfg, scenario,
                          self.n_agents, difficulty)
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

            next_obs_dict, rewards_dict, terminated, truncated, next_info = env.step(actions)
            done = terminated or truncated
            next_masks = next_info["action_masks"]

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

        total_td = np.zeros(len(transitions))

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
                target_q = rew_t + self.gamma * next_q_val * (1.0 - dones_t)

            current_q = agent.online_net(obs_t, mask_t).gather(1, act_t.unsqueeze(1)).squeeze(-1)
            td_errors = (target_q - current_q).detach().cpu().numpy()
            total_td += np.abs(td_errors)

            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

            agent.optimizer.zero_grad()
            if agent.scaler is not None:
                agent.scaler.scale(loss).backward()
                agent.scaler.unscale_(agent.optimizer)
                torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), self.grad_clip)
                agent.scaler.step(agent.optimizer)
                agent.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), self.grad_clip)
                agent.optimizer.step()

            agent.update_target()

        self.buffer.update_priorities(indices, total_td / self.n_agents)
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for agent in self.agents:
                agent.hard_update_target()

        # Update centralized critic
        all_actions_oh = []
        for i in range(self.n_agents):
            ah = F.one_hot(torch.LongTensor(act_batch[i]), self.n_actions).float().to(self.device)
            all_actions_oh.append(ah)
        actions_cat = torch.cat(all_actions_oh, dim=-1)
        shared_rew = torch.FloatTensor([
            sum(t["rewards"]) / self.n_agents for t in transitions
        ]).to(self.device)
        v = self.critic(gs_t, actions_cat).squeeze(-1)
        critic_loss = F.mse_loss(v, shared_rew)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

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
