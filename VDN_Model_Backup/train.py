"""
Training entry point.
Usage:
  python train.py                                        # default: all scenarios, 50 epochs
  python train.py --scenario building_collapse           # single scenario
  python train.py --epochs 10 --difficulty easy          # quick test
  python train.py --resume                               # continue from latest checkpoint
"""
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="all", choices=["building_collapse", "wildfire", "flood", "all"])
    p.add_argument("--difficulty", default=None, help="Override curriculum difficulty")
    p.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--env_cfg", default="config/env_config.yaml")
    p.add_argument("--agent_cfg", default="config/agent_config.yaml")
    p.add_argument("--training_cfg", default="config/training_config.yaml")
    p.add_argument("--reward_cfg", default="config/reward_config.yaml")
    return p.parse_args()


def main():
    import ctypes
    try:
        # Prevent Windows sleep (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("Windows sleep mode disabled for uninterrupted training.")
    except Exception as e:
        print(f"Could not disable sleep mode: {e}")

    args = parse_args()
    import yaml

    # Override training config if needed
    if args.epochs or args.difficulty:
        with open(args.training_cfg) as f:
            tcfg = yaml.safe_load(f)
        if args.epochs:
            tcfg["total_epochs"] = args.epochs
        import io, yaml as _yaml
        with open(args.training_cfg, "w") as f:
            _yaml.dump(tcfg, f)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    from training import CTDETrainer
    from rewards.reward import Reward
    trainer = CTDETrainer(args.env_cfg, args.agent_cfg, args.training_cfg, args.reward_cfg)
    
    # Set the single shared reward function instance (version locked)
    shared_reward_fn = Reward()
    trainer.set_reward_fn(shared_reward_fn)

    if args.difficulty:
        trainer.curriculum.override_difficulty = args.difficulty

    start_epoch = 0
    # Safety autoload: if latest checkpoint exists, resume from the last completed epoch automatically
    ckpt = os.path.join("checkpoints", "latest.pt")
    if os.path.exists(ckpt):
        import torch
        try:
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            start_epoch = int(meta.get("epoch", 0)) + 1
            print(f"Safety Autoload: Found checkpoint. Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load latest checkpoint: {e}. Starting from epoch 0.")

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
