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
    trainer = CTDETrainer(args.env_cfg, args.agent_cfg, args.training_cfg, args.reward_cfg)

    start_epoch = 0
    if args.resume:
        ckpt = os.path.join("checkpoints", "latest.pt")
        if os.path.exists(ckpt):
            import torch
            meta = torch.load(ckpt, map_location="cpu")
            start_epoch = int(meta.get("epoch", 0)) + 1
            print(f"Resuming from epoch {start_epoch}")

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
