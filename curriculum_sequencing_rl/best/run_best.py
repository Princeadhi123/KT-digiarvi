import os
import json
import argparse
import sys
from typing import Dict, Any

# We import lazily inside main to avoid import errors when doing --dry_run

def load_best_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Ensure the parent directory (curriculum_sequencing_rl) is on sys.path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))
    # Defaults relative to being in a 'best' subfolder under curriculum_sequencing_rl
    default_json = os.path.join(here, "..", "best_hyperparams.json")
    default_csv = os.path.join(here, "..", "experiment_metrics.csv")
    default_data = os.path.abspath(os.path.join(here, "..", "..", "preprocessed_kt_data.csv"))

    parser = argparse.ArgumentParser(description="Run curriculum experiments with best hyperparams per model")
    parser.add_argument("--model", choices=["ql", "dqn", "a2c", "a3c", "ppo"], required=True,
                        help="Which model to run using best_hyperparams.json")
    parser.add_argument("--config", default=default_json, help="Path to best_hyperparams.json")
    parser.add_argument("--data", default=default_data, help="Path to preprocessed_kt_data.csv")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (defaults to best config's seed)")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--metrics_csv", default=default_csv, help="Where to append metrics CSV")
    parser.add_argument("--no_chance", dest="include_chance", action="store_false", help="Disable chance baseline")
    parser.set_defaults(include_chance=True)
    parser.add_argument("--dry_run", action="store_true", help="Only print resolved params and exit")

    args = parser.parse_args()

    cfg = load_best_config(args.config)
    if "best" not in cfg or args.model not in cfg["best"]:
        raise ValueError(f"Model '{args.model}' not found in best config at {args.config}")

    best_entry = cfg["best"][args.model]
    best_seed = int(best_entry.get("seed", 42))
    params = dict(best_entry.get("params", {}))

    # Global reward weights
    global_cfg = cfg.get("global", {})
    reward_correct_w = float(global_cfg.get("reward_correct_w", 0.5))
    reward_score_w = float(global_cfg.get("reward_score_w", 0.5))

    # Build kwargs for run_all_and_report()
    kwargs: Dict[str, Any] = {
        "models": [args.model],
        "eval_episodes": args.eval_episodes,
        "seed": args.seed if args.seed is not None else best_seed,
        "metrics_csv": args.metrics_csv,
        "include_chance": args.include_chance,
        "reward_correct_w": reward_correct_w,
        "reward_score_w": reward_score_w,
        # Spread model-specific params (keys already match run_all_and_report signature)
        **params,
    }

    if args.dry_run:
        print("Data:", args.data)
        print("Resolved kwargs for run_all_and_report():")
        for k in sorted(kwargs.keys()):
            print(f"  {k}: {kwargs[k]}")
        return

    # Import here to avoid import if user is only doing --dry_run
    from curriculum_rl_experiments import run_all_and_report

    print(f"Using data: {args.data}")
    results = run_all_and_report(data_path=args.data, **kwargs)
    print("\n=== Done (from run_best.py) ===")
    for name, (acc, rew) in results.items():
        print(f"{name:<10}: acc={acc:.3f}, reward={rew:.3f}")


if __name__ == "__main__":
    main()
