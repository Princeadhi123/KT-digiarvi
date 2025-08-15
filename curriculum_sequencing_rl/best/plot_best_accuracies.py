import os
import json
import argparse
import matplotlib.pyplot as plt


def load_best(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Assume this script is placed under curriculum_sequencing_rl/best
    default_json = os.path.join(here, "..", "best_hyperparams.json")
    default_out = os.path.join(here, "best_accuracies.png")

    parser = argparse.ArgumentParser(description="Plot bar chart of best accuracies from best_hyperparams.json")
    parser.add_argument("--config", default=default_json, help="Path to best_hyperparams.json")
    parser.add_argument("--output", default=default_out, help="Path to save bar chart PNG")
    args = parser.parse_args()

    cfg = load_best(args.config)
    best = cfg.get("best", {})

    # Preserve common model order
    order = ["ql", "dqn", "a2c", "a3c", "ppo"]
    labels_map = {"ql": "Q-Learning", "dqn": "DQN", "a2c": "A2C", "a3c": "A3C", "ppo": "PPO"}

    xs, ys, seeds = [], [], []
    for key in order:
        if key in best:
            entry = best[key]
            xs.append(labels_map.get(key, key.upper()))
            ys.append(float(entry.get("acc", 0.0)))
            seeds.append(str(entry.get("seed", "")))

    if not xs:
        raise ValueError("No 'best' entries found in config")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(xs, ys, color=["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262fd"])  # nice palette
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (test)")
    plt.title("Best Test Accuracies by Model")
    plt.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate bars with exact values and seed
    for rect, acc, seed in zip(bars, ys, seeds):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                 f"{acc:.3f}\n(seed {seed})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved bar chart to: {args.output}")


if __name__ == "__main__":
    main()
