import os
import argparse
from typing import List
from env import CurriculumEnvV2
from evaluation import eval_policy_category_accuracy
from q_learning import train_q_learning, greedy_from_qtable
from dqn import train_dqn, dqn_policy
from a2c import train_a2c, a2c_policy_fn
from a3c import train_a3c
from ppo import train_ppo
import random
import numpy as np
import torch
import csv
from datetime import datetime, timezone

# -----------------------------
# Orchestration utilities
# -----------------------------

def run_all_and_report(
    data_path: str,
    ql_epochs: int = 5,
    dqn_episodes: int = 50,
    a2c_episodes: int = 50,
    a3c_episodes: int = 50,
    ppo_episodes: int = 50,
    eval_episodes: int = 300,
    models: List[str] = None,
    reward_correct_w: float = 0.5,
    reward_score_w: float = 0.5,
    # Reproducibility
    seed: int = 42,
    # Q-Learning params
    ql_alpha: float = 0.2,
    ql_gamma: float = 0.9,
    ql_eps_start: float = 0.3,
    ql_eps_end: float = 0.0,
    ql_eps_decay_epochs: int = 3,
    ql_select_best_on_val: bool = True,
    ql_val_episodes: int = 300,
    # DQN params
    dqn_lr: float = 1e-3,
    dqn_gamma: float = 0.99,
    dqn_batch_size: int = 128,
    dqn_buffer_size: int = 20000,
    dqn_hidden_dim: int = 128,
    dqn_eps_start: float = 1.0,
    dqn_eps_end: float = 0.05,
    dqn_eps_decay_steps: int = 20000,
    dqn_target_tau: float = 0.01,
    dqn_target_update_interval: int = 1,
    dqn_select_best_on_val: bool = True,
    dqn_val_episodes: int = 300,
    # A2C params
    a2c_lr: float = 1e-3,
    a2c_entropy: float = 0.01,
    a2c_value_coef: float = 0.5,
    a2c_bc_warmup: int = 1,
    a2c_bc_weight: float = 0.5,
    a2c_batch_episodes: int = 4,
    # A3C params
    a3c_lr: float = 1e-3,
    a3c_entropy: float = 0.01,
    a3c_value_coef: float = 0.5,
    a3c_gae_lambda: float = 0.95,
    a3c_bc_warmup: int = 1,
    a3c_bc_weight: float = 0.5,
    a3c_rollouts: int = 4,
    # PPO params
    ppo_lr: float = 3e-4,
    ppo_epochs: int = 4,
    ppo_batch_episodes: int = 8,
    ppo_minibatch_size: int = 2048,
    ppo_entropy: float = 0.01,
    ppo_value_coef: float = 0.5,
    ppo_gae_lambda: float = 0.95,
    ppo_bc_warmup: int = 2,
    ppo_bc_weight: float = 1.0,
    # Reporting
    include_chance: bool = True,
    metrics_csv: str = None,
):
    # Seed control
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = CurriculumEnvV2(data_path, reward_correct_w=reward_correct_w, reward_score_w=reward_score_w, seed=seed)
    if models is None:
        models = ["ql", "dqn", "a2c", "a3c", "ppo"]

    results = {}

    # Chance baseline via uniform random policy
    def _random_policy_fn(env_ref):
        def _p(state, cur_cat: int) -> int:
            return random.randrange(env_ref.action_size)
        return _p

    if include_chance:
        chance_acc, chance_reward = eval_policy_category_accuracy(env, _random_policy_fn(env), mode="test", episodes=eval_episodes)
        results["Chance"] = (chance_acc, chance_reward)

    if "ql" in models:
        ql = train_q_learning(
            env,
            epochs=ql_epochs,
            alpha=ql_alpha,
            gamma=ql_gamma,
            eps_start=ql_eps_start,
            eps_end=ql_eps_end,
            eps_decay_epochs=ql_eps_decay_epochs,
            seed=seed,
            select_best_on_val=ql_select_best_on_val,
            val_episodes=ql_val_episodes,
        )
        ql_acc, ql_reward = eval_policy_category_accuracy(env, greedy_from_qtable(ql), mode="test", episodes=eval_episodes)
        results["Q-Learning"] = (ql_acc, ql_reward)

    if "dqn" in models:
        dqn = train_dqn(
            env,
            episodes=dqn_episodes,
            eps_start=dqn_eps_start,
            eps_end=dqn_eps_end,
            eps_decay_steps=dqn_eps_decay_steps,
            target_tau=dqn_target_tau,
            target_update_interval=dqn_target_update_interval,
            lr=dqn_lr,
            gamma=dqn_gamma,
            batch_size=dqn_batch_size,
            buffer_size=dqn_buffer_size,
            hidden_dim=dqn_hidden_dim,
            select_best_on_val=dqn_select_best_on_val,
            val_episodes=dqn_val_episodes,
        )
        dqn_acc, dqn_reward = eval_policy_category_accuracy(env, dqn_policy(dqn), mode="test", episodes=eval_episodes)
        results["DQN"] = (dqn_acc, dqn_reward)

    if "a2c" in models:
        a2c_net = train_a2c(
            env,
            episodes=a2c_episodes,
            lr=a2c_lr,
            entropy_coef=a2c_entropy,
            value_coef=a2c_value_coef,
            bc_warmup_epochs=a2c_bc_warmup,
            bc_weight=a2c_bc_weight,
            batch_episodes=a2c_batch_episodes,
        )
        a2c_acc, a2c_reward = eval_policy_category_accuracy(env, a2c_policy_fn(a2c_net, device=str(next(a2c_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["A2C"] = (a2c_acc, a2c_reward)

    if "a3c" in models:
        a3c_net = train_a3c(
            env,
            episodes=a3c_episodes,
            lr=a3c_lr,
            entropy_coef=a3c_entropy,
            value_coef=a3c_value_coef,
            gae_lambda=a3c_gae_lambda,
            bc_warmup_epochs=a3c_bc_warmup,
            bc_weight=a3c_bc_weight,
            rollouts_per_update=a3c_rollouts,
        )
        a3c_acc, a3c_reward = eval_policy_category_accuracy(env, a2c_policy_fn(a3c_net, device=str(next(a3c_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["A3C"] = (a3c_acc, a3c_reward)

    if "ppo" in models:
        ppo_net = train_ppo(
            env,
            episodes=ppo_episodes,
            lr=ppo_lr,
            ppo_epochs=ppo_epochs,
            batch_episodes=ppo_batch_episodes,
            minibatch_size=ppo_minibatch_size,
            entropy_coef=ppo_entropy,
            value_coef=ppo_value_coef,
            gae_lambda=ppo_gae_lambda,
            bc_warmup_epochs=ppo_bc_warmup,
            bc_weight=ppo_bc_weight,
        )
        ppo_acc, ppo_reward = eval_policy_category_accuracy(env, a2c_policy_fn(ppo_net, device=str(next(ppo_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["PPO"] = (ppo_acc, ppo_reward)

    print("\n=== Test Metrics (category accuracy, avg reward) ===")
    for name, (acc, rew) in results.items():
        print(f"{name:<10}: acc={acc:.3f}, reward={rew:.3f}")
    
    # Optional CSV logging
    if metrics_csv:
        fieldnames = [
            "timestamp", "model", "acc", "reward", "seed",
            # env/reward weights
            "reward_correct_w", "reward_score_w",
            # Q-Learning
            "ql_epochs", "ql_alpha", "ql_gamma", "ql_eps_start", "ql_eps_end", "ql_eps_decay_epochs", "ql_select_best_on_val", "ql_val_episodes",
            # DQN
            "dqn_episodes", "dqn_lr", "dqn_gamma", "dqn_batch_size", "dqn_buffer_size", "dqn_hidden_dim",
            "dqn_eps_start", "dqn_eps_end", "dqn_eps_decay_steps", "dqn_target_tau", "dqn_target_update_interval", "dqn_select_best_on_val", "dqn_val_episodes",
            # A2C
            "a2c_episodes", "a2c_lr", "a2c_entropy", "a2c_value_coef", "a2c_bc_warmup", "a2c_bc_weight", "a2c_batch_episodes",
            # A3C
            "a3c_episodes", "a3c_lr", "a3c_entropy", "a3c_value_coef", "a3c_gae_lambda", "a3c_bc_warmup", "a3c_bc_weight", "a3c_rollouts",
            # PPO
            "ppo_episodes", "ppo_lr", "ppo_epochs", "ppo_batch_episodes", "ppo_minibatch_size", "ppo_entropy", "ppo_value_coef", "ppo_gae_lambda", "ppo_bc_warmup", "ppo_bc_weight",
        ]
        ts = datetime.now(timezone.utc).isoformat()
        rows = []
        for model_name, (acc, rew) in results.items():
            rows.append({
                "timestamp": ts,
                "model": model_name,
                "acc": acc,
                "reward": rew,
                "seed": seed,
                "reward_correct_w": reward_correct_w,
                "reward_score_w": reward_score_w,
                "ql_epochs": ql_epochs,
                "ql_alpha": ql_alpha,
                "ql_gamma": ql_gamma,
                "ql_eps_start": ql_eps_start,
                "ql_eps_end": ql_eps_end,
                "ql_eps_decay_epochs": ql_eps_decay_epochs,
                "ql_select_best_on_val": ql_select_best_on_val,
                "ql_val_episodes": ql_val_episodes,
                "dqn_episodes": dqn_episodes,
                "dqn_lr": dqn_lr,
                "dqn_gamma": dqn_gamma,
                "dqn_batch_size": dqn_batch_size,
                "dqn_buffer_size": dqn_buffer_size,
                "dqn_hidden_dim": dqn_hidden_dim,
                "dqn_eps_start": dqn_eps_start,
                "dqn_eps_end": dqn_eps_end,
                "dqn_eps_decay_steps": dqn_eps_decay_steps,
                "dqn_target_tau": dqn_target_tau,
                "dqn_target_update_interval": dqn_target_update_interval,
                "dqn_select_best_on_val": dqn_select_best_on_val,
                "dqn_val_episodes": dqn_val_episodes,
                "a2c_episodes": a2c_episodes,
                "a2c_lr": a2c_lr,
                "a2c_entropy": a2c_entropy,
                "a2c_value_coef": a2c_value_coef,
                "a2c_bc_warmup": a2c_bc_warmup,
                "a2c_bc_weight": a2c_bc_weight,
                "a2c_batch_episodes": a2c_batch_episodes,
                "a3c_episodes": a3c_episodes,
                "a3c_lr": a3c_lr,
                "a3c_entropy": a3c_entropy,
                "a3c_value_coef": a3c_value_coef,
                "a3c_gae_lambda": a3c_gae_lambda,
                "a3c_bc_warmup": a3c_bc_warmup,
                "a3c_bc_weight": a3c_bc_weight,
                "a3c_rollouts": a3c_rollouts,
                "ppo_episodes": ppo_episodes,
                "ppo_lr": ppo_lr,
                "ppo_epochs": ppo_epochs,
                "ppo_batch_episodes": ppo_batch_episodes,
                "ppo_minibatch_size": ppo_minibatch_size,
                "ppo_entropy": ppo_entropy,
                "ppo_value_coef": ppo_value_coef,
                "ppo_gae_lambda": ppo_gae_lambda,
                "ppo_bc_warmup": ppo_bc_warmup,
                "ppo_bc_weight": ppo_bc_weight,
            })
        # Append mode; create header if file does not exist
        need_header = not os.path.exists(metrics_csv)
        with open(metrics_csv, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run curriculum sequencing RL experiments")
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.abspath(os.path.join(here, "..", "preprocessed_kt_data.csv"))
    parser.add_argument("--data", type=str, default=default_data, help="Path to preprocessed CSV")
    parser.add_argument("--ql_epochs", type=int, default=5)
    parser.add_argument("--dqn_episodes", type=int, default=50)
    parser.add_argument("--a2c_episodes", type=int, default=50)
    parser.add_argument("--a3c_episodes", type=int, default=50)
    parser.add_argument("--ppo_episodes", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=300)
    parser.add_argument("--reward_correct_w", type=float, default=0.5, help="Weight for correctness in reward")
    parser.add_argument("--reward_score_w", type=float, default=0.5, help="Weight for next score in reward")
    parser.add_argument("--models", type=str, default="ql,dqn,a2c,a3c,ppo", help="Comma-separated models to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_csv", type=str, default=None, help="Path to CSV file to append metrics")
    parser.add_argument("--no_chance", dest="include_chance", action="store_false", help="Disable chance baseline evaluation")
    parser.set_defaults(include_chance=True)
    # Q-Learning
    parser.add_argument("--ql_alpha", type=float, default=0.2)
    parser.add_argument("--ql_gamma", type=float, default=0.9)
    parser.add_argument("--ql_eps_start", type=float, default=0.3)
    parser.add_argument("--ql_eps_end", type=float, default=0.0)
    parser.add_argument("--ql_eps_decay_epochs", type=int, default=3)
    parser.add_argument("--ql_val_episodes", type=int, default=300)
    parser.add_argument("--no_ql_select_best", dest="ql_select_best_on_val", action="store_false", help="Disable validation-based selection for Q-Learning")
    parser.set_defaults(ql_select_best_on_val=True)
    # DQN
    parser.add_argument("--dqn_lr", type=float, default=1e-3)
    parser.add_argument("--dqn_gamma", type=float, default=0.99)
    parser.add_argument("--dqn_batch_size", type=int, default=128)
    parser.add_argument("--dqn_buffer_size", type=int, default=20000)
    parser.add_argument("--dqn_hidden_dim", type=int, default=128)
    parser.add_argument("--dqn_eps_start", type=float, default=1.0)
    parser.add_argument("--dqn_eps_end", type=float, default=0.05)
    parser.add_argument("--dqn_eps_decay_steps", type=int, default=20000)
    parser.add_argument("--dqn_target_tau", type=float, default=0.01)
    parser.add_argument("--dqn_target_update_interval", type=int, default=1)
    parser.add_argument("--dqn_val_episodes", type=int, default=300)
    parser.add_argument("--no_dqn_select_best", dest="dqn_select_best_on_val", action="store_false", help="Disable validation-based selection for DQN")
    parser.set_defaults(dqn_select_best_on_val=True)
    # A2C
    parser.add_argument("--a2c_lr", type=float, default=1e-3)
    parser.add_argument("--a2c_entropy", type=float, default=0.01)
    parser.add_argument("--a2c_value_coef", type=float, default=0.5)
    parser.add_argument("--a2c_bc_warmup", type=int, default=1)
    parser.add_argument("--a2c_bc_weight", type=float, default=0.5)
    parser.add_argument("--a2c_batch_episodes", type=int, default=4)
    # A3C
    parser.add_argument("--a3c_lr", type=float, default=1e-3)
    parser.add_argument("--a3c_entropy", type=float, default=0.01)
    parser.add_argument("--a3c_value_coef", type=float, default=0.5)
    parser.add_argument("--a3c_gae_lambda", type=float, default=0.95)
    parser.add_argument("--a3c_bc_warmup", type=int, default=1)
    parser.add_argument("--a3c_bc_weight", type=float, default=0.5)
    parser.add_argument("--a3c_rollouts", type=int, default=4)
    # PPO
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_episodes", type=int, default=8)
    parser.add_argument("--ppo_minibatch_size", type=int, default=2048)
    parser.add_argument("--ppo_entropy", type=float, default=0.01)
    parser.add_argument("--ppo_value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_bc_warmup", type=int, default=2)
    parser.add_argument("--ppo_bc_weight", type=float, default=1.0)
    args = parser.parse_args()

    model_list = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    print(f"Using data: {args.data}")
    run_all_and_report(
        data_path=args.data,
        ql_epochs=args.ql_epochs,
        dqn_episodes=args.dqn_episodes,
        a2c_episodes=args.a2c_episodes,
        a3c_episodes=args.a3c_episodes,
        ppo_episodes=args.ppo_episodes,
        eval_episodes=args.eval_episodes,
        models=model_list,
        reward_correct_w=args.reward_correct_w,
        reward_score_w=args.reward_score_w,
        seed=args.seed,
        metrics_csv=args.metrics_csv,
        include_chance=args.include_chance,
        # QL
        ql_alpha=args.ql_alpha,
        ql_gamma=args.ql_gamma,
        ql_eps_start=args.ql_eps_start,
        ql_eps_end=args.ql_eps_end,
        ql_eps_decay_epochs=args.ql_eps_decay_epochs,
        ql_select_best_on_val=args.ql_select_best_on_val,
        ql_val_episodes=args.ql_val_episodes,
        # DQN
        dqn_lr=args.dqn_lr,
        dqn_gamma=args.dqn_gamma,
        dqn_batch_size=args.dqn_batch_size,
        dqn_buffer_size=args.dqn_buffer_size,
        dqn_hidden_dim=args.dqn_hidden_dim,
        dqn_eps_start=args.dqn_eps_start,
        dqn_eps_end=args.dqn_eps_end,
        dqn_eps_decay_steps=args.dqn_eps_decay_steps,
        dqn_target_tau=args.dqn_target_tau,
        dqn_target_update_interval=args.dqn_target_update_interval,
        dqn_select_best_on_val=args.dqn_select_best_on_val,
        dqn_val_episodes=args.dqn_val_episodes,
        a2c_lr=args.a2c_lr,
        a2c_entropy=args.a2c_entropy,
        a2c_value_coef=args.a2c_value_coef,
        a2c_bc_warmup=args.a2c_bc_warmup,
        a2c_bc_weight=args.a2c_bc_weight,
        a2c_batch_episodes=args.a2c_batch_episodes,
        a3c_lr=args.a3c_lr,
        a3c_entropy=args.a3c_entropy,
        a3c_value_coef=args.a3c_value_coef,
        a3c_gae_lambda=args.a3c_gae_lambda,
        a3c_bc_warmup=args.a3c_bc_warmup,
        a3c_bc_weight=args.a3c_bc_weight,
        a3c_rollouts=args.a3c_rollouts,
        ppo_lr=args.ppo_lr,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_episodes=args.ppo_batch_episodes,
        ppo_minibatch_size=args.ppo_minibatch_size,
        ppo_entropy=args.ppo_entropy,
        ppo_value_coef=args.ppo_value_coef,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_bc_warmup=args.ppo_bc_warmup,
        ppo_bc_weight=args.ppo_bc_weight,
    )
