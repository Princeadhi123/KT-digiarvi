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
):
    env = CurriculumEnvV2(data_path, reward_correct_w=reward_correct_w, reward_score_w=reward_score_w)
    if models is None:
        models = ["ql", "dqn", "a2c", "a3c", "ppo"]

    results = {}

    if "ql" in models:
        ql = train_q_learning(env, epochs=ql_epochs)
        ql_acc, ql_reward = eval_policy_category_accuracy(env, greedy_from_qtable(ql), mode="test", episodes=eval_episodes)
        results["Q-Learning"] = (ql_acc, ql_reward)

    if "dqn" in models:
        dqn = train_dqn(env, episodes=dqn_episodes)
        dqn_acc, dqn_reward = eval_policy_category_accuracy(env, dqn_policy(dqn), mode="test", episodes=eval_episodes)
        results["DQN"] = (dqn_acc, dqn_reward)

    if "a2c" in models:
        a2c_net = train_a2c(env, episodes=a2c_episodes)
        a2c_acc, a2c_reward = eval_policy_category_accuracy(env, a2c_policy_fn(a2c_net, device=str(next(a2c_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["A2C"] = (a2c_acc, a2c_reward)

    if "a3c" in models:
        a3c_net = train_a3c(env, episodes=a3c_episodes)
        a3c_acc, a3c_reward = eval_policy_category_accuracy(env, a2c_policy_fn(a3c_net, device=str(next(a3c_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["A3C"] = (a3c_acc, a3c_reward)

    if "ppo" in models:
        ppo_net = train_ppo(env, episodes=ppo_episodes)
        ppo_acc, ppo_reward = eval_policy_category_accuracy(env, a2c_policy_fn(ppo_net, device=str(next(ppo_net.parameters()).device)), mode="test", episodes=eval_episodes)
        results["PPO"] = (ppo_acc, ppo_reward)

    print("\n=== Test Metrics (category accuracy, avg reward) ===")
    for name, (acc, rew) in results.items():
        print(f"{name:<10}: acc={acc:.3f}, reward={rew:.3f}")
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
    )
