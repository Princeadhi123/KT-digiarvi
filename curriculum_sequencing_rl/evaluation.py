from typing import Tuple
import numpy as np


def eval_policy_category_accuracy(env, policy_fn, mode: str = "test", episodes: int = 200) -> Tuple[float, float]:
    """Returns (accuracy, avg_reward)."""
    correct = 0
    total = 0
    rewards = []
    for _ in range(episodes):
        state = env.reset(mode)
        done = False
        while not done:
            cur_cat = int(np.argmax(state[: env.action_size]))
            action = policy_fn(state, cur_cat)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            correct += int(info.get("correct", 0))
            total += 1
            state = next_state if not done else state
    acc = (correct / total) if total > 0 else 0.0
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return acc, avg_reward
