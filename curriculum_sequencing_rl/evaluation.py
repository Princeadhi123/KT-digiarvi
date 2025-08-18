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


def print_sample_rollouts(env, policy_fn, mode: str = "test", episodes: int = 1, max_steps: int = 15, model_name: str = None):
    """Print small, human-readable episode traces:
    shows current category, chosen action, ground-truth next category, correctness, and reward.
    """
    import numpy as _np
    name = f"[{model_name}] " if model_name else ""
    def _cat_name(idx: int) -> str:
        try:
            return f"{env.categories[int(idx)]}({int(idx)})"
        except Exception:
            return str(idx)
    for ep in range(episodes):
        print(f"{name}Episode {ep+1} [{mode}]")
        state = env.reset(mode)
        done = False
        steps = 0
        correct = 0
        rewards = []
        while not done and steps < max_steps:
            cur_cat = int(_np.argmax(state[: env.action_size]))
            action = policy_fn(state, cur_cat)
            next_state, reward, done, info = env.step(action)
            target = int(info.get('target', -1))
            corr = int(info.get('correct', 0))
            correct += corr
            rewards.append(float(reward))
            print(f"  step={steps+1:02d} cur={_cat_name(cur_cat)} -> action={_cat_name(action)} target={_cat_name(target)} correct={corr} reward={reward:.3f}")
            state = next_state if not done else state
            steps += 1
        ep_acc = (correct / steps) if steps > 0 else 0.0
        ep_avg_r = float(_np.mean(rewards)) if rewards else 0.0
        print(f"  summary: steps={steps}, acc={ep_acc:.3f}, avg_reward={ep_avg_r:.3f}")
