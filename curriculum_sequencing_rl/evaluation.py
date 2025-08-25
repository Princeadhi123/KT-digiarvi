from typing import Any, Callable, Optional, Tuple
import numpy as np


def print_sample_rollouts(env: Any, policy_fn: Callable[[Any, int], int], mode: str = "test", episodes: int = 1, max_steps: int = 15, model_name: Optional[str] = None) -> None:
    """Print small, human-readable episode traces.

    Args:
        env: Environment exposing `reset(mode)`, `step(action)`, `categories`, and `action_size`.
        policy_fn: Callable(state, cur_cat) -> action.
        mode: Split to evaluate: 'train' | 'val' | 'test'.
        episodes: Number of episodes to print.
        max_steps: Max steps per episode.
        model_name: Optional label prefix for prints.
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
        rewards = []
        while not done and steps < max_steps:
            cur_cat = int(_np.argmax(state[: env.action_size]))
            action = policy_fn(state, cur_cat)
            next_state, reward, done, info = env.step(action)
            rewards.append(float(reward))
            if 'target' in info:
                target = int(info.get('target', -1))
                print(f"  step={steps+1:02d} cur={_cat_name(cur_cat)} -> suggested={_cat_name(action)} target={_cat_name(target)} reward={reward:.3f}")
            else:
                # Interactive env path: show valid_action instead
                va = info.get('valid_action', None)
                va_txt = f" valid={int(va)}" if va is not None else ""
                # Try to show remaining actionable types if available
                rem_txt = ""
                try:
                    rem = len(getattr(env, 'valid_action_ids')())
                    rem_txt = f" remaining_action_types={rem}"
                except Exception:
                    pass
                print(f"  step={steps+1:02d} cur={_cat_name(cur_cat)} -> suggested={_cat_name(action)}{va_txt}{rem_txt} reward={reward:.3f}")
            state = next_state if not done else state
            steps += 1
        ep_avg_r = float(_np.mean(rewards)) if rewards else 0.0
        print(f"  summary: steps={steps}, avg_reward={ep_avg_r:.3f}")


def eval_policy_avg_score(env: Any, policy_fn: Callable[[Any, int], int], mode: str = "test", episodes: int = 200) -> float:
    """Evaluate a policy by averaging per-step rewards (e.g., normalized_score) across episodes.

    Returns avg_reward (float).
    """
    rewards = []
    for _ in range(episodes):
        state = env.reset(mode)
        done = False
        while not done:
            cur_cat = int(np.argmax(state[: env.action_size]))
            action = policy_fn(state, cur_cat)
            next_state, reward, done, info = env.step(action)
            rewards.append(float(reward))
            state = next_state if not done else state
    return float(np.mean(rewards)) if rewards else 0.0


# -----------------------------
# Interactive diagnostics
# -----------------------------

def _best_remaining_immediate_reward(env: Any) -> float:
    """Return the best immediate reward available from current interactive state.

    Prefers env.estimate_immediate_reward(aid) when available (captures shaping).
    Falls back to score-only based on normalized_score. Returns NaN on failure.
    """
    try:
        valid_ids = list(env.valid_action_ids())
        # Preferred path: environment-provided estimator (includes shaping if configured)
        if hasattr(env, "estimate_immediate_reward"):
            if not valid_ids:
                return 0.0
            best_est = max(float(env.estimate_immediate_reward(aid)) for aid in valid_ids)
            return float(best_est)
        # Fallback: compute score-only best from env internals
        best = 0.0
        for aid in valid_ids:
            idxs = env._rows_by_action.get(int(aid), [])  # type: ignore[attr-defined]
            nxt_idx = None
            for i in idxs:
                if i not in env._consumed:  # type: ignore[attr-defined]
                    nxt_idx = i
                    break
            if nxt_idx is None:
                continue
            row = env.current_student_df.iloc[nxt_idx]
            norm_score = float(row.get("normalized_score", 0.0))
            best = max(best, env.rw_score * norm_score)
        return float(best)
    except Exception:
        return float("nan")


def _run_interactive_diagnostics(env: Any, policy_fn: Callable[[Any, int], int], mode: str, episodes: int) -> Tuple[float, float, float, float]:
    """Internal helper computing VPR, regret, regret_ratio, avg_reward."""
    total_steps = 0
    vpr_hits = 0.0
    regret_sum = 0.0
    regret_ratio_sum = 0.0
    reward_sum = 0.0
    interactive_ok = hasattr(env, "valid_action_ids") and hasattr(env, "_rows_by_action")
    for _ in range(episodes):
        state = env.reset(mode)
        done = False
        while not done:
            best_pre = _best_remaining_immediate_reward(env) if interactive_ok else float("nan")
            cur_cat = int(np.argmax(state[: env.action_size]))
            action = policy_fn(state, cur_cat)
            next_state, reward, done, info = env.step(action)
            reward = float(reward)
            reward_sum += reward
            total_steps += 1
            # VPR
            if isinstance(info, dict) and ("valid_action" in info):
                vpr_hits += 1.0 if bool(info.get("valid_action")) else 0.0
            # Regret
            if not np.isnan(best_pre):
                reg = max(0.0, best_pre - reward)
                regret_sum += reg
                if best_pre > 1e-9:
                    regret_ratio_sum += max(0.0, reg / best_pre)
            state = next_state if not done else state
    avg_reward = (reward_sum / total_steps) if total_steps > 0 else 0.0
    vpr = (vpr_hits / total_steps) if total_steps > 0 else float("nan")
    avg_regret = (regret_sum / total_steps) if total_steps > 0 else float("nan")
    avg_regret_ratio = (regret_ratio_sum / total_steps) if total_steps > 0 else float("nan")
    return vpr, avg_regret, avg_regret_ratio, avg_reward


def eval_policy_valid_pick_rate(env: Any, policy_fn: Callable[[Any, int], int], mode: str = "test", episodes: int = 200) -> float:
    """Fraction of steps where the chosen action was valid (Interactive env only).

    Returns NaN if environment doesn't provide valid_action info.
    """
    vpr, _, _, _ = _run_interactive_diagnostics(env, policy_fn, mode, episodes)
    return float(vpr)


def eval_policy_regret(env: Any, policy_fn: Callable[[Any, int], int], mode: str = "test", episodes: int = 200) -> Tuple[float, float]:
    """Compute average instantaneous regret and regret ratio for interactive env.

    - avg_regret: E[max_best_possible_reward - obtained_reward]
    - avg_regret_ratio: E[regret / best_possible_reward], 0 if best_possible_reward == 0
    Returns (avg_regret, avg_regret_ratio). NaN if not interactive.
    """
    _, avg_regret, avg_regret_ratio, _ = _run_interactive_diagnostics(env, policy_fn, mode, episodes)
    return float(avg_regret), float(avg_regret_ratio)
