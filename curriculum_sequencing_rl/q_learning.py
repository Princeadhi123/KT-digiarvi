import numpy as np
from collections import defaultdict
from typing import Any

from evaluation import eval_policy_avg_score


class QLearningBaseline:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.1, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int):
        td_target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * td_target


def train_q_learning(
    env: Any,
    epochs: int = 5,
    alpha: float = 0.2,
    gamma: float = 0.9,
    eps_start: float = 0.3,
    eps_end: float = 0.0,
    eps_decay_epochs: int = 3,
    seed: int = 42,
    select_best_on_val: bool = True,
    val_episodes: int = 300,
) -> QLearningBaseline:
    agent = QLearningBaseline(env.action_size, env.action_size, alpha=alpha, gamma=gamma, epsilon=eps_start, seed=seed)
    best_metric = -1e18
    best_Q = None
    # Iterate over train students epochs
    for ep in range(epochs):
        # Linear epsilon decay over epochs
        frac = 1.0 - min(ep, eps_decay_epochs) / max(1, eps_decay_epochs)
        agent.epsilon = eps_end + (eps_start - eps_end) * frac
        for sid in env.splits.train_students:
            s_df = env.df[env.df["student_id"] == sid]
            s_df = s_df.sort_values("order")
            for i in range(len(s_df) - 1):
                cur_cat = int(s_df.iloc[i]["category_id"])
                next_row = s_df.iloc[i + 1]
                next_cat = int(next_row["category_id"])
                a = agent.select_action(cur_cat)
                corr = 1.0 if a == next_cat else 0.0
                r = env.rw_correct * corr + env.rw_score * float(next_row.get("normalized_score", 0.0))
                agent.update(cur_cat, a, r, next_cat)
        if select_best_on_val:
            val_avg = eval_policy_avg_score(env, greedy_from_qtable(agent), mode="val", episodes=val_episodes)
            if val_avg > best_metric:
                best_metric = val_avg
                best_Q = agent.Q.copy()
    if select_best_on_val and best_Q is not None:
        agent.Q = best_Q
    return agent


def greedy_from_qtable(agent: QLearningBaseline):
    def _policy(state, cur_cat: int) -> int:
        return int(np.argmax(agent.Q[cur_cat]))
    return _policy
