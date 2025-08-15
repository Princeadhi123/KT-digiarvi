import numpy as np
from collections import defaultdict

from env import CurriculumEnvV2


class QLearningBaseline:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(42)

    def select_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int):
        td_target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * td_target


def train_q_learning(env: CurriculumEnvV2, epochs: int = 5) -> QLearningBaseline:
    agent = QLearningBaseline(env.action_size, env.action_size)
    # Iterate over train students epochs
    for _ in range(epochs):
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
    return agent


def greedy_from_qtable(agent: QLearningBaseline):
    def _policy(state, cur_cat: int) -> int:
        return int(np.argmax(agent.Q[cur_cat]))
    return _policy
