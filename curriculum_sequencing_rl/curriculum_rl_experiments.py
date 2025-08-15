import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict

# -----------------------------
# Environment using training-safe features only
# -----------------------------


def _standardize_sex(val: str) -> float:
    if pd.isna(val):
        return 0.5
    s = str(val).strip().lower()
    if s in ("boy", "male", "m"):
        return 0.0
    if s in ("girl", "gir", "female", "f"):
        return 1.0
    return 0.5


@dataclass
class SplitData:
    train_students: np.ndarray
    val_students: np.ndarray
    test_students: np.ndarray


class CurriculumEnvV2:
    """
    Offline environment built from preprocessed CSV.

    State (training-safe):
    - one-hot current category
    - order_norm (global 0-1)
    - normalized_score (current)
    - grade_encoded (label-encoded)
    - sex_binary (Girl=1, Boy=0, unknown=0.5)
    - home_school_lang_match (0/1, NaN->0.5)
    - missing_all, missing_beginning30, missing_last50 (NaN->0)

    Action space: category id to present next.

    Reward (shaped): 0.5 * 1{action == target_next_category} + 0.5 * next_normalized_score
    """

    def __init__(self, data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42,
                 reward_correct_w: float = 0.5, reward_score_w: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.df = pd.read_csv(data_path)

        # Reward weighting (normalized to sum to 1 when possible)
        total_w = float(reward_correct_w) + float(reward_score_w)
        if total_w <= 0:
            self.rw_correct, self.rw_score = 1.0, 0.0
        else:
            self.rw_correct = float(reward_correct_w) / total_w
            self.rw_score = float(reward_score_w) / total_w

        # Required columns check
        required = [
            "student_id", "exercise_id", "category", "order", "normalized_score",
            "grade", "sex", "home_school_lang_match",
            "missing_all", "missing_beginning30", "missing_last50"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Category encoding and action space
        self.df["category"] = self.df["category"].fillna("Unknown")
        self.categories: List[str] = sorted(self.df["category"].unique())
        self.cat2id: Dict[str, int] = {c: i for i, c in enumerate(self.categories)}
        self.action_size = len(self.categories)

        # Global order normalization (consistent across students)
        order_min = self.df["order"].min()
        order_max = self.df["order"].max()
        self.df["order_norm"] = (self.df["order"] - order_min) / (order_max - order_min + 1e-9)

        # Sex
        self.df["sex_bin"] = self.df["sex"].apply(_standardize_sex)

        # Grade encoding
        self.df["grade"] = self.df["grade"].astype(str).fillna("Unknown")
        grades = sorted(self.df["grade"].unique())
        self.grade2id = {g: i for i, g in enumerate(grades)}
        self.df["grade_enc"] = self.df["grade"].map(self.grade2id).astype(float)
        if len(grades) > 1:
            self.df["grade_enc"] = self.df["grade_enc"] / (len(grades) - 1)

        # Language match and missingness
        self.df["home_school_lang_match"] = self.df["home_school_lang_match"].fillna(0.5).astype(float)
        for col in ["missing_all", "missing_beginning30", "missing_last50"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)

        # One-hot category for state
        self.df["category_id"] = self.df["category"].map(self.cat2id).astype(int)

        # Build per-student ordered data
        self.df.sort_values(["student_id", "order"], inplace=True)

        # Split by students
        self.splits = self._split_students(train_ratio, val_ratio, seed)

        # State dimension = one-hot(category) + [order_norm, norm_score, grade_enc, sex_bin, lang_match, missing_all, missing_beginning30, missing_last50]
        self.state_dim = self.action_size + 8

        # Episode internals
        self.current_student_id = None
        self.current_student_df = None
        self.ptr = 0

    def _split_students(self, train_ratio: float, val_ratio: float, seed: int) -> SplitData:
        students = self.df["student_id"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(students)
        n = len(students)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_students = students[:n_train]
        val_students = students[n_train:n_train + n_val]
        test_students = students[n_train + n_val:]
        return SplitData(train_students, val_students, test_students)

    def _one_hot(self, idx: int, n: int) -> np.ndarray:
        v = np.zeros(n, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _build_state_from_row(self, row: pd.Series) -> np.ndarray:
        cat_oh = self._one_hot(int(row["category_id"]), self.action_size)
        x = np.array([
            float(row["order_norm"]),
            float(row.get("normalized_score", 0.0)),
            float(row["grade_enc"]),
            float(row["sex_bin"]),
            float(row["home_school_lang_match"]),
            float(row["missing_all"]),
            float(row["missing_beginning30"]),
            float(row["missing_last50"]),
        ], dtype=np.float32)
        return np.concatenate([cat_oh, x], dtype=np.float32)

    def reset(self, mode: str = "train") -> np.ndarray:
        if mode == "train":
            sids = self.splits.train_students
        elif mode == "val":
            sids = self.splits.val_students
        elif mode == "test":
            sids = self.splits.test_students
        else:
            raise ValueError("mode must be one of {'train','val','test'}")
        if len(sids) == 0:
            raise ValueError(f"No students in split '{mode}'")
        self.current_student_id = int(self.rng.choice(sids))
        self.current_student_df = self.df[self.df["student_id"] == self.current_student_id]
        self.ptr = 0
        return self._build_state_from_row(self.current_student_df.iloc[self.ptr])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # If at the last row for this student, episode ends
        if self.ptr >= len(self.current_student_df) - 1:
            return None, 0.0, True, {}

        next_row = self.current_student_df.iloc[self.ptr + 1]
        target_cat = int(next_row["category_id"])
        corr = 1.0 if int(action) == target_cat else 0.0
        next_norm_score = float(next_row.get("normalized_score", 0.0))
        reward = self.rw_correct * corr + self.rw_score * next_norm_score
        
        self.ptr += 1
        done = self.ptr >= len(self.current_student_df) - 1
        next_state = self._build_state_from_row(self.current_student_df.iloc[self.ptr])
        return next_state, float(reward), bool(done), {"correct": corr, "target": target_cat}


# -----------------------------
# Q-Learning baseline (tabular on category->category transitions)
# -----------------------------

class QLearningBaseline:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.1):
        # state = current category id, action = next category id
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
                # action chosen epsilon-greedy on current state
                a = agent.select_action(cur_cat)
                # environment transition is fixed by dataset; reward depends on chosen a
                corr = 1.0 if a == next_cat else 0.0
                r = 0.5 * corr + 0.5 * float(next_row.get("normalized_score", 0.0))
                agent.update(cur_cat, a, r, next_cat)
    return agent


def eval_policy_category_accuracy(env: CurriculumEnvV2, policy_fn, mode: str = "test", episodes: int = 200) -> Tuple[float, float]:
    """Returns (accuracy, avg_reward)."""
    correct = 0
    total = 0
    rewards = []
    for _ in range(episodes):
        state = env.reset(mode)
        done = False
        while not done:
            # current category = argmax of one-hot prefix
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


# -----------------------------
# DQN
# -----------------------------

class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.adv = nn.Linear(128, n_actions)
        self.val = nn.Linear(128, 1)

    def forward(self, x):
        z = self.feature(x)
        adv = self.adv(z)
        val = self.val(z)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, device: str = "cpu",
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 20000,
                 target_tau: float = 0.01, target_update_interval: int = 1):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.q = DuelingQNet(state_dim, n_actions).to(self.device)
        self.q_tgt = DuelingQNet(state_dim, n_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=1e-3)
        self.gamma = 0.99
        # Epsilon schedule (linear)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.steps_done = 0
        self.batch_size = 128
        self.buf = deque(maxlen=20000)
        # Target update
        self.target_tau = float(target_tau)
        self.target_update_interval = int(target_update_interval)

    def _epsilon(self):
        t = min(self.steps_done, self.eps_decay_steps)
        frac = 1.0 - (t / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    def act(self, state: np.ndarray, training: bool = True) -> int:
        eps = self._epsilon()
        if training and random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)
            return int(torch.argmax(qv, dim=1).item())

    def remember(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def _soft_update_target(self):
        with torch.no_grad():
            for tgt, src in zip(self.q_tgt.parameters(), self.q.parameters()):
                tgt.data.copy_((1.0 - self.target_tau) * tgt.data + self.target_tau * src.data)

    def replay(self):
        if len(self.buf) < self.batch_size:
            return 0.0
        batch = random.sample(self.buf, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        non_final_mask = torch.tensor([ns_i is not None for ns_i in ns], device=self.device, dtype=torch.bool)
        ns_non_final = torch.tensor(
            np.stack([ns_i for ns_i in ns if ns_i is not None]) if any(non_final_mask.cpu().numpy()) else np.zeros((0, s.shape[1])),
            dtype=torch.float32, device=self.device)
        q_sa = self.q(s).gather(1, a)
        q_next = torch.zeros(self.batch_size, 1, device=self.device)
        if ns_non_final.shape[0] > 0:
            # Double DQN: action from online, value from target
            next_online_q = self.q(ns_non_final)
            next_actions = torch.argmax(next_online_q, dim=1)
            next_target_q = self.q_tgt(ns_non_final).gather(1, next_actions.unsqueeze(1))
            q_next[non_final_mask] = next_target_q
        y = r + (1 - torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)) * self.gamma * q_next
        loss = F.smooth_l1_loss(q_sa, y.detach())
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()
        self.steps_done += 1
        if self.target_update_interval > 0 and (self.steps_done % self.target_update_interval == 0):
            self._soft_update_target()
        return float(loss.item())

    def update_target(self):
        self.q_tgt.load_state_dict(self.q.state_dict())


def train_dqn(env: CurriculumEnvV2, episodes: int = 50, device: str = None,
              eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 20000,
              target_tau: float = 0.01, target_update_interval: int = 1) -> DQNAgent:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_dim, env.action_size, device=device,
                     eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
                     target_tau=target_tau, target_update_interval=target_update_interval)
    for ep in range(episodes):
        s = env.reset("train")
        done = False
        steps = 0
        while not done and steps < 1000:
            a = agent.act(s, training=True)
            ns, r, done, _ = env.step(a)
            agent.remember(s, a, r, ns, done)
            agent.replay()
            s = ns if not done else s
            steps += 1
        # Hard sync every few episodes to reduce drift
        if (ep + 1) % 20 == 0:
            agent.q_tgt.load_state_dict(agent.q.state_dict())
    return agent


# -----------------------------
# A2C (synchronous advantage actor-critic)
# -----------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        z = self.body(x)
        logits = self.pi(z)
        value = self.v(z)
        return logits, value


def train_a2c(env: CurriculumEnvV2, episodes: int = 50, gamma: float = 0.99, lr: float = 1e-3, device: str = None) -> ActorCritic:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.state_dim, env.action_size).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    for _ in range(episodes):
        s = env.reset("train")
        done = False
        logps, values, rewards = [], [], []
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits, v = net(st)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            a = int(dist.sample().item())
            ns, r, done, _ = env.step(a)
            logps.append(dist.log_prob(torch.tensor(a, device=device)))
            values.append(v.squeeze(0))
            rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            s = ns if not done else s

        # Compute returns and advantages
        R = torch.tensor(0.0, device=device)
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        returns = torch.stack(returns)
        values = torch.stack(values).squeeze(-1)
        adv = returns - values.detach()

        policy_loss = -(torch.stack(logps) * adv).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = -torch.mean(torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1))
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


def a2c_policy_fn(net: ActorCritic, device: str):
    def _policy(state: np.ndarray, cur_cat: int) -> int:
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(st)
            return int(torch.argmax(logits, dim=-1).item())
    return _policy


# -----------------------------
# A3C (single-process, multi-episode update variant)
# -----------------------------

def train_a3c(env: CurriculumEnvV2, episodes: int = 50, rollouts_per_update: int = 4, gamma: float = 0.99, lr: float = 1e-3, device: str = None) -> ActorCritic:
    """
    Simplified A3C-style trainer: collects multiple on-policy episodes per update and
    applies a shared-parameter update. In offline setting, this approximates A3C.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.state_dim, env.action_size).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    episodes_done = 0
    while episodes_done < episodes:
        batch_logps, batch_values, batch_returns = [], [], []
        for _ in range(min(rollouts_per_update, episodes - episodes_done)):
            s = env.reset("train")
            done = False
            logps, values, rewards = [], [], []
            while not done:
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = net(st)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
                ns, r, done, _ = env.step(a)
                logps.append(dist.log_prob(torch.tensor(a, device=device)))
                values.append(v.squeeze(0))
                rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
                s = ns if not done else s

            # Returns
            R = torch.tensor(0.0, device=device)
            ret = []
            for rr in reversed(rewards):
                R = rr + gamma * R
                ret.append(R)
            ret = list(reversed(ret))
            batch_logps.append(torch.stack(logps))
            batch_values.append(torch.stack(values).squeeze(-1))
            batch_returns.append(torch.stack(ret))
            episodes_done += 1

        # Concatenate rollouts and update
        logps_t = torch.cat(batch_logps)
        values_t = torch.cat(batch_values)
        returns_t = torch.cat(batch_returns)
        adv = returns_t - values_t.detach()
        policy_loss = -(logps_t * adv).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        # Entropy bonus using last logits for simplicity
        # (approximate; proper A3C would compute per-step entropies)
        entropy = 0.0
        loss = policy_loss + 0.5 * value_loss - 0.0 * (entropy if isinstance(entropy, torch.Tensor) else 0.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    return net


# -----------------------------
# PPO (clipped surrogate)
# -----------------------------

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z)


def train_ppo(env: CurriculumEnvV2, episodes: int = 50, gamma: float = 0.99, clip_eps: float = 0.2, lr: float = 3e-4, device: str = None) -> PPOActorCritic:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = PPOActorCritic(env.state_dim, env.action_size).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    for _ in range(episodes):
        # Collect one episode (on-policy)
        s = env.reset("train")
        done = False
        states, actions, rewards, logps_old, values = [], [], [], [], []
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits, v = net(st)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            a = int(dist.sample().item())
            ns, r, done, _ = env.step(a)
            states.append(st.squeeze(0))
            actions.append(torch.tensor(a, device=device))
            rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            # Important: store old log-probs detached to avoid backprop through their graph in later epochs
            logps_old.append(dist.log_prob(torch.tensor(a, device=device)).detach())
            values.append(v.squeeze(0))
            s = ns if not done else s

        # Compute returns and advantages
        R = torch.tensor(0.0, device=device)
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        returns = torch.stack(returns)
        values = torch.stack(values)
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Optimize with clipped objective
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        # Old log-probs must be treated as constants for PPO ratio
        logps_old_t = torch.stack(logps_old).detach()
        for _ in range(3):  # a few epochs
            logits, v = net(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            logps = dist.log_prob(actions_t)
            ratio = torch.exp(logps - logps_old_t)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            obj = torch.min(ratio * adv, clipped).mean()
            vf_loss = F.mse_loss(v.squeeze(-1), returns)
            entropy = dist.entropy().mean()
            loss = -(obj - 0.5 * vf_loss + 0.01 * entropy)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


def greedy_from_qtable(agent: QLearningBaseline):
    def _policy(state: np.ndarray, cur_cat: int) -> int:
        return int(np.argmax(agent.Q[cur_cat]))
    return _policy


def dqn_policy(agent: DQNAgent):
    def _policy(state: np.ndarray, cur_cat: int) -> int:
        return agent.act(state, training=False)
    return _policy


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
