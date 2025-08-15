import random
from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from evaluation import eval_policy_category_accuracy


class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.adv = nn.Linear(hidden_dim, n_actions)
        self.val = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.feature(x)
        adv = self.adv(z)
        val = self.val(z)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, device: str = "cpu",
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 20000,
                 target_tau: float = 0.01, target_update_interval: int = 1,
                 lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 128, buffer_size: int = 20000,
                 hidden_dim: int = 128):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.q = DuelingQNet(state_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_tgt = DuelingQNet(state_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        # Epsilon schedule (linear)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.steps_done = 0
        self.batch_size = int(batch_size)
        self.buf = deque(maxlen=int(buffer_size))
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
        loss = torch.nn.functional.smooth_l1_loss(q_sa, y.detach())
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


def train_dqn(env, episodes: int = 50, device: str = None,
              eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 20000,
              target_tau: float = 0.01, target_update_interval: int = 1,
              lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 128, buffer_size: int = 20000,
              hidden_dim: int = 128, select_best_on_val: bool = True, val_episodes: int = 300) -> DQNAgent:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(env.state_dim, env.action_size, device=device,
                         eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
                         target_tau=target_tau, target_update_interval=target_update_interval,
                         lr=lr, gamma=gamma, batch_size=batch_size, buffer_size=buffer_size,
                         hidden_dim=hidden_dim)
        best_acc = -1.0
        best_state = None
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
            # Validation selection
            if select_best_on_val and ((ep + 1) % 5 == 0 or ep == episodes - 1):
                val_acc, _ = eval_policy_category_accuracy(env, dqn_policy(agent), mode="val", episodes=val_episodes)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = copy.deepcopy(agent.q.state_dict())
        if select_best_on_val and best_state is not None:
            agent.q.load_state_dict(best_state)
            agent.q_tgt.load_state_dict(agent.q.state_dict())
        return agent


def dqn_policy(agent: DQNAgent):
    def _policy(state, cur_cat: int) -> int:
        return agent.act(state, training=False)
    return _policy
