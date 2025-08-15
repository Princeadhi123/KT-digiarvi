import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def train_a2c(env, episodes: int = 50, gamma: float = 0.99, lr: float = 1e-3, device: str = None) -> ActorCritic:
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
    def _policy(state, cur_cat: int) -> int:
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(st)
            return int(torch.argmax(logits, dim=-1).item())
    return _policy
