import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def train_ppo(env, episodes: int = 50, gamma: float = 0.99, clip_eps: float = 0.2, lr: float = 3e-4, device: str = None) -> PPOActorCritic:
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
