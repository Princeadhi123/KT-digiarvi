import torch
import torch.nn.functional as F
import torch.optim as optim

from a2c import ActorCritic


def train_a3c(env, episodes: int = 50, rollouts_per_update: int = 4, gamma: float = 0.99, lr: float = 1e-3, device: str = None) -> ActorCritic:
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
        import torch as _torch
        logps_t = _torch.cat(batch_logps)
        values_t = _torch.cat(batch_values)
        returns_t = _torch.cat(batch_returns)
        adv = returns_t - values_t.detach()
        policy_loss = -(logps_t * adv).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy = 0.0
        loss = policy_loss + 0.5 * value_loss - 0.0 * (entropy if isinstance(entropy, _torch.Tensor) else 0.0)
        opt.zero_grad()
        loss.backward()
        _torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    return net
