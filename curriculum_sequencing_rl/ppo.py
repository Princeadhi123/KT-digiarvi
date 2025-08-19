import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c import _bc_pretrain_policy

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


def train_ppo(
    env,
    episodes: int = 50,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    device: str = None,
    ppo_epochs: int = 4,
    batch_episodes: int = 8,
    minibatch_size: int = 2048,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    gae_lambda: float = 0.95,
    bc_warmup_epochs: int = 2,
    bc_weight: float = 1.0,
) -> PPOActorCritic:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = PPOActorCritic(env.state_dim, env.action_size).to(device)

    # Optional behavior cloning warm start using dataset transitions
    if bc_warmup_epochs > 0:
        _bc_pretrain_policy(net, env, epochs=bc_warmup_epochs, lr=lr, device=device)

    opt = optim.Adam(net.parameters(), lr=lr)

    episodes_done = 0
    while episodes_done < episodes:
        # Collect a batch of on-policy trajectories
        traj_states, traj_actions, traj_rewards, traj_values, traj_logp_old, traj_targets = [], [], [], [], [], []
        for _ in range(min(batch_episodes, episodes - episodes_done)):
            s = env.reset("train")
            done = False
            ep_states, ep_actions, ep_rewards, ep_values, ep_logp_old, ep_targets = [], [], [], [], [], []
            while not done:
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = net(st)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
                ns, r, done, info = env.step(a)
                logp_a = dist.log_prob(torch.tensor(a, device=device)).detach()
                ep_states.append(st.squeeze(0))
                ep_actions.append(torch.tensor(a, device=device))
                ep_rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
                # Detach stored values to avoid linking the data collection graph to PPO update steps
                ep_values.append(v.squeeze(0).detach())
                ep_logp_old.append(logp_a)
                ep_targets.append(int(info.get("target", 0)))
                s = ns if not done else s
            traj_states.append(torch.stack(ep_states))
            traj_actions.append(torch.stack(ep_actions))
            traj_rewards.append(torch.stack(ep_rewards))
            traj_values.append(torch.stack(ep_values))
            traj_logp_old.append(torch.stack(ep_logp_old))
            traj_targets.append(torch.tensor(ep_targets, dtype=torch.long, device=device))
            episodes_done += 1

        # Flatten trajectories and compute GAE advantages and returns
        all_states, all_actions, all_returns, all_adv, all_logp_old, all_targets = [], [], [], [], [], []
        for states, actions, rewards, values, logp_old, targets in zip(traj_states, traj_actions, traj_rewards, traj_values, traj_logp_old, traj_targets):
            with torch.no_grad():
                values = values.squeeze(-1)
                T = values.shape[0]
                adv = torch.zeros(T, device=device)
                lastgaelam = 0.0
                for t in reversed(range(T)):
                    next_v = torch.tensor(0.0, device=device) if t == T - 1 else values[t + 1]
                    delta = rewards[t] + gamma * next_v - values[t]
                    lastgaelam = delta + gamma * gae_lambda * lastgaelam
                    adv[t] = lastgaelam
                returns = adv + values
                # Normalize advantages per-episode
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            all_states.append(states)
            all_actions.append(actions)
            all_returns.append(returns.detach())
            all_adv.append(adv.detach())
            all_logp_old.append(logp_old.detach())
            all_targets.append(targets)

        states_cat = torch.cat(all_states)
        actions_cat = torch.cat(all_actions)
        returns_cat = torch.cat(all_returns)
        adv_cat = torch.cat(all_adv)
        logp_old_cat = torch.cat(all_logp_old).detach()
        targets_cat = torch.cat(all_targets)

        # PPO optimization with minibatches
        N = states_cat.shape[0]
        idx = torch.randperm(N, device=device)
        for _ in range(ppo_epochs):
            for start in range(0, N, minibatch_size):
                mb_idx = idx[start:start + minibatch_size]
                mb_states = states_cat[mb_idx]
                mb_actions = actions_cat[mb_idx]
                mb_returns = returns_cat[mb_idx]
                mb_adv = adv_cat[mb_idx]
                mb_logp_old = logp_old_cat[mb_idx]
                mb_targets = targets_cat[mb_idx]

                logits, v = net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                logps = dist.log_prob(mb_actions)
                ratio = torch.exp(logps - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_obj = torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(v.squeeze(-1), mb_returns)
                entropy = dist.entropy().mean()
                # Disable online BC for interactive env (no supervised targets)
                if hasattr(env, "valid_action_ids"):
                    ce_loss = torch.tensor(0.0, device=device)
                else:
                    ce_loss = F.cross_entropy(logits, mb_targets)
                loss = -(policy_obj) + value_coef * vf_loss - entropy_coef * entropy + bc_weight * ce_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
    return net
