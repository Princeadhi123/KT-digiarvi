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


def _bc_pretrain_policy(net: ActorCritic, env, epochs: int, lr: float, device: str):
    """Behavior cloning warmup: supervised next-category prediction from dataset transitions."""
    opt = optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(max(0, epochs)):
        states_batch, targets_batch = [], []
        for sid in env.splits.train_students:
            s_df = env.df[env.df["student_id"] == sid].sort_values("order")
            for i in range(len(s_df) - 1):
                cur_row = s_df.iloc[i]
                nxt_row = s_df.iloc[i + 1]
                state = env._build_state_from_row(cur_row)
                target = int(nxt_row["category_id"])
                states_batch.append(torch.tensor(state, dtype=torch.float32, device=device))
                targets_batch.append(target)
                # Minibatch update to bound memory
                if len(states_batch) >= 512:
                    st = torch.stack(states_batch)
                    logits, _ = net(st)
                    loss = F.cross_entropy(logits, torch.tensor(targets_batch, dtype=torch.long, device=device))
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    states_batch, targets_batch = [], []
        if states_batch:
            st = torch.stack(states_batch)
            logits, _ = net(st)
            loss = F.cross_entropy(logits, torch.tensor(targets_batch, dtype=torch.long, device=device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def train_a2c(
    env,
    episodes: int = 50,
    gamma: float = 0.99,
    lr: float = 1e-3,
    device: str = None,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    bc_warmup_epochs: int = 1,
    bc_weight: float = 0.5,
    batch_episodes: int = 4,
) -> ActorCritic:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.state_dim, env.action_size).to(device)

    # Optional supervised warm start
    if bc_warmup_epochs > 0:
        _bc_pretrain_policy(net, env, epochs=bc_warmup_epochs, lr=lr, device=device)

    opt = optim.Adam(net.parameters(), lr=lr)
    episodes_done = 0
    while episodes_done < episodes:
        # Collect a small batch of episodes
        batch_states, batch_actions, batch_rewards, batch_values, batch_targets = [], [], [], [], []
        for _ in range(min(batch_episodes, episodes - episodes_done)):
            s = env.reset("train")
            done = False
            ep_states, ep_actions, ep_rewards, ep_values, ep_targets = [], [], [], [], []
            while not done:
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = net(st)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
                ns, r, done, info = env.step(a)
                ep_states.append(st.squeeze(0))
                ep_actions.append(a)
                ep_rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
                ep_values.append(v.squeeze(0))
                ep_targets.append(int(info.get("target", 0)))
                s = ns if not done else s
            # Append episode to batch
            batch_states.append(torch.stack(ep_states))
            batch_actions.append(torch.tensor(ep_actions, dtype=torch.long, device=device))
            batch_rewards.append(torch.stack(ep_rewards))
            batch_values.append(torch.stack(ep_values))
            batch_targets.append(torch.tensor(ep_targets, dtype=torch.long, device=device))
            episodes_done += 1

        # Compute returns and advantages for the batch
        all_log_probs, all_adv, all_values, all_states, all_targets = [], [], [], [], []
        for states, actions, rewards, values, targets in zip(batch_states, batch_actions, batch_rewards, batch_values, batch_targets):
            R = torch.tensor(0.0, device=device)
            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.append(R)
            returns = list(reversed(returns))
            returns = torch.stack(returns)
            adv = returns - values.detach().squeeze(-1)
            # Advantage normalization per-episode
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            logits, _ = net(states)
            dist = torch.distributions.Categorical(logits=logits)
            logps = dist.log_prob(actions)

            all_log_probs.append(logps)
            all_adv.append(adv)
            all_values.append(values.squeeze(-1))
            all_states.append(states)
            all_targets.append(targets)

        logps_cat = torch.cat(all_log_probs)
        adv_cat = torch.cat(all_adv)
        values_cat = torch.cat(all_values)
        states_cat = torch.cat(all_states)
        targets_cat = torch.cat(all_targets)

        # Recompute for losses
        logits_cat, v_cat = net(states_cat)
        dist_cat = torch.distributions.Categorical(logits=logits_cat)
        entropy = dist_cat.entropy().mean()
        policy_loss = -(logps_cat * adv_cat).mean()
        # For value loss, rebuild returns to align with values_cat shape
        # Approximate by (adv + values.detach()) target
        returns_est = adv_cat.detach() + values_cat.detach()
        value_loss = F.mse_loss(v_cat.squeeze(-1), returns_est)
        # Auxiliary behavior cloning loss
        ce_loss = F.cross_entropy(logits_cat, targets_cat)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + bc_weight * ce_loss
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
