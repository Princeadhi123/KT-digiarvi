import torch
import torch.nn.functional as F
import torch.optim as optim

from a2c import ActorCritic, _bc_pretrain_policy


def train_a3c(
    env,
    episodes: int = 50,
    rollouts_per_update: int = 4,
    gamma: float = 0.99,
    lr: float = 1e-3,
    device: str = None,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    gae_lambda: float = 0.95,
    bc_warmup_epochs: int = 1,
    bc_weight: float = 0.5,
) -> ActorCritic:
    """Improved A3C with GAE, entropy regularization, and auxiliary behavior cloning."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.state_dim, env.action_size).to(device)

    if bc_warmup_epochs > 0:
        _bc_pretrain_policy(net, env, epochs=bc_warmup_epochs, lr=lr, device=device)

    opt = optim.Adam(net.parameters(), lr=lr)

    episodes_done = 0
    while episodes_done < episodes:
        batch_states, batch_actions, batch_rewards, batch_values, batch_targets = [], [], [], [], []
        for _ in range(min(rollouts_per_update, episodes - episodes_done)):
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
            batch_states.append(torch.stack(ep_states))
            batch_actions.append(torch.tensor(ep_actions, dtype=torch.long, device=device))
            batch_rewards.append(torch.stack(ep_rewards))
            batch_values.append(torch.stack(ep_values))
            batch_targets.append(torch.tensor(ep_targets, dtype=torch.long, device=device))
            episodes_done += 1

        # Compute GAE advantages and returns
        all_log_probs, all_adv, all_returns, all_states, all_actions, all_targets = [], [], [], [], [], []
        for states, actions, rewards, values, targets in zip(batch_states, batch_actions, batch_rewards, batch_values, batch_targets):
            values = values.squeeze(-1)
            T = values.shape[0]
            adv = torch.zeros(T, device=device)
            lastgaelam = 0.0
            next_value = torch.tensor(0.0, device=device)
            for t in reversed(range(T)):
                delta = rewards[t] + gamma * (next_value if t == T - 1 else values[t + 1]) - values[t]
                lastgaelam = delta + gamma * gae_lambda * lastgaelam
                adv[t] = lastgaelam
            returns = adv + values
            # Normalize advantages per-episode
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            logits, _ = net(states)
            dist = torch.distributions.Categorical(logits=logits)
            logps = dist.log_prob(actions)

            all_log_probs.append(logps)
            all_adv.append(adv)
            all_returns.append(returns)
            all_states.append(states)
            all_actions.append(actions)
            all_targets.append(targets)

        logps_cat = torch.cat(all_log_probs)
        adv_cat = torch.cat(all_adv)
        returns_cat = torch.cat(all_returns)
        states_cat = torch.cat(all_states)
        actions_cat = torch.cat(all_actions)
        targets_cat = torch.cat(all_targets)

        logits_cat, v_cat = net(states_cat)
        dist_cat = torch.distributions.Categorical(logits=logits_cat)
        entropy = dist_cat.entropy().mean()
        policy_loss = -(logps_cat * adv_cat).mean()
        value_loss = F.mse_loss(v_cat.squeeze(-1), returns_cat)
        # Disable online BC for interactive env (no supervised targets)
        if hasattr(env, "valid_action_ids"):
            ce_loss = torch.tensor(0.0, device=device)
        else:
            ce_loss = F.cross_entropy(logits_cat, targets_cat)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + bc_weight * ce_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    return net
