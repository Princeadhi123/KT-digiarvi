"""Refactored policy gradient methods (A2C/A3C/PPO) using new architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Optional, List, Tuple
import numpy as np

from ..core.base import BaseAgent, BaseTrainer, BaseNetwork, PolicyFunction
from ..core.config import A2CConfig, A3CConfig, PPOConfig
from ..core.utils import setup_device, compute_gae, normalize_advantages
from ..core.factory import register_trainer


class ActorCriticNetwork(BaseNetwork):
    """Shared actor-critic network for policy gradient methods."""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Policy head
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Value head
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and values."""
        features = self.shared(x)
        logits = self.actor(features)
        values = self.critic(features)
        return logits, values


class PolicyGradientAgent(BaseAgent):
    """Base agent for policy gradient methods."""
    
    def __init__(self, state_dim: int, action_dim: int, config, hidden_dim: int = 128):
        super().__init__(state_dim, action_dim, setup_device(config.device))
        self.config = config
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim
        ).to_device(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Training state
        self.steps_done = 0
    
    def act(self, state: np.ndarray, training: bool = True, 
            valid_ids: Optional[Any] = None) -> int:
        """Select action using policy network."""
        # Ensure dropout/batchnorm behave correctly: eval for inference, train for sampling
        prev_mode = self.network.training
        if training:
            self.network.train(True)
        else:
            self.network.train(False)  # equivalent to eval()

        try:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                logits, _ = self.network(state_tensor)

                # Sanitize raw logits first
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = torch.clamp(logits, min=-20, max=20)

                # Apply action masking AFTER sanitization so invalid remain -inf
                if valid_ids is not None and len(valid_ids) > 0:
                    mask = torch.full((self.action_dim,), float('-inf'), device=self.device)
                    mask[torch.tensor(list(valid_ids), device=self.device)] = 0.0
                    logits = logits + mask

                # Fallback: if all -inf after masking, use zeros to create a uniform policy
                if torch.isneginf(logits).all():
                    logits = torch.zeros_like(logits)
                
                if training:
                    # Sample from policy
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    # Greedy action
                    action = logits.argmax(dim=-1)
                
                action_id = action.item()
        finally:
            # Restore original train/eval mode
            self.network.train(prev_mode)

        return action_id
    
    def get_policy(self, env: Optional[Any] = None) -> PolicyFunction:
        """Return policy function for evaluation."""
        def policy(state: np.ndarray, cur_cat: int) -> int:
            valid_ids = None
            if env is not None and hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
            return self.act(state, training=False, valid_ids=valid_ids)
        return policy
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
    
    def update(self, *args, **kwargs) -> dict:
        """Base update method - to be implemented by specific trainers."""
        raise NotImplementedError("Update method should be implemented by specific trainers")


def behavior_cloning_pretrain(network: ActorCriticNetwork, env: Any, 
                             epochs: int, lr: float, device: torch.device) -> None:
    """Behavior cloning pretraining on dataset transitions."""
    if epochs <= 0:
        return
    
    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.train()
    
    for _ in range(epochs):
        batch_states, batch_targets = [], []
        
        for student_id in env.splits.train_students:
            student_df = env.df[env.df["student_id"] == student_id].sort_values("order")
            
            for i in range(len(student_df) - 1):
                current_row = student_df.iloc[i]
                next_row = student_df.iloc[i + 1]
                
                state = env._build_state_from_row(current_row)
                target = int(next_row["category_id"])
                
                batch_states.append(torch.from_numpy(state).float().to(device))
                batch_targets.append(target)
                
                # Process in mini-batches to manage memory
                if len(batch_states) >= 512:
                    states_tensor = torch.stack(batch_states)
                    targets_tensor = torch.tensor(batch_targets, dtype=torch.long, device=device)
                    
                    logits, _ = network(states_tensor)
                    loss = F.cross_entropy(logits, targets_tensor)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    
                    batch_states, batch_targets = [], []
        
        # Process remaining batch
        if batch_states:
            states_tensor = torch.stack(batch_states)
            targets_tensor = torch.tensor(batch_targets, dtype=torch.long, device=device)
            
            logits, _ = network(states_tensor)
            loss = F.cross_entropy(logits, targets_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()


@register_trainer('a2c')
class A2CTrainer(BaseTrainer):
    """A2C trainer with improved batch processing."""
    
    def __init__(self, config: A2CConfig):
        super().__init__(config)
        self.config = config
    
    def create_agent(self, env: Any) -> PolicyGradientAgent:
        """Create A2C agent."""
        agent = PolicyGradientAgent(env.state_dim, env.action_size, self.config)
        
        # Behavior cloning pretraining
        if self.config.bc_warmup_epochs > 0:
            behavior_cloning_pretrain(
                agent.network, env, self.config.bc_warmup_epochs, 
                self.config.lr, agent.device
            )
        
        return agent
    
    def train_step(self, env: Any, agent: PolicyGradientAgent) -> dict:
        """Execute A2C training step with batch of episodes."""
        batch_data = self._collect_batch(env, agent)
        loss_info = self._update_agent(agent, batch_data, env)
        return loss_info
    
    def _collect_batch(self, env: Any, agent: PolicyGradientAgent) -> dict:
        """Collect batch of episodes."""
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_values, batch_targets = [], []
        
        # Use rollouts for A3C, batch_episodes for A2C
        num_episodes = (
            getattr(self.config, 'rollouts_per_update', None)
            or getattr(self.config, 'rollouts', None)
            or getattr(self.config, 'batch_episodes', 4)
        )
        for _ in range(num_episodes):
            states, actions, rewards, values, targets = self._collect_episode(env, agent)
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_values.append(values)
            batch_targets.append(targets)
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'values': batch_values,
            'targets': batch_targets
        }
    
    def _collect_episode(self, env: Any, agent: PolicyGradientAgent) -> Tuple:
        """Collect single episode."""
        state = env.reset("train")
        done = False
        
        states, actions, rewards, values, targets = [], [], [], [], []
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
            logits, value = agent.network(state_tensor)

            # Sanitize raw logits before masking
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = torch.clamp(logits, min=-20, max=20)

            # Apply action masking after sanitization
            masked_logits = logits
            if hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
                if len(valid_ids) > 0:
                    mask = torch.full((env.action_size,), float('-inf'), device=agent.device)
                    mask[torch.tensor(valid_ids, device=agent.device)] = 0.0
                    masked_logits = logits + mask

            # Fallback if all actions masked
            if torch.isneginf(masked_logits).all():
                masked_logits = torch.zeros_like(masked_logits)

            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample().item()
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state_tensor.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            values.append(value.squeeze(0))
            targets.append(int(info.get("target", 0)))
            
            state = next_state if not done else state
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long, device=agent.device),
            torch.tensor(rewards, dtype=torch.float32, device=agent.device),
            torch.stack(values),
            torch.tensor(targets, dtype=torch.long, device=agent.device)
        )
    
    def _update_agent(self, agent: PolicyGradientAgent, batch_data: dict, env: Any) -> dict:
        """Update agent using collected batch."""
        all_states, all_actions, all_advantages = [], [], []
        all_returns, all_targets = [], []
        
        # Process each episode in batch
        for states, actions, rewards, values, targets in zip(
            batch_data['states'], batch_data['actions'], batch_data['rewards'],
            batch_data['values'], batch_data['targets']
        ):
            # Compute returns
            returns = []
            R = 0.0
            for reward in reversed(rewards):
                R = reward + self.config.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=agent.device)
            
            # Compute advantages
            advantages = returns - values.detach().squeeze(-1)
            advantages = normalize_advantages(advantages)
            
            all_states.append(states)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_targets.append(targets)
        
        # Concatenate all data
        states_cat = torch.cat(all_states)
        actions_cat = torch.cat(all_actions)
        advantages_cat = torch.cat(all_advantages)
        returns_cat = torch.cat(all_returns)
        targets_cat = torch.cat(all_targets)
        
        # Forward pass
        logits, values = agent.network(states_cat)
        # Sanitize logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, min=-20, max=20)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Compute losses
        log_probs = dist.log_prob(actions_cat)
        policy_loss = -(log_probs * advantages_cat).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns_cat)
        entropy_loss = -dist.entropy().mean()
        
        # Behavior cloning loss (disabled for interactive env)
        bc_loss = torch.tensor(0.0, device=agent.device)
        if not hasattr(env, 'valid_action_ids'):
            bc_loss = F.cross_entropy(logits, targets_cat)
        
        total_loss = (policy_loss + 
                     self.config.value_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss +
                     self.config.bc_weight * bc_loss)
        
        # Update
        agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)
        agent.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'bc_loss': bc_loss.item(),
            'total_loss': total_loss.item()
        }


@register_trainer('a3c')
class A3CTrainer(A2CTrainer):
    """A3C trainer with GAE advantages."""
    
    def __init__(self, config: A3CConfig):
        super().__init__(config)
        self.config = config
    
    def _update_agent(self, agent: PolicyGradientAgent, batch_data: dict, env: Any) -> dict:
        """Update agent using GAE advantages."""
        all_states, all_actions, all_advantages = [], [], []
        all_returns, all_targets = [], []
        
        # Process each episode with GAE
        for states, actions, rewards, values, targets in zip(
            batch_data['states'], batch_data['actions'], batch_data['rewards'],
            batch_data['values'], batch_data['targets']
        ):
            values_np = values.detach().squeeze(-1)
            
            # Compute GAE advantages
            advantages = compute_gae(
                rewards, values_np, self.config.gamma, self.config.gae_lambda
            )
            returns = advantages + values_np
            advantages = normalize_advantages(advantages)
            
            all_states.append(states)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_targets.append(targets)
        
        # Rest is same as A2C
        states_cat = torch.cat(all_states)
        actions_cat = torch.cat(all_actions)
        advantages_cat = torch.cat(all_advantages)
        returns_cat = torch.cat(all_returns)
        targets_cat = torch.cat(all_targets)
        
        logits, values = agent.network(states_cat)
        dist = torch.distributions.Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions_cat)
        policy_loss = -(log_probs * advantages_cat).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns_cat)
        entropy_loss = -dist.entropy().mean()
        
        bc_loss = torch.tensor(0.0, device=agent.device)
        if not hasattr(env, 'valid_action_ids'):
            bc_loss = F.cross_entropy(logits, targets_cat)
        
        total_loss = (policy_loss + 
                     self.config.value_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss +
                     self.config.bc_weight * bc_loss)
        
        agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)
        agent.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'bc_loss': bc_loss.item(),
            'total_loss': total_loss.item()
        }


@register_trainer('ppo')
class PPOTrainer(A2CTrainer):
    """PPO trainer with clipped objective."""
    
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.config = config
    
    def _collect_batch(self, env: Any, agent: PolicyGradientAgent) -> dict:
        """Collect batch with per-step masks and old log probabilities (masked)."""
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_values, batch_targets, batch_masks = [], [], []
        batch_old_log_probs = []

        # Use same batch sizing logic as A2C/A3C
        num_episodes = (
            getattr(self.config, 'rollouts_per_update', None)
            or getattr(self.config, 'rollouts', None)
            or getattr(self.config, 'batch_episodes', 4)
        )

        for _ in range(num_episodes):
            state = env.reset("train")
            done = False
            states, actions, rewards, values, targets, masks = [], [], [], [], [], []

            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                logits, value = agent.network(state_tensor)

                # Sanitize raw logits first
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = torch.clamp(logits, min=-20, max=20)

                # Build mask from current valid actions (if any) and apply AFTER sanitization
                valid_ids = env.valid_action_ids() if hasattr(env, 'valid_action_ids') else None
                if valid_ids is not None and len(valid_ids) > 0:
                    mask_vec = torch.full((env.action_size,), float('-inf'), device=agent.device)
                    mask_vec[torch.tensor(valid_ids, device=agent.device)] = 0.0
                else:
                    mask_vec = torch.zeros((env.action_size,), device=agent.device)
                masked_logits = logits + mask_vec

                # Fallback if all actions masked
                if torch.isneginf(masked_logits).all():
                    masked_logits = torch.zeros_like(masked_logits)

                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample().item()

                next_state, reward, done, info = env.step(action)

                states.append(state_tensor.squeeze(0))
                actions.append(action)
                rewards.append(reward)
                values.append(value.squeeze(0))
                targets.append(int(info.get("target", 0)))
                masks.append(mask_vec)

                state = next_state if not done else state

            # Tensorize episode
            states_t = torch.stack(states)
            actions_t = torch.tensor(actions, dtype=torch.long, device=agent.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
            values_t = torch.stack(values)
            targets_t = torch.tensor(targets, dtype=torch.long, device=agent.device)
            masks_t = torch.stack(masks)  # [T, A]

            # Old log probs under masked policy
            with torch.no_grad():
                logits_ep, _ = agent.network(states_t)
                if torch.isnan(logits_ep).any() or torch.isinf(logits_ep).any():
                    logits_ep = torch.nan_to_num(logits_ep, nan=0.0, posinf=20.0, neginf=-20.0)
                logits_ep = torch.clamp(logits_ep, min=-20, max=20)
                logits_ep = logits_ep + masks_t
                all_neg_inf = torch.isneginf(logits_ep).all(dim=1)
                if all_neg_inf.any():
                    logits_ep[all_neg_inf] = 0.0
                dist_ep = torch.distributions.Categorical(logits=logits_ep)
                old_log_probs = dist_ep.log_prob(actions_t).detach()

            batch_states.append(states_t)
            batch_actions.append(actions_t)
            batch_rewards.append(rewards_t)
            batch_values.append(values_t)
            batch_targets.append(targets_t)
            batch_masks.append(masks_t)
            batch_old_log_probs.append(old_log_probs)

        return {
            'states': batch_states,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'values': batch_values,
            'targets': batch_targets,
            'old_log_probs': batch_old_log_probs,
            'masks': batch_masks,
        }
    
    def _update_agent(self, agent: PolicyGradientAgent, batch_data: dict, env: Any) -> dict:
        """Update agent using PPO clipped objective."""
        # Prepare data with GAE
        all_states, all_actions, all_advantages = [], [], []
        all_returns, all_old_log_probs, all_targets = [], [], []
        all_masks = []
        
        for states, actions, rewards, values, targets, old_log_probs, masks in zip(
            batch_data['states'], batch_data['actions'], batch_data['rewards'],
            batch_data['values'], batch_data['targets'], batch_data['old_log_probs'],
            batch_data['masks']
        ):
            values_np = values.detach().squeeze(-1)
            
            # GAE computation
            advantages = compute_gae(
                rewards, values_np, self.config.gamma, self.config.gae_lambda
            )
            returns = advantages + values_np
            advantages = normalize_advantages(advantages)
            
            all_states.append(states)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_old_log_probs.append(old_log_probs)
            all_targets.append(targets)
            all_masks.append(masks)
        
        # Concatenate
        states_cat = torch.cat(all_states)
        actions_cat = torch.cat(all_actions)
        advantages_cat = torch.cat(all_advantages)
        returns_cat = torch.cat(all_returns)
        old_log_probs_cat = torch.cat(all_old_log_probs)
        targets_cat = torch.cat(all_targets)
        masks_cat = torch.cat(all_masks)  # [N, action_dim]
        
        # PPO updates
        total_loss_info = {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'bc_loss': 0}
        
        for _ in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states_cat), device=agent.device)
            
            for start in range(0, len(states_cat), self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, len(states_cat))
                mb_indices = indices[start:end]
                
                mb_states = states_cat[mb_indices]
                mb_actions = actions_cat[mb_indices]
                mb_advantages = advantages_cat[mb_indices]
                mb_returns = returns_cat[mb_indices]
                mb_old_log_probs = old_log_probs_cat[mb_indices]
                mb_targets = targets_cat[mb_indices]
                mb_masks = masks_cat[mb_indices]
                
                # Forward pass
                logits, values = agent.network(mb_states)
                # Sanitize and apply masks
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = torch.clamp(logits, min=-20, max=20)
                logits = logits + mb_masks
                all_neg_inf = torch.isneginf(logits).all(dim=1)
                if all_neg_inf.any():
                    logits[all_neg_inf] = 0.0
                dist = torch.distributions.Categorical(logits=logits)
                
                # PPO loss
                log_probs = dist.log_prob(mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                entropy_loss = -dist.entropy().mean()
                
                bc_loss = torch.tensor(0.0, device=agent.device)
                if not hasattr(env, 'valid_action_ids'):
                    bc_loss = F.cross_entropy(logits, mb_targets)
                
                total_loss = (policy_loss + 
                             self.config.value_coef * value_loss + 
                             self.config.entropy_coef * entropy_loss +
                             self.config.bc_weight * bc_loss)
                
                # Update
                agent.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)
                agent.optimizer.step()
                
                # Accumulate losses
                total_loss_info['policy_loss'] += policy_loss.item()
                total_loss_info['value_loss'] += value_loss.item()
                total_loss_info['entropy_loss'] += entropy_loss.item()
                total_loss_info['bc_loss'] += bc_loss.item()
        
        # Average losses
        num_updates = self.config.ppo_epochs * ((len(states_cat) + self.config.minibatch_size - 1) // self.config.minibatch_size)
        for key in total_loss_info:
            total_loss_info[key] /= num_updates
        
        return total_loss_info
