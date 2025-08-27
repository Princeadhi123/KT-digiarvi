"""Refactored DQN implementation using new architecture."""

import random
import copy
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core.base import BaseAgent, BaseTrainer, BaseNetwork, ReplayBuffer, PolicyFunction
from ..core.config import DQNConfig
from ..core.utils import setup_device, EpsilonScheduler
from ..core.factory import register_trainer

try:
    from ..evaluation import eval_policy_avg_score
except ImportError:
    from evaluation import eval_policy_avg_score


class DuelingQNetwork(BaseNetwork):
    """Dueling DQN network with improved architecture."""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Dueling streams
        self.advantage = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.feature(x)
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Dueling aggregation
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent(BaseAgent):
    """Improved DQN agent with better memory management."""
    
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        super().__init__(state_dim, action_dim, setup_device(config.device))
        self.config = config
        
        # Networks
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, config.hidden_dim
        ).to_device(self.device)
        
        self.target_network = DuelingQNetwork(
            state_dim, action_dim, config.hidden_dim
        ).to_device(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.freeze()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, state_dim, str(self.device)
        )
        
        # Exploration
        self.epsilon_scheduler = EpsilonScheduler(
            config.eps_start, config.eps_end, config.eps_decay_steps
        )
        
        # Training state
        self.steps_done = 0
        self.update_count = 0
    
    def act(self, state: np.ndarray, training: bool = True, 
            valid_ids: Optional[Any] = None) -> int:
        """Select action using epsilon-greedy with optional masking."""
        if training and random.random() < self.epsilon_scheduler.get_epsilon():
            # Random exploration
            if valid_ids is not None and len(valid_ids) > 0:
                return random.choice(list(valid_ids))
            return random.randrange(self.action_dim)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Apply action masking if provided
            if valid_ids is not None and len(valid_ids) > 0:
                mask = torch.full((self.action_dim,), float('-inf'), device=self.device)
                mask[torch.tensor(list(valid_ids), device=self.device)] = 0.0
                q_values = q_values + mask
            
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: Optional[np.ndarray], done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """Perform one update step and return loss."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using Double DQN
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            next_q[dones.unsqueeze(1)] = 0.0
            target_q = rewards.unsqueeze(1) + self.config.gamma * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update exploration
        self.epsilon_scheduler.step()
        self.steps_done += 1
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.config.target_update_interval == 0:
            self._soft_update_target()
        
        return loss.item()
    
    def _soft_update_target(self) -> None:
        """Soft update of target network."""
        tau = self.config.target_tau
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
    
    def hard_update_target(self) -> None:
        """Hard update of target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_policy(self, env: Optional[Any] = None) -> PolicyFunction:
        """Return greedy policy function."""
        def policy(state: np.ndarray, cur_cat: int) -> int:
            valid_ids = None
            if env is not None and hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
            return self.act(state, training=False, valid_ids=valid_ids)
        return policy
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'update_count': self.update_count,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.update_count = checkpoint['update_count']


@register_trainer('dqn')
class DQNTrainer(BaseTrainer):
    """DQN trainer with improved training loop."""
    
    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self.config = config
        self.best_agent_state = None
        self.best_score = float('-inf')
    
    def create_agent(self, env: Any) -> DQNAgent:
        """Create DQN agent."""
        return DQNAgent(env.state_dim, env.action_size, self.config)
    
    def train_step(self, env: Any, agent: DQNAgent) -> dict:
        """Execute one training episode."""
        state = env.reset("train")
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        
        while not done and steps < 1000:  # Max steps per episode
            # Get valid actions if available
            valid_ids = None
            if hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
            
            # Select action
            action = agent.act(state, training=True, valid_ids=valid_ids)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.remember(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            
            episode_reward += reward
            episode_loss += loss
            state = next_state if not done else state
            steps += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_loss': episode_loss / max(steps, 1),
            'episode_steps': steps,
            'epsilon': agent.epsilon_scheduler.get_epsilon()
        }
    
    def _evaluate_agent(self, env: Any, episode: int) -> None:
        """Evaluate agent and save best model."""
        if not self.config.select_best_on_val:
            return
        
        policy = self.agent.get_policy(env)
        val_score = eval_policy_avg_score(
            env, policy, mode="val", episodes=self.config.val_episodes
        )
        
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_agent_state = copy.deepcopy(self.agent.q_network.state_dict())
    
    def train(self, env: Any) -> DQNAgent:
        """Main training loop with improvements."""
        agent = self.create_agent(env)
        
        for episode in range(self.config.episodes):
            metrics = self.train_step(env, agent)
            self._update_metrics(metrics, episode)
            
            # Hard target update periodically
            if (episode + 1) % 20 == 0:
                agent.hard_update_target()
            
            # Evaluation
            if self._should_evaluate(episode):
                self._evaluate_agent(env, episode)
        
        # Load best model if validation was used
        if self.config.select_best_on_val and self.best_agent_state is not None:
            agent.q_network.load_state_dict(self.best_agent_state)
            agent.hard_update_target()
        
        self.agent = agent
        return agent
