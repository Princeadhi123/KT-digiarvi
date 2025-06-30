import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import pickle

# Experience replay buffer
Transition = namedtuple('Transition', 
                       ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, *args):
        """Save a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """Q-network for the DQN agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SelfAdaptiveAgent:
    """Self-Adapting RL agent for adaptive curriculum learning."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config: 'RLConfig',
                 curriculum_config: 'CurriculumConfig',
                 self_adaptive_config: 'SelfAdaptiveConfig'):
        """Initialize the agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.curriculum_config = curriculum_config
        self.self_adaptive_config = self_adaptive_config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=config.learning_rate
        )
        
        # Experience replay
        self.memory = ReplayBuffer(config.buffer_size)
        
        # Exploration
        self.epsilon = config.exploration_initial_eps
        self.epsilon_final = config.exploration_final_eps
        self.epsilon_decay = 1.0 / (config.total_timesteps * config.exploration_fraction)
        
        # Training tracking
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Curriculum adaptation
        self.current_difficulty = curriculum_config.initial_difficulty
        self.success_history = deque(maxlen=curriculum_config.window_size)
        
        # Meta-learning components
        self.meta_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self_adaptive_config.meta_learning_rate
        )
        self.meta_buffer = ReplayBuffer(self_adaptive_config.experience_replay_size)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action using epsilon-greedy policy."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(
            self.epsilon_final,
            self.epsilon - self.epsilon_decay
        )
    
    def update_curriculum(self, success: bool):
        """Update the curriculum difficulty based on recent performance."""
        self.success_history.append(1 if success else 0)
        
        if len(self.success_history) < self.curriculum_config.window_size // 2:
            return  # Not enough data to make decisions
        
        success_rate = np.mean(self.success_history)
        
        # Adjust difficulty based on success rate
        if success_rate > self.curriculum_config.success_threshold:
            # Increase difficulty
            self.current_difficulty = min(
                self.curriculum_config.max_difficulty,
                self.current_difficulty + self.curriculum_config.difficulty_step
            )
            # Reset history after adjustment
            self.success_history = deque(maxlen=self.curriculum_config.window_size)
            
        elif success_rate < self.curriculum_config.failure_threshold:
            # Decrease difficulty
            self.current_difficulty = max(
                self.curriculum_config.min_difficulty,
                self.current_difficulty - self.curriculum_config.difficulty_step
            )
            # Reset history after adjustment
            self.success_history = deque(maxlen=self.curriculum_config.window_size)
    
    def adapt(self, experiences: List[Transition]):
        """Adapt the policy using meta-learning on recent experiences."""
        if len(experiences) < self.self_adaptive_config.meta_batch_size:
            return  # Not enough experiences for adaptation
        
        # Sample a batch of experiences
        batch = random.sample(experiences, self.self_adaptive_config.meta_batch_size)
        
        # Perform a few steps of gradient descent
        for _ in range(self.self_adaptive_config.adaptation_steps):
            # Compute loss on the batch
            loss = self._compute_loss(batch)
            
            # Update parameters
            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()
    
    def _compute_loss(self, batch: List[Transition]) -> torch.Tensor:
        """Compute the loss for a batch of transitions."""
        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        return loss
    
    def update(self):
        """Update the agent's parameters using experience replay."""
        if len(self.memory) < self.config.batch_size:
            return  # Not enough samples
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Compute loss and update
        loss = self._compute_loss(transitions)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save the agent's parameters."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'current_difficulty': self.current_difficulty,
            'success_history': list(self.success_history)
        }, path)
    
    def load(self, path: str):
        """Load the agent's parameters."""
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            self.current_difficulty = checkpoint['current_difficulty']
            self.success_history = deque(
                checkpoint['success_history'],
                maxlen=self.curriculum_config.window_size
            )
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"No checkpoint found at {path}")
