import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple, Deque
import os

# Define experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNDifficultyAdapter:
    def __init__(self, data_path: str, batch_size: int = 64, gamma: float = 0.95,  # Reduced gamma for shorter-term focus
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: float = 0.998,  # Higher end epsilon, slower decay
                 target_update: int = 20, hidden_size: int = 64, memory_size: int = 10000,  # Less frequent target updates
                 learning_rate: float = 1e-4):  # Added learning rate parameter
        """
        Initialize the DQN-based Difficulty Adaptation agent.
        
        Args:
            data_path: Path to the preprocessed KT data CSV file
            batch_size: Number of transitions sampled from the replay buffer
            gamma: Discount factor
            epsilon_start: Starting value of epsilon
            epsilon_end: Minimum value of epsilon
            epsilon_decay: Multiplicative factor of epsilon decay
            target_update: How often to update the target network
            hidden_size: Size of hidden layers in DQN
            memory_size: Size of the replay buffer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        
        # Data loading and preprocessing
        self.data = self._load_data(data_path)
        self.num_difficulty_levels = 5  # Number of difficulty levels
        
        # State size: [current_difficulty, performance_metric, pass_rate, time_spent_normalized]
        self.state_size = 4
        # Action space: 0 (decrease), 1 (maintain), 2 (increase)
        self.action_size = 3  # Now using 0-2 instead of -1 to 1
        
        # Networks
        self.policy_net = DQN(self.state_size, hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with lower learning rate for more conservative updates
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added L2 regularization
        self.memory = ReplayMemory(memory_size)
        
        # Training tracking
        self.steps_done = 0
        self.episode_rewards = []
        self.performance_history = []
        self.difficulty_history = []
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the data."""
        df = pd.read_csv(data_path)
        
        # Convert pass_status to binary (1 for Pass, 0 for Fail)
        df['pass_binary'] = df['pass_status'].apply(lambda x: 1 if x == 'Pass' else 0)
        
        # Normalize numerical features
        df['time_spent_normalized'] = (df['time_spent'] - df['time_spent'].min()) / \
                                    (df['time_spent'].max() - df['time_spent'].min() + 1e-8)
        
        return df
    
    def _get_state(self, current_difficulty: int, performance_metric: float, 
                  pass_rate: float, time_spent: float) -> torch.Tensor:
        """Convert state to tensor."""
        # Normalize values to [0, 1]
        state = torch.FloatTensor([
            current_difficulty / (self.num_difficulty_levels - 1),  # Normalize difficulty
            performance_metric,  # Already in [0, 1]
            pass_rate,  # Already in [0, 1]
            time_spent  # Already normalized
        ]).unsqueeze(0).to(self.device)
        return state
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy with action smoothing."""
        if training and random.random() < self.epsilon:
            # More likely to choose 'maintain' action (1) when exploring
            actions = [1] * 5 + [0, 2] * 2  # 0: decrease, 1: maintain, 2: increase
            return random.choice(actions)
            
        # Exploitation: choose best action from model
        with torch.no_grad():
            # Forward pass through the network
            q_values = self.policy_net(state)
            # Convert to action (0, 1, or 2)
            return q_values.argmax().item()
        
        with torch.no_grad():
            # Forward pass through the network
            q_values = self.policy_net(state)
            # Convert action index to -1, 0, or 1
            return q_values.argmax().item() - 1
    
    def _compute_td_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the TD loss for a batch of transitions."""
        if len(self.memory) < batch_size:
            return None
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert batch arrays to tensors
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, 
                                  device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, 
                                  device=self.device, dtype=torch.float32)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        return loss
    
    def update_model(self) -> float:
        """Perform a single optimization step."""
        loss = self._compute_td_loss(self.batch_size)
        if loss is None:
            return 0.0
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        return loss.item()
    
    def _get_reward(self, performance: float, difficulty_change: int, current_difficulty: int) -> float:
        """Calculate reward based on performance and difficulty change.
        
        Args:
            performance: Student's performance score (0-1)
            difficulty_change: Change in difficulty (-1, 0, or 1)
            current_difficulty: Current difficulty level (0-4)
            
        Returns:
            float: Reward value that encourages appropriate difficulty adjustments
        """
        # Target performance range (0.6-0.8 is considered optimal)
        optimal_min = 0.6
        optimal_max = 0.8
        
        # Base reward based on performance
        if performance < 0.3:
            # Very poor performance - should decrease difficulty
            reward = -1.0 if difficulty_change >= 0 else 0.5
        elif performance < optimal_min:
            # Below optimal - small reward for decreasing or maintaining
            reward = 0.5 if difficulty_change <= 0 else -0.5
        elif performance <= optimal_max:
            # In optimal range - reward for maintaining
            reward = 1.0 if difficulty_change == 0 else -0.2
        else:
            # Above optimal - reward for increasing difficulty
            reward = 0.8 if difficulty_change > 0 else -0.3
        
        # Additional penalties/constraints
        if current_difficulty == 0 and difficulty_change < 0:
            reward -= 0.5  # Penalize trying to decrease from minimum
        elif current_difficulty == 4 and difficulty_change > 0:
            reward -= 0.5  # Penalize trying to increase from maximum
            
        # Small penalty for changing difficulty to encourage stability
        if difficulty_change != 0:
            reward -= 0.1
            
        return reward
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """Train the DQN agent."""
        print("Starting DQN training...")
        
        for episode in range(num_episodes):
            # Get a random student's data
            student_id = random.choice(self.data['student_id'].unique())
            student_data = self.data[self.data['student_id'] == student_id].sort_values('order')
            
            # Initialize state
            current_difficulty = 2  # Start with medium difficulty
            episode_reward = 0
            
            for i in range(len(student_data) - 1):
                current_row = student_data.iloc[i]
                next_row = student_data.iloc[i + 1]
                
                # Get current state
                state = self._get_state(
                    current_difficulty,
                    current_row['score'],
                    current_row['pass_binary'],
                    current_row['time_spent_normalized']
                )
                
                # Select and perform an action
                action = self.select_action(state, training=True)
                difficulty_change = action - 1  # Convert 0,1,2 to -1,0,1
                
                # Ensure action is within valid range
                action = max(0, min(2, action))  # Keep actions in [0, 1, 2]
                
                # Apply action (ensure difficulty stays within bounds)
                new_difficulty = max(0, min(self.num_difficulty_levels - 1, current_difficulty + difficulty_change))
                
                # Get reward based on performance and difficulty change
                reward = self._get_reward(
                    performance=next_row['score'],
                    difficulty_change=difficulty_change,  # Pass the actual change (-1, 0, 1)
                    current_difficulty=current_difficulty
                )
                episode_reward += reward
                
                # Get next state
                next_state = self._get_state(
                    new_difficulty,
                    next_row['score'],
                    next_row['pass_binary'],
                    next_row['time_spent_normalized']
                )
                
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                
                # Move to the next state
                current_difficulty = new_difficulty
                
                # Perform one step of the optimization
                loss = self.update_model()
                
                # Track history for analysis
                self.performance_history.append(next_row['score'])
                self.difficulty_history.append(current_difficulty)
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon more slowly for continued exploration
            if len(self.memory) > self.batch_size:  # Only decay after we have enough samples
                self.epsilon = max(self.epsilon_end, 
                                 self.epsilon * self.epsilon_decay)
            
            # Save model checkpoint
            if (episode + 1) % save_interval == 0:
                self.save_model(f"dqn_checkpoint_ep{episode + 1}.pth")
            
            # Log progress
            self.episode_rewards.append(episode_reward)
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print("Training completed!")
        
        # Save final model
        self.save_model("dqn_final.pth")
        
        return self.episode_rewards
    
    def save_model(self, path: str):
        """Save the model weights."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'performance_history': self.performance_history,
            'difficulty_history': self.difficulty_history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.performance_history = checkpoint.get('performance_history', [])
        self.difficulty_history = checkpoint.get('difficulty_history', [])
        print(f"Model loaded from {path}")
    
    def plot_learning_curve(self, window_size: int = 100):
        """Plot the learning curve and difficulty adaptation."""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, alpha=0.3)
        plt.plot(pd.Series(self.episode_rewards).rolling(window_size).mean())
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot performance vs difficulty
        plt.subplot(2, 1, 2)
        smooth_perf = pd.Series(self.performance_history).rolling(window_size).mean()
        smooth_diff = pd.Series(self.difficulty_history).rolling(window_size).mean()
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        color = 'tab:blue'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Performance (smoothed)', color=color)
        ax1.plot(smooth_perf, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        
        color = 'tab:red'
        ax2.set_ylabel('Difficulty Level (smoothed)', color=color)
        ax2.plot(smooth_diff, color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Performance vs. Difficulty Over Time')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('dqn_learning_curve.png')
        plt.close()
        print("Learning curve saved as 'dqn_learning_curve.png'")

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize the DQN agent
    data_path = '../preprocessed_kt_data.csv'  # Adjust path as needed
    agent = DQNDifficultyAdapter(data_path)
    
    # Train the agent
    print("Starting training...")
    rewards = agent.train(num_episodes=500)
    
    # Save and plot results
    agent.plot_learning_curve()
    
    # Example of using the trained model
    print("\nExample difficulty adaptation:")
    current_difficulty = 2
    performance_metrics = [0.2, 0.5, 0.8, 0.9, 0.4]
    pass_rates = [0.3, 0.6, 0.8, 0.9, 0.5]
    time_spent = [0.5, 0.4, 0.3, 0.2, 0.4]  # Normalized values
    
    for perf, pass_rate, time_s in zip(performance_metrics, pass_rates, time_spent):
        state = agent._get_state(current_difficulty, perf, pass_rate, time_s)
        action = agent.select_action(state, training=False)
        next_difficulty = max(0, min(4, current_difficulty + action))
        
        print(f"Performance: {perf:.2f}, "
              f"Current Difficulty: {current_difficulty}, "
              f"Action: {action}, "
              f"Next Difficulty: {next_difficulty}")
        
        current_difficulty = next_difficulty

if __name__ == "__main__":
    main()
