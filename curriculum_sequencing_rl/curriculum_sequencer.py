import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class CurriculumEnvironment:
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15):
        """
        Initialize the Curriculum Environment.
        
        Args:
            data_path (str): Path to the preprocessed data CSV file
            train_ratio (float): Ratio of data to use for training
            val_ratio (float): Ratio of data to use for validation
        """
        print("Initializing Curriculum Environment...")
        
        # Load and preprocess data
        print(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} rows")
        
        # Basic data validation
        print("\nData columns:", self.data.columns.tolist())
        print("\nFirst few rows of data:")
        print(self.data.head())
        
        # Preprocess data
        print("\nPreprocessing data...")
        self._preprocess_data()
        
        # Verify preprocessing
        print("\nAfter preprocessing - first few rows:")
        print(self.data[['pass_rate', 'category_encoded', 'order_norm', 'time_spent_norm', 'total_attempts_norm']].head())
        
        # Check for NaN values
        print("\nNaN values in key columns:")
        print(self.data[['pass_rate', 'category_encoded', 'order_norm', 'time_spent_norm', 'total_attempts_norm']].isna().sum())
        
        # Split data into train/val/test
        print("\nSplitting data...")
        self._split_data(train_ratio, val_ratio)
        
        # Set initial mode to training
        self.mode = 'train'
        self.current_student_data = None
        self.current_exercise_idx = 0
        
        # Define action and state spaces
        self.action_size = len(self.data['category_encoded'].unique())
        self.state_size = 5  # pass_rate, category_encoded, order_norm, time_spent_norm, total_attempts_norm
        
        print(f"\nEnvironment initialized with {self.action_size} actions and state size {self.state_size}")
        
    def _preprocess_data(self):
        """Preprocess the input data with robust handling of missing values."""
        print("\nPreprocessing data...")
        
        # Handle missing pass_status (assume 'Fail' if missing)
        self.data['pass_status'] = self.data['pass_status'].fillna('Fail')
        self.data['pass_status'] = (self.data['pass_status'] == 'Pass').astype(int)
        
        # Handle missing categories (fill with 'Unknown')
        self.data['category'] = self.data['category'].fillna('Unknown')
        self.data['category_encoded'] = self.data['category'].astype('category').cat.codes
        
        # Handle missing time_spent (fill with median)
        time_median = self.data['time_spent'].median()
        self.data['time_spent'] = self.data['time_spent'].fillna(time_median)
        
        # Handle missing total_attempts (fill with 1)
        self.data['total_attempts'] = self.data['total_attempts'].fillna(1)
        
        # Normalize numerical features with robust scaling
        def robust_scale(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:  # Avoid division by zero
                return (series - series.median()) / (series.max() - series.min() + 1e-10)
            return (series - series.median()) / iqr
            
        self.data['time_spent_norm'] = robust_scale(self.data['time_spent'])
        self.data['total_attempts_norm'] = robust_scale(self.data['total_attempts'])
        
        # Clip extreme values
        for col in ['time_spent_norm', 'total_attempts_norm']:
            self.data[col] = np.clip(self.data[col], -3, 3)
        
        # Normalize order (0-1 range)
        min_order = self.data['order'].min()
        max_order = self.data['order'].max()
        self.data['order_norm'] = (self.data['order'] - min_order) / (max_order - min_order + 1e-10)
        
        # Calculate pass rate per student
        self.data['pass_rate'] = self.data.groupby('student_id')['pass_status'].transform('mean')
        
        print("Data preprocessing complete")
        
    def _split_data(self, train_ratio, val_ratio):
        """Split data into train, validation, and test sets."""
        test_ratio = 1 - train_ratio - val_ratio
        train_size = int(len(self.data) * train_ratio)
        val_size = int(len(self.data) * val_ratio)
        test_size = len(self.data) - train_size - val_size
        
        self.train_data, self.val_data, self.test_data = np.split(self.data, [train_size, train_size + val_size])
        
    def reset(self, student_id=None, mode='train'):
        """
        Reset the environment for a new student.
        
        Args:
            student_id: ID of the student to use. If None, a random student is selected.
            mode: 'train', 'val', or 'test' to select the dataset
            
        Returns:
            Initial state of the environment
        """
        # Select dataset based on mode
        if mode == 'train':
            dataset = self.train_data
        elif mode == 'val':
            dataset = self.val_data
        elif mode == 'test':
            dataset = self.test_data
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")
        
        # Select a random student if none provided
        if student_id is None:
            self.current_student_id = np.random.choice(dataset['student_id'].unique())
        else:
            self.current_student_id = student_id
            
        # Get student data and sort by exercise order
        self.current_student_data = dataset[dataset['student_id'] == self.current_student_id]\
            .sort_values('order')
            
        self.current_exercise_idx = 0
        
        # Get initial state
        initial_state = self._get_state(0)
        return initial_state
    
    def _get_state(self, exercise_idx):
        """
        Get the state representation for the given exercise index with robust error handling.
        
        Args:
            exercise_idx: Index of the exercise in the current student's data
            
        Returns:
            np.ndarray: State vector or None if invalid index
        """
        try:
            if exercise_idx >= len(self.current_student_data) or exercise_idx < 0:
                return None
                
            exercise = self.current_student_data.iloc[exercise_idx]
            
            # Safely extract and validate each feature
            def safe_get(feature, default=0.0):
                try:
                    value = exercise.get(feature, default)
                    return float(value) if pd.notnull(value) else default
                except (ValueError, TypeError):
                    return default
            
            # Construct state vector with validation
            state = [
                safe_get('pass_rate', 0.5),           # Student's overall pass rate
                safe_get('category_encoded', 0),      # Current exercise category
                safe_get('order_norm', 0),            # Normalized exercise order
                np.tanh(safe_get('time_spent_norm', 0)),      # Tanh for bounded values
                np.tanh(safe_get('total_attempts_norm', 0))   # Tanh for bounded values
            ]
            
            # Convert to numpy array with float32 precision
            state_array = np.array(state, dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(state_array)) or np.any(np.isinf(state_array)):
                print(f"Warning: Invalid state values - replacing with zeros")
                state_array = np.zeros_like(state_array, dtype=np.float32)
                
            return state_array
            
        except Exception as e:
            print(f"Error in _get_state: {e}")
            # Return a zero state with correct dimensions
            return np.zeros(5, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The category to present next (0 to num_categories-1)
            
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: np.ndarray or None if episode done
                - reward: float, reward for the action
                - done: bool, whether the episode is complete
                - info: dict, additional information
        """
        try:
            # Validate action
            if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.action_size:
                print(f"Invalid action: {action}, using random action")
                action = np.random.randint(0, self.action_size)
            
            # Check if episode is done
            if self.current_exercise_idx >= len(self.current_student_data) - 1:
                return None, 0.0, True, {}
            
            # Get current and next exercise data
            current_exercise = self.current_student_data.iloc[self.current_exercise_idx].to_dict()
            next_exercise_idx = self.current_exercise_idx + 1
            next_exercise = self.current_student_data.iloc[next_exercise_idx].to_dict()
            
            # Calculate reward
            reward = self._calculate_reward(current_exercise, next_exercise, action)
            
            # Update state
            self.current_exercise_idx = next_exercise_idx
            
            # Check if episode is done
            done = (self.current_exercise_idx >= len(self.current_student_data) - 1)
            
            # Get next state if not done
            next_state = self._get_state(self.current_exercise_idx) if not done else None
            
            return next_state, float(reward), done, {}
            
        except Exception as e:
            print(f"Error in step: {e}")
            return None, 0.0, True, {}
    
    def _calculate_reward(self, current_exercise, next_exercise, action):
        """
        Calculate the reward for the given action with robust handling of edge cases.
        
        Args:
            current_exercise: Dictionary containing current exercise data
            next_exercise: Dictionary containing next exercise data
            action: The action taken (category to present next)
            
        Returns:
            float: Calculated reward
        """
        try:
            reward = 0.0
            
            # 1. Category selection reward
            target_category = next_exercise.get('category_encoded', -1)
            if action == target_category:
                reward += 0.5  # Reward for correct category prediction
            else:
                reward -= 0.1  # Small penalty for incorrect prediction
            
            # 2. Performance-based rewards
            next_pass = next_exercise.get('pass_status', 0)
            if next_pass == 1:  # If student passed the exercise
                reward += 0.5
                
                # Bonus for consecutive passes
                current_pass = current_exercise.get('pass_status', 0)
                if current_pass == 1:
                    reward += 0.2
            else:
                # Small penalty for failing, but not too harsh
                reward -= 0.1
            
            # 3. Efficiency rewards (normalized between -1 and 1)
            attempts_norm = next_exercise.get('total_attempts_norm', 0)
            time_norm = next_exercise.get('time_spent_norm', 0)
            
            # Scale efficiency rewards to be smaller than performance rewards
            reward += 0.1 * (1.0 - np.tanh(attempts_norm))  # Fewer attempts is better
            reward += 0.05 * (1.0 - np.tanh(time_norm))     # Less time is better
            
            # Clip final reward to reasonable range
            reward = np.clip(reward, -1.0, 1.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            return 0.0  # Neutral reward in case of errors


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)  # Smaller replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slightly faster decay
        self.learning_rate = 0.0001  # Much smaller learning rate
        self.batch_size = 64  # Smaller batch size
        self.target_update = 5  # Update target network more frequently
        self.gradient_clip = 1.0  # Gradient clipping value
        
        # Initialize Q-network and target network
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append(Transition(state, action, next_state, reward, done))
    
    def act(self, state, training=True):
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Return 0 loss if not enough samples
        
        try:
            # Sample a minibatch from memory
            transitions = random.sample(self.memory, self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Filter out None states (terminal states)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=self.device, dtype=torch.bool)
            
            # Handle empty next states
            if sum(non_final_mask) > 0:
                non_final_next_states = torch.FloatTensor(
                    np.array([s for s in batch.next_state if s is not None])
                ).to(self.device)
            
            # Convert to PyTorch tensors with gradient tracking
            state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
            action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).to(self.device)
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # Compute V(s_{t+1}) for all next states
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            if sum(non_final_mask) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            
            # Reshape for loss calculation
            expected_state_action_values = expected_state_action_values.unsqueeze(1)
            
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {e}")
            return 0.0
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update the target network with weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save the model weights."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load the model weights."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)


def train_agent(env, episodes=200, batch_size=64, save_path='dqn_curriculum.pth'):
    """
    Train the DQN agent.
    
    Args:
        env: The curriculum environment
        episodes (int): Number of episodes to train for
        batch_size (int): Batch size for training
        save_path (str): Path to save the trained model
    """
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent with smaller learning rate
    agent = DQNAgent(env.state_size, env.action_size, device=device)
    
    # Track metrics
    rewards = []
    val_rewards = []
    losses = []
    
    # For early stopping
    best_val_reward = -np.inf
    patience = 20
    patience_counter = 0
    
    for e in range(episodes):
        # Reset environment for a new student
        state = env.reset()
        episode_rewards = []
        episode_losses = []
        done = False
        
        while not done:
            # Select an action
            action = agent.act(state)
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Store the experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent on a batch of past experiences
            loss = agent.replay()
            if loss > 0:  # Only append if training happened
                episode_losses.append(loss)
            
            # Update state and track rewards
            state = next_state if not done else None
            episode_rewards.append(reward)
        
        # Calculate average reward and loss for the episode
        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Update the target network periodically
        if e % agent.target_update == 0:
            agent.update_target_network()
        
        # Evaluate on validation set
        val_reward = evaluate_agent(agent, env, n_evals=3, mode='val')
        
        # Track metrics
        rewards.append(avg_episode_reward)
        val_rewards.append(val_reward)
        losses.append(avg_episode_loss)
        
        # Early stopping check
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.save(save_path)
            patience_counter = 0
            print(f"New best model saved with validation reward: {best_val_reward:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at episode {e+1} - no improvement for {patience} episodes")
                break
        
        # Print progress
        print(f"Episode: {e+1}/{episodes}, "
              f"Avg Reward: {avg_episode_reward:.4f}, "
              f"Val Reward: {val_reward:.4f}, "
              f"Avg Loss: {avg_episode_loss:.6f}, "
              f"Epsilon: {agent.epsilon:.4f}")
    
    # Load the best model
    agent.load(save_path)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Training Reward')
    plt.plot(val_rewards, label='Validation Reward')
    plt.title('Training and Validation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    return agent


def evaluate_agent(agent, env, n_evals=5, mode='test'):
    """
    Evaluate the agent's performance.
    
    Args:
        agent: The DQN agent
        env: The curriculum environment
        n_evals (int): Number of evaluation episodes
        mode: 'train', 'val', or 'test' to select the dataset
        
    Returns:
        Average reward over evaluation episodes
    """
    rewards = []
    
    for _ in range(n_evals):
        state = env.reset(mode=mode)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:  # Prevent infinite loops
            action = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state if not done else None
            steps += 1
        
        rewards.append(episode_reward)
    
    return np.mean(rewards) if rewards else 0.0


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Initialize environment
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessed_kt_data.csv')
    env = CurriculumEnvironment(data_path)
    
    # Train the agent
    print("Starting training...")
    agent = train_agent(env, episodes=500, save_path='dqn_curriculum.pth')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_reward = evaluate_agent(agent, env, n_evals=20, mode='test')
    print(f"Average test reward: {test_reward:.2f}")
    
    # Example of using the trained agent to recommend exercises
    print("\nExample exercise sequence for a new student:")
    state = env.reset(mode='test')
    done = False
    step = 0
    
    # Get category mapping from the data
    category_mapping = dict(enumerate(env.data['category'].astype('category').cat.categories))
    
    while not done and step < 20:  # Limit to 20 steps for the example
        action = agent.act(state, training=False)
        
        # Get the category name from the mapping
        category_name = category_mapping.get(action, f"Unknown({action})")
        print(f"Step {step + 1}: Recommended category = {category_name} (ID: {action})")
        
        # Simulate student performance (in practice, this would come from real student data)
        next_state, reward, done, _ = env.step(action)
        print(f"  Reward: {reward:.2f}, Done: {done}")
        
        state = next_state if not done else None
        step += 1
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the main function
    main()
