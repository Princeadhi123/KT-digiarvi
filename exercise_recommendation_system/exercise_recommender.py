import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define a transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ExerciseEnvironment:
    def __init__(self, data_path):
        """Initialize the exercise recommendation environment."""
        self.data = pd.read_csv(data_path)
        self.students = self.data['student_id'].unique()
        self.exercises = self.data['exercise_id'].unique()
        self.categories = self.data['category'].unique()
        
        # Define state space components
        self.state_size = 8  # past_5_attempts (5) + time_spent + pass_rate + mean_perception
        
        # Define action space (all possible exercises)
        self.action_size = len(self.exercises)
        self.action_to_exercise = {i: ex for i, ex in enumerate(self.exercises)}
        self.exercise_to_action = {ex: i for i, ex in enumerate(self.exercises)}
        
        # Track current student and their history
        self.current_student = None
        self.student_history = {}
        self.reset()
    
    def reset(self, student_id=None):
        """Reset the environment for a new student or episode."""
        if student_id is None:
            self.current_student = np.random.choice(self.students)
        else:
            self.current_student = student_id
            
        # Initialize student history
        student_data = self.data[self.data['student_id'] == self.current_student]
        self.student_history = {
            'attempts': deque(maxlen=5),  # Track last 5 attempts (1 for pass, 0 for fail)
            'time_spent': 0.0,            # Average time spent on exercises
            'pass_rate': 0.0,              # Current pass rate
            'mean_perception': student_data['mean_perception'].iloc[0],  # Student's perception
            'exercises_done': set(),       # Track completed exercises
            'total_attempts': 0,           # Total attempts made
            'total_passes': 0              # Total passes
        }
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state representation with proper validation and normalization.
        Ensures all values are finite and within expected ranges.
        """
        try:
            # Initialize with default values
            state = np.zeros(8, dtype=np.float32)  # 5 attempts + time_spent + pass_rate + mean_perception
            
            # 1. Handle past attempts (last 5)
            past_attempts = list(self.student_history['attempts'])
            num_attempts = len(past_attempts)
            
            # Fill in the available attempts (up to 5)
            for i in range(min(5, num_attempts)):
                # Ensure the value is 0 or 1
                state[i] = 1.0 if past_attempts[-(i+1)] else 0.0
            
            # 2. Handle time spent (normalized to 0-1)
            time_spent = float(self.student_history.get('time_spent', 0.0))
            # Cap at 5 minutes (300 seconds) for normalization
            state[5] = min(max(0.0, time_spent / 300.0), 1.0)
            
            # 3. Handle pass rate (should be between 0 and 1)
            pass_rate = float(self.student_history.get('pass_rate', 0.0))
            state[6] = min(max(0.0, pass_rate), 1.0)
            
            # 4. Handle mean perception (normalized to 0-1, assuming max 4.0)
            mean_perception = float(self.student_history.get('mean_perception', 2.0))  # Default to neutral
            state[7] = min(max(0.0, mean_perception / 4.0), 1.0)
            
            # Final validation
            if not np.isfinite(state).all():
                print(f"Warning: Non-finite values in state: {state}")
                # Return a safe default state if there are any issues
                return np.zeros(8, dtype=np.float32)
                
            return state
            
        except Exception as e:
            print(f"Error in _get_state: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a safe default state if there are any issues
            return np.zeros(8, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action (recommend an exercise) and return the next state, reward, and done flag.
        
        Args:
            action: Index of the exercise to recommend
            
        Returns:
            next_state: Next state after taking the action
            reward: Reward for taking the action
            done: Whether the episode is done
            info: Additional information
        """
        exercise_id = self.action_to_exercise[action]
        
        # Get the exercise data
        exercise_data = self.data[
            (self.data['student_id'] == self.current_student) & 
            (self.data['exercise_id'] == exercise_id)
        ]
        
        if exercise_data.empty:
            # If exercise not found for student, use average values
            reward = -0.1  # Penalize for recommending non-existent exercises
            pass_status = 0
            time_spent = 30.0  # Default time spent
        else:
            # Use the first occurrence of the exercise for the student
            exercise_row = exercise_data.iloc[0]
            pass_status = 1 if exercise_row['pass_status'] == 'Pass' else 0
            time_spent = exercise_row['time_spent']
            
            # Calculate reward
            if pass_status == 1:
                reward = 1.0  # Positive reward for pass
            else:
                reward = 0.0   # No reward for fail
        
        # Update student history
        self.student_history['attempts'].append(pass_status)
        self.student_history['total_attempts'] += 1
        self.student_history['total_passes'] += pass_status
        self.student_history['pass_rate'] = self.student_history['total_passes'] / self.student_history['total_attempts']
        self.student_history['time_spent'] = (
            self.student_history['time_spent'] * (self.student_history['total_attempts'] - 1) + time_spent
        ) / self.student_history['total_attempts']
        
        # Check if done (all exercises completed or max attempts reached)
        self.student_history['exercises_done'].add(exercise_id)
        done = len(self.student_history['exercises_done']) >= len(self.exercises) or \
               self.student_history['total_attempts'] >= 100  # Max 100 attempts per episode
        
        next_state = self._get_state()
        info = {'exercise_id': exercise_id, 'pass_status': pass_status}
        
        # Store the transition in memory
        if hasattr(self, 'memory'):
            self.memory.append(
                Transition(self._get_state(), action, next_state, reward, done)
            )
        
        return next_state, reward, done, info


class DQNAgent:
    def _build_model(self):
        """Build the neural network model for the DQN with proper initialization."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        
        # Initialize weights using Kaiming initialization for ReLU
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
                
        return model
        
    def __init__(self, state_size, action_size):
        """Initialize the DQN agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_every = 10  # Update target network every 10 steps
        self.steps_done = 0
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize policy and target networks
        self.policy_net = self._build_model().to(self.device)
        self.target_net = self._build_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    
    def remember(self, state, action, next_state, reward, done):
        # Ensure all inputs are valid before storing
        try:
            state = np.array(state, dtype=np.float32)
            if next_state is not None:
                next_state = np.array(next_state, dtype=np.float32)
            
            # Verify values are finite
            if not (np.isfinite(state).all() and 
                   (next_state is None or np.isfinite(next_state).all()) and 
                   np.isfinite(reward)):
                print(f"Warning: Non-finite values detected in memory. State: {state}, Reward: {reward}")
                return
                
            self.memory.append((state, action, next_state, float(reward), bool(done)))
            
            # Keep memory size in check
            if len(self.memory) > 10000:
                self.memory.pop(0)
                
        except Exception as e:
            print(f"Error in remember: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        try:
            # Sample a batch of transitions
            batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            
            # Unpack the batch with validation
            states = []
            actions = []
            next_states = []
            rewards = []
            dones = []
            
            for t in batch:
                state, action, next_state, reward, done = t
                
                # Skip invalid transitions
                if not (isinstance(state, (np.ndarray, list)) and 
                       (next_state is None or isinstance(next_state, (np.ndarray, list))) and
                       isinstance(reward, (int, float)) and 
                       isinstance(done, (bool, np.bool_))):
                    continue
                    
                # Convert to numpy arrays if needed
                state = np.array(state, dtype=np.float32)
                if next_state is not None:
                    next_state = np.array(next_state, dtype=np.float32)
                
                # Skip if any values are not finite
                if not (np.isfinite(state).all() and 
                       (next_state is None or np.isfinite(next_state).all()) and 
                       np.isfinite(reward)):
                    continue
                
                states.append(state)
                actions.append(int(action))
                next_states.append(next_state if next_state is not None else np.zeros_like(state))
                rewards.append(float(reward))
                dones.append(float(done))
            
            if not states:  # Skip if no valid transitions
                return 0.0
            
            # Convert to tensors with gradient tracking
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Verify tensor shapes
            if states.dim() != 2 or states.size(1) != self.state_size:
                print(f"Invalid state shape: {states.shape}, expected (batch_size, {self.state_size})")
                return 0.0
                
            # Forward pass
            current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
            
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].detach()
                target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # Verify no NaN or infinite values
            if not (torch.isfinite(current_q_values).all() and torch.isfinite(target_q_values).all()):
                print("Warning: Non-finite values in Q-values")
                return 0.0
            
            # Compute loss with clipping
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='mean')
            
            # Skip update if loss is not finite
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss: {loss}")
                return 0.0
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            # Check for exploding gradients
            total_norm = 0.0
            for p in self.policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 1000:  # Very large gradient norm
                print(f"Warning: Large gradient norm: {total_norm}, skipping update")
                return 0.0
                
            self.optimizer.step()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if self.steps_done % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.steps_done += 1
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def load(self, path):
        """Load model weights."""
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
    
    def save(self, path):
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), path)


def train_agent(env, episodes=1000):
    """Train the DQN agent."""
    # Get state and action sizes from the environment
    env.reset()  # Initialize the environment
    state_size = len(env._get_state())  # Get the actual state size
    action_size = len(env.exercises)  # Number of possible exercises
    
    # Initialize the agent
    agent = DQNAgent(state_size, action_size)
    
    # Create a directory to save model checkpoints
    os.makedirs('models', exist_ok=True)
    
    # Track rewards and losses
    episode_rewards = []
    episode_losses = []
    
    for e in range(episodes):
        # Reset the environment for a new episode
        env.reset()
        total_reward = 0
        total_loss = 0
        done = False
        step = 0
        
        while not done:
            # Get current state
            state = env._get_state()
            
            # Get action from the agent
            action = agent.act(state)
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Store the experience in the replay memory
            agent.remember(state, action, next_state, reward, done)
            
            # Train the agent on a batch of past experiences
            loss = agent.replay()
            total_loss += loss if loss is not None else 0
            
            total_reward += reward
            step += 1
        
        # Track metrics
        avg_loss = total_loss / step if step > 0 else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        
        print(f"Episode: {e+1}/{episodes}, "
              f"Total Reward: {total_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.2f}, "
              f"Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint every 100 episodes
        if (e + 1) % 100 == 0:
            agent.save(f'models/dqn_agent_episode_{e+1}.pth')
    
    # Save the final model
    agent.save('models/dqn_agent_final.pth')
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_losses)
    plt.title('Episode Losses')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    return agent

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate the trained agent."""
    total_rewards = []
    total_passes = 0
    total_attempts = 0
    
    for e in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get available actions (indices of exercises not yet done)
            available_actions = [i for i in range(env.action_size) 
                               if env.action_to_exercise[i] not in env.student_history['exercises_done']]
            
            # If no available actions, break to avoid infinite loop
            if not available_actions:
                print("No available actions, ending episode early.")
                break
                
            # Choose action using the agent
            action = agent.act(state)
            
            # If the chosen action is not available, select a random available action
            if action not in available_actions:
                action = np.random.choice(available_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            # Track pass/fail if available in info
            if isinstance(info, dict) and 'pass_status' in info:
                total_passes += info['pass_status']
                total_attempts += 1
        
        total_rewards.append(episode_reward)
        print(f"Evaluation episode {e+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_pass_rate = total_passes / total_attempts if total_attempts > 0 else 0.0
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Pass Rate: {avg_pass_rate:.2f} ({total_passes}/{total_attempts})")
    
    return avg_reward, avg_pass_rate

def main():
    # Initialize environment
    data_path = r"c:\Users\pdaadh\Desktop\KT digiarvi\preprocessed_kt_data.csv"
    env = ExerciseEnvironment(data_path)
    
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    print(f"Number of students: {len(env.students)}")
    print(f"Number of exercises: {len(env.exercises)}")
    
    # Train the agent
    print("Training the agent...")
    agent = train_agent(env, episodes=500)
    
    # Evaluate the agent
    print("\nEvaluating the agent...")
    avg_reward, avg_pass_rate = evaluate_agent(env, agent)
    
    # Example of using the trained model to recommend exercises
    print("\nExample recommendation:")
    state = env.reset(student_id=1)  # Reset for student with ID 1
    
    # Get available actions (exercises not yet done)
    available_actions = [i for i in range(env.action_size) 
                        if env.action_to_exercise[i] not in env.student_history['exercises_done']]
    
    if not available_actions:
        print("No exercises available to recommend (all exercises completed).")
        return
    
    # Get Q-values for all actions
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor).squeeze().cpu().numpy()
    
    # Only consider available actions for recommendation
    available_q_values = q_values[available_actions]
    top_indices = np.argsort(available_q_values)[::-1][:3]  # Get indices of top 3 available actions
    top_actions = [available_actions[i] for i in top_indices]  # Map back to original action indices
    
    print("\nTop 3 recommended exercises:")
    for i, action in enumerate(top_actions):
        exercise_id = env.action_to_exercise[action]
        exercise_data = env.data[env.data['exercise_id'] == exercise_id].iloc[0]
        print(f"{i+1}. Exercise ID: {exercise_id}, "
              f"Category: {exercise_data['category']}, "
              f"Grade: {exercise_data['grade']}, "
              f"Q-value: {q_values[action]:.4f}")

if __name__ == "__main__":
    main()
