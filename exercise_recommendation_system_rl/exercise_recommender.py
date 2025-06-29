# Standard library imports
import os
import random
import sys
from collections import deque, namedtuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Deque, Any, Union
import numpy.typing as npt
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define a transition in our environment

# Define a transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ExerciseEnvironment:
    """Environment for exercise recommendation using reinforcement learning."""
    
    def __init__(self, data_path: str) -> None:
        """Initialize the exercise recommendation environment.
        
        Args:
            data_path: Path to the CSV file containing exercise data
        """
        # Load and preprocess data
        self.data = pd.read_csv(data_path)
        self.students = self.data['student_id'].unique()
        self.exercises = self.data['exercise_id'].unique()
        
        # State space configuration
        self.state_size = 8  # past_5_attempts (5) + time_spent + pass_rate + mean_perception
        
        # Action space configuration
        self.action_size = len(self.exercises)
        self.action_to_exercise = {i: ex for i, ex in enumerate(self.exercises)}
        self.exercise_to_action = {ex: i for i, ex in enumerate(self.exercises)}
        
        # Initialize tracking variables
        self.current_student: Optional[int] = None
        self.student_history: Dict[str, Any] = {}
        
        # Reset to initialize the environment
        self.reset()
    
    def reset(self, student_id: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset the environment for a new student or episode.
        
        Args:
            student_id: Optional student ID to reset to. If None, selects a random student.
            
        Returns:
            The initial state vector for the new episode.
            
        Raises:
            ValueError: If the specified student_id is not found in the data.
        """
        try:
            # Select student
            if student_id is None:
                self.current_student = np.random.choice(self.students)
            else:
                if student_id not in self.students:
                    raise ValueError(f"Student ID {student_id} not found in dataset")
                self.current_student = student_id
            
            # Get student data
            student_data = self.data[self.data['student_id'] == self.current_student]
            if student_data.empty:
                raise ValueError(f"No data available for student {self.current_student}")
            
            # Initialize student history with type hints
            self.student_history = {
                'attempts': deque(maxlen=5),  # type: Deque[int]  # 1 for pass, 0 for fail
                'time_spent': 0.0,            # float: Average time spent on exercises
                'pass_rate': 0.0,              # float: Current pass rate
                'mean_perception': float(student_data['mean_perception'].iloc[0]),  # float: Student's perception
                'exercises_done': set(),       # Set[str]: Track completed exercise IDs
                'total_attempts': 0,           # int: Total attempts made
                'total_passes': 0,             # int: Total successful attempts
                'exercise_history': []          # List[Dict]: Full history of exercises attempted
            }
            
            return self._get_state()
            
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            raise
    
    def _get_state(self) -> npt.NDArray[np.float32]:
        """Get the current state representation as a normalized vector.
        
        The state consists of:
        - Last 5 exercise attempts (padded with 0 if <5 attempts)
        - Normalized time spent (0-1)
        - Current pass rate (0-1)
        - Normalized mean perception (0-1)
        
        Returns:
            numpy.ndarray: Normalized state vector of shape (8,)
        """
        try:
            state = np.zeros(8, dtype=np.float32)
            
            # 1. Last 5 attempts (padded with 0 if <5 attempts)
            past_attempts = list(self.student_history['attempts'])
            for i in range(min(5, len(past_attempts))):
                state[i] = 1.0 if past_attempts[-(i+1)] else 0.0
            
            # 2. Normalized time spent (capped at 300 seconds = 5 minutes)
            state[5] = min(max(0.0, float(self.student_history['time_spent']) / 300.0), 1.0)
            
            # 3. Pass rate (0-1)
            state[6] = min(max(0.0, float(self.student_history['pass_rate'])), 1.0)
            
            # 4. Normalized mean perception (assuming scale 0-4)
            state[7] = min(max(0.0, float(self.student_history['mean_perception']) / 4.0), 1.0)
            
            # Validate state
            if not np.isfinite(state).all():
                print("Warning: Invalid state values, returning zeros")
                return np.zeros(8, dtype=np.float32)
                
            return state
            
        except Exception as e:
            print(f"Error in _get_state: {str(e)}")
            return np.zeros(8, dtype=np.float32)
    
    def _get_exercise_data(self, exercise_id: str) -> Tuple[bool, float]:
        """Retrieve exercise data for the current student.
        
        Args:
            exercise_id: ID of the exercise to retrieve data for
            
        Returns:
            Tuple of (pass_status, time_spent) for the exercise
        """
        exercise_data = self.data[
            (self.data['student_id'] == self.current_student) & 
            (self.data['exercise_id'] == exercise_id)
        ]
        
        if exercise_data.empty:
            return False, 30.0  # Default values if exercise not found
            
        row = exercise_data.iloc[0]
        pass_status = bool(row['pass_status'] == 'Pass' if isinstance(row['pass_status'], str) 
                        else row['pass_status'])
        return pass_status, float(row['time_spent'])
    
    def _update_student_history(self, exercise_id: str, pass_status: bool, time_spent: float) -> None:
        """Update the student's history with a new exercise attempt."""
        self.student_history['attempts'].append(int(pass_status))
        self.student_history['total_attempts'] += 1
        self.student_history['total_passes'] += int(pass_status)
        
        # Update pass rate
        self.student_history['pass_rate'] = (self.student_history['total_passes'] / 
                                           max(1, self.student_history['total_attempts']))
        
        # Update moving average of time spent
        prev_attempts = self.student_history['total_attempts'] - 1
        self.student_history['time_spent'] = (
            self.student_history['time_spent'] * prev_attempts + time_spent
        ) / self.student_history['total_attempts']
        
        # Track completed exercises
        self.student_history['exercises_done'].add(exercise_id)
        
        # Add to exercise history
        self.student_history['exercise_history'].append({
            'exercise_id': exercise_id,
            'pass_status': pass_status,
            'time_spent': time_spent,
            'timestamp': len(self.student_history['exercise_history'])
        })
    
    def _is_done(self) -> bool:
        """Check if the episode is done."""
        max_attempts = 100
        max_exercises = len(self.exercises)
        
        return (self.student_history['total_attempts'] >= max_attempts or
                len(self.student_history['exercises_done']) >= max_exercises)
    
    def step(self, action: int) -> Tuple[npt.NDArray[np.float32], float, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            action: Index of the exercise to recommend (0 to action_size-1)
            
        Returns:
            Tuple containing:
                - next_state: The state after taking the action
                - reward: The reward for taking the action
                - done: Whether the episode is complete
                - info: Additional information about the step
                
        Raises:
            ValueError: If the action is invalid
        """
        if not 0 <= action < self.action_size:
            raise ValueError(f"Invalid action: {action}. Must be in [0, {self.action_size-1}]")
            
        try:
            # Get exercise ID from action
            exercise_id = self.action_to_exercise[action]
            
            # Get exercise data
            pass_status, time_spent = self._get_exercise_data(exercise_id)
            
            # Calculate reward (higher for passes, lower for fails)
            reward = 1.0 if pass_status else 0.1
            
            # Update student history with this attempt
            self._update_student_history(exercise_id, pass_status, time_spent)
            
            # Check if episode is done
            done = self._is_done()
            
            # Get next state
            next_state = self._get_state()
            
            # Prepare info dictionary
            info = {
                'exercise_id': exercise_id,
                'pass_status': pass_status,
                'time_spent': time_spent,
                'total_attempts': self.student_history['total_attempts'],
                'exercises_done': len(self.student_history['exercises_done'])
            }
            
            # Store transition if memory is available
            if hasattr(self, 'memory'):
                self.memory.append(
                    Transition(self._get_state(), action, next_state, reward, done)
                )
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise


class DQNAgent:
    """Deep Q-Network agent for exercise recommendation.
    
    This agent uses experience replay and target network updates to stabilize training.
    """
    
    def __init__(self, state_size: int, action_size: int) -> None:
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimensionality of the state space
            action_size: Number of possible actions (exercises)
        """
        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience replay buffer
        self.memory: Deque[Transition] = deque(maxlen=2000)
        
        # Training hyperparameters
        self.gamma = 0.95       # Discount factor for future rewards
        self.epsilon = 1.0      # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_every = 10  # Steps between target network updates
        self.steps_done = 0
        
        # Device configuration (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural networks
        self.policy_net = self._build_model().to(self.device)
        self.target_net = self._build_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def _build_model(self) -> nn.Module:
        """Build the neural network model for the DQN.
        
        Returns:
            torch.nn.Module: The policy network
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        
        # Initialize weights using Xavier/Glorot initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)  # Small positive bias
                
        return model
        
    def remember(self, state: npt.NDArray[np.float32], action: int, 
                  next_state: npt.NDArray[np.float32], reward: float, done: bool) -> None:
        """Store an experience in the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state after taking the action
            reward: Reward received
            done: Whether the episode terminated after this step
            
        Raises:
            ValueError: If any input is invalid
        """
        try:
            # Convert inputs to correct types
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            done = bool(done)
            
            # Validate shapes and values
            if state.shape != (self.state_size,):
                raise ValueError(f"Invalid state shape: {state.shape}, expected ({self.state_size},)")
            if next_state.shape != (self.state_size,):
                raise ValueError(f"Invalid next_state shape: {next_state.shape}, expected ({self.state_size},)")
            if not (0 <= action < self.action_size):
                raise ValueError(f"Invalid action: {action}, must be in [0, {self.action_size-1}]")
                
            self.memory.append(Transition(state, action, next_state, reward, done))
            
        except Exception as e:
            print(f"Error in remember: {str(e)}")
            # Don't store invalid transitions to maintain data quality
    
    def act(self, state: npt.NDArray[np.float32]) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
        
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()
    
    def replay(self) -> float:
        """Train the agent on a batch of experiences from replay memory.
        
        Returns:
            float: The loss value for this training step
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        try:
            # Sample a batch of transitions
            transitions = random.sample(self.memory, self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(batch.state)).to(self.device)
            actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(batch.reward).to(self.device)
            next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
            dones = torch.FloatTensor(batch.done).to(self.device)
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            current_q_values = self.policy_net(states).gather(1, actions)
            
            # Compute V(s_{t+1}) for all next states
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].detach()
                # Compute the expected Q values
                expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute Huber loss
            loss = self.loss_fn(current_q_values.squeeze(), expected_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update target network
            self.steps_done += 1
            if self.steps_done % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {str(e)}")
            return 0.0
    
    def save(self, path: str) -> None:
        """Save the model weights to a file.
        
        Args:
            path: Path to save the model weights
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str) -> None:
        """Load model weights from a file.
        
        Args:
            path: Path to load the model weights from
            
        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        # Set to evaluation mode
        self.policy_net.eval()
        self.target_net.eval()
    
    def save(self, path: str) -> None:
        """Save the model weights and optimizer state to a file.
        
        Args:
            path: Path to save the model weights and optimizer state
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)


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

def get_student_recommendations(env, agent, student_id):
    """Get exercise recommendations for a specific student."""
    print(f"\nGetting recommendations for student {student_id}...")
    state = env.reset(student_id=student_id)
    
    # Get available actions (exercises not yet done)
    available_actions = [i for i in range(env.action_size) 
                        if env.action_to_exercise[i] not in env.student_history['exercises_done']]
    
    if not available_actions:
        print(f"No exercises available to recommend for student {student_id} (all exercises completed).")
        return
    
    # Get Q-values for all actions
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor).squeeze().cpu().numpy()
    
    # Only consider available actions for recommendation
    available_q_values = q_values[available_actions]
    top_indices = np.argsort(available_q_values)[::-1][:3]  # Get indices of top 3 available actions
    top_actions = [available_actions[i] for i in top_indices]  # Map back to original action indices
    
    # Get student's current performance
    attempts = list(env.student_history['attempts'])
    pass_rate = env.student_history['pass_rate']
    
    print(f"\nStudent {student_id} current performance:")
    print(f"- Pass rate: {pass_rate*100:.1f}%")
    print(f"- Recent attempts (1=pass, 0=fail): {attempts[-5:] if len(attempts) > 0 else 'None'}")
    
    print("\nTop 3 recommended exercises:")
    for i, action in enumerate(top_actions):
        exercise_id = env.action_to_exercise[action]
        exercise_data = env.data[env.data['exercise_id'] == exercise_id].iloc[0]
        print(f"{i+1}. Exercise ID: {exercise_id}")
        print(f"   Category: {exercise_data['category']}")
        print(f"   Grade: {exercise_data['grade']}")
        print(f"   Predicted success probability: {1/(1+np.exp(-q_values[action]))*100:.1f}%")
        print(f"   Q-value: {q_values[action]:.4f}")

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
    
    # Show example recommendations for a few students
    print("\nExample recommendations for sample students:")
    example_students = [1, 100, 200, 300, 400, 500]
    
    for student_id in example_students:
        if 1 <= student_id <= 545:
            get_student_recommendations(env, agent, student_id)
            print("\n" + "="*50 + "\n")
    
    # Interactive mode if running in a terminal
    if sys.stdin.isatty():
        try:
            while True:
                try:
                    student_id = int(input("\nEnter student ID (1-545) or 0 to exit: "))
                    if student_id == 0:
                        break
                    if 1 <= student_id <= 545:
                        get_student_recommendations(env, agent, student_id)
                    else:
                        print("Please enter a student ID between 1 and 545.")
                except ValueError:
                    print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
    else:
        print("\nRunning in non-interactive mode. Showing example recommendations only.")
        # Show a few more example recommendations
        additional_students = [50, 150, 250, 350, 450]
        for student_id in additional_students:
            if 1 <= student_id <= 545:
                get_student_recommendations(env, agent, student_id)
                print("\n" + "="*50 + "\n")
    
    # Interactive mode removed due to state reference issue

if __name__ == "__main__":
    main()
