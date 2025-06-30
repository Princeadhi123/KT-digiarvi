import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from gym import spaces
import torch
import torch.nn as nn
from collections import deque
import random

class StudentLearningEnv(gym.Env):
    """
    Custom Gym environment for student learning with adaptive curriculum.
    The environment simulates a student learning process where the agent (tutor)
    selects exercises of varying difficulty for the student.
    """
    
    def __init__(self, data_path: str, config: 'CurriculumConfig', split: str = 'train', seed: int = 42):
        super(StudentLearningEnv, self).__init__()
        
        # Set random seed for reproducibility
        self.seed = seed
        np.random.seed(seed)
        
        # Load and preprocess student data with split
        self.data = self._load_and_split_data(data_path, split)
        self.student_ids = self.data['student_id'].unique()
        self.split = split
        self.current_student_id = None
        self.current_student_data = None
        self.current_exercise_idx = 0
        
        # Configuration
        self.config = config
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # 5 difficulty levels
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(10,),  # State representation dimension
            dtype=np.float32
        )
        
        # Student model (simple neural network to simulate student learning)
        self.student_model = self._build_student_model()
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=config.student_learning_rate
        )
        self.loss_fn = nn.BCELoss()
        
        # Track performance
        self.performance_history = deque(maxlen=100)
        self.current_difficulty = config.initial_difficulty
        
        # Reset environment
        self.reset()
    
    def _load_and_split_data(self, data_path: str, split: str) -> pd.DataFrame:
        """Load and preprocess student data with train/validation/test split."""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Convert categorical variables to numerical
        df['sex'] = df['sex'].map({'Boy': 0, 'Gir': 1})
        
        # Normalize numerical features
        numerical_cols = ['score', 'time_spent', 'total_attempts', 'cumulative_passes', 'pass_rate']
        for col in numerical_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        
        # Split by student IDs to prevent data leakage
        student_ids = df['student_id'].unique()
        np.random.shuffle(student_ids)
        
        # 70% train, 15% validation, 15% test split
        train_size = int(0.7 * len(student_ids))
        val_size = int(0.15 * len(student_ids))
        
        if split == 'train':
            selected_ids = student_ids[:train_size]
        elif split == 'val':
            selected_ids = student_ids[train_size:train_size + val_size]
        elif split == 'test':
            selected_ids = student_ids[train_size + val_size:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        return df[df['student_id'].isin(selected_ids)]
    
    # Keep backward compatibility
    def _load_data(self, data_path: str) -> pd.DataFrame:
        return self._load_and_split_data(data_path, 'train')
    
    def _build_student_model(self) -> nn.Module:
        """Build a simple neural network to simulate student learning."""
        class StudentModel(nn.Module):
            def __init__(self, input_dim=5, hidden_dim=32):
                super(StudentModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
                
        return StudentModel()
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation."""
        if self.current_student_data is None or self.current_exercise_idx >= len(self.current_student_data):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get current exercise data
        current_row = self.current_student_data.iloc[self.current_exercise_idx]
        
        # Create state vector
        state = np.array([
            current_row['score'],
            current_row['time_spent'],
            current_row['total_attempts'],
            current_row['cumulative_passes'],
            current_row['pass_rate'],
            current_row['mean_perception'],
            current_row['sex'],
            self.current_difficulty,
            self.current_exercise_idx / len(self.current_student_data),  # Progress
            len(self.performance_history) > 0 and np.mean(self.performance_history) or 0  # Recent performance
        ], dtype=np.float32)
        
        return state
    
    def _simulate_student_response(self, difficulty: float) -> Tuple[bool, float]:
        """Simulate student's response to an exercise of given difficulty."""
        # Get current exercise data
        current_row = self.current_student_data.iloc[self.current_exercise_idx]
        
        # Prepare input features for student model
        features = torch.FloatTensor([
            current_row['score'],
            current_row['time_spent'],
            current_row['total_attempts'],
            current_row['cumulative_passes'],
            difficulty
        ]).unsqueeze(0)
        
        # Get student's probability of success
        with torch.no_grad():
            success_prob = self.student_model(features).item()
        
        # Add some noise to make it more realistic
        success_prob = np.clip(success_prob + np.random.normal(0, 0.1), 0, 1)
        
        # Determine success/failure
        success = success_prob > 0.5
        
        # Update student model based on experience
        self._update_student_model(features, success)
        
        return success, success_prob
    
    def _update_student_model(self, features: torch.Tensor, success: bool):
        """Update the student model based on the latest experience."""
        # Convert success to tensor
        target = torch.FloatTensor([[1.0 if success else 0.0]])
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.student_model(features)
        loss = self.loss_fn(output, target)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer representing the difficulty level (0-4)
            
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Convert action to difficulty level (0.1 to 1.0)
        difficulty = 0.1 + (action / 4.0) * 0.9
        self.current_difficulty = difficulty
        
        # Simulate student response
        success, success_prob = self._simulate_student_response(difficulty)
        
        # Calculate reward
        if success:
            reward = 1.0 * difficulty  # Higher reward for more difficult exercises
        else:
            reward = -0.5 * difficulty  # Penalty for failure, scaled by difficulty
        
        # Update performance history
        self.performance_history.append(1 if success else 0)
        
        # Move to next exercise
        self.current_exercise_idx += 1
        
        # Check if episode is done
        done = self.current_exercise_idx >= len(self.current_student_data)
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'success': success,
            'difficulty': difficulty,
            'success_prob': success_prob,
            'student_id': self.current_student_id,
            'exercise_idx': self.current_exercise_idx
        }
        
        return next_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        # Select a random student
        self.current_student_id = np.random.choice(self.student_ids)
        self.current_student_data = self.data[self.data['student_id'] == self.current_student_id]
        self.current_exercise_idx = 0
        
        # Reset student model for new student
        self.student_model = self._build_student_model()
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=self.config.student_learning_rate
        )
        
        # Reset performance tracking
        self.performance_history = deque(maxlen=100)
        self.current_difficulty = self.config.initial_difficulty
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Student ID: {self.current_student_id}")
            print(f"Current Exercise: {self.current_exercise_idx + 1}/{len(self.current_student_data)}")
            print(f"Current Difficulty: {self.current_difficulty:.2f}")
            if len(self.performance_history) > 0:
                print(f"Recent Success Rate: {np.mean(self.performance_history):.2f}")
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")
