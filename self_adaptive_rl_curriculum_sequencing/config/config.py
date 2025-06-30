import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class RLConfig:
    # Environment parameters
    state_dim: int = 10  # Dimension of state space
    action_dim: int = 5   # Number of difficulty levels
    max_steps: int = 1000  # Maximum steps per episode
    
    # Training parameters
    total_timesteps: int = 100000  # Increased for 10,000 episodes (10M timesteps)
    learning_rate: float = 0.00025  # Slightly reduced for more stable training
    batch_size: int = 128  # Increased batch size for more stable gradients
    buffer_size: int = 200000  # Larger replay buffer for better experience replay
    learning_starts: int = 5000  # More initial random steps for better exploration
    train_freq: int = 4  # Update the model every 4 steps
    target_update_interval: int = 1000  # Keep target network update frequency
    gamma: float = 0.99  # Discount factor (good default)
    tau: float = 0.01  # Slightly higher for more stable target network updates
    gradient_steps: int = 1  # Keep as is for now
    
    # Exploration parameters
    exploration_initial_eps: float = 1.0  # Start with full exploration
    exploration_final_eps: float = 0.02  # Slightly higher final epsilon for better exploration
    exploration_fraction: float = 0.15  # Increased exploration duration
    
    # Model architecture
    policy_kwargs: Dict = None
    
    # Logging
    log_interval: int = 100
    tensorboard_log: str = "./tensorboard_logs/"
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = dict(
                net_arch=[dict(pi=[128, 128], qf=[128, 128])]  # Deeper network with more units
            )

@dataclass
class CurriculumConfig:
    # Curriculum learning parameters
    initial_difficulty: float = 0.2  # Start with slightly easier difficulty
    min_difficulty: float = 0.05    # Allow for easier exercises
    max_difficulty: float = 1.2     # Allow for more challenging exercises
    difficulty_step: float = 0.1    # Larger steps for more aggressive adjustments
    success_threshold: float = 0.7  # Lower success threshold to increase difficulty more quickly
    failure_threshold: float = 0.4  # Higher failure threshold to decrease difficulty more readily
    window_size: int = 50           # Smaller window for more responsive difficulty adjustments
    
    # Student model parameters
    student_learning_rate: float = 0.001
    student_batch_size: int = 32
    student_epochs: int = 10

@dataclass
class SelfAdaptiveConfig:
    # Meta-learning parameters
    meta_learning_rate: float = 0.0001
    meta_batch_size: int = 32
    meta_update_freq: int = 1000
    
    # Adaptation parameters
    adaptation_steps: int = 100
    adaptation_learning_rate: float = 0.001
    
    # Uncertainty estimation
    uncertainty_threshold: float = 0.2
    
    # Experience replay
    experience_replay_size: int = 10000
    experience_replay_batch_size: int = 64

@dataclass
class ExperimentConfig:
    # Experiment setup
    experiment_name: str = "self_adaptive_rl_curriculum"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    
    # Data parameters
    train_test_split: float = 0.8
    
    # Create directories if they don't exist
    def __post_init__(self):
        # Directory paths are still defined but not created automatically
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")

# Create configuration instances
rl_config = RLConfig()
curriculum_config = CurriculumConfig()
self_adaptive_config = SelfAdaptiveConfig()
experiment_config = ExperimentConfig()
