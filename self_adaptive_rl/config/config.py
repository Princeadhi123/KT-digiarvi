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
    total_timesteps: int = 50000  # Increased for significantly more extensive training
    learning_rate: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 100000
    learning_starts: int = 1000
    train_freq: int = 1
    target_update_interval: int = 1000
    gamma: float = 0.99
    tau: float = 0.005
    gradient_steps: int = 1
    
    # Exploration parameters
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    exploration_fraction: float = 0.1
    
    # Model architecture
    policy_kwargs: Dict = None
    
    # Logging
    log_interval: int = 100
    tensorboard_log: str = "./tensorboard_logs/"
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = dict(
                net_arch=[dict(pi=[64, 64], qf=[64, 64])]
            )

@dataclass
class CurriculumConfig:
    # Curriculum learning parameters
    initial_difficulty: float = 0.3
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    difficulty_step: float = 0.05
    success_threshold: float = 0.8  # Success rate to increase difficulty
    failure_threshold: float = 0.3  # Failure rate to decrease difficulty
    window_size: int = 100  # Window size for success rate calculation
    
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
