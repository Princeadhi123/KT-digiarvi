"""Curriculum Sequencing RL - Refactored Architecture.

This package provides a clean, modular architecture for curriculum sequencing
reinforcement learning experiments with improved performance and maintainability.

Key Components:
- core: Base classes, configuration, and utilities
- agents: Refactored RL algorithm implementations
- environment: Optimized interactive environment
- evaluation: Comprehensive evaluation utilities
- experiment_runner: Streamlined experiment orchestration

Backward Compatibility:
The original API is maintained through compatibility imports.
"""

from .core import Config, TrainerFactory, setup_device, set_seed
from .environment import OptimizedInteractiveEnv, BaselinePolicies
from .agents import DQNAgent, QLearningAgent, PolicyGradientAgent
from .experiment_runner import ExperimentRunner
from .main import main

# Backward compatibility imports
from .evaluation import (
    eval_policy_avg_score,
    eval_policy_valid_pick_rate,
    eval_policy_regret,
    eval_policy_interactive_metrics,
    print_sample_rollouts
)

__version__ = "2.0.0"
__all__ = [
    # New architecture
    'Config',
    'TrainerFactory',
    'setup_device',
    'set_seed',
    'OptimizedInteractiveEnv',
    'BaselinePolicies',
    'DQNAgent',
    'QLearningAgent',
    'PolicyGradientAgent',
    'ExperimentRunner',
    'main',
    # Backward compatibility
    'eval_policy_avg_score',
    'eval_policy_valid_pick_rate',
    'eval_policy_regret',
    'eval_policy_interactive_metrics',
    'print_sample_rollouts',
    # Legacy modules (evaluation only)
    "evaluation",
]
