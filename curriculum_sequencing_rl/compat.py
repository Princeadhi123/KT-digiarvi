"""Backward compatibility layer for legacy API usage.

This module provides wrapper functions and classes that maintain the original
API while using the new refactored architecture underneath.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .core import Config, TrainerFactory, setup_device, set_seed
from .environment import OptimizedInteractiveEnv
from .agents import DQNAgent, QLearningAgent, PolicyGradientAgent
from .experiment_runner import ExperimentRunner


def _deprecation_warning(old_func: str, new_func: str) -> None:
    """Issue deprecation warning for legacy functions."""
    warnings.warn(
        f"{old_func} is deprecated. Use {new_func} instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Legacy training function wrappers
def train_dqn(env: Any, episodes: int = 50, lr: float = 1e-3, 
              gamma: float = 0.99, batch_size: int = 128,
              buffer_size: int = 20000, hidden_dim: int = 128,
              eps_start: float = 1.0, eps_end: float = 0.05,
              eps_decay_steps: int = 20000, target_tau: float = 0.01,
              target_update_interval: int = 1, device: Optional[str] = None,
              **kwargs) -> Tuple[DQNAgent, Dict[str, Any]]:
    """Legacy DQN training function."""
    _deprecation_warning("train_dqn", "DQNTrainer.train")
    
    if device is None:
        device = setup_device()
    
    config = Config()
    config.dqn.episodes = episodes
    config.dqn.learning_rate = lr
    config.dqn.gamma = gamma
    config.dqn.batch_size = batch_size
    config.dqn.buffer_size = buffer_size
    config.dqn.hidden_dim = hidden_dim
    config.dqn.eps_start = eps_start
    config.dqn.eps_end = eps_end
    config.dqn.eps_decay_steps = eps_decay_steps
    config.dqn.target_tau = target_tau
    config.dqn.target_update_interval = target_update_interval
    
    trainer = TrainerFactory.create_trainer("dqn", config, env, device)
    agent = trainer.train()
    
    return agent, {"config": config}


def train_q_learning(env: Any, epochs: int = 5, alpha: float = 0.2,
                    gamma: float = 0.9, eps_start: float = 0.3,
                    eps_end: float = 0.0, eps_decay_epochs: int = 3,
                    **kwargs) -> Tuple[QLearningAgent, Dict[str, Any]]:
    """Legacy Q-Learning training function."""
    _deprecation_warning("train_q_learning", "QLearningTrainer.train")
    
    config = Config()
    config.q_learning.epochs = epochs
    config.q_learning.alpha = alpha
    config.q_learning.gamma = gamma
    config.q_learning.eps_start = eps_start
    config.q_learning.eps_end = eps_end
    config.q_learning.eps_decay_epochs = eps_decay_epochs
    
    trainer = TrainerFactory.create_trainer("q_learning", config, env)
    agent = trainer.train()
    
    return agent, {"config": config}


def train_a2c(env: Any, episodes: int = 50, lr: float = 1e-3,
              entropy_coef: float = 0.01, value_coef: float = 0.5,
              bc_warmup: int = 1, bc_weight: float = 0.5,
              batch_episodes: int = 4, device: Optional[str] = None,
              **kwargs) -> Tuple[PolicyGradientAgent, Dict[str, Any]]:
    """Legacy A2C training function."""
    _deprecation_warning("train_a2c", "A2CTrainer.train")
    
    if device is None:
        device = setup_device()
    
    config = Config()
    config.a2c.episodes = episodes
    config.a2c.learning_rate = lr
    config.a2c.entropy_coef = entropy_coef
    config.a2c.value_coef = value_coef
    config.a2c.bc_warmup = bc_warmup
    config.a2c.bc_weight = bc_weight
    config.a2c.batch_episodes = batch_episodes
    
    trainer = TrainerFactory.create_trainer("a2c", config, env, device)
    agent = trainer.train()
    
    return agent, {"config": config}


def train_a3c(env: Any, episodes: int = 50, lr: float = 1e-3,
              entropy_coef: float = 0.01, value_coef: float = 0.5,
              gae_lambda: float = 0.95, bc_warmup: int = 1,
              bc_weight: float = 0.5, rollouts: int = 4,
              device: Optional[str] = None, **kwargs) -> Tuple[PolicyGradientAgent, Dict[str, Any]]:
    """Legacy A3C training function."""
    _deprecation_warning("train_a3c", "A3CTrainer.train")
    
    if device is None:
        device = setup_device()
    
    config = Config()
    config.a3c.episodes = episodes
    config.a3c.learning_rate = lr
    config.a3c.entropy_coef = entropy_coef
    config.a3c.value_coef = value_coef
    config.a3c.gae_lambda = gae_lambda
    config.a3c.bc_warmup = bc_warmup
    config.a3c.bc_weight = bc_weight
    config.a3c.rollouts = rollouts
    
    trainer = TrainerFactory.create_trainer("a3c", config, env, device)
    agent = trainer.train()
    
    return agent, {"config": config}


def train_ppo(env: Any, episodes: int = 50, lr: float = 3e-4,
              clip_eps: float = 0.2, epochs: int = 4,
              batch_episodes: int = 8, minibatch_size: int = 2048,
              entropy_coef: float = 0.01, value_coef: float = 0.5,
              gae_lambda: float = 0.95, bc_warmup: int = 2,
              bc_weight: float = 1.0, device: Optional[str] = None,
              **kwargs) -> Tuple[PolicyGradientAgent, Dict[str, Any]]:
    """Legacy PPO training function."""
    _deprecation_warning("train_ppo", "PPOTrainer.train")
    
    if device is None:
        device = setup_device()
    
    config = Config()
    config.ppo.episodes = episodes
    config.ppo.learning_rate = lr
    config.ppo.clip_eps = clip_eps
    config.ppo.epochs = epochs
    config.ppo.batch_episodes = batch_episodes
    config.ppo.minibatch_size = minibatch_size
    config.ppo.entropy_coef = entropy_coef
    config.ppo.value_coef = value_coef
    config.ppo.gae_lambda = gae_lambda
    config.ppo.bc_warmup = bc_warmup
    config.ppo.bc_weight = bc_weight
    
    trainer = TrainerFactory.create_trainer("ppo", config, env, device)
    agent = trainer.train()
    
    return agent, {"config": config}


# Legacy policy function wrappers
def dqn_policy(agent: DQNAgent) -> Callable[[Any, int], int]:
    """Create legacy DQN policy function."""
    def policy_fn(state: Any, current_category: int) -> int:
        return agent.act(state, deterministic=True)
    return policy_fn


def q_learning_policy(agent: QLearningAgent) -> Callable[[Any, int], int]:
    """Create legacy Q-Learning policy function."""
    def policy_fn(state: Any, current_category: int) -> int:
        return agent.act(state, deterministic=True)
    return policy_fn


def a2c_policy(agent: PolicyGradientAgent) -> Callable[[Any, int], int]:
    """Create legacy A2C policy function."""
    def policy_fn(state: Any, current_category: int) -> int:
        return agent.act(state, deterministic=True)
    return policy_fn


def a3c_policy(agent: PolicyGradientAgent) -> Callable[[Any, int], int]:
    """Create legacy A3C policy function."""
    def policy_fn(state: Any, current_category: int) -> int:
        return agent.act(state, deterministic=True)
    return policy_fn


def ppo_policy(agent: PolicyGradientAgent) -> Callable[[Any, int], int]:
    """Create legacy PPO policy function."""
    def policy_fn(state: Any, current_category: int) -> int:
        return agent.act(state, deterministic=True)
    return policy_fn


# Legacy environment wrapper for backward compatibility
def create_legacy_environment(data_path: str, **kwargs):
    """Create environment with legacy interface."""
    _deprecation_warning("create_legacy_environment", "OptimizedInteractiveEnv")
    
    config = Config()
    config.environment.data_path = data_path
    
    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config.environment, key):
            setattr(config.environment, key, value)
    
    return OptimizedInteractiveEnv(config.environment)


# Utility functions for backward compatibility
def get_legacy_trainer_results(trainer_type: str, env: Any, config: Config, 
                              device: str = "cpu") -> Dict[str, Any]:
    """Get training results in legacy format."""
    trainer = TrainerFactory.create_trainer(trainer_type, config, env, device)
    agent = trainer.train()
    
    # Create legacy policy function
    if trainer_type == "dqn":
        policy_fn = dqn_policy(agent)
    elif trainer_type == "q_learning":
        policy_fn = q_learning_policy(agent)
    elif trainer_type in ["a2c", "a3c", "ppo"]:
        policy_fn = globals()[f"{trainer_type}_policy"](agent)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    return {
        "agent": agent,
        "policy_fn": policy_fn,
        "config": config,
        "trainer": trainer
    }
