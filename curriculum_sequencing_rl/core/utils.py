"""Utility functions for RL experiments."""

import random
import numpy as np
import torch
from typing import Optional, Union
import logging


def setup_device(device: Optional[str] = None) -> torch.device:
    """Setup and return the appropriate device for computation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device_obj = torch.device("cpu")
    
    return device_obj


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, 
                gamma: float, gae_lambda: float, 
                next_value: float = 0.0) -> torch.Tensor:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        lastgaelam = delta + gamma * gae_lambda * lastgaelam
        advantages[t] = lastgaelam
    
    return advantages


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def linear_schedule(start: float, end: float, current_step: int, total_steps: int) -> float:
    """Linear interpolation between start and end values."""
    if total_steps <= 0:
        return end
    
    progress = min(current_step / total_steps, 1.0)
    return start + (end - start) * progress


def exponential_schedule(start: float, end: float, current_step: int, 
                        decay_steps: int, decay_rate: float = 0.95) -> float:
    """Exponential decay schedule."""
    if decay_steps <= 0:
        return end
    
    decay_factor = decay_rate ** (current_step / decay_steps)
    return end + (start - end) * decay_factor


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, 
                   dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Apply softmax with masking for invalid actions."""
    masked_logits = logits / temperature
    masked_logits = masked_logits + mask
    return torch.softmax(masked_logits, dim=dim)


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip gradient norm and return the total norm."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class MovingAverage:
    """Efficient moving average computation."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = []
        self.sum = 0.0
    
    def update(self, value: float) -> float:
        """Add new value and return current average."""
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        
        return self.sum / len(self.values)
    
    def get_average(self) -> float:
        """Get current average."""
        return self.sum / len(self.values) if self.values else 0.0


class EpsilonScheduler:
    """Epsilon-greedy exploration scheduler."""
    
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return linear_schedule(self.start, self.end, self.step_count, self.decay_steps)
    
    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:
    """Setup logging configuration."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler()]
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
