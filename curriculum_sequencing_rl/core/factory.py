"""Factory pattern for creating trainers and agents."""

from typing import Dict, Type, Any
from .base import BaseTrainer, BaseAgent
from .config import TrainingConfig, QLearningConfig, DQNConfig, A2CConfig, A3CConfig, PPOConfig


class TrainerFactory:
    """Factory for creating appropriate trainers based on configuration."""
    
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    
    @classmethod
    def register(cls, name: str, trainer_class: Type[BaseTrainer]) -> None:
        """Register a trainer class."""
        cls._trainers[name] = trainer_class
    
    @classmethod
    def create(cls, model_name: str, config: TrainingConfig, **kwargs) -> BaseTrainer:
        """Create a trainer instance."""
        if model_name not in cls._trainers:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._trainers.keys())}")
        
        trainer_class = cls._trainers[model_name]
        return trainer_class(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all available trainer types."""
        return list(cls._trainers.keys())


def register_trainer(name: str):
    """Decorator to register a trainer class."""
    def decorator(trainer_class: Type[BaseTrainer]):
        TrainerFactory.register(name, trainer_class)
        return trainer_class
    return decorator
