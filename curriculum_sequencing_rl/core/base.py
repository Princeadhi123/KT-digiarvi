"""Base classes and interfaces for RL agents and trainers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch

# Type aliases
PolicyFunction = Callable[[np.ndarray, int], int]
StateType = np.ndarray
ActionType = int
RewardType = float
InfoType = Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
    
    @abstractmethod
    def act(self, state: StateType, training: bool = True, valid_ids: Optional[Any] = None) -> ActionType:
        """Select an action given the current state."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """Update the agent's parameters and return loss."""
        pass
    
    @abstractmethod
    def get_policy(self, env: Optional[Any] = None) -> PolicyFunction:
        """Return a policy function for evaluation."""
        pass
    
    def save(self, path: str) -> None:
        """Save agent state to file."""
        pass
    
    def load(self, path: str) -> None:
        """Load agent state from file."""
        pass


class BaseTrainer(ABC):
    """Abstract base class for training different RL algorithms."""
    
    def __init__(self, config: 'TrainingConfig'):
        self.config = config
        self.agent: Optional[BaseAgent] = None
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def create_agent(self, env: Any) -> BaseAgent:
        """Create and initialize the agent."""
        pass
    
    @abstractmethod
    def train_step(self, env: Any, agent: BaseAgent) -> Dict[str, float]:
        """Execute one training step and return metrics."""
        pass
    
    def train(self, env: Any) -> BaseAgent:
        """Main training loop."""
        self.agent = self.create_agent(env)
        
        for episode in range(self.config.episodes):
            metrics = self.train_step(env, self.agent)
            self._update_metrics(metrics, episode)
            
            if self._should_evaluate(episode):
                self._evaluate_agent(env, episode)
        
        return self.agent
    
    def _update_metrics(self, metrics: Dict[str, float], episode: int) -> None:
        """Update training metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def _should_evaluate(self, episode: int) -> bool:
        """Determine if agent should be evaluated at this episode."""
        return (episode + 1) % getattr(self.config, 'eval_interval', 10) == 0
    
    def _evaluate_agent(self, env: Any, episode: int) -> None:
        """Evaluate agent performance."""
        pass


class BaseNetwork(torch.nn.Module):
    """Base neural network with common functionality."""
    
    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
    
    def to_device(self, device: Union[str, torch.device]) -> 'BaseNetwork':
        """Move network to specified device."""
        self._device = torch.device(device)
        return self.to(self._device)
    
    @property
    def device(self) -> torch.device:
        """Get the device this network is on."""
        return self._device
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ReplayBuffer:
    """Efficient replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.ptr = 0
        
        # Pre-allocate tensors for efficiency
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: Optional[np.ndarray], done: bool) -> None:
        """Add a transition to the buffer."""
        self.states[self.ptr] = torch.from_numpy(state).float()
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        if next_state is not None:
            self.next_states[self.ptr] = torch.from_numpy(next_state).float()
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        return self.size
