"""Configuration management for RL experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import yaml
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    data_path: str
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    reward_correct_w: float = 0.0
    reward_score_w: float = 1.0
    action_on: str = "category"
    
    # Multi-objective shaping
    rew_improve_w: float = 0.0
    rew_deficit_w: float = 0.0
    rew_spacing_w: float = 0.0
    rew_diversity_w: float = 0.0
    rew_challenge_w: float = 0.0
    
    # Shaping hyperparameters
    ema_alpha: float = 0.3
    need_threshold: float = 0.6
    spacing_window: int = 5
    diversity_recent_k: int = 5
    challenge_target: float = 0.7
    challenge_band: float = 0.4
    invalid_penalty: float = 0.0
    
    # Hybrid weights
    hybrid_base_w: float = 1.0
    hybrid_mastery_w: float = 1.0
    hybrid_motivation_w: float = 1.0


@dataclass
class TrainingConfig:
    """Base training configuration."""
    episodes: int = 50
    eval_episodes: int = 300
    # Optional cap to avoid overly long training episodes
    train_max_steps_per_episode: Optional[int] = None
    # Optional caps to guarantee fast evaluation/smoke tests
    eval_max_steps_per_episode: Optional[int] = None
    val_max_steps_per_episode: Optional[int] = None
    eval_interval: int = 5
    device: Optional[str] = None
    seed: int = 42
    select_best_on_val: bool = False
    val_episodes: int = 300


@dataclass
class QLearningConfig(TrainingConfig):
    """Q-Learning specific configuration."""
    epochs: int = 5
    alpha: float = 0.2
    gamma: float = 0.9
    eps_start: float = 0.3
    eps_end: float = 0.0
    eps_decay_epochs: int = 3


@dataclass
class DQNConfig(TrainingConfig):
    """DQN specific configuration."""
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 20000
    hidden_dim: int = 128
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20000
    target_tau: float = 0.01
    target_update_interval: int = 1


@dataclass
class PolicyGradientConfig(TrainingConfig):
    """Base config for policy gradient methods (A2C/A3C/PPO)."""
    lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    bc_warmup_epochs: int = 1
    bc_weight: float = 0.5


@dataclass
class A2CConfig(PolicyGradientConfig):
    """A2C specific configuration."""
    batch_episodes: int = 4


@dataclass
class A3CConfig(PolicyGradientConfig):
    """A3C specific configuration."""
    gae_lambda: float = 0.95
    rollouts_per_update: int = 4


@dataclass
class PPOConfig(PolicyGradientConfig):
    """PPO specific configuration."""
    lr: float = 3e-4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    batch_episodes: int = 8
    minibatch_size: int = 2048
    gae_lambda: float = 0.95
    bc_warmup_epochs: int = 2
    bc_weight: float = 1.0


@dataclass
class SARLDQNConfig(DQNConfig):
    """Self-adaptive DQN configuration with adaptation knobs."""
    # Adaptation cadence
    adapt_interval: int = 5

    # Target hybrid contribution shares (sum not necessarily 1; will be normalized)
    hybrid_base_target_share: float = 0.7
    hybrid_mastery_target_share: float = 0.2
    hybrid_motivation_target_share: float = 0.1

    # Hybrid weight update settings
    weight_lr: float = 0.3
    weight_min: float = 0.0
    weight_max: float = 3.0
    share_tolerance: float = 0.05

    # Exploration adaptation via epsilon schedule
    invalid_rate_hi: float = 0.15
    invalid_rate_lo: float = 0.02
    epsilon_up_step: float = 0.05
    epsilon_down_step: float = 0.02
    eps_min: float = 0.01
    eps_max: float = 1.0

    # Optimizer LR adaptation
    lr_patience: int = 5
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-5

    # Curriculum/difficulty adaptation
    challenge_increase_threshold: float = 0.85
    challenge_decrease_threshold: float = 0.55
    challenge_target_step: float = 0.05
    challenge_target_min: float = 0.4
    challenge_target_max: float = 0.9
    challenge_band_step: float = -0.05  # negative narrows band when easy
    challenge_band_min: float = 0.2
    challenge_band_max: float = 0.6


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    environment: EnvironmentConfig
    models: List[str] = field(default_factory=lambda: ["ql", "dqn", "a2c", "a3c", "ppo"])
    
    # Model-specific configs
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    a3c: A3CConfig = field(default_factory=A3CConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sarl: 'SARLDQNConfig' = field(default_factory=SARLDQNConfig)
    
    # Evaluation settings
    include_chance: bool = True
    include_trivial: bool = True
    include_markov: bool = True
    
    # Output settings
    metrics_csv: Optional[str] = None
    demo: bool = False
    demo_episodes: int = 1
    demo_steps: int = 12
    demo_mode: str = "test"


class Config:
    """Configuration manager with file I/O support."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig(
            environment=EnvironmentConfig(data_path="")
        )
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying config."""
        return getattr(self.config, name)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Handle nested configs
        env_data = data.get('environment', {})
        env_config = EnvironmentConfig(**env_data)
        
        # Create model configs
        model_configs = {}
        for model in ['q_learning', 'dqn', 'a2c', 'a3c', 'ppo']:
            model_data = data.get(model, {})
            if model == 'q_learning':
                model_configs[model] = QLearningConfig(**model_data)
            elif model == 'dqn':
                model_configs[model] = DQNConfig(**model_data)
            elif model == 'a2c':
                model_configs[model] = A2CConfig(**model_data)
            elif model == 'a3c':
                model_configs[model] = A3CConfig(**model_data)
            elif model == 'ppo':
                model_configs[model] = PPOConfig(**model_data)
        # SARL config (optional)
        sarl_data = data.get('sarl', {})
        model_configs['sarl'] = SARLDQNConfig(**sarl_data)
        
        # Create experiment config
        exp_data = {k: v for k, v in data.items() 
                   if k not in ['environment', 'q_learning', 'dqn', 'a2c', 'a3c', 'ppo', 'sarl']}
        exp_config = ExperimentConfig(
            environment=env_config,
            **model_configs,
            **exp_data
        )
        
        return cls(exp_config)
    
    @classmethod
    def from_file(cls, path: str) -> 'Config':
        """Load configuration from file (JSON or YAML)."""
        path_obj = Path(path)
        
        with open(path_obj, 'r') as f:
            if path_obj.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        def _asdict_recursive(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = _asdict_recursive(value)
                    else:
                        result[key] = value
                return result
            return obj
        
        return _asdict_recursive(self.config)
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        path_obj = Path(path)
        data = self.to_dict()
        
        with open(path_obj, 'w') as f:
            if path_obj.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    def get_model_config(self, model_name: str) -> TrainingConfig:
        """Get configuration for a specific model."""
        model_map = {
            'ql': self.config.q_learning,
            'dqn': self.config.dqn,
            'a2c': self.config.a2c,
            'a3c': self.config.a3c,
            'ppo': self.config.ppo,
            'sarl': self.config.sarl
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name]
    
    def update_from_args(self, args) -> None:
        """Update config from command line arguments."""
        # Update environment config
        env_attrs = [attr for attr in dir(self.config.environment) 
                    if not attr.startswith('_')]
        for attr in env_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                if value is not None:
                    setattr(self.config.environment, attr, value)
        
        # Update model configs
        for model_name in ['q_learning', 'dqn', 'a2c', 'a3c', 'ppo', 'sarl']:
            model_config = getattr(self.config, model_name)
            prefix = model_name.replace('_', '') if model_name != 'q_learning' else 'ql'

            for attr in dir(model_config):
                if not attr.startswith('_'):
                    arg_name = f"{prefix}_{attr}"
                    if hasattr(args, arg_name):
                        value = getattr(args, arg_name)
                        # Skip None to avoid overriding config values with CLI defaults
                        if value is None:
                            continue
                        setattr(model_config, attr, value)

            # Handle known CLI-to-config alias mismatches for PG families and PPO
            # - bc_warmup (CLI) -> bc_warmup_epochs (config)
            # - a3c_rollouts (CLI) -> rollouts_per_update (config)
            # - ppo_epochs (CLI) -> ppo_epochs (config) without double prefix
            if model_name in ['a2c', 'a3c', 'ppo']:
                alias_arg = f"{prefix}_bc_warmup"
                if hasattr(args, alias_arg):
                    value = getattr(args, alias_arg)
                    if value is not None:
                        setattr(model_config, 'bc_warmup_epochs', value)
            if model_name == 'a3c':
                if hasattr(args, 'a3c_rollouts'):
                    value = getattr(args, 'a3c_rollouts')
                    if value is not None:
                        setattr(model_config, 'rollouts_per_update', value)
            if model_name == 'ppo':
                if hasattr(args, 'ppo_epochs'):
                    value = getattr(args, 'ppo_epochs')
                    if value is not None:
                        setattr(model_config, 'ppo_epochs', value)
        
        # Update experiment-level settings
        exp_attrs = ['models', 'include_chance', 'include_trivial', 'include_markov',
                    'metrics_csv', 'demo', 'demo_episodes', 'demo_steps', 'demo_mode']
        for attr in exp_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                if value is None:
                    continue
                # Special handling: allow comma-separated string for models
                if attr == 'models' and isinstance(value, str):
                    value = [m.strip().lower() for m in value.split(',') if m.strip()]
                setattr(self.config, attr, value)

        # Optional global override for evaluation episodes across all models
        if hasattr(args, 'eval_episodes') and getattr(args, 'eval_episodes') is not None:
            for model_name in ['q_learning', 'dqn', 'a2c', 'a3c', 'ppo', 'sarl']:
                model_config = getattr(self.config, model_name)
                setattr(model_config, 'eval_episodes', getattr(args, 'eval_episodes'))
