# Curriculum Sequencing RL - Refactored Architecture

A comprehensive reinforcement learning framework for curriculum sequencing with improved efficiency, modularity, and maintainability.

## 🚀 Key Improvements

- **Modular Architecture**: Clean separation of concerns with base classes and interfaces
- **Configuration Management**: Centralized config system with JSON/YAML support
- **Factory Pattern**: Easy extensibility for new RL algorithms
- **Optimized Environment**: Precomputed state vectors for faster training
- **Streamlined Orchestration**: Clean experiment runner with comprehensive logging
- **Backward Compatibility**: Legacy API maintained for existing scripts

## 📁 Architecture Overview

```
curriculum_sequencing_rl/
├── core/                    # Core architecture components
│   ├── base.py             # Abstract base classes for agents/trainers
│   ├── config.py           # Configuration management system
│   ├── factory.py          # Trainer factory pattern
│   └── utils.py            # Utility functions and helpers
├── agents/                  # Refactored RL algorithm implementations
│   ├── dqn_agent.py        # DQN with dueling network and replay buffer
│   ├── q_learning_agent.py # Tabular Q-Learning with epsilon decay
│   └── policy_gradient_agent.py # A2C/A3C/PPO with shared architecture
├── environment/             # Optimized environment and baselines
│   ├── optimized_env.py    # Fast interactive environment
│   └── baseline_policies.py # Chance, trivial, and Markov baselines
├── experiment_runner.py     # Streamlined experiment orchestration
├── main.py                 # New CLI entry point
├── compat.py               # Backward compatibility layer
└── [legacy files]          # Original implementations maintained
```

## 🛠 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install specific packages
pip install torch numpy pandas matplotlib seaborn
```

## 🎯 Quick Start

### New API (Recommended)

```python
from curriculum_sequencing_rl import ExperimentRunner, Config

# Create configuration
config = Config()
config.dqn.episodes = 100
config.dqn.learning_rate = 1e-3

# Run experiment
runner = ExperimentRunner("data.csv", config)
results = runner.run_experiment()
```

### Command Line Interface

```bash
# Run with default settings
python -m curriculum_sequencing_rl.main

# Customize training
python -m curriculum_sequencing_rl.main \
    --models dqn,ppo \
    --dqn_episodes 200 \
    --ppo_episodes 150 \
    --eval_episodes 500 \
    --metrics_csv results.csv

# Enable demo mode
python -m curriculum_sequencing_rl.main --demo --demo_episodes 2
```

### Legacy API (Backward Compatible)

```python
# Original API still works
from curriculum_sequencing_rl import train_dqn, dqn_policy
from curriculum_sequencing_rl.env import InteractiveReorderEnv

env = InteractiveReorderEnv("data.csv")
agent, info = train_dqn(env, episodes=100)
policy_fn = dqn_policy(agent)
```

## 🔧 Configuration

The new configuration system supports:

- **Dataclass-based configs** with type hints and validation
- **JSON/YAML serialization** for reproducible experiments
- **CLI integration** with automatic argument parsing
- **Model-specific parameters** organized by algorithm

```python
from curriculum_sequencing_rl.core import Config

config = Config()

# Environment settings
config.environment.reward_correct_w = 1.0
config.environment.reward_score_w = 0.0

# DQN hyperparameters
config.dqn.learning_rate = 3e-4
config.dqn.gamma = 0.995
config.dqn.hidden_dim = 256

# Save/load configuration
config.save("experiment_config.json")
config = Config.load("experiment_config.json")
```

## 🧠 Supported Algorithms

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **Q-Learning** | Tabular | Epsilon-greedy, validation selection |
| **DQN** | Deep Q-Network | Dueling network, replay buffer, target network |
| **A2C** | Actor-Critic | Advantage estimation, behavior cloning warmup |
| **A3C** | Async Actor-Critic | GAE advantages, parallel rollouts |
| **PPO** | Policy Optimization | Clipped objective, minibatch updates |

## 📊 Evaluation & Metrics

Comprehensive evaluation includes:

- **Accuracy**: Correct category selection rate
- **Valid Pick Rate (VPR)**: Proportion of valid actions
- **Regret**: Performance gap vs optimal policy
- **Reward Components**: Base, shaping, and normalized rewards
- **Baseline Comparisons**: Chance, trivial, and Markov policies

## 🎛 Advanced Features

### Multi-Objective Reward Shaping

```bash
python -m curriculum_sequencing_rl.main \
    --rew_improve_w 0.1 \
    --rew_deficit_w 0.05 \
    --rew_spacing_w 0.08 \
    --rew_diversity_w 0.04
```

### Hyperparameter Tuning

```bash
# DQN with custom hyperparameters
python -m curriculum_sequencing_rl.main \
    --models dqn \
    --dqn_lr 2e-4 \
    --dqn_gamma 0.997 \
    --dqn_hidden_dim 512 \
    --dqn_batch_size 512 \
    --dqn_buffer_size 120000
```

### Demo Mode

```bash
# Print sample rollouts
python -m curriculum_sequencing_rl.main \
    --demo \
    --demo_episodes 3 \
    --demo_steps 15 \
    --demo_mode test
```

## 📈 Performance Improvements

- **~3x faster environment steps** through precomputed state vectors
- **Reduced memory usage** with efficient replay buffers
- **Optimized training loops** with batched operations
- **Configurable evaluation** to balance speed vs accuracy

## 🔄 Migration Guide

### From Legacy to New API

```python
# Old way
from curriculum_sequencing_rl.dqn import train_dqn, dqn_policy
agent, _ = train_dqn(env, episodes=100, lr=1e-3)
policy = dqn_policy(agent)

# New way
from curriculum_sequencing_rl import TrainerFactory, Config
config = Config()
config.dqn.episodes = 100
config.dqn.learning_rate = 1e-3
trainer = TrainerFactory.create_trainer("dqn", config, env)
agent = trainer.train()
```

### Experiment Runner Migration

```python
# Old orchestrator
from curriculum_sequencing_rl.curriculum_rl_experiments import main
main()  # Complex CLI with many arguments

# New orchestrator
from curriculum_sequencing_rl import ExperimentRunner
runner = ExperimentRunner("data.csv")
results = runner.run_experiment()  # Clean, configurable
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Set `--device cpu` if GPU unavailable
3. **Memory Issues**: Reduce batch sizes or buffer sizes
4. **Slow Training**: Use optimized environment and smaller eval episodes

### Performance Tips

- Use `OptimizedInteractiveEnv` for faster training
- Reduce `eval_episodes` for quicker experiments
- Enable GPU acceleration when available
- Use precomputed baselines to skip redundant evaluation

## 📝 Citation

If you use this refactored framework, please cite:

```bibtex
@software{curriculum_sequencing_rl_v2,
  title={Curriculum Sequencing RL - Refactored Architecture},
  author={Your Name},
  year={2024},
  version={2.0.0},
  url={https://github.com/your-repo/curriculum-sequencing-rl}
}
```

## 🤝 Contributing

1. Follow the modular architecture patterns
2. Add type hints and docstrings
3. Update tests for new features
4. Maintain backward compatibility
5. Document configuration options

## 📄 License

[Your License Here]
