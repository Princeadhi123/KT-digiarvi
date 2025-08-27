# Curriculum Sequencing RL - Usage Guide

## 🚀 Quick Start

### Using Configuration Files

```bash
# Run with a predefined configuration
python -m curriculum_sequencing_rl.main --config configs/quick_test.json

# Run high-performance experiment
python -m curriculum_sequencing_rl.main --config configs/high_performance.json

# Test reward shaping
python -m curriculum_sequencing_rl.main --config configs/reward_shaping.json

# Compare all algorithms
python -m curriculum_sequencing_rl.main --config configs/all_algorithms.yaml
```

### Command Line Interface

```bash
# Quick DQN test
python -m curriculum_sequencing_rl.main --models dqn --dqn_episodes 50 --eval_episodes 200

# Multiple algorithms with custom settings
python -m curriculum_sequencing_rl.main \
    --models dqn,ppo \
    --dqn_episodes 100 --dqn_lr 2e-4 \
    --ppo_episodes 100 --ppo_lr 3e-4 \
    --eval_episodes 500 \
    --metrics_csv results.csv

# Demo mode for visualization
python -m curriculum_sequencing_rl.main \
    --models dqn \
    --dqn_episodes 20 \
    --demo --demo_episodes 3 --demo_steps 10
```

### Programmatic API

```python
from curriculum_sequencing_rl import ExperimentRunner, Config

# Create and customize configuration
config = Config()
config.dqn.episodes = 100
config.dqn.learning_rate = 1e-3
config.eval_episodes = 500

# Run experiment
runner = ExperimentRunner("preprocessed_kt_data.csv", config)
results = runner.run_experiment()

# Access results
print(f"DQN accuracy: {results['dqn']['accuracy']:.3f}")
```

## 🔧 Configuration Options

### Environment Settings

```json
{
  "environment": {
    "reward_correct_w": 1.0,      // Weight for correctness reward
    "reward_score_w": 0.0,        // Weight for score reward
    "action_on": "category",      // Action space: "category" or "category_group"
    "seed": 42                    // Random seed
  }
}
```

### Reward Shaping

```json
{
  "environment": {
    "rew_improve_w": 0.1,         // Improvement shaping weight
    "rew_deficit_w": 0.05,        // Deficit addressing weight
    "rew_spacing_w": 0.05,        // Spacing optimization weight
    "rew_diversity_w": 0.02,      // Diversity encouragement weight
    "rew_challenge_w": 0.03,      // Challenge level weight
    "spacing_window": 16,         // Window for spacing calculation
    "diversity_recent_k": 10      // Recent choices for diversity
  }
}
```

### Algorithm-Specific Settings

**DQN:**
```json
{
  "dqn": {
    "episodes": 200,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 20000,
    "hidden_dim": 128,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10000,
    "target_tau": 0.01
  }
}
```

**PPO:**
```json
{
  "ppo": {
    "episodes": 200,
    "learning_rate": 3e-4,
    "clip_eps": 0.2,
    "epochs": 4,
    "batch_episodes": 8,
    "minibatch_size": 1024,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "gae_lambda": 0.95,
    "bc_warmup": 2,
    "bc_weight": 1.0
  }
}
```

## 📊 Performance Tuning

### For Speed
- Reduce `eval_episodes` (e.g., 100-300)
- Use smaller networks (`hidden_dim`: 64-128)
- Reduce training episodes for quick tests
- Disable baselines: `--no_chance --no_trivial --no_markov`

### For Accuracy
- Increase training episodes (200-400)
- Use larger networks (`hidden_dim`: 256-512)
- Increase evaluation episodes (500-1000)
- Fine-tune learning rates and other hyperparameters

### Memory Optimization
- Reduce DQN buffer size if memory constrained
- Use smaller batch sizes
- Reduce PPO minibatch size

## 🎯 Common Use Cases

### 1. Quick Algorithm Comparison
```bash
python -m curriculum_sequencing_rl.main \
    --models ql,dqn,ppo \
    --eval_episodes 200 \
    --metrics_csv comparison.csv
```

### 2. Hyperparameter Tuning
```bash
# Test different learning rates
for lr in 1e-4 2e-4 5e-4 1e-3; do
    python -m curriculum_sequencing_rl.main \
        --models dqn \
        --dqn_lr $lr \
        --dqn_episodes 100 \
        --metrics_csv "tuning_lr_${lr}.csv"
done
```

### 3. Reward Shaping Experiments
```bash
python -m curriculum_sequencing_rl.main \
    --models dqn \
    --rew_improve_w 0.1 \
    --rew_spacing_w 0.05 \
    --spacing_window 20 \
    --metrics_csv shaping_experiment.csv
```

### 4. Production Training
```bash
python -m curriculum_sequencing_rl.main \
    --config configs/high_performance.json \
    --seed 123 \
    --metrics_csv production_results.csv
```

## 🔍 Debugging and Analysis

### Enable Demo Mode
```bash
python -m curriculum_sequencing_rl.main \
    --models dqn \
    --demo --demo_episodes 3 --demo_steps 15
```

### Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from curriculum_sequencing_rl import ExperimentRunner
runner = ExperimentRunner("data.csv")
results = runner.run_experiment()
```

### Performance Profiling
```python
import cProfile
from curriculum_sequencing_rl import ExperimentRunner

def run_experiment():
    runner = ExperimentRunner("data.csv")
    return runner.run_experiment()

cProfile.run('run_experiment()', 'profile_stats')
```

## 📈 Results Analysis

### CSV Output Format
The metrics CSV contains:
- `timestamp`: Experiment timestamp
- `model`: Algorithm name
- `seed`: Random seed used
- `accuracy`: Test accuracy
- `avg_reward`: Average shaped reward
- `reward_base`: Base reward component
- `reward_norm`: Normalized reward
- `vpr`: Valid pick rate
- `regret_ratio`: Performance gap vs optimal
- Hyperparameter columns for each model

### Programmatic Access
```python
import pandas as pd

# Load results
df = pd.read_csv('results.csv')

# Best performing models
best_accuracy = df.loc[df['accuracy'].idxmax()]
print(f"Best model: {best_accuracy['model']} with {best_accuracy['accuracy']:.3f} accuracy")

# Compare algorithms
comparison = df.groupby('model')['accuracy'].agg(['mean', 'std', 'max'])
print(comparison)
```

## 🛠 Advanced Features

### Custom Trainers
```python
from curriculum_sequencing_rl.core import BaseTrainer, TrainerFactory

@TrainerFactory.register("custom_dqn")
class CustomDQNTrainer(BaseTrainer):
    def train(self):
        # Custom training logic
        pass

# Use in experiments
config.models = ["custom_dqn"]
```

### Environment Extensions
```python
from curriculum_sequencing_rl.environment import OptimizedInteractiveEnv

class CustomEnv(OptimizedInteractiveEnv):
    def _compute_reward(self, action, info):
        # Custom reward logic
        return super()._compute_reward(action, info)
```

### Configuration Inheritance
```python
from curriculum_sequencing_rl import Config

# Load base config
base_config = Config.from_file("configs/base.json")

# Modify for experiment
base_config.dqn.learning_rate = 2e-4
base_config.eval_episodes = 1000

# Save variant
base_config.save("configs/experiment_variant.json")
```

This guide covers the essential usage patterns for the refactored curriculum sequencing RL system. The new architecture provides flexibility while maintaining ease of use for both quick experiments and production training.
