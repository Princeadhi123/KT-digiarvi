# Difficulty Adaptation using Deep Q-Learning

This project implements a Deep Q-Network (DQN) for adaptive difficulty adjustment in educational exercises based on student performance metrics.

## Overview

The system monitors student performance (scores, pass rates) and dynamically adjusts exercise difficulty to maintain an optimal challenge level. The DQN agent learns to make these adjustments through reinforcement learning.

## Features

- **Adaptive Difficulty**: Adjusts exercise difficulty in real-time
- **Performance-Based**: Uses student scores and pass rates to inform decisions
- **Stable Learning**: Implements experience replay and target networks
- **Configurable**: Easy to adjust hyperparameters and reward functions

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset in CSV format with columns: `score`, `pass_rate`, etc.
2. Run the training script:
   ```bash
   python dqn_difficulty_adaptation.py
   ```
3. The model will save checkpoints and learning curves during training

## Model Architecture

- **Input Layer**: 4 features (current_difficulty, performance_metric, pass_rate, time_spent_normalized)
- **Hidden Layers**: 2 fully connected layers with ReLU activation
- **Output Layer**: 3 actions (decrease, maintain, increase difficulty)

## Reward Function

The reward function is designed to:
- Encourage maintaining difficulty when performance is in the optimal range (0.6-0.8)
- Reward increasing difficulty for high performance (>0.8)
- Penalize maintaining high difficulty with poor performance (<0.3)
- Discourage frequent large difficulty changes

## Training

- **Episodes**: 500
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Gamma (discount factor)**: 0.95
- **Epsilon**: Starts at 1.0, decays to 0.1
- **Target Network Update**: Every 20 episodes

## Outputs

- `dqn_final.pth`: Trained model weights
- `dqn_learning_curve.png`: Training performance over time
- Checkpoints saved every 100 episodes

## Example Usage

```python
# Load the trained model
model = DQNDifficultyAdapter(data_path='your_data.csv')
model.policy_net.load_state_dict(torch.load('dqn_final.pth'))

# Get difficulty adjustment for a student
performance = 0.75
current_difficulty = 2
state = model._get_state(current_difficulty, performance)
action = model.select_action(state, training=False)
new_difficulty = max(0, min(4, current_difficulty + (action - 1)))
print(f"New difficulty: {new_difficulty}")
```

## Performance

The model shows:
- Stable convergence during training
- Logical difficulty adjustments based on performance
- Good balance between exploration and exploitation

## License

MIT License - See LICENSE file for details
