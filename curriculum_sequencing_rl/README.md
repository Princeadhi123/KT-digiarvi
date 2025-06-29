# Curriculum Sequencing using Deep Q-Learning

This project implements a Deep Q-Network (DQN) for intelligent curriculum sequencing in educational applications. The system learns to recommend the optimal sequence of learning activities based on student performance data.

## Overview

The curriculum sequencing agent uses reinforcement learning to dynamically adjust the learning path for students, ensuring optimal knowledge acquisition and retention. It analyzes various student interaction metrics to make informed sequencing decisions.

## Features

- **Adaptive Learning Paths**: Creates personalized learning sequences
- **Performance-Based Sequencing**: Uses student performance metrics to inform decisions
- **Deep Reinforcement Learning**: Implements DQN with experience replay
- **Modular Design**: Easy to extend with new learning activities or metrics

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `curriculum_sequencer.py`: Main implementation of the DQN agent and environment
- `dqn_curriculum.pth`: Pre-trained model weights
- `training_history.png`: Training progress visualization
- `training_metrics.png`: Performance metrics visualization
- `requirements.txt`: Python dependencies

## Usage

1. Prepare your dataset in CSV format with the following columns:
   - `pass_rate`: Student's success rate (0-1)
   - `category_encoded`: Encoded category of the learning activity
   - `order_norm`: Normalized order of activities
   - `time_spent_norm`: Normalized time spent on activities
   - `total_attempts_norm`: Normalized number of attempts

2. Train the model:
   ```bash
   python curriculum_sequencer.py
   ```

## Model Architecture

- **Input Layer**: State features (pass rate, category, order, time spent, attempts)
- **Hidden Layers**: 2 fully connected layers with 64 units each (ReLU activation)
- **Output Layer**: Q-values for each possible action

## Training Configuration

- **Episodes**: 200
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Epsilon**: Starts at 1.0, decays to 0.01
- **Target Network Update**: Every 5 episodes
- **Replay Memory**: 10,000 transitions

## Outputs

- `dqn_curriculum.pth`: Trained model weights
- `training_history.png`: Training progress visualization
- `training_metrics.png`: Performance metrics over time

## Example Usage

```python
# Initialize environment
env = CurriculumEnvironment('your_data.csv')

# Initialize agent
agent = DQNAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load pre-trained weights (optional)
agent.load('dqn_curriculum.pth')

# Get recommended action for a state
state = env.reset()
action = agent.act(state, training=False)
print(f"Recommended action: {action}")
```

## Evaluation

The model can be evaluated on training, validation, or test sets using the `evaluate_agent` function, which provides average rewards and performance metrics.

## License

MIT License - See LICENSE file for details
