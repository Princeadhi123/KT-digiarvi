# Self-Adapting Reinforcement Learning for Adaptive Curriculum Learning

This project implements a Self-Adapting Reinforcement Learning (RL) system for adaptive curriculum learning, designed to personalize educational content for students based on their performance.

## Project Structure

```
self_adaptive_rl/
├── config/
│   └── config.py         # Configuration parameters
├── models/
│   └── self_adaptive_agent.py  # Self-Adapting RL agent implementation
├── utils/
│   └── student_env.py    # Custom Gym environment for student learning
├── data/                 # Directory for storing data
├── models/               # Directory for saving trained models
├── train.py              # Training script
└── requirements.txt      # Python dependencies
```

## Features

- **Self-Adapting RL Agent**: Automatically adjusts its learning strategy based on the student's performance.
- **Adaptive Curriculum**: Dynamically adjusts exercise difficulty based on the student's success rate.
- **Meta-Learning**: Enables the agent to adapt to new students more efficiently.
- **Experience Replay**: Improves learning stability and sample efficiency.
- **Curriculum Learning**: Gradually increases difficulty as the student improves.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd self_adaptive_rl
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data**:
   - Place your preprocessed student data file (CSV format) in the project directory.
   - The expected format should include columns like 'student_id', 'exercise_id', 'score', 'time_spent', etc.

2. **Configure the system** (optional):
   - Modify parameters in `config/config.py` to adjust the learning process, curriculum adaptation, etc.

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Monitor training**:
   - Training progress will be displayed in the console.
   - Model checkpoints and training plots will be saved in the `models/` directory.

## Configuration

Key configuration parameters (in `config/config.py`):

- **RL Parameters**: Learning rate, batch size, discount factor, etc.
- **Curriculum Parameters**: Initial difficulty, difficulty adjustment steps, success/failure thresholds.
- **Self-Adaptation**: Meta-learning rate, adaptation steps, uncertainty threshold.

## Results

After training, the following will be saved in the model directory:
- Model checkpoints
- Training metrics (rewards, success rates, difficulty progression)
- Plots visualizing the training progress

## Customization

To adapt this system to your specific needs:

1. **Environment**: Modify `student_env.py` to match your specific educational scenario.
2. **State/Action Space**: Adjust the state representation and action space in the environment.
3. **Reward Function**: Customize the reward function to better match your educational objectives.
4. **Curriculum Design**: Modify the curriculum adaptation logic in the agent.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is inspired by recent advances in reinforcement learning for education.
- Built with PyTorch and Gym.
