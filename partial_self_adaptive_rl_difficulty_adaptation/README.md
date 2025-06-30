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

TODO for fully self adaptive model
-------------------------------------------------------------------------------------------
Let me analyze each point based on our actual implementation:

1. Dynamic Difficulty Adjustment ✅
Implemented: Yes

The code adjusts difficulty based on student performance
Uses current_difficulty that gets updated in the curriculum
Difficulty is adjusted based on success/failure thresholds
2. Meta-Learning Components ❌
Not Fully Implemented:

While there's a 
SelfAdaptiveConfig
 with meta-learning parameters, the actual meta-learning logic is not fully implemented
The agent doesn't currently adapt its learning algorithm based on student patterns
3. Student-Specific Adaptation ✅
Partially Implemented:

The environment tracks individual students via current_student_id
However, the agent doesn't maintain long-term memory of individual students between episodes
Each student's session is independent
4. Reward Shaping ✅
Implemented:

Rewards are scaled by difficulty
Different rewards for success (1.0 * difficulty) vs failure (-0.5 * difficulty)
5. Experience Replay with Adaptation ❌
Basic Implementation:

Uses standard experience replay
No prioritization of experiences
No special adaptation of the replay buffer
6. Uncertainty Estimation ❌
Not Implemented:

No explicit uncertainty estimation
The agent doesn't adjust exploration based on prediction confidence
7. Curriculum Learning Integration ✅
Implemented:

Has a 
CurriculumConfig
c:\Users\pdaadh\Desktop\KT digiarvi\self_adaptive_rl\config\config.py
 class
Implements adaptive difficulty progression
Adjusts based on success/failure rates
Summary:
✅ Fully Implemented: 1, 4, 7
❌ Not Fully Implemented: 2, 3, 5, 6

Key Missing Components:
Meta-Learning: The agent doesn't adapt its learning algorithm
Student Memory: No long-term student modeling
Advanced Replay: Basic experience replay without prioritization
Uncertainty: No confidence estimation in predictions
Would you like me to help implement any of these missing components? For example, we could:

Add a meta-learning update step
Implement a student memory system
Add prioritized experience replay
Include uncertainty estimation
