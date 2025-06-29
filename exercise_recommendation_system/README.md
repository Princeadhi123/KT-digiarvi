# Exercise Recommendation System

A reinforcement learning-based exercise recommendation system that personalizes exercise selection for students based on their learning history and performance.

## Features

- **Personalized Recommendations**: Provides tailored exercise suggestions for individual students
- **Deep Q-Learning**: Implements a DQN (Deep Q-Network) agent for optimal exercise selection
- **Performance Tracking**: Monitors student progress and adjusts recommendations accordingly
- **Interactive Mode**: Allows testing with specific student IDs for demonstration

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- pandas 1.3.0+
- scikit-learn 0.24.2+
- matplotlib 3.4.2+
- tqdm 4.62.0+

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd exercise_recommendation_system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Model**:
   ```bash
   python exercise_recommender.py
   ```
   This will train the DQN agent and save the model weights.

2. **Getting Recommendations**:
   The system will automatically display example recommendations for sample students after training.
   
   To get recommendations for a specific student (ID 1-545):
   ```
   Enter student ID (1-545) or 0 to exit: 42
   ```

## Model Architecture

The system uses a Deep Q-Network with the following components:
- **State Space**: 8-dimensional vector including past attempts, time spent, pass rate, and mean perception
- **Action Space**: Set of available exercises
- **Reward Function**: Based on exercise completion and success
- **Neural Network**: Fully connected network with ReLU activations

## Training Process

1. The agent interacts with the environment (student data)
2. Experiences are stored in a replay buffer
3. The model is trained using mini-batch gradient descent
4. Target network updates are performed periodically for stability

## Evaluation

The model is evaluated based on:
- Average reward per episode
- Pass rate on evaluation tasks
- Quality of recommendations

## Results

- Average evaluation reward: ~19.7
- Average pass rate: ~76%
- Training time: ~X minutes (depends on hardware)

## Files

- `exercise_recommender.py`: Main implementation of the DQN agent and environment
- `requirements.txt`: List of required Python packages
- `data/`: Directory containing student exercise data

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
