# RL-Based Exercise Recommendation System

This project implements a Reinforcement Learning (RL) based exercise recommendation system that suggests the next exercise to maximize student learning outcomes. The system uses Deep Q-Learning to learn optimal exercise sequences based on student performance data.

## Features

- **State Representation**: Captures student's learning state using:
  - Past 5 exercise attempts (pass/fail)
  - Average time spent on exercises
  - Current pass rate
  - Student's mean perception score

- **Action Space**: Recommends the next exercise from available exercises

- **Reward Function**:
  - +1 for correct answers
  - 0 for incorrect answers
  - Penalties for recommending already-completed exercises

## Project Structure

```
exercise_recommendation_system/
├── exercise_recommender.py  # Main implementation
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Place your preprocessed exercise data file at `c:\Users\pdaadh\Desktop\KT digiarvi\preprocessed_kt_data.csv`
   - The expected CSV format should include columns: student_id, exercise_id, category, order, score, pass_status, grade, mean_perception, time_spent

## Usage

### Training the Model

To train the RL agent:

```bash
python exercise_recommender.py
```

This will:
1. Load and preprocess the exercise data
2. Train a Deep Q-Network (DQN) agent
3. Save the trained model in the `models/` directory
4. Generate training performance plots

### Model Outputs

- **Trained Models**: Saved in the `models/` directory
- **Training Plots**:
  - `training_results.png`: Shows reward and pass rate over training episodes

## How It Works

1. **Environment**:
   - The `ExerciseEnvironment` class simulates the student learning environment
   - Tracks student state and provides rewards based on exercise outcomes

2. **RL Agent**:
   - Implements a Deep Q-Network (DQN) with experience replay
   - Uses epsilon-greedy exploration strategy
   - Updates target network periodically for stable training

3. **State Representation**:
   - Normalized features for consistent learning
   - Tracks both short-term (last 5 attempts) and long-term (pass rate) performance

## Customization

You can modify the following parameters in `exercise_recommender.py`:

- Training parameters (episodes, batch size, etc.)
- Neural network architecture
- Reward function
- State representation

## Evaluation

The system includes an evaluation function that tests the trained model on unseen student data and reports:
- Average reward per episode
- Overall pass rate

## Future Improvements

- Add more sophisticated state representations
- Implement curriculum learning
- Add student clustering for personalized recommendations
- Include exercise difficulty and concept mapping
- Add more detailed evaluation metrics
