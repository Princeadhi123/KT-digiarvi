import os
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Import configurations and components
from config.config import rl_config, curriculum_config, self_adaptive_config, experiment_config
from models.self_adaptive_agent import SelfAdaptiveAgent
from utils.student_env import StudentLearningEnv

def evaluate(agent, env, num_episodes=10):
    """Evaluate the agent on the given environment."""
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(experiment_config.device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

def train():
    # Set random seeds for reproducibility
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)
    
    # Create environments
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessed_kt_data.csv")
    
    # Training environment
    train_env = StudentLearningEnv(
        data_path, 
        curriculum_config, 
        split='train',
        seed=experiment_config.seed
    )
    
    # Validation environment
    val_env = StudentLearningEnv(
        data_path,
        curriculum_config,
        split='val',
        seed=experiment_config.seed + 1  # Different seed for validation
    )
    
    # Initialize agent
    agent = SelfAdaptiveAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        config=rl_config,
        curriculum_config=curriculum_config,
        self_adaptive_config=self_adaptive_config
    )
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(experiment_config.model_dir, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    train_rewards = []
    val_rewards = []
    best_val_reward = -np.inf
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    # Calculate total episodes based on timesteps
    total_episodes = rl_config.total_timesteps // 1000
    
    for episode in tqdm(range(total_episodes)):
        # Training phase
        state = train_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = train_env.step(action)
            
            # Store transition in replay buffer
            agent.memory.push(
                state, action, next_state, 
                float(reward), float(done)
            )
            
            # Store in meta buffer for adaptation if it exists
            if hasattr(agent, 'meta_buffer'):
                agent.meta_buffer.push(
                    state, action, next_state, 
                    float(reward), float(done)
                )
            
            # Update agent
            loss = agent.update()
            
            # Update curriculum if success info is available
            if 'success' in info:
                agent.update_curriculum(info['success'])
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            agent.steps_done += 1
            
            # Update exploration rate
            agent.update_epsilon()
            
            # Periodic meta-updates
            if hasattr(agent, 'meta_buffer') and agent.steps_done % agent.self_adaptive_config.meta_update_freq == 0:
                agent.adapt(agent.meta_buffer.buffer)
        
        # Track training rewards
        train_rewards.append(episode_reward)
        
        # Validation phase (every 10 episodes)
        if episode % 10 == 0:
            val_reward = evaluate(agent, val_env)
            val_rewards.append((episode, val_reward))
            
            # Save best model based on validation reward
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_model_path = os.path.join(save_dir, f"best_model_ep{episode}_reward{val_reward:.2f}.pt")
                agent.save(best_model_path)
                print(f"\nNew best model saved with validation reward: {val_reward:.2f}")
            
            # Print training progress
            print(f"\n--- Episode {episode}/{total_episodes} ---")
            print(f"Train Reward: {episode_reward:.2f}")
            print(f"Val Reward: {val_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Current Difficulty: {getattr(agent, 'current_difficulty', 0):.4f}")
            
            # Plot training progress
            plot_training_progress(train_rewards, val_rewards, save_dir)
    
    # Training complete
    training_time = (time.time() - start_time) / 60  # in minutes
    print(f"\nTraining completed in {training_time:.2f} minutes")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Final evaluation on test set
    print("\nRunning final evaluation on test set...")
    test_env = StudentLearningEnv(
        data_path,
        curriculum_config,
        split='test',
        seed=experiment_config.seed + 2  # Different seed for test
    )
    test_reward = evaluate(agent, test_env, num_episodes=20)
    print(f"Test Reward: {test_reward:.2f}")
    
    # Save test results
    with open(os.path.join(save_dir, "test_results.txt"), 'w') as f:
        f.write(f"Test Reward: {test_reward:.2f}\n")
        f.write(f"Training Time: {training_time:.2f} minutes\n")
    
    return agent, train_env

def plot_training_progress(train_rewards, val_rewards, save_dir):
    """Plot training and validation rewards over time."""
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot training rewards
    ax1.plot(train_rewards, label='Training Reward')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot validation rewards if available
    if val_rewards:
        episodes, vals = zip(*val_rewards)
        ax2.plot(episodes, vals, 'r-', label='Validation Reward')
        ax2.set_title('Validation Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "training_progress.png"))
    plt.close()

if __name__ == "__main__":
    train()
