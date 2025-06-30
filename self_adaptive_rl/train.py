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

def train():
    # Set random seeds for reproducibility
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)
    
    # Create environment
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessed_kt_data.csv")
    env = StudentLearningEnv(data_path, curriculum_config)
    
    # Initialize agent
    agent = SelfAdaptiveAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=rl_config,
        curriculum_config=curriculum_config,
        self_adaptive_config=self_adaptive_config
    )
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(experiment_config.model_dir, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    difficulties = []
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for episode in tqdm(range(rl_config.total_timesteps // 1000)):  # Adjust based on your needs
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.memory.push(
                state, action, next_state, 
                float(reward), float(done)
            )
            
            # Store in meta buffer for adaptation
            if hasattr(agent, 'meta_buffer'):
                agent.meta_buffer.push(
                    state, action, next_state, 
                    float(reward), float(done)
                )
            
            # Update agent
            loss = agent.update()
            
            # Update curriculum
            agent.update_curriculum(info['success'])
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            agent.steps_done += 1
            
            # Update exploration rate
            agent.update_epsilon()
            
            # Periodic updates
            if agent.steps_done % agent.self_adaptive_config.meta_update_freq == 0:
                agent.adapt(agent.meta_buffer.buffer)
            
            # Logging
            if agent.steps_done % rl_config.log_interval == 0:
                print(f"\nStep: {agent.steps_done}")
                print(f"Episode: {episode}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(f"Current Difficulty: {agent.current_difficulty:.4f}")
                if len(episode_rewards) > 0:
                    print(f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
                
                # Save model checkpoint
                checkpoint_path = os.path.join(save_dir, f"checkpoint_{agent.steps_done}.pt")
                agent.save(checkpoint_path)
        
        # Update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate success rate for the episode
        if hasattr(env, 'performance_history') and len(env.performance_history) > 0:
            success_rate = np.mean(env.performance_history)
            success_rates.append(success_rate)
            difficulties.append(agent.current_difficulty)
        
        # Print episode summary
        if episode % 10 == 0:
            print(f"\n--- Episode {episode} ---")
            print(f"Steps: {agent.steps_done}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Length: {episode_length}")
            print(f"Success Rate: {success_rate:.2f}%" if 'success_rate' in locals() else "")
            print(f"Current Difficulty: {agent.current_difficulty:.4f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training metrics
    plot_training_metrics(
        episode_rewards, 
        success_rates, 
        difficulties,
        save_dir=save_dir
    )
    
    return agent, env

def plot_training_metrics(episode_rewards, success_rates, difficulties, save_dir):
    """Plot training metrics and save to file."""
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "plots", "episode_rewards.png"))
    plt.close()
    
    # Plot success rates
    if success_rates:
        plt.figure(figsize=(12, 6))
        plt.plot(success_rates)
        plt.title("Success Rate Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "plots", "success_rates.png"))
        plt.close()
    
    # Plot difficulty progression
    if difficulties:
        plt.figure(figsize=(12, 6))
        plt.plot(difficulties)
        plt.title("Curriculum Difficulty Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Difficulty Level")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "plots", "difficulty_progression.png"))
        plt.close()

if __name__ == "__main__":
    train()
