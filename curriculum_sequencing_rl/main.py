"""New streamlined main entry point for curriculum sequencing RL experiments."""

import argparse
import os
from pathlib import Path

from experiment_runner import create_experiment_from_args


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="Curriculum Sequencing RL Experiments - Refactored Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file support
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file (JSON/YAML)")
    
    # Data and basic settings
    here = Path(__file__).parent
    default_data = here.parent / "preprocessed_kt_data.csv"
    
    parser.add_argument("--data", type=str, default=str(default_data),
                       help="Path to preprocessed CSV data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--models", type=str, default="ql,dqn,a2c,a3c,ppo",
                       help="Comma-separated models to run")
    
    # Environment configuration
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--reward_correct_w", type=float, default=0.0,
                          help="Weight for correctness reward")
    env_group.add_argument("--reward_score_w", type=float, default=1.0,
                          help="Weight for score reward")
    env_group.add_argument("--action_on", type=str, default="category",
                          choices=["category", "category_group"],
                          help="Action space basis")
    
    # Multi-objective shaping
    shaping_group = parser.add_argument_group("Reward Shaping")
    shaping_group.add_argument("--rew_improve_w", type=float, default=0.0,
                              help="Weight for improvement shaping")
    shaping_group.add_argument("--rew_deficit_w", type=float, default=0.0,
                              help="Weight for deficit shaping")
    shaping_group.add_argument("--rew_spacing_w", type=float, default=0.0,
                              help="Weight for spacing shaping")
    shaping_group.add_argument("--rew_diversity_w", type=float, default=0.0,
                              help="Weight for diversity shaping")
    shaping_group.add_argument("--rew_challenge_w", type=float, default=0.0,
                              help="Weight for challenge shaping")
    
    # Shaping hyperparameters
    shaping_group.add_argument("--ema_alpha", type=float, default=0.3,
                              help="EMA alpha for improvement baseline")
    shaping_group.add_argument("--need_threshold", type=float, default=0.6,
                              help="Mastery threshold for deficit computation")
    shaping_group.add_argument("--spacing_window", type=int, default=5,
                              help="Window for spacing normalization")
    shaping_group.add_argument("--diversity_recent_k", type=int, default=5,
                              help="Recent choices window for diversity")
    shaping_group.add_argument("--challenge_target", type=float, default=0.7,
                              help="Target score for challenge proximity")
    shaping_group.add_argument("--challenge_band", type=float, default=0.4,
                              help="Bandwidth around challenge target")
    shaping_group.add_argument("--invalid_penalty", type=float, default=0.0,
                              help="Penalty for invalid actions")
    
    # Hybrid weights
    hybrid_group = parser.add_argument_group("Hybrid Weights")
    hybrid_group.add_argument("--hybrid_base_w", type=float, default=1.0,
                             help="Hybrid weight for base reward group")
    hybrid_group.add_argument("--hybrid_mastery_w", type=float, default=1.0,
                             help="Hybrid weight for mastery group")
    hybrid_group.add_argument("--hybrid_motivation_w", type=float, default=1.0,
                             help="Hybrid weight for motivation group")
    
    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval_episodes", type=int, default=300,
                           help="Episodes for evaluation")
    eval_group.add_argument("--no_chance", dest="include_chance", action="store_false",
                           help="Disable chance baseline")
    eval_group.add_argument("--no_trivial", dest="include_trivial", action="store_false",
                           help="Disable trivial baseline")
    eval_group.add_argument("--no_markov", dest="include_markov", action="store_false",
                           help="Disable Markov baseline")
    parser.set_defaults(include_chance=True, include_trivial=True, include_markov=True)
    
    # Output settings
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--metrics_csv", type=str, default=None,
                             help="CSV file to save metrics")
    output_group.add_argument("--demo", action="store_true",
                             help="Print sample rollouts")
    output_group.add_argument("--demo_episodes", type=int, default=1,
                             help="Demo episodes per model")
    output_group.add_argument("--demo_steps", type=int, default=12,
                             help="Max steps per demo episode")
    output_group.add_argument("--demo_mode", type=str, default="test",
                             choices=["train", "val", "test"],
                             help="Split for demo rollouts")
    
    # Q-Learning parameters
    ql_group = parser.add_argument_group("Q-Learning")
    ql_group.add_argument("--ql_epochs", type=int, default=5,
                         help="Q-Learning training epochs")
    ql_group.add_argument("--ql_alpha", type=float, default=0.2,
                         help="Q-Learning learning rate")
    ql_group.add_argument("--ql_gamma", type=float, default=0.9,
                         help="Q-Learning discount factor")
    ql_group.add_argument("--ql_eps_start", type=float, default=0.3,
                         help="Q-Learning initial epsilon")
    ql_group.add_argument("--ql_eps_end", type=float, default=0.0,
                         help="Q-Learning final epsilon")
    ql_group.add_argument("--ql_eps_decay_epochs", type=int, default=3,
                         help="Q-Learning epsilon decay epochs")
    ql_group.add_argument("--ql_select_best", dest="ql_select_best_on_val", 
                         action="store_true", help="Enable Q-Learning validation selection")
    ql_group.add_argument("--ql_val_episodes", type=int, default=300,
                         help="Q-Learning validation episodes")
    parser.set_defaults(ql_select_best_on_val=False)
    
    # DQN parameters
    dqn_group = parser.add_argument_group("DQN")
    dqn_group.add_argument("--dqn_episodes", type=int, default=50,
                          help="DQN training episodes")
    dqn_group.add_argument("--dqn_lr", type=float, default=1e-3,
                          help="DQN learning rate")
    dqn_group.add_argument("--dqn_gamma", type=float, default=0.99,
                          help="DQN discount factor")
    dqn_group.add_argument("--dqn_batch_size", type=int, default=128,
                          help="DQN batch size")
    dqn_group.add_argument("--dqn_buffer_size", type=int, default=20000,
                          help="DQN replay buffer size")
    dqn_group.add_argument("--dqn_hidden_dim", type=int, default=128,
                          help="DQN hidden layer dimension")
    dqn_group.add_argument("--dqn_eps_start", type=float, default=1.0,
                          help="DQN initial epsilon")
    dqn_group.add_argument("--dqn_eps_end", type=float, default=0.05,
                          help="DQN final epsilon")
    dqn_group.add_argument("--dqn_eps_decay_steps", type=int, default=20000,
                          help="DQN epsilon decay steps")
    dqn_group.add_argument("--dqn_target_tau", type=float, default=0.01,
                          help="DQN target network soft update rate")
    dqn_group.add_argument("--dqn_target_update_interval", type=int, default=1,
                          help="DQN target network update interval")
    dqn_group.add_argument("--dqn_select_best", dest="dqn_select_best_on_val",
                          action="store_true", help="Enable DQN validation selection")
    dqn_group.add_argument("--dqn_val_episodes", type=int, default=300,
                          help="DQN validation episodes")
    parser.set_defaults(dqn_select_best_on_val=False)
    
    # A2C parameters
    a2c_group = parser.add_argument_group("A2C")
    a2c_group.add_argument("--a2c_episodes", type=int, default=50,
                          help="A2C training episodes")
    a2c_group.add_argument("--a2c_lr", type=float, default=1e-3,
                          help="A2C learning rate")
    a2c_group.add_argument("--a2c_entropy", type=float, default=0.01,
                          help="A2C entropy coefficient")
    a2c_group.add_argument("--a2c_value_coef", type=float, default=0.5,
                          help="A2C value loss coefficient")
    a2c_group.add_argument("--a2c_bc_warmup", type=int, default=1,
                          help="A2C behavior cloning warmup epochs")
    a2c_group.add_argument("--a2c_bc_weight", type=float, default=0.5,
                          help="A2C behavior cloning weight")
    a2c_group.add_argument("--a2c_batch_episodes", type=int, default=4,
                          help="A2C batch episodes")
    
    # A3C parameters
    a3c_group = parser.add_argument_group("A3C")
    a3c_group.add_argument("--a3c_episodes", type=int, default=50,
                          help="A3C training episodes")
    a3c_group.add_argument("--a3c_lr", type=float, default=1e-3,
                          help="A3C learning rate")
    a3c_group.add_argument("--a3c_entropy", type=float, default=0.01,
                          help="A3C entropy coefficient")
    a3c_group.add_argument("--a3c_value_coef", type=float, default=0.5,
                          help="A3C value loss coefficient")
    a3c_group.add_argument("--a3c_gae_lambda", type=float, default=0.95,
                          help="A3C GAE lambda")
    a3c_group.add_argument("--a3c_bc_warmup", type=int, default=1,
                          help="A3C behavior cloning warmup epochs")
    a3c_group.add_argument("--a3c_bc_weight", type=float, default=0.5,
                          help="A3C behavior cloning weight")
    a3c_group.add_argument("--a3c_rollouts", type=int, default=4,
                          help="A3C rollouts per update")
    
    # PPO parameters
    ppo_group = parser.add_argument_group("PPO")
    ppo_group.add_argument("--ppo_episodes", type=int, default=50,
                          help="PPO training episodes")
    ppo_group.add_argument("--ppo_lr", type=float, default=3e-4,
                          help="PPO learning rate")
    ppo_group.add_argument("--ppo_clip_eps", type=float, default=0.2,
                          help="PPO clipping parameter")
    ppo_group.add_argument("--ppo_epochs", type=int, default=4,
                          help="PPO optimization epochs per batch")
    ppo_group.add_argument("--ppo_batch_episodes", type=int, default=8,
                          help="PPO batch episodes")
    ppo_group.add_argument("--ppo_minibatch_size", type=int, default=2048,
                          help="PPO minibatch size")
    ppo_group.add_argument("--ppo_entropy", type=float, default=0.01,
                          help="PPO entropy coefficient")
    ppo_group.add_argument("--ppo_value_coef", type=float, default=0.5,
                          help="PPO value loss coefficient")
    ppo_group.add_argument("--ppo_gae_lambda", type=float, default=0.95,
                          help="PPO GAE lambda")
    ppo_group.add_argument("--ppo_bc_warmup", type=int, default=2,
                          help="PPO behavior cloning warmup epochs")
    ppo_group.add_argument("--ppo_bc_weight", type=float, default=1.0,
                          help="PPO behavior cloning weight")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        from core import Config
        config = Config.from_file(args.config)
        # Override with any command line arguments
        config.update_from_args(args)
        experiment = create_experiment_from_args(args, config)
    else:
        # Parse models list
        args.models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
        # Create experiment from args only
        experiment = create_experiment_from_args(args)
    
    results = experiment.run_experiment()
    return results


if __name__ == "__main__":
    main()
