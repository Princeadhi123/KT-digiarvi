"""New streamlined main entry point for curriculum sequencing RL experiments."""

import argparse
import os
from pathlib import Path

from .experiment_runner import create_experiment_from_args


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
    parser.add_argument("--models", type=str, default="ql,dqn,a2c,a3c,ppo,sarl",
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
    
    # SARL parameters (Self-Adaptive DQN)
    sarl_group = parser.add_argument_group("SARL (Self-Adaptive DQN)")
    # Base DQN-like settings
    sarl_group.add_argument("--sarl_episodes", type=int, default=50,
                           help="SARL training episodes")
    sarl_group.add_argument("--sarl_lr", type=float, default=1e-3,
                           help="SARL learning rate")
    sarl_group.add_argument("--sarl_gamma", type=float, default=0.99,
                           help="SARL discount factor")
    sarl_group.add_argument("--sarl_batch_size", type=int, default=128,
                           help="SARL batch size")
    sarl_group.add_argument("--sarl_buffer_size", type=int, default=20000,
                           help="SARL replay buffer size")
    sarl_group.add_argument("--sarl_hidden_dim", type=int, default=128,
                           help="SARL hidden layer dimension")
    sarl_group.add_argument("--sarl_eps_start", type=float, default=1.0,
                           help="SARL initial epsilon")
    sarl_group.add_argument("--sarl_eps_end", type=float, default=0.05,
                           help="SARL final epsilon")
    sarl_group.add_argument("--sarl_eps_decay_steps", type=int, default=20000,
                           help="SARL epsilon decay steps")
    sarl_group.add_argument("--sarl_target_tau", type=float, default=0.01,
                           help="SARL target network soft update rate")
    sarl_group.add_argument("--sarl_target_update_interval", type=int, default=1,
                           help="SARL target network update interval")
    sarl_group.add_argument("--sarl_select_best", dest="sarl_select_best_on_val",
                           action="store_true", help="Enable SARL validation selection")
    sarl_group.add_argument("--sarl_val_episodes", type=int, default=300,
                           help="SARL validation episodes")
    # Optional: direct control of SARL eval episodes without touching config
    sarl_group.add_argument("--sarl_eval_episodes", type=int, default=None,
                           help="SARL evaluation episodes (overrides config if provided)")
    # Optional: per-episode step caps for SARL (None means use config/default)
    sarl_group.add_argument("--sarl_eval_max_steps_per_episode", type=int, default=None,
                           help="Max steps per evaluation episode for SARL")
    sarl_group.add_argument("--sarl_val_max_steps_per_episode", type=int, default=None,
                           help="Max steps per validation episode for SARL")
    sarl_group.add_argument("--sarl_train_max_steps_per_episode", type=int, default=None,
                           help="Max steps per training episode for SARL")
    parser.set_defaults(sarl_select_best_on_val=False)
    # Adaptation cadence
    sarl_group.add_argument("--sarl_adapt_interval", type=int, default=5,
                           help="Episodes between adaptation steps")
    # Hybrid contribution targets
    sarl_group.add_argument("--sarl_hybrid_base_target_share", type=float, default=0.7,
                           help="Target share for base contribution")
    sarl_group.add_argument("--sarl_hybrid_mastery_target_share", type=float, default=0.2,
                           help="Target share for mastery contribution")
    sarl_group.add_argument("--sarl_hybrid_motivation_target_share", type=float, default=0.1,
                           help="Target share for motivation contribution")
    # Hybrid weight update settings
    sarl_group.add_argument("--sarl_weight_lr", type=float, default=0.3,
                           help="Learning rate for hybrid weight updates")
    sarl_group.add_argument("--sarl_weight_min", type=float, default=0.0,
                           help="Minimum hybrid weight")
    sarl_group.add_argument("--sarl_weight_max", type=float, default=3.0,
                           help="Maximum hybrid weight")
    sarl_group.add_argument("--sarl_share_tolerance", type=float, default=0.05,
                           help="Tolerance for target share matching")
    # Epsilon adaptation
    sarl_group.add_argument("--sarl_invalid_rate_hi", type=float, default=0.15,
                           help="Invalid action rate threshold to increase epsilon")
    sarl_group.add_argument("--sarl_invalid_rate_lo", type=float, default=0.02,
                           help="Invalid action rate threshold to decrease epsilon")
    sarl_group.add_argument("--sarl_epsilon_up_step", type=float, default=0.05,
                           help="Increment to epsilon when invalid rate is high")
    sarl_group.add_argument("--sarl_epsilon_down_step", type=float, default=0.02,
                           help="Decrement to epsilon when invalid rate is low")
    sarl_group.add_argument("--sarl_eps_min", type=float, default=0.01,
                           help="Minimum epsilon bound for adaptation")
    sarl_group.add_argument("--sarl_eps_max", type=float, default=1.0,
                           help="Maximum epsilon bound for adaptation")
    # Optimizer LR adaptation
    sarl_group.add_argument("--sarl_lr_patience", type=int, default=5,
                           help="Episodes without improvement before reducing LR")
    sarl_group.add_argument("--sarl_lr_reduce_factor", type=float, default=0.5,
                           help="Factor to multiply LR by on plateau")
    sarl_group.add_argument("--sarl_min_lr", type=float, default=1e-5,
                           help="Minimum LR floor")
    # Curriculum/difficulty adaptation
    sarl_group.add_argument("--sarl_challenge_increase_threshold", type=float, default=0.85,
                           help="Normalized performance to increase difficulty")
    sarl_group.add_argument("--sarl_challenge_decrease_threshold", type=float, default=0.55,
                           help="Normalized performance to decrease difficulty")
    sarl_group.add_argument("--sarl_challenge_target_step", type=float, default=0.05,
                           help="Step size to change challenge target")
    sarl_group.add_argument("--sarl_challenge_target_min", type=float, default=0.4,
                           help="Minimum challenge target")
    sarl_group.add_argument("--sarl_challenge_target_max", type=float, default=0.9,
                           help="Maximum challenge target")
    sarl_group.add_argument("--sarl_challenge_band_step", type=float, default=-0.05,
                           help="Step size to change challenge band (negative narrows)")
    sarl_group.add_argument("--sarl_challenge_band_min", type=float, default=0.2,
                           help="Minimum challenge band")
    sarl_group.add_argument("--sarl_challenge_band_max", type=float, default=0.6,
                           help="Maximum challenge band")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        from .core import Config
        config = Config.from_file(args.config)
        # Override with any command line arguments
        # Prevent CLI defaults from overriding config-provided values
        # Neutralize env-level defaults that would override config
        args.seed = None
        args.data = None
        args.models = None
        args.include_chance = None
        args.include_trivial = None
        args.include_markov = None
        # Environment fields: ensure CLI defaults don't clobber file values
        args.reward_correct_w = None
        args.reward_score_w = None
        args.action_on = None
        # Shaping weights
        args.rew_improve_w = None
        args.rew_deficit_w = None
        args.rew_spacing_w = None
        args.rew_diversity_w = None
        args.rew_challenge_w = None
        # Shaping hyperparameters
        args.ema_alpha = None
        args.need_threshold = None
        args.spacing_window = None
        args.diversity_recent_k = None
        args.challenge_target = None
        args.challenge_band = None
        args.invalid_penalty = None
        # Initial hybrid weights
        args.hybrid_base_w = None
        args.hybrid_mastery_w = None
        args.hybrid_motivation_w = None
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
