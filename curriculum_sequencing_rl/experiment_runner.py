"""New streamlined experiment runner with improved architecture."""

import os
import csv
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

from .core import Config, setup_device, set_seed, TrainerFactory
from .environment import OptimizedInteractiveEnv, BaselinePolicies
from .evaluation import eval_policy_interactive_metrics, print_sample_rollouts


class ExperimentRunner:
    """Streamlined experiment runner with better separation of concerns."""
    
    def __init__(self, config: Config):
        self.config = config
        self.env: Optional[OptimizedInteractiveEnv] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment logging."""
        logger = logging.getLogger('ExperimentRunner')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def setup_environment(self) -> OptimizedInteractiveEnv:
        """Setup and return the environment."""
        if self.env is None:
            self.env = OptimizedInteractiveEnv(self.config.config.environment)
            self.logger.info(f"Environment created with {self.env.action_size} actions")
        return self.env
    
    def run_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Run baseline policies."""
        env = self.setup_environment()
        results = {}
        
        # Determine evaluation episodes and max steps for baselines from first configured model
        eval_episodes = None
        eval_max_steps_per_episode = None
        for m in self.config.config.models:
            if m in {'ql', 'dqn', 'a2c', 'a3c', 'ppo', 'sarl'}:
                try:
                    mc = self.config.get_model_config(m)
                    eval_episodes = mc.eval_episodes
                    # Optional cap to guarantee fast/hanging-proof evaluation
                    eval_max_steps_per_episode = getattr(mc, 'eval_max_steps_per_episode', None)
                    break
                except Exception:
                    pass
        if eval_episodes is None:
            eval_episodes = 300

        # Build only requested baselines to avoid heavy precomputation (e.g., Markov)
        baseline_policies = {}
        if self.config.config.include_chance:
            baseline_policies['Chance'] = BaselinePolicies.random_policy(
                env, self.config.config.environment.seed
            )
        if self.config.config.include_trivial:
            baseline_policies['TrivialSame'] = BaselinePolicies.trivial_same_policy(env)
        if self.config.config.include_markov:
            baseline_policies['Markov1-Train'] = BaselinePolicies.markov_policy(env)

        if not baseline_policies:
            return results

        for name, policy in baseline_policies.items():
            self.logger.info(f"Evaluating baseline: {name}")
            metrics = eval_policy_interactive_metrics(
                env, policy, mode="test", episodes=eval_episodes,
                max_steps_per_episode=eval_max_steps_per_episode
            )
            results[name] = metrics
            
            # Demo if requested
            if self.config.config.demo:
                print_sample_rollouts(
                    env, policy, mode=self.config.config.demo_mode,
                    episodes=self.config.config.demo_episodes,
                    max_steps=self.config.config.demo_steps,
                    model_name=name
                )
        
        return results
    
    def run_rl_models(self) -> Dict[str, Dict[str, Any]]:
        """Run RL model training and evaluation."""
        env = self.setup_environment()
        results = {}
        
        for model_name in self.config.config.models:
            if model_name not in TrainerFactory.list_available():
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            self.logger.info(f"Training model: {model_name}")
            
            # Get model config
            model_config = self.config.get_model_config(model_name)
            
            # Set up reproducibility
            set_seed(model_config.seed)
            
            # Create and train agent
            trainer = TrainerFactory.create(model_name, model_config)
            agent = trainer.train(env)
            
            # Evaluate agent
            policy = agent.get_policy(env)
            metrics = eval_policy_interactive_metrics(
                env, policy, mode="test", episodes=model_config.eval_episodes,
                max_steps_per_episode=getattr(model_config, 'eval_max_steps_per_episode', None)
            )
            results[model_name.upper()] = metrics
            
            # Demo if requested
            if self.config.config.demo:
                print_sample_rollouts(
                    env, policy, mode=self.config.config.demo_mode,
                    episodes=self.config.config.demo_episodes,
                    max_steps=self.config.config.demo_steps,
                    model_name=model_name.upper()
                )
            
            self.logger.info(f"Completed {model_name}: reward={metrics['reward']:.3f}")
        
        return results
    
    def run_experiment(self) -> Dict[str, Dict[str, Any]]:
        """Run complete experiment."""
        self.logger.info("Starting experiment")
        
        # Setup environment and reproducibility
        env = self.setup_environment()
        set_seed(self.config.config.environment.seed)
        
        # Run baselines
        baseline_results = self.run_baselines()
        self.results.update(baseline_results)
        
        # Run RL models
        rl_results = self.run_rl_models()
        self.results.update(rl_results)
        
        # Print summary
        self._print_results_summary()
        
        # Save results if requested
        if self.config.config.metrics_csv:
            self._save_results_to_csv()
        
        self.logger.info("Experiment completed")
        return self.results
    
    def _print_results_summary(self) -> None:
        """Print formatted results summary."""
        print("\n=== Test Metrics (interactive: avg_reward (shaped), VPR, regret) ===")
        
        for name, metrics in self.results.items():
            reward = metrics.get('reward', 0.0)
            reward_base = metrics.get('reward_base', float('nan'))
            reward_norm = metrics.get('reward_norm', float('nan'))
            vpr = metrics.get('vpr', float('nan'))
            regret = metrics.get('regret', float('nan'))
            regret_ratio = metrics.get('regret_ratio', float('nan'))
            
            # Hybrid contributions
            base_contrib = metrics.get('reward_base_contrib', float('nan'))
            mastery = metrics.get('reward_mastery', float('nan'))
            motivation = metrics.get('reward_motivation', float('nan'))
            
            print(
                f"{name:<12}: avg_reward={reward:.3f}  base={reward_base:.3f}  "
                f"norm={reward_norm:.3f}  vpr={vpr:.3f}  regret={regret:.3f}  "
                f"regret_ratio={regret_ratio:.3f}  |  "
                f"hyb_base={base_contrib:.3f}  hyb_mastery={mastery:.3f}  "
                f"hyb_motiv={motivation:.3f}  |  "
                f"base%={reward_base*100:.1f}  norm%={reward_norm*100:.1f}  "
                f"vpr%={vpr*100:.1f}  regret%={regret_ratio*100:.1f}"
            )
    
    def _save_results_to_csv(self) -> None:
        """Save results to CSV file."""
        csv_path = self.config.config.metrics_csv
        
        # Define comprehensive fieldnames
        fieldnames = [
            "timestamp", "model", "reward", "vpr", "vpr_pct", "regret", 
            "regret_ratio", "regret_ratio_pct", "seed", "env_type",
            "reward_base", "reward_shaping", "reward_norm", 
            "reward_base_pct", "reward_norm_pct",
            "reward_base_contrib", "reward_mastery", "reward_motivation",
            "reward_base_contrib_pct", "reward_mastery_pct", "reward_motivation_pct",
            "term_improve", "term_deficit", "term_spacing", "term_diversity", "term_challenge"
        ]
        
        # Add environment and model config fields
        env_config = self.config.config.environment
        fieldnames.extend([
            "reward_correct_w", "reward_score_w", "hybrid_base_w", 
            "hybrid_mastery_w", "hybrid_motivation_w",
            "rew_improve_w", "rew_deficit_w", "rew_spacing_w", 
            "rew_diversity_w", "rew_challenge_w",
            "ema_alpha", "need_threshold", "spacing_window", 
            "diversity_recent_k", "challenge_target", "challenge_band", "invalid_penalty"
        ])
        
        # Check if file exists and has header
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            for model_name, metrics in self.results.items():
                row = {
                    "timestamp": timestamp,
                    "model": model_name,
                    "seed": env_config.seed,
                    "env_type": "interactive"
                }
                
                # Add metrics
                for key, value in metrics.items():
                    row[key] = value
                
                # Add percentage versions
                for base_key in ['vpr', 'regret_ratio', 'reward_base', 'reward_norm']:
                    if base_key in metrics and not np.isnan(metrics[base_key]):
                        row[f"{base_key}_pct"] = metrics[base_key] * 100.0
                
                # Add hybrid contribution percentages
                for key in ['reward_base_contrib', 'reward_mastery', 'reward_motivation']:
                    if key in metrics and not np.isnan(metrics[key]):
                        row[f"{key}_pct"] = metrics[key] * 100.0
                
                # Add environment config
                for attr in ['reward_correct_w', 'reward_score_w', 'hybrid_base_w',
                           'hybrid_mastery_w', 'hybrid_motivation_w', 'rew_improve_w',
                           'rew_deficit_w', 'rew_spacing_w', 'rew_diversity_w', 
                           'rew_challenge_w', 'ema_alpha', 'need_threshold',
                           'spacing_window', 'diversity_recent_k', 'challenge_target',
                           'challenge_band', 'invalid_penalty']:
                    row[attr] = getattr(env_config, attr)
                
                writer.writerow(row)
        
        self.logger.info(f"Results saved to {csv_path}")


def create_experiment_from_args(args, config: Optional[Config] = None) -> ExperimentRunner:
    """Create experiment runner from command line arguments."""
    if config is None:
        # Create default config
        from .core.config import ExperimentConfig, EnvironmentConfig
        
        env_config = EnvironmentConfig(data_path=args.data)
        exp_config = ExperimentConfig(environment=env_config)
        config = Config(exp_config)
        
        # Update config from args
        config.update_from_args(args)
    
    return ExperimentRunner(config)
