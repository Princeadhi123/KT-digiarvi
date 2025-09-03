"""New streamlined experiment runner with improved architecture."""

import os
import csv
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np
from copy import deepcopy

from .core import Config, setup_device, set_seed, TrainerFactory
from .environment import OptimizedInteractiveEnv, BaselinePolicies
from .evaluation import eval_policy_interactive_metrics, print_sample_rollouts
from . import agents  # noqa: F401 - ensure trainer registration via import side-effects


class ExperimentRunner:
    """Streamlined experiment runner with better separation of concerns."""
    
    def __init__(self, config: Config):
        self.config = config
        self.env: Optional[OptimizedInteractiveEnv] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.run_records: List[Dict[str, Any]] = []
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
        speed_threshold_norm = None
        for m in self.config.config.models:
            if m in {'ql', 'dqn', 'a2c', 'a3c', 'ppo', 'sarl'}:
                try:
                    mc = self.config.get_model_config(m)
                    eval_episodes = mc.eval_episodes
                    # Optional cap to guarantee fast/hanging-proof evaluation
                    eval_max_steps_per_episode = getattr(mc, 'eval_max_steps_per_episode', None)
                    speed_threshold_norm = getattr(mc, 'speed_threshold_norm', None)
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
                max_steps_per_episode=eval_max_steps_per_episode,
                speed_threshold_norm=speed_threshold_norm
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
        # Ensure an environment exists for initial information/logging
        _ = self.setup_environment()
        results = {}
        # Helpful debug: list all trainers that are currently registered
        self.logger.info("Available trainers: %s", TrainerFactory.list_available())
        
        for model_name in self.config.config.models:
            if model_name not in TrainerFactory.list_available():
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            self.logger.info(f"Training model: {model_name}")
            
            # Get model config
            model_config = self.config.get_model_config(model_name)
            
            # Determine seeds to run
            seeds: List[int] = (
                list(self.config.config.seeds)
                if (self.config.config.seeds is not None)
                else [int(getattr(model_config, 'seed', 42))]
            )

            model_uc = model_name.upper()
            ep_means_across_seeds: List[float] = []
            speed_means_across_seeds: List[float] = []
            speed_success_across_seeds: List[float] = []
            scal_ep_ratio_across_seeds: List[float] = []  # small/base per-seed
            adapt_ep_ratio_across_seeds: List[float] = []  # post/pre per-seed
            scal_reward_ratio_across_seeds: List[float] = []  # small/base reward
            adapt_reward_ratio_across_seeds: List[float] = []  # post/pre reward

            for seed in seeds:
                # Set up reproducibility per seed
                set_seed(seed)

                # Fresh base environment per seed
                base_env_cfg = deepcopy(self.config.config.environment)
                base_env_cfg.seed = int(seed)
                base_env = OptimizedInteractiveEnv(base_env_cfg)

                # Train agent on base env
                trainer = TrainerFactory.create(model_name, model_config)
                agent = trainer.train(base_env)

                # Evaluate on base
                policy_base = agent.get_policy(base_env)
                base_metrics = eval_policy_interactive_metrics(
                    base_env, policy_base, mode="test", episodes=model_config.eval_episodes,
                    max_steps_per_episode=getattr(model_config, 'eval_max_steps_per_episode', None),
                    speed_threshold_norm=getattr(model_config, 'speed_threshold_norm', None)
                )
                base_metrics['seed'] = int(seed)
                base_metrics['variant'] = 'base'
                base_metrics['student_fraction'] = float(getattr(base_env_cfg, 'student_fraction', 1.0))
                results[f"{model_uc}__seed{seed}__base"] = base_metrics
                self.run_records.append({**base_metrics, 'model': model_uc, 'variant': 'base'})

                # Track episodic accuracy per seed
                ep_mean = float(base_metrics.get('ep_return_mean', np.nan))
                ep_means_across_seeds.append(ep_mean)
                # Track speed per seed (mean steps to threshold and success rate)
                try:
                    speed_means_across_seeds.append(float(base_metrics.get('speed_steps_to_threshold_mean', np.nan)))
                except Exception:
                    speed_means_across_seeds.append(float('nan'))
                try:
                    speed_success_across_seeds.append(float(base_metrics.get('speed_success_rate', np.nan)))
                except Exception:
                    speed_success_across_seeds.append(float('nan'))

                # Demo if requested
                if self.config.config.demo:
                    print_sample_rollouts(
                        base_env, policy_base, mode=self.config.config.demo_mode,
                        episodes=self.config.config.demo_episodes,
                        max_steps=self.config.config.demo_steps,
                        model_name=f"{model_uc}-seed{seed}-BASE"
                    )

                # Scalability: evaluate on smaller environment and compute ratios
                scalability_reward_ratio = float('nan')
                scalability_ep_return_ratio = float('nan')
                if bool(getattr(self.config.config, 'evaluate_scalability', False)):
                    small_env_cfg = deepcopy(base_env_cfg)
                    small_env_cfg.student_fraction = float(getattr(self.config.config, 'scalability_small_fraction', 0.5))
                    try:
                        small_env = OptimizedInteractiveEnv(small_env_cfg)
                        if (small_env.action_size == base_env.action_size) and (small_env.state_dim == base_env.state_dim):
                            policy_small = agent.get_policy(small_env)
                            small_metrics = eval_policy_interactive_metrics(
                                small_env, policy_small, mode="test", episodes=model_config.eval_episodes,
                                max_steps_per_episode=getattr(model_config, 'eval_max_steps_per_episode', None),
                                speed_threshold_norm=getattr(model_config, 'speed_threshold_norm', None)
                            )
                            small_metrics['seed'] = int(seed)
                            small_metrics['variant'] = 'scalability_small'
                            small_metrics['student_fraction'] = float(small_env_cfg.student_fraction)
                            results[f"{model_uc}__seed{seed}__scalesmall"] = small_metrics
                            self.run_records.append({**small_metrics, 'model': model_uc, 'variant': 'scalability_small'})

                            # Ratios small/base
                            b_r = float(base_metrics.get('reward', np.nan))
                            s_r = float(small_metrics.get('reward', np.nan))
                            if (not np.isnan(b_r)) and (abs(b_r) > 1e-12) and (not np.isnan(s_r)):
                                scalability_reward_ratio = s_r / b_r
                            b_e = float(base_metrics.get('ep_return_mean', np.nan))
                            s_e = float(small_metrics.get('ep_return_mean', np.nan))
                            if (not np.isnan(b_e)) and (abs(b_e) > 1e-12) and (not np.isnan(s_e)):
                                scalability_ep_return_ratio = s_e / b_e
                        else:
                            self.logger.warning(
                                "Skipping scalability eval for %s seed=%s due to action/state dim mismatch (base_a=%d, small_a=%d, base_s=%d, small_s=%d)",
                                model_uc, str(seed), base_env.action_size, small_env.action_size, base_env.state_dim, small_env.state_dim
                            )
                    except Exception as e:
                        self.logger.warning("Scalability evaluation failed for %s seed=%s: %s", model_uc, str(seed), str(e))

                base_metrics['scalability_reward_ratio'] = scalability_reward_ratio
                base_metrics['scalability_ep_return_ratio'] = scalability_ep_return_ratio
                # Track scalability ratio per seed (small/base)
                try:
                    scal_ep_ratio_across_seeds.append(float(base_metrics.get('scalability_ep_return_ratio', np.nan)))
                except Exception:
                    scal_ep_ratio_across_seeds.append(float('nan'))
                try:
                    scal_reward_ratio_across_seeds.append(float(base_metrics.get('scalability_reward_ratio', np.nan)))
                except Exception:
                    scal_reward_ratio_across_seeds.append(float('nan'))

                # Adaptability: evaluate pre vs post environment change
                adaptability_reward_ratio = float('nan')
                adaptability_ep_return_ratio = float('nan')
                if bool(getattr(self.config.config, 'evaluate_adaptability', False)):
                    pre_eps = int(getattr(self.config.config, 'adapt_pre_episodes', model_config.eval_episodes))
                    pre_metrics = eval_policy_interactive_metrics(
                        base_env, policy_base, mode="test", episodes=pre_eps,
                        max_steps_per_episode=getattr(model_config, 'eval_max_steps_per_episode', None),
                        speed_threshold_norm=getattr(model_config, 'speed_threshold_norm', None)
                    )
                    pre_metrics['seed'] = int(seed)
                    pre_metrics['variant'] = 'adapt_pre'
                    pre_metrics['student_fraction'] = float(getattr(base_env_cfg, 'student_fraction', 1.0))
                    results[f"{model_uc}__seed{seed}__adaptpre"] = pre_metrics
                    self.run_records.append({**pre_metrics, 'model': model_uc, 'variant': 'adapt_pre'})

                    post_env_cfg = deepcopy(base_env_cfg)
                    if getattr(self.config.config, 'adapt_post_challenge_target', None) is not None:
                        post_env_cfg.challenge_target = float(getattr(self.config.config, 'adapt_post_challenge_target'))
                    if getattr(self.config.config, 'adapt_post_challenge_band', None) is not None:
                        post_env_cfg.challenge_band = float(getattr(self.config.config, 'adapt_post_challenge_band'))
                    try:
                        post_env = OptimizedInteractiveEnv(post_env_cfg)
                        policy_post = agent.get_policy(post_env)
                        post_eps = int(getattr(self.config.config, 'adapt_post_episodes', model_config.eval_episodes))
                        post_metrics = eval_policy_interactive_metrics(
                            post_env, policy_post, mode="test", episodes=post_eps,
                            max_steps_per_episode=getattr(model_config, 'eval_max_steps_per_episode', None),
                            speed_threshold_norm=getattr(model_config, 'speed_threshold_norm', None)
                        )
                        post_metrics['seed'] = int(seed)
                        post_metrics['variant'] = 'adapt_post'
                        post_metrics['student_fraction'] = float(getattr(post_env_cfg, 'student_fraction', 1.0))
                        results[f"{model_uc}__seed{seed}__adaptpost"] = post_metrics
                        self.run_records.append({**post_metrics, 'model': model_uc, 'variant': 'adapt_post'})

                        # Ratios post/pre
                        p_r = float(pre_metrics.get('reward', np.nan))
                        q_r = float(post_metrics.get('reward', np.nan))
                        if (not np.isnan(p_r)) and (abs(p_r) > 1e-12) and (not np.isnan(q_r)):
                            adaptability_reward_ratio = q_r / p_r
                        p_e = float(pre_metrics.get('ep_return_mean', np.nan))
                        q_e = float(post_metrics.get('ep_return_mean', np.nan))
                        if (not np.isnan(p_e)) and (abs(p_e) > 1e-12) and (not np.isnan(q_e)):
                            adaptability_ep_return_ratio = q_e / p_e
                    except Exception as e:
                        self.logger.warning("Adaptability evaluation failed for %s seed=%s: %s", model_uc, str(seed), str(e))

                base_metrics['adaptability_reward_ratio'] = adaptability_reward_ratio
                base_metrics['adaptability_ep_return_ratio'] = adaptability_ep_return_ratio
                # Track adaptability ratio per seed (post/pre)
                try:
                    adapt_ep_ratio_across_seeds.append(float(base_metrics.get('adaptability_ep_return_ratio', np.nan)))
                except Exception:
                    adapt_ep_ratio_across_seeds.append(float('nan'))
                try:
                    adapt_reward_ratio_across_seeds.append(float(base_metrics.get('adaptability_reward_ratio', np.nan)))
                except Exception:
                    adapt_reward_ratio_across_seeds.append(float('nan'))

                self.logger.info(
                    f"Completed {model_name} seed={seed}: reward={base_metrics.get('reward', float('nan')):.3f} ep_return_mean={base_metrics.get('ep_return_mean', float('nan')):.3f}"
                )

            # Aggregate episodic returns across seeds (Accuracy & Consistency)
            try:
                vals = np.array([v for v in ep_means_across_seeds if not np.isnan(v)], dtype=float)
                if vals.size > 0:
                    agg_mean = float(np.mean(vals))
                    agg_std = float(np.std(vals))
                else:
                    agg_mean, agg_std = float('nan'), float('nan')
            except Exception:
                agg_mean, agg_std = float('nan'), float('nan')

            # Compute additional aggregates across seeds for axes
            def _nanmean(arr: List[float]) -> float:
                try:
                    a = np.array(arr, dtype=float)
                    m = ~np.isnan(a)
                    return float(np.mean(a[m])) if np.any(m) else float('nan')
                except Exception:
                    return float('nan')

            agg_speed_mean = _nanmean(speed_means_across_seeds)
            agg_speed_success = _nanmean(speed_success_across_seeds)
            agg_scal_ratio = _nanmean(scal_ep_ratio_across_seeds)  # small/base
            agg_adapt_ratio = _nanmean(adapt_ep_ratio_across_seeds)  # post/pre
            agg_scal_reward_ratio = _nanmean(scal_reward_ratio_across_seeds)
            agg_adapt_reward_ratio = _nanmean(adapt_reward_ratio_across_seeds)

            results[f"{model_uc}__AGG"] = {
                'seed': 'multi',
                'variant': 'aggregate',
                'ep_return_mean': agg_mean,
                'ep_return_std': agg_std,
                # Store aggregates needed for axis computations
                'speed_steps_to_threshold_mean_agg': agg_speed_mean,
                'speed_success_rate_agg': agg_speed_success,
                'scalability_ep_return_ratio_agg': agg_scal_ratio,
                'adaptability_ep_return_ratio_agg': agg_adapt_ratio,
                'scalability_reward_ratio_agg': agg_scal_reward_ratio,
                'adaptability_reward_ratio_agg': agg_adapt_reward_ratio,
            }
            self.run_records.append({'model': model_uc, 'seed': 'multi', 'variant': 'aggregate', 'ep_return_mean': agg_mean, 'ep_return_std': agg_std})

        # Cross-model normalization of aggregated episodic returns (max-only)
        # Accuracy(model) = ep_return_mean(model) / max(ep_return_mean(all models))
        try:
            agg_items = [(k, v) for k, v in results.items() if k.endswith('__AGG')]
            ep_vals = np.array([float(v.get('ep_return_mean', np.nan)) for _, v in agg_items], dtype=float)
            mask = ~np.isnan(ep_vals)
            if np.any(mask):
                vmax = float(np.max(ep_vals[mask]))
                for (k, v) in agg_items:
                    cur = float(v.get('ep_return_mean', np.nan))
                    norm = (cur / vmax) if (not np.isnan(cur) and vmax > 1e-12) else float('nan')
                    v['ep_return_mean_norm_across_models'] = norm
        except Exception:
            pass

        # Compute 0–100 evaluation axis scores on aggregated entries
        try:
            agg_items = [(k, v) for k, v in results.items() if k.endswith('__AGG')]
            # Consistency and Speed need cross-model maxima
            std_vals = np.array([float(v.get('ep_return_std', np.nan)) for _, v in agg_items], dtype=float)
            std_max = float(np.nanmax(std_vals)) if np.any(~np.isnan(std_vals)) else float('nan')

            speed_vals = np.array([float(v.get('speed_steps_to_threshold_mean_agg', np.nan)) for _, v in agg_items], dtype=float)
            speed_max = float(np.nanmax(speed_vals)) if np.any(~np.isnan(speed_vals)) else float('nan')

            for (k, v) in agg_items:
                # Accuracy: normalized ep_return_mean across models (0–1) -> percent
                acc_norm = float(v.get('ep_return_mean_norm_across_models', float('nan')))
                axis_accuracy = 100.0 * acc_norm if not np.isnan(acc_norm) else float('nan')

                # Consistency: lower std is better -> 100 - (std / max_std)*100
                cur_std = float(v.get('ep_return_std', float('nan')))
                if not np.isnan(cur_std) and (not np.isnan(std_max)) and std_max > 1e-12:
                    axis_consistency = 100.0 - (cur_std / std_max) * 100.0
                else:
                    axis_consistency = float('nan')

                # Speed: lower steps is better -> 100 - (steps / max_steps)*100
                cur_steps = float(v.get('speed_steps_to_threshold_mean_agg', float('nan')))
                if not np.isnan(cur_steps) and (not np.isnan(speed_max)) and speed_max > 1e-12:
                    axis_speed = 100.0 - (cur_steps / speed_max) * 100.0
                else:
                    axis_speed = float('nan')

                # Scalability: invert small/base reward ratio to large/small, then percent
                # Use reward ratio per user's definition (large/small)
                scal_ratio = float(v.get('scalability_reward_ratio_agg', float('nan')))
                if not np.isnan(scal_ratio) and abs(scal_ratio) > 1e-12:
                    axis_scalability = (1.0 / scal_ratio) * 100.0
                else:
                    axis_scalability = float('nan')

                # Adaptability: post/pre reward ratio -> percent
                adapt_ratio = float(v.get('adaptability_reward_ratio_agg', float('nan')))
                axis_adaptability = adapt_ratio * 100.0 if not np.isnan(adapt_ratio) else float('nan')

                # Optional clamp to [0, 100] to satisfy 0–100 range
                def _clamp_0_100(x: float) -> float:
                    try:
                        if np.isnan(x):
                            return x
                        return float(max(0.0, min(100.0, x)))
                    except Exception:
                        return float('nan')

                v['axis_accuracy'] = _clamp_0_100(axis_accuracy)
                v['axis_consistency'] = _clamp_0_100(axis_consistency)
                v['axis_speed'] = _clamp_0_100(axis_speed)
                v['axis_scalability'] = _clamp_0_100(axis_scalability)
                v['axis_adaptability'] = _clamp_0_100(axis_adaptability)
        except Exception:
            pass
        
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
            mask_violation_rate = metrics.get('mask_violation_rate', float('nan'))
            mask_violations = metrics.get('mask_violations', float('nan'))
            
            # Hybrid contributions
            base_contrib = metrics.get('reward_base_contrib', float('nan'))
            mastery = metrics.get('reward_mastery', float('nan'))
            motivation = metrics.get('reward_motivation', float('nan'))

            # Hybrid contribution percentage shares (normalized)
            vals = np.array([base_contrib, mastery, motivation], dtype=float)
            if np.any(np.isnan(vals)):
                base_share, mastery_share, motivation_share = float('nan'), float('nan'), float('nan')
            else:
                total = float(np.sum(vals))
                if total > 1e-9:
                    base_share = 100.0 * (base_contrib / total)
                    mastery_share = 100.0 * (mastery / total)
                    motivation_share = 100.0 * (motivation / total)
                else:
                    base_share, mastery_share, motivation_share = float('nan'), float('nan'), float('nan')

            # Percent versions for selected metrics
            norm_pct = reward_norm * 100.0 if not np.isnan(reward_norm) else float('nan')
            regret_ratio_pct = regret_ratio * 100.0 if not np.isnan(regret_ratio) else float('nan')
            
            mv_pct_txt = f" mask_violation_rate%={mask_violation_rate*100.0:.2f} mask_violations={int(mask_violations)}" \
                if not np.isnan(mask_violation_rate) else ""
            print(
                f"{name:<12}: avg_after_shaping_reward={reward:.3f}  base_reward_before_shaping={reward_base:.3f}  "
                f"normalized_after_shaping_reward%={norm_pct:.1f}  vpr={vpr:.3f}  regret={regret:.3f}  "
                f"regret_ratio_after_shaping%={regret_ratio_pct:.1f}{mv_pct_txt}  |  "
                f"after_shaping_base_contrib={base_contrib:.3f}  after_shaping_mastery_contrib={mastery:.3f}  "
                f"after_shaping_motivation_contrib={motivation:.3f}  |  "
                f"after_shaping_base_share%={base_share:.1f}  after_shaping_mastery_share%={mastery_share:.1f}  "
                f"after_shaping_motivation_share%={motivation_share:.1f}"
            )

            # Brief speed summary
            sp_thr = metrics.get('speed_threshold_norm', float('nan'))
            sp_mean = metrics.get('speed_steps_to_threshold_mean', float('nan'))
            sp_med = metrics.get('speed_steps_to_threshold_median', float('nan'))
            sp_succ = metrics.get('speed_success_rate', float('nan'))
            sp_succ_pct = sp_succ * 100.0 if not np.isnan(sp_succ) else float('nan')
            if not (np.isnan(sp_thr) and np.isnan(sp_mean) and np.isnan(sp_succ)):
                print(
                    f"  speed: threshold_norm={sp_thr:.2f} steps_mean={sp_mean:.1f} steps_median={sp_med:.1f} success%={sp_succ_pct:.1f}"
                )

        # Aggregated episodic returns across seeds
        print("\n=== Accuracy/Consistency (episodic returns aggregated across seeds) ===")
        for name, metrics in self.results.items():
            if not name.endswith('__AGG'):
                continue
            acc = metrics.get('ep_return_mean', float('nan'))
            cons = metrics.get('ep_return_std', float('nan'))
            acc_norm = metrics.get('ep_return_mean_norm_across_models', float('nan'))
            acc_norm_pct = acc_norm * 100.0 if not np.isnan(acc_norm) else float('nan')
            print(
                f"{name:<20}: ep_return_mean={acc:.4f}  ep_return_std={cons:.4f}  ep_return_mean_norm_across_models%={acc_norm_pct:.1f}"
            )
    
    def _save_results_to_csv(self) -> None:
        """Save results to CSV file."""
        csv_path = self.config.config.metrics_csv
        
        # Define comprehensive fieldnames (used when creating a new CSV)
        default_fieldnames = [
            "timestamp", "model", "reward", "vpr", "vpr_pct", "regret", 
            "regret_ratio", "regret_ratio_pct", "seed", "env_type",
            "reward_base", "reward_shaping", "reward_norm", 
            "reward_base_pct", "reward_norm_pct",
            "reward_base_contrib", "reward_mastery", "reward_motivation",
            "reward_base_contrib_pct", "reward_mastery_pct", "reward_motivation_pct",
            "hybrid_base_share_pct", "hybrid_mastery_share_pct", "hybrid_motivation_share_pct",
            "mask_violations", "mask_violation_rate",
            "term_improve", "term_deficit", "term_spacing", "term_diversity", "term_challenge",
            # Speed metrics
            "speed_threshold_norm", "speed_steps_to_threshold_mean", "speed_steps_to_threshold_median", "speed_success_rate",
            # Episodic return aggregates
            "ep_return_mean", "ep_return_std", "ep_return_mean_norm_across_models",
            # Extended evaluation axes metadata
            "variant", "scalability_reward_ratio", "scalability_ep_return_ratio",
            "adaptability_reward_ratio", "adaptability_ep_return_ratio",
            # Aggregates across seeds for axes
            "speed_steps_to_threshold_mean_agg", "speed_success_rate_agg",
            "scalability_ep_return_ratio_agg", "adaptability_ep_return_ratio_agg",
            "scalability_reward_ratio_agg", "adaptability_reward_ratio_agg",
            # Final axis scores (0–100)
            "axis_accuracy", "axis_consistency", "axis_speed", "axis_scalability", "axis_adaptability",
            # Helper label
            "model_base"
        ]
        
        # Add environment and model config fields
        env_config = self.config.config.environment
        default_fieldnames.extend([
            "reward_correct_w", "reward_score_w", "hybrid_base_w", 
            "hybrid_mastery_w", "hybrid_motivation_w",
            "rew_improve_w", "rew_deficit_w", "rew_spacing_w", 
            "rew_diversity_w", "rew_challenge_w",
            "ema_alpha", "need_threshold", "spacing_window", 
            "diversity_recent_k", "challenge_target", "challenge_band", "invalid_penalty",
            # Environment sampling
            "student_fraction"
        ])
        
        # Check if file exists and has header
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # If file exists, reuse its header to avoid column mismatch; else use default_fieldnames
        fieldnames_to_use = None
        if file_exists:
            try:
                with open(csv_path, mode='r', newline='') as rf:
                    reader = csv.reader(rf)
                    existing_header = next(reader)
                    if isinstance(existing_header, list) and len(existing_header) > 0:
                        fieldnames_to_use = existing_header
                        # Warn if important new fields are missing; they will be omitted
                        missing_cols = [c for c in default_fieldnames if c not in existing_header]
                        if missing_cols:
                            self.logger.warning(
                                "Existing CSV missing columns; new fields will be omitted: %s",
                                ", ".join(missing_cols)
                            )
            except Exception:
                fieldnames_to_use = None
        if fieldnames_to_use is None:
            fieldnames_to_use = default_fieldnames

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_to_use, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            for model_name, metrics in self.results.items():
                row = {
                    "timestamp": timestamp,
                    "model": model_name,
                    "seed": metrics.get('seed', env_config.seed),
                    "env_type": "interactive"
                }
                
                # Add metrics
                for key, value in metrics.items():
                    row[key] = value
                
                # Add base model label (strip any variant suffix like __seed or __AGG)
                try:
                    row["model_base"] = str(row["model"]).split("__")[0].upper()
                except Exception:
                    row["model_base"] = str(row["model"]).upper()

                # Add percentage versions
                for base_key in ['vpr', 'regret_ratio', 'reward_base', 'reward_norm']:
                    if base_key in metrics and not np.isnan(metrics[base_key]):
                        row[f"{base_key}_pct"] = metrics[base_key] * 100.0

                # Add hybrid contribution percentages (absolute, not normalized)
                for key in ['reward_base_contrib', 'reward_mastery', 'reward_motivation']:
                    if key in metrics and not np.isnan(metrics[key]):
                        row[f"{key}_pct"] = metrics[key] * 100.0

                # Add hybrid contribution percentage shares (normalized to sum of hybrid parts)
                base_contrib = metrics.get('reward_base_contrib', float('nan'))
                mastery = metrics.get('reward_mastery', float('nan'))
                motivation = metrics.get('reward_motivation', float('nan'))
                vals = np.array([base_contrib, mastery, motivation], dtype=float)
                if not np.any(np.isnan(vals)):
                    total = float(np.sum(vals))
                    if total > 1e-9:
                        row['hybrid_base_share_pct'] = 100.0 * (base_contrib / total)
                        row['hybrid_mastery_share_pct'] = 100.0 * (mastery / total)
                        row['hybrid_motivation_share_pct'] = 100.0 * (motivation / total)

                # Add environment config
                for attr in ['reward_correct_w', 'reward_score_w', 'hybrid_base_w',
                           'hybrid_mastery_w', 'hybrid_motivation_w', 'rew_improve_w',
                           'rew_deficit_w', 'rew_spacing_w', 'rew_diversity_w', 
                           'rew_challenge_w', 'ema_alpha', 'need_threshold',
                           'spacing_window', 'diversity_recent_k', 'challenge_target',
                           'challenge_band', 'invalid_penalty', 'student_fraction']:
                    # Preserve any run-specific values already placed by metrics
                    if attr not in row:
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
