"""Self-Adaptive RL (SARL) trainer built on top of DQN.

Implements periodic adaptation of:
- Hybrid reward weights (base/mastery/motivation) toward target contribution shares
- Exploration epsilon based on invalid action rate
- Optimizer learning rate using patience-based scheduler on validation reward
- Curriculum difficulty via challenge_target and challenge_band
"""

from __future__ import annotations

from typing import Any, Optional, Deque, Dict
from collections import deque
import copy
import math

import numpy as np

from ..core.base import BaseTrainer
from ..core.config import SARLDQNConfig
from ..core.factory import register_trainer
from ..core.utils import EpsilonScheduler
from .dqn_agent import DQNAgent

try:
    from ..evaluation import eval_policy_interactive_metrics
except ImportError:  # pragma: no cover
    from evaluation import eval_policy_interactive_metrics  # type: ignore


@register_trainer('sarl')
class SARLDQNTrainer(BaseTrainer):
    """Self-Adaptive DQN trainer with curriculum and hyperparameter adaptation."""

    def __init__(self, config: SARLDQNConfig):
        super().__init__(config)
        self.config: SARLDQNConfig = config
        self.best_agent_state = None
        self.best_score = float('-inf')
        self.no_improve_count = 0
        self._recent_stats: Deque[Dict[str, float]] = deque(maxlen=max(1, self.config.adapt_interval))

    # -----------------------------
    # Boilerplate
    # -----------------------------
    def create_agent(self, env: Any) -> DQNAgent:
        return DQNAgent(env.state_dim, env.action_size, self.config)

    # -----------------------------
    # Training loop with signals collection
    # -----------------------------
    def train_step(self, env: Any, agent: DQNAgent) -> dict:
        state = env.reset("train")
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        invalid_count = 0

        # Aggregate hybrid contributions and reward_norm
        base_contrib_sum = 0.0
        mastery_sum = 0.0
        motivation_sum = 0.0
        norm_sum = 0.0
        norm_count = 0

        # Enforce configurable per-episode step cap to prevent long/hanging episodes
        max_steps = self.config.train_max_steps_per_episode or 1000
        while not done and steps < max_steps:
            valid_ids = env.valid_action_ids() if hasattr(env, 'valid_action_ids') else None

            action = agent.act(state, training=True, valid_ids=valid_ids)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()

            # Collect signals
            episode_reward += float(reward)
            episode_loss += float(loss)
            steps += 1
            if isinstance(info, dict):
                if not bool(info.get('valid_action', True)):
                    invalid_count += 1
                base_contrib_sum += float(info.get('reward_base_contrib', 0.0))
                mastery_sum += float(info.get('reward_mastery', 0.0))
                motivation_sum += float(info.get('reward_motivation', 0.0))
                rn = info.get('reward_norm', float('nan'))
                try:
                    rn_f = float(rn)
                    if not math.isnan(rn_f):
                        norm_sum += rn_f
                        norm_count += 1
                except Exception:
                    pass

            state = next_state if not done else state

        invalid_rate = (invalid_count / steps) if steps > 0 else 0.0
        avg_loss = episode_loss / max(steps, 1)
        avg_reward_norm = (norm_sum / norm_count) if norm_count > 0 else float('nan')

        # Stash into recent stats for adaptation
        self._recent_stats.append({
            'invalid_rate': invalid_rate,
            'base_contrib': base_contrib_sum,
            'mastery_contrib': mastery_sum,
            'motivation_contrib': motivation_sum,
            'avg_reward_norm': avg_reward_norm if not math.isnan(avg_reward_norm) else 0.0,
        })

        # Return metrics for logging
        return {
            'episode_reward': episode_reward,
            'episode_loss': avg_loss,
            'episode_steps': steps,
            'epsilon': agent.epsilon_scheduler.get_epsilon(),
            'lr': float(agent.optimizer.param_groups[0]['lr']),
            'invalid_rate': invalid_rate,
            'hyb_reward_base': base_contrib_sum,
            'hyb_reward_mastery': mastery_sum,
            'hyb_reward_motivation': motivation_sum,
            'avg_reward_norm': avg_reward_norm,
            'hyb_w_base': getattr(env.config, 'hybrid_base_w', float('nan')),
            'hyb_w_mastery': getattr(env.config, 'hybrid_mastery_w', float('nan')),
            'hyb_w_motivation': getattr(env.config, 'hybrid_motivation_w', float('nan')),
            'challenge_target': getattr(env.config, 'challenge_target', float('nan')),
            'challenge_band': getattr(env.config, 'challenge_band', float('nan')),
        }

    # -----------------------------
    # Evaluation + LR scheduler
    # -----------------------------
    def _evaluate_agent(self, env: Any, episode: int) -> None:
        if not self.config.select_best_on_val:
            return

        policy = self.agent.get_policy(env)  # type: ignore[attr-defined]
        metrics = eval_policy_interactive_metrics(
            env, policy, mode="val", episodes=self.config.val_episodes,
            max_steps_per_episode=self.config.val_max_steps_per_episode
        )
        val_reward = float(metrics.get('reward', 0.0))

        improved = val_reward > (self.best_score + 1e-6)
        if improved:
            self.best_score = val_reward
            # snapshot best network
            self.best_agent_state = copy.deepcopy(self.agent.q_network.state_dict())  # type: ignore[attr-defined]
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # LR adaptation on plateau
        if self.no_improve_count >= max(1, self.config.lr_patience):
            self._reduce_lr()
            self.no_improve_count = 0

    def _reduce_lr(self) -> None:
        """Reduce optimizer LR by factor with floor."""
        try:
            opt = self.agent.optimizer  # type: ignore[attr-defined]
            for group in opt.param_groups:
                new_lr = max(self.config.min_lr, group['lr'] * self.config.lr_reduce_factor)
                group['lr'] = new_lr
        except Exception:
            pass

    # -----------------------------
    # Adaptation logic (called periodically)
    # -----------------------------
    def _aggregate_recent(self) -> Dict[str, float]:
        n = max(1, len(self._recent_stats))
        agg: Dict[str, float] = {
            'invalid_rate': 0.0,
            'base_contrib': 0.0,
            'mastery_contrib': 0.0,
            'motivation_contrib': 0.0,
            'avg_reward_norm': 0.0,
        }
        for s in self._recent_stats:
            for k in agg.keys():
                agg[k] += float(s.get(k, 0.0))
        for k in agg.keys():
            agg[k] /= float(n)
        return agg

    def _adapt(self, env: Any, agent: DQNAgent) -> None:
        stats = self._aggregate_recent()

        # 1) Adapt exploration epsilon from invalid rate
        self._adapt_epsilon(agent, stats['invalid_rate'])

        # 2) Adapt hybrid weights from contribution shares
        self._adapt_hybrid_weights(env, stats)

        # 3) Adapt curriculum difficulty from normalized reward
        self._adapt_curriculum(env, stats['avg_reward_norm'])

        # Refresh environment reward setup if weights changed
        try:
            env._setup_reward_config()
        except Exception:
            pass

    def _adapt_epsilon(self, agent: DQNAgent, invalid_rate: float) -> None:
        eps = agent.epsilon_scheduler.get_epsilon()
        if invalid_rate > self.config.invalid_rate_hi:
            eps = min(self.config.eps_max, eps + self.config.epsilon_up_step)
            agent.epsilon_scheduler.set_epsilon(eps)
        elif invalid_rate < self.config.invalid_rate_lo:
            eps = max(self.config.eps_min, eps - self.config.epsilon_down_step)
            agent.epsilon_scheduler.set_epsilon(eps)

    def _adapt_hybrid_weights(self, env: Any, stats: Dict[str, float]) -> None:
        base = float(stats.get('base_contrib', 0.0))
        mastery = float(stats.get('mastery_contrib', 0.0))
        motivation = float(stats.get('motivation_contrib', 0.0))
        total = base + mastery + motivation
        if total <= 1e-9:
            return

        # Current shares measured
        s_base = base / total
        s_mas = mastery / total
        s_mot = motivation / total

        # Target shares normalized
        t_sum = (
            self.config.hybrid_base_target_share +
            self.config.hybrid_mastery_target_share +
            self.config.hybrid_motivation_target_share
        )
        if t_sum <= 1e-9:
            t_base, t_mas, t_mot = 0.7, 0.2, 0.1
        else:
            t_base = self.config.hybrid_base_target_share / t_sum
            t_mas = self.config.hybrid_mastery_target_share / t_sum
            t_mot = self.config.hybrid_motivation_target_share / t_sum

        # Adjust weights toward targets
        def _clip(x: float) -> float:
            return float(min(self.config.weight_max, max(self.config.weight_min, x)))

        hb = getattr(env.config, 'hybrid_base_w', 1.0)
        hm = getattr(env.config, 'hybrid_mastery_w', 1.0)
        hv = getattr(env.config, 'hybrid_motivation_w', 1.0)

        hb_new = _clip(hb + self.config.weight_lr * (t_base - s_base))
        hm_new = _clip(hm + self.config.weight_lr * (t_mas - s_mas))
        hv_new = _clip(hv + self.config.weight_lr * (t_mot - s_mot))

        env.config.hybrid_base_w = hb_new
        env.config.hybrid_mastery_w = hm_new
        env.config.hybrid_motivation_w = hv_new

    def _adapt_curriculum(self, env: Any, perf: float) -> None:
        if math.isnan(perf):
            return
        # Increase difficulty if performing well
        if perf >= self.config.challenge_increase_threshold:
            env.config.challenge_target = float(min(self.config.challenge_target_max,
                                                   env.config.challenge_target + self.config.challenge_target_step))
            # band step is negative to narrow when easy
            env.config.challenge_band = float(max(self.config.challenge_band_min,
                                                  env.config.challenge_band + self.config.challenge_band_step))
        # Decrease difficulty if struggling
        elif perf <= self.config.challenge_decrease_threshold:
            env.config.challenge_target = float(max(self.config.challenge_target_min,
                                                   env.config.challenge_target - self.config.challenge_target_step))
            # widen band: subtract negative step to widen
            env.config.challenge_band = float(min(self.config.challenge_band_max,
                                                  env.config.challenge_band - self.config.challenge_band_step))

    # -----------------------------
    # Main training loop: add periodic adaptation and model selection
    # -----------------------------
    def train(self, env: Any) -> DQNAgent:
        agent = self.create_agent(env)

        for episode in range(self.config.episodes):
            metrics = self.train_step(env, agent)
            self._update_metrics(metrics, episode)

            # Periodic target update (hard) for stability
            if (episode + 1) % 20 == 0:
                agent.hard_update_target()

            # Periodic adaptation
            if (episode + 1) % max(1, self.config.adapt_interval) == 0:
                self._adapt(env, agent)

            # Evaluation and LR scheduling
            if self._should_evaluate(episode):
                self._evaluate_agent(env, episode)

        # Load best model if validation was used
        if self.config.select_best_on_val and self.best_agent_state is not None:
            agent.q_network.load_state_dict(self.best_agent_state)
            agent.hard_update_target()

        self.agent = agent
        return agent
