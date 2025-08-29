"""Refactored Q-Learning implementation using new architecture."""

import numpy as np
from typing import Any, Optional
import copy

from ..core.base import BaseAgent, BaseTrainer, PolicyFunction
from ..core.config import QLearningConfig
from ..core.utils import linear_schedule
from ..core.factory import register_trainer

try:
    from ..evaluation import eval_policy_avg_score
except ImportError:
    from evaluation import eval_policy_avg_score


class QLearningAgent(BaseAgent):
    """Improved tabular Q-Learning agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: QLearningConfig):
        super().__init__(state_dim, action_dim)
        self.config = config
        
        # Q-table initialization
        self.q_table = np.zeros((action_dim, action_dim), dtype=np.float32)
        
        # Exploration
        self.epsilon = config.eps_start
        self.rng = np.random.default_rng(config.seed)
        
        # Training state
        self.steps_done = 0
    
    def act(self, state: np.ndarray, training: bool = True, 
            valid_ids: Optional[Any] = None) -> int:
        """Select action using epsilon-greedy."""
        # Extract current category from state (one-hot encoded)
        current_category = int(np.argmax(state[:self.action_dim]))
        
        if training and self.rng.random() < self.epsilon:
            # Random exploration
            if valid_ids is not None and len(valid_ids) > 0:
                return int(self.rng.choice(list(valid_ids)))
            return int(self.rng.integers(0, self.action_dim))
        
        # Greedy action selection
        q_values = self.q_table[current_category].copy()
        
        # Apply action masking if provided
        if valid_ids is not None and len(valid_ids) > 0:
            mask = np.full_like(q_values, -np.inf)
            mask[list(valid_ids)] = 0.0
            q_values = q_values + mask
        
        return int(np.argmax(q_values))
    
    def act_from_cat(self, current_category: int, training: bool = True,
                     valid_ids: Optional[Any] = None) -> int:
        """Like act(), but skip state decoding; use current category directly.
        This avoids reconstructing state and taking argmax during offline QL training.
        """
        if training and self.rng.random() < self.epsilon:
            # Random exploration
            if valid_ids is not None and len(valid_ids) > 0:
                return int(self.rng.choice(list(valid_ids)))
            return int(self.rng.integers(0, self.action_dim))
        
        # Greedy action selection
        q_values = self.q_table[current_category].copy()
        
        # Apply action masking if provided
        if valid_ids is not None and len(valid_ids) > 0:
            mask = np.full_like(q_values, -np.inf)
            mask[list(valid_ids)] = 0.0
            q_values = q_values + mask
        
        return int(np.argmax(q_values))
    
    def update(self, state: int, action: int, reward: float, next_state: int) -> float:
        """Update Q-table using Q-learning rule."""
        # Q-learning update
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        target_q = reward + self.config.gamma * max_next_q
        
        # Update with learning rate
        self.q_table[state, action] += self.config.alpha * (target_q - current_q)
        
        self.steps_done += 1
        return abs(target_q - current_q)  # Return TD error as loss
    
    def update_epsilon(self, epoch: int) -> None:
        """Update exploration rate."""
        self.epsilon = linear_schedule(
            self.config.eps_start,
            self.config.eps_end,
            epoch,
            self.config.eps_decay_epochs
        )
    
    def get_policy(self, env: Optional[Any] = None) -> PolicyFunction:
        """Return greedy policy function."""
        def policy(state: np.ndarray, cur_cat: int) -> int:
            valid_ids = None
            if env is not None and hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
            return self.act(state, training=False, valid_ids=valid_ids)
        return policy
    
    def save(self, path: str) -> None:
        """Save Q-table to file."""
        np.save(path, self.q_table)
    
    def load(self, path: str) -> None:
        """Load Q-table from file."""
        self.q_table = np.load(path)


@register_trainer('ql')
class QLearningTrainer(BaseTrainer):
    """Q-Learning trainer with improved data processing."""
    
    def __init__(self, config: QLearningConfig):
        super().__init__(config)
        self.config = config
        self.best_q_table = None
        self.best_score = float('-inf')
        # Cache of offline training transitions for speed
        self._cached_transitions = None
    
    def create_agent(self, env: Any) -> QLearningAgent:
        """Create Q-Learning agent."""
        return QLearningAgent(env.state_dim, env.action_size, self.config)
    
    def _cache_transitions(self, env: Any) -> None:
        """Precompute offline training transitions once to avoid per-epoch sorting.
        Builds a flat list of (current_cat, next_cat, next_score) tuples for all
        consecutive pairs across training students.
        """
        transitions = []
        # Sort once globally for deterministic per-student order
        df_sorted = env.df.sort_values(["student_id", "order"])
        for student_id in env.splits.train_students:
            student_df = df_sorted[df_sorted["student_id"] == student_id]
            if len(student_df) < 2:
                continue
            # Categories
            cats = student_df["category_id"].astype(int).values
            cur_cats = cats[:-1]
            next_cats = cats[1:]
            # Scores (default to zeros if column missing)
            if "normalized_score" in student_df.columns:
                scores = student_df["normalized_score"].fillna(0.0).astype(float).values
                next_scores = scores[1:]
            else:
                next_scores = np.zeros(len(next_cats), dtype=float)
            for c, n, s in zip(cur_cats, next_cats, next_scores):
                transitions.append((int(c), int(n), float(s)))
        self._cached_transitions = transitions
    
    def train_step(self, env: Any, agent: QLearningAgent) -> dict:
        """Execute one training epoch (pass through all students)."""
        total_loss = 0.0
        total_transitions = 0
        
        if getattr(self, "_cached_transitions", None):
            # Fast path: use precomputed transitions and skip state reconstruction
            for current_cat, next_cat, next_score in self._cached_transitions:
                action = agent.act_from_cat(current_cat, training=True)
                if hasattr(env, "valid_action_ids"):
                    # Interactive environment: score-only reward
                    reward = env.rw_score * float(next_score)
                else:
                    # Standard reward: correctness + score
                    correctness = 1.0 if action == next_cat else 0.0
                    reward = (env.rw_correct * correctness + env.rw_score * float(next_score))
                loss = agent.update(current_cat, action, reward, next_cat)
                total_loss += loss
                total_transitions += 1
        else:
            # Fallback: original slow path
            for student_id in env.splits.train_students:
                student_df = env.df[env.df["student_id"] == student_id].sort_values("order")
                for i in range(len(student_df) - 1):
                    current_row = student_df.iloc[i]
                    next_row = student_df.iloc[i + 1]
                    current_cat = int(current_row["category_id"])
                    next_cat = int(next_row["category_id"])
                    current_state = env._build_state_from_row(current_row)
                    action = agent.act(current_state, training=True)
                    if hasattr(env, "valid_action_ids"):
                        reward = env.rw_score * float(next_row.get("normalized_score", 0.0))
                    else:
                        correctness = 1.0 if action == next_cat else 0.0
                        reward = (env.rw_correct * correctness + env.rw_score * float(next_row.get("normalized_score", 0.0)))
                    loss = agent.update(current_cat, action, reward, next_cat)
                    total_loss += loss
                    total_transitions += 1
        
        return {
            'avg_loss': total_loss / max(total_transitions, 1),
            'total_transitions': total_transitions,
            'epsilon': agent.epsilon
        }
    
    def train(self, env: Any) -> QLearningAgent:
        """Main training loop for Q-Learning."""
        agent = self.create_agent(env)
        # Precompute transitions once for speed (avoids per-epoch DataFrame sorts)
        if getattr(self, "_cached_transitions", None) is None:
            self._cache_transitions(env)
        
        for epoch in range(self.config.epochs):
            # Update exploration rate
            agent.update_epsilon(epoch)
            
            # Train on all students
            metrics = self.train_step(env, agent)
            self._update_metrics(metrics, epoch)
            
            # Evaluation for model selection
            if self.config.select_best_on_val:
                policy = agent.get_policy(env)
                val_score = eval_policy_avg_score(
                    env, policy, mode="val", episodes=self.config.val_episodes
                )
                
                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_q_table = agent.q_table.copy()
        
        # Load best Q-table if validation was used
        if self.config.select_best_on_val and self.best_q_table is not None:
            agent.q_table = self.best_q_table
        
        self.agent = agent
        return agent
