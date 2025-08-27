"""Baseline policies for curriculum sequencing evaluation."""

import random
import numpy as np
from typing import Any, Callable, Dict
from ..core.base import PolicyFunction


class BaselinePolicies:
    """Collection of baseline policies for evaluation."""
    
    @staticmethod
    def random_policy(env: Any, seed: int = 42) -> PolicyFunction:
        """Uniform random policy."""
        rng = random.Random(seed)
        
        def policy(state: np.ndarray, cur_cat: int) -> int:
            if hasattr(env, 'valid_action_ids'):
                valid_ids = env.valid_action_ids()
                if len(valid_ids) > 0:
                    return rng.choice(list(valid_ids))
            return rng.randrange(env.action_size)
        
        return policy
    
    @staticmethod
    def trivial_same_policy(env: Any) -> PolicyFunction:
        """Always predict the current category."""
        def policy(state: np.ndarray, cur_cat: int) -> int:
            return cur_cat
        
        return policy
    
    @staticmethod
    def markov_policy(env: Any) -> PolicyFunction:
        """Markov-1 policy learned from training transitions."""
        # Build transition counts
        counts = np.zeros((env.action_size, env.action_size), dtype=np.int64)
        
        for student_id in env.splits.train_students:
            student_mask = env.df["student_id"] == student_id
            student_df = env.df[student_mask].sort_values("order")
            
            for i in range(len(student_df) - 1):
                cur_cat = int(student_df.iloc[i]["category_id"])
                next_cat = int(student_df.iloc[i + 1]["category_id"])
                counts[cur_cat, next_cat] += 1
        
        # Get most frequent next category for each current category
        most_next = np.argmax(counts, axis=1)
        
        def policy(state: np.ndarray, cur_cat: int) -> int:
            # Fallback to current category if no training data
            if counts[cur_cat].sum() == 0:
                return cur_cat
            return int(most_next[cur_cat])
        
        return policy
    
    @staticmethod
    def get_all_baselines(env: Any, seed: int = 42) -> Dict[str, PolicyFunction]:
        """Get all baseline policies."""
        return {
            'Chance': BaselinePolicies.random_policy(env, seed),
            'TrivialSame': BaselinePolicies.trivial_same_policy(env),
            'Markov1-Train': BaselinePolicies.markov_policy(env)
        }
