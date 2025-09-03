"""Optimized interactive environment with improved memory management."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import torch

from ..core.config import EnvironmentConfig


@dataclass
class SplitData:
    """Student split data structure."""
    train_students: np.ndarray
    val_students: np.ndarray
    test_students: np.ndarray


class OptimizedInteractiveEnv:
    """Optimized interactive environment with better performance and memory usage."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        
        # Initialize episode state
        self._reset_episode_state()
    
    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess the dataset with optimizations."""
        # Load data
        self.df = pd.read_csv(self.config.data_path)
        
        # Validate required columns
        required_cols = [
            "student_id", "exercise_id", "category", "category_group", "order", 
            "normalized_score", "grade", "sex", "home_school_lang_match",
            "missing_all", "missing_beginning30", "missing_last50"
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Optional: downsample students for scalability experiments
        # We filter the dataframe to a random subset of students before any further preprocessing
        try:
            frac = float(getattr(self.config, "student_fraction", 1.0))
        except Exception:
            frac = 1.0
        if frac <= 0.0:
            raise ValueError("student_fraction must be > 0")
        if frac < 1.0:
            students = np.array(self.df["student_id"].unique())
            # Use environment RNG (seeded) for reproducibility
            self.rng.shuffle(students)
            import numpy as _np  # local alias to avoid confusion
            n_keep = max(1, int(_np.ceil(len(students) * frac)))
            keep_ids = set(students[:n_keep].tolist())
            self.df = self.df[self.df["student_id"].isin(keep_ids)].reset_index(drop=True)
        
        # Set up action space
        self.category_col = self.config.action_on
        self.df[self.category_col] = self.df[self.category_col].fillna("Unknown")
        self.categories = sorted(self.df[self.category_col].unique())
        self.cat2id = {c: i for i, c in enumerate(self.categories)}
        self.action_size = len(self.categories)
        
        # State dimension (needed before precomputing features)
        self.state_dim = self.action_size + 8
        
        # Precompute normalized features
        self._precompute_features()
        
        # Create student splits
        self.splits = self._create_splits()
        
        # Precompute category statistics
        self._precompute_category_stats()
        
        # Set up reward configuration
        self._setup_reward_config()
    
    def _precompute_features(self) -> None:
        """Precompute and cache normalized features."""
        # Order normalization
        order_min, order_max = self.df["order"].min(), self.df["order"].max()
        self.df["order_norm"] = (self.df["order"] - order_min) / (order_max - order_min + 1e-9)
        # Guard against NaNs in order_norm
        self.df["order_norm"] = pd.to_numeric(self.df["order_norm"], errors="coerce").fillna(0.0).astype(float)
        
        # Sex encoding
        self.df["sex_bin"] = self.df["sex"].apply(self._standardize_sex)
        
        # Grade encoding
        self.df["grade"] = self.df["grade"].astype(str).fillna("Unknown")
        grades = sorted(self.df["grade"].unique())
        grade2id = {g: i for i, g in enumerate(grades)}
        self.df["grade_enc"] = self.df["grade"].map(grade2id).astype(float)
        if len(grades) > 1:
            self.df["grade_enc"] /= (len(grades) - 1)
        
        # Ensure normalized_score is numeric and bounded; fill missing with neutral 0.5
        self.df["normalized_score"] = pd.to_numeric(self.df["normalized_score"], errors="coerce")
        self.df["normalized_score"] = self.df["normalized_score"].clip(lower=0.0, upper=1.0).fillna(0.5).astype(float)
        
        # Other features
        self.df["home_school_lang_match"] = self.df["home_school_lang_match"].fillna(0.5).astype(float)
        for col in ["missing_all", "missing_beginning30", "missing_last50"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)
        
        # Category IDs
        self.df["category_id"] = self.df[self.category_col].map(self.cat2id).astype(int)
        
        # Sort by student and order for efficient access
        self.df = self.df.sort_values(["student_id", "order"]).reset_index(drop=True)
        
        # Precompute state vectors for all rows (memory vs computation tradeoff)
        self._precompute_state_vectors()
    
    def _precompute_state_vectors(self) -> None:
        """Precompute state vectors for all data points."""
        n_rows = len(self.df)
        self.state_cache = np.zeros((n_rows, self.state_dim), dtype=np.float32)
        
        # Vectorized computation
        category_ids = self.df["category_id"].values
        one_hot = np.eye(self.action_size, dtype=np.float32)[category_ids]
        
        numeric_features = self.df[[
            "order_norm", "normalized_score", "grade_enc", "sex_bin",
            "home_school_lang_match", "missing_all", "missing_beginning30", "missing_last50"
        ]].values.astype(np.float32)
        
        self.state_cache = np.concatenate([one_hot, numeric_features], axis=1)
    
    def _standardize_sex(self, val: str) -> float:
        """Standardize sex values."""
        if pd.isna(val):
            return 0.5
        s = str(val).strip().lower()
        if s in ("boy", "male", "m"):
            return 0.0
        if s in ("girl", "gir", "female", "f"):
            return 1.0
        return 0.5
    
    def _create_splits(self) -> SplitData:
        """Create train/val/test splits."""
        students = self.df["student_id"].unique()
        self.rng.shuffle(students)
        
        n = len(students)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        
        return SplitData(
            train_students=students[:n_train],
            val_students=students[n_train:n_train + n_val],
            test_students=students[n_train + n_val:]
        )
    
    def _precompute_category_stats(self) -> None:
        """Precompute category statistics for faster access."""
        self.category_mean_score = {}
        for aid in range(self.action_size):
            scores = self.df[self.df["category_id"] == aid]["normalized_score"]
            mean_score = scores.mean() if len(scores) > 0 else 0.5
            self.category_mean_score[aid] = float(mean_score)
    
    def _setup_reward_config(self) -> None:
        """Setup reward configuration."""
        total_w = self.config.reward_correct_w + self.config.reward_score_w
        if total_w <= 0:
            self.rw_correct, self.rw_score = 0.0, 1.0
        else:
            self.rw_correct = self.config.reward_correct_w / total_w
            self.rw_score = self.config.reward_score_w / total_w
        
        # Precompute reward normalization denominator
        self._reward_den = (
            (self.config.hybrid_base_w * self.rw_score) +
            (self.config.hybrid_mastery_w * (
                self.config.rew_improve_w + self.config.rew_deficit_w + self.config.rew_spacing_w
            )) +
            (self.config.hybrid_motivation_w * (
                self.config.rew_diversity_w + self.config.rew_challenge_w
            ))
        )
    
    def _reset_episode_state(self) -> None:
        """Reset episode-specific state variables."""
        self.current_student_id: Optional[int] = None
        self.current_student_indices: Optional[np.ndarray] = None
        self._rows_by_action: Dict[int, List[int]] = {}
        self._consumed: set = set()
        self._cur_idx: int = 0
        
        # Shaping trackers
        self._ema_scores: Dict[int, float] = {}
        self._last_step_seen: Dict[int, Optional[int]] = {}
        self._recent_choices: deque = deque(maxlen=self.config.diversity_recent_k)
        self._step_t: int = 0
    
    def reset(self, mode: str = "train") -> np.ndarray:
        """Reset environment for new episode."""
        # Select student split
        if mode == "train":
            students = self.splits.train_students
        elif mode == "val":
            students = self.splits.val_students
        elif mode == "test":
            students = self.splits.test_students
        else:
            raise ValueError("mode must be one of {'train', 'val', 'test'}")
        
        if len(students) == 0:
            raise ValueError(f"No students in split '{mode}'")
        
        # Select random student
        self.current_student_id = int(self.rng.choice(students))
        
        # Get student data indices (pre-sorted)
        student_mask = self.df["student_id"] == self.current_student_id
        self.current_student_indices = np.where(student_mask)[0]
        
        if len(self.current_student_indices) == 0:
            raise ValueError("Selected student has no data")
        
        # Build action mapping
        self._build_rows_by_action()
        
        # Initialize episode state
        self._consumed = {0}  # First exercise already done
        self._cur_idx = 0
        
        # Initialize shaping trackers
        self._ema_scores = {aid: self.category_mean_score[aid] for aid in range(self.action_size)}
        self._last_step_seen = {aid: None for aid in range(self.action_size)}
        self._recent_choices.clear()
        self._step_t = 0
        
        # Return initial state
        return self.state_cache[self.current_student_indices[0]].copy()
    
    def _build_rows_by_action(self) -> None:
        """Build mapping from actions to row indices."""
        self._rows_by_action = {}
        for local_idx, global_idx in enumerate(self.current_student_indices):
            action_id = self.df.iloc[global_idx]["category_id"]
            self._rows_by_action.setdefault(action_id, []).append(local_idx)
    
    def valid_action_ids(self) -> List[int]:
        """Get valid action IDs for current state."""
        valid_ids = []
        for action_id, local_indices in self._rows_by_action.items():
            if any(idx not in self._consumed for idx in local_indices):
                valid_ids.append(action_id)
        return valid_ids
    
    def step(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, Dict[str, Any]]:
        """Take environment step."""
        if len(self._consumed) >= len(self.current_student_indices):
            return None, 0.0, True, {}
        
        # Find valid row for action
        chosen_idx = self._find_valid_row(action)
        valid_action = chosen_idx is not None
        
        if chosen_idx is None:
            # Invalid action: auto-advance
            chosen_idx = self._get_next_unconsumed()
            if chosen_idx is None:
                return None, 0.0, True, {}
            reward, info = self._compute_invalid_reward(chosen_idx)
        else:
            # Valid action
            reward, info = self._compute_valid_reward(chosen_idx, action)
        
        # Update state
        self._consumed.add(chosen_idx)
        self._cur_idx = chosen_idx
        self._step_t += 1
        
        # Check if done
        done = len(self._consumed) >= len(self.current_student_indices)
        
        # Get next state
        next_state = None
        if not done:
            next_state = self.state_cache[self.current_student_indices[chosen_idx]].copy()
        
        info["valid_action"] = valid_action
        return next_state, reward, done, info
    
    def _find_valid_row(self, action: int) -> Optional[int]:
        """Find valid row index for given action."""
        local_indices = self._rows_by_action.get(action, [])
        for idx in local_indices:
            if idx not in self._consumed:
                return idx
        return None
    
    def _get_next_unconsumed(self) -> Optional[int]:
        """Get next unconsumed row index."""
        for idx in range(len(self.current_student_indices)):
            if idx not in self._consumed:
                return idx
        return None
    
    def _compute_valid_reward(self, local_idx: int, action: int) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for valid action."""
        global_idx = self.current_student_indices[local_idx]
        row = self.df.iloc[global_idx]
        
        cat_id = int(row["category_id"])
        norm_score = float(row["normalized_score"])
        
        # Base reward
        base_raw = self.rw_score * norm_score
        
        # Shaping components
        improve, deficit, spacing, diversity, challenge = self._compute_shaping_terms(cat_id, norm_score)
        
        # Group contributions
        mastery_raw = (
            self.config.rew_improve_w * improve +
            self.config.rew_deficit_w * deficit +
            self.config.rew_spacing_w * spacing
        )
        motivation_raw = (
            self.config.rew_diversity_w * diversity +
            self.config.rew_challenge_w * challenge
        )
        
        # Apply hybrid weights
        base_contrib = self.config.hybrid_base_w * base_raw
        mastery_contrib = self.config.hybrid_mastery_w * mastery_raw
        motivation_contrib = self.config.hybrid_motivation_w * motivation_raw
        
        total_reward = base_contrib + mastery_contrib + motivation_contrib
        
        # Update trackers
        self._update_trackers(cat_id, norm_score)
        
        # Build info dict
        reward_norm = total_reward / self._reward_den if self._reward_den > 1e-9 else float('nan')
        
        info = {
            "base_reward": base_raw,
            "improve": improve,
            "deficit": deficit,
            "spacing": spacing,
            "diversity": diversity,
            "challenge": challenge,
            "shaping_reward": mastery_contrib + motivation_contrib,
            "reward_base_contrib": base_contrib,
            "reward_mastery": mastery_contrib,
            "reward_motivation": motivation_contrib,
            "reward_norm": reward_norm,
        }
        
        return float(total_reward), info
    
    def _compute_invalid_reward(self, local_idx: int) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for invalid action (auto-advance)."""
        global_idx = self.current_student_indices[local_idx]
        row = self.df.iloc[global_idx]
        
        cat_id = int(row["category_id"])
        norm_score = float(row["normalized_score"])
        
        # Update trackers even for invalid actions
        self._update_trackers(cat_id, norm_score)
        
        info = {
            "base_reward": 0.0,
            "improve": 0.0,
            "deficit": 0.0,
            "spacing": 0.0,
            "diversity": 0.0,
            "challenge": 0.0,
            "shaping_reward": 0.0,
            "reward_base_contrib": 0.0,
            "reward_mastery": 0.0,
            "reward_motivation": 0.0,
            "reward_norm": float('nan'),
        }
        
        return float(self.config.invalid_penalty), info
    
    def _compute_shaping_terms(self, cat_id: int, norm_score: float) -> Tuple[float, float, float, float, float]:
        """Compute all shaping reward terms."""
        # Improvement over EMA baseline
        ema_prev = self._ema_scores.get(cat_id, self.category_mean_score[cat_id])
        improve = max(0.0, norm_score - ema_prev)
        
        # Deficit relative to mastery threshold
        deficit = max(0.0, self.config.need_threshold - ema_prev)
        
        # Spacing bonus
        last_seen = self._last_step_seen.get(cat_id)
        if last_seen is None:
            spacing = 0.5
        else:
            gap = max(0, self._step_t - last_seen)
            spacing = gap / (gap + max(1, self.config.spacing_window))
        
        # Diversity bonus
        recent_list = list(self._recent_choices)
        if len(recent_list) == 0:
            diversity = 0.5
        else:
            freq = recent_list.count(cat_id)
            diversity = 1.0 - (freq / len(recent_list)) if freq > 0 else 0.5
        
        # Challenge proximity
        band = max(1e-6, self.config.challenge_band)
        challenge = max(0.0, 1.0 - abs(norm_score - self.config.challenge_target) / band)
        
        return improve, deficit, spacing, diversity, challenge
    
    def _update_trackers(self, cat_id: int, norm_score: float) -> None:
        """Update episode trackers."""
        # Update EMA
        prev_ema = self._ema_scores.get(cat_id, self.category_mean_score[cat_id])
        new_ema = (1.0 - self.config.ema_alpha) * prev_ema + self.config.ema_alpha * norm_score
        self._ema_scores[cat_id] = new_ema
        
        # Update last seen
        self._last_step_seen[cat_id] = self._step_t
        
        # Update recent choices
        self._recent_choices.append(cat_id)
    
    def estimate_immediate_reward(self, action: int) -> float:
        """Estimate immediate reward without state mutation."""
        try:
            local_idx = self._find_valid_row(action)
            if local_idx is None:
                return float(self.config.invalid_penalty)
            
            global_idx = self.current_student_indices[local_idx]
            row = self.df.iloc[global_idx]
            
            cat_id = int(row["category_id"])
            norm_score = float(row["normalized_score"])
            
            # Compute without updating trackers
            base_raw = self.rw_score * norm_score
            improve, deficit, spacing, diversity, challenge = self._compute_shaping_terms(cat_id, norm_score)
            
            mastery_raw = (
                self.config.rew_improve_w * improve +
                self.config.rew_deficit_w * deficit +
                self.config.rew_spacing_w * spacing
            )
            motivation_raw = (
                self.config.rew_diversity_w * diversity +
                self.config.rew_challenge_w * challenge
            )
            
            base_contrib = self.config.hybrid_base_w * base_raw
            mastery_contrib = self.config.hybrid_mastery_w * mastery_raw
            motivation_contrib = self.config.hybrid_motivation_w * motivation_raw
            
            return float(base_contrib + mastery_contrib + motivation_contrib)
            
        except Exception:
            return 0.0
    
    def _build_state_from_row(self, row) -> np.ndarray:
        """Build state vector from row (for compatibility)."""
        global_idx = self.df.index[self.df["exercise_id"] == row["exercise_id"]].tolist()[0]
        return self.state_cache[global_idx].copy()
