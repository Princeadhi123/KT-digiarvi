from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from collections import deque
import numpy as np
import pandas as pd


def _standardize_sex(val: str) -> float:
    if pd.isna(val):
        return 0.5
    s = str(val).strip().lower()
    if s in ("boy", "male", "m"):
        return 0.0
    if s in ("girl", "gir", "female", "f"):
        return 1.0
    return 0.5


@dataclass
class SplitData:
    train_students: np.ndarray
    val_students: np.ndarray
    test_students: np.ndarray

class InteractiveReorderEnv:
    """
    Interactive environment that lets the agent choose WHICH exercise to do next
    for the current student by selecting an action from the global action space.

    Key idea: We treat the per-student logged exercises as a pool. At each step,
    the agent picks an action (exercise category id), and the environment jumps to
    that exercise for this student (if it exists and wasn't used yet) and grants
    reward equal to that exercise's recorded normalized_score. This evaluates
    learning impact directly (average normalized_score) and makes actions affect
    what happens next, unlike simple logged playback.

    Notes/assumptions:
    - Action space uses fine-grained `category` by default to stay consistent with
      existing models; you can change to `category_group` by setting `action_on`.
    - If the agent chooses an action that this student never attempted (or is
      already consumed), the env auto-advances to the next remaining exercise in
      the student's original order with zero reward for that step. This guarantees
      progress and penalizes invalid choices.
    - Reward ignores next-label correctness and equals rw_score * chosen_norm_score.
    - State format: one-hot(category) + 8 numeric feats; consistent across models.
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        reward_correct_w: float = 0.0,
        reward_score_w: float = 1.0,
        action_on: str = "category",  # or "category_group"
        # Multi-objective shaping weights (all default to 0.0 for backward-compat)
        rew_improve_w: float = 0.0,
        rew_deficit_w: float = 0.0,
        rew_spacing_w: float = 0.0,
        rew_diversity_w: float = 0.0,
        rew_challenge_w: float = 0.0,
        # Shaping hyperparameters
        ema_alpha: float = 0.3,
        need_threshold: float = 0.6,
        spacing_window: int = 5,
        diversity_recent_k: int = 5,
        challenge_target: float = 0.7,
        challenge_band: float = 0.4,
        invalid_penalty: float = 0.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.df = pd.read_csv(data_path)

        # Reward weights: default to score-only
        total_w = float(reward_correct_w) + float(reward_score_w)
        if total_w <= 0:
            self.rw_correct, self.rw_score = 0.0, 1.0
        else:
            self.rw_correct = float(reward_correct_w) / total_w
            self.rw_score = float(reward_score_w) / total_w

        # Store shaping weights and params
        self.rew_improve_w = float(rew_improve_w)
        self.rew_deficit_w = float(rew_deficit_w)
        self.rew_spacing_w = float(rew_spacing_w)
        self.rew_diversity_w = float(rew_diversity_w)
        self.rew_challenge_w = float(rew_challenge_w)

        self.ema_alpha = float(ema_alpha)
        self.need_threshold = float(need_threshold)
        self.spacing_window = int(spacing_window)
        self.diversity_recent_k = int(diversity_recent_k)
        self.challenge_target = float(challenge_target)
        self.challenge_band = float(challenge_band)
        self.invalid_penalty = float(invalid_penalty)

        # Precompute reward normalization denominator for reporting (sum of weights)
        self._reward_den = float(
            self.rw_score
            + self.rew_improve_w
            + self.rew_deficit_w
            + self.rew_spacing_w
            + self.rew_diversity_w
            + self.rew_challenge_w
        )

        # Basic checks
        required = [
            "student_id", "exercise_id", "category", "category_group", "order", "normalized_score",
            "grade", "sex", "home_school_lang_match",
            "missing_all", "missing_beginning30", "missing_last50",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Action key (category vs category_group)
        if action_on not in ("category", "category_group"):
            raise ValueError("action_on must be 'category' or 'category_group'")
        self.category_col = action_on
        self.df[self.category_col] = self.df[self.category_col].fillna("Unknown")
        self.categories: List[str] = sorted(self.df[self.category_col].unique())
        self.cat2id: Dict[str, int] = {c: i for i, c in enumerate(self.categories)}
        self.action_size = len(self.categories)

        # Normalizations/encodings (same as V2)
        order_min = self.df["order"].min()
        order_max = self.df["order"].max()
        self.df["order_norm"] = (self.df["order"] - order_min) / (order_max - order_min + 1e-9)

        self.df["sex_bin"] = self.df["sex"].apply(_standardize_sex)
        self.df["grade"] = self.df["grade"].astype(str).fillna("Unknown")
        grades = sorted(self.df["grade"].unique())
        self.grade2id = {g: i for i, g in enumerate(grades)}
        self.df["grade_enc"] = self.df["grade"].map(self.grade2id).astype(float)
        if len(grades) > 1:
            self.df["grade_enc"] = self.df["grade_enc"] / (len(grades) - 1)

        self.df["home_school_lang_match"] = self.df["home_school_lang_match"].fillna(0.5).astype(float)
        for col in ["missing_all", "missing_beginning30", "missing_last50"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)

        # Action id for state one-hot
        self.df["category_id"] = self.df[self.category_col].map(self.cat2id).astype(int)

        # Sort by student/order and split students
        self.df.sort_values(["student_id", "order"], inplace=True)
        self.splits = self._split_students(train_ratio, val_ratio, seed)

        # State: one-hot + 8 numeric feats
        self.state_dim = self.action_size + 8

        # Precompute category mean scores for EMA initialization
        self.category_mean_score: Dict[int, float] = {}
        for aid in range(self.action_size):
            m = float(self.df[self.df["category_id"] == aid]["normalized_score"].astype(float).dropna().mean())
            if np.isnan(m):
                m = 0.5
            self.category_mean_score[aid] = m

        # Episode vars
        self.current_student_id: Optional[int] = None
        self.current_student_df: Optional[pd.DataFrame] = None
        self._rows_by_action: Dict[int, List[int]] = {}
        self._consumed: set = set()
        self._cur_idx: int = 0
        # Shaping trackers (episode-scoped)
        self._ema_scores: Dict[int, float] = {}
        self._last_step_seen: Dict[int, Optional[int]] = {}
        self._recent_choices: deque = deque(maxlen=self.diversity_recent_k)
        self._step_t: int = 0

    def _split_students(self, train_ratio: float, val_ratio: float, seed: int) -> SplitData:
        students = self.df["student_id"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(students)
        n = len(students)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_students = students[:n_train]
        val_students = students[n_train:n_train + n_val]
        test_students = students[n_train + n_val:]
        return SplitData(train_students, val_students, test_students)

    def _one_hot(self, idx: int, n: int) -> np.ndarray:
        v = np.zeros(n, dtype=np.float32)
        if 0 <= idx < n:
            v[idx] = 1.0
        return v

    def _build_state_from_row(self, row) -> np.ndarray:
        cat_oh = self._one_hot(int(row["category_id"]), self.action_size)
        x = np.array([
            float(row["order_norm"]),
            float(row.get("normalized_score", 0.0)),
            float(row["grade_enc"]),
            float(row["sex_bin"]),
            float(row["home_school_lang_match"]),
            float(row["missing_all"]),
            float(row["missing_beginning30"]),
            float(row["missing_last50"]),
        ], dtype=np.float32)
        return np.concatenate([cat_oh, x], dtype=np.float32)

    def _build_rows_by_action(self):
        self._rows_by_action = {}
        for idx, row in self.current_student_df.reset_index(drop=True).iterrows():
            aid = int(row["category_id"])
            self._rows_by_action.setdefault(aid, []).append(idx)

    def valid_action_ids(self) -> List[int]:
        """Return actions that map to at least one unconsumed row for this student."""
        out = []
        for aid, idxs in self._rows_by_action.items():
            if any((i not in self._consumed) for i in idxs):
                out.append(aid)
        return out

    def reset(self, mode: str = "train") -> np.ndarray:
        if mode == "train":
            sids = self.splits.train_students
        elif mode == "val":
            sids = self.splits.val_students
        elif mode == "test":
            sids = self.splits.test_students
        else:
            raise ValueError("mode must be one of {'train','val','test'}")
        if len(sids) == 0:
            raise ValueError(f"No students in split '{mode}'")
        self.current_student_id = int(self.rng.choice(sids))
        self.current_student_df = self.df[self.df["student_id"] == self.current_student_id].sort_values("order").reset_index(drop=True)
        if len(self.current_student_df) == 0:
            raise ValueError("Selected student has no rows")

        # Initialize per-episode structures
        self._build_rows_by_action()
        self._consumed = set([0])  # mark the first row as already done
        self._cur_idx = 0
        # Initialize shaping trackers
        self._ema_scores = {aid: float(self.category_mean_score.get(aid, 0.5)) for aid in range(self.action_size)}
        self._last_step_seen = {aid: None for aid in range(self.action_size)}
        self._recent_choices = deque(maxlen=self.diversity_recent_k)
        self._step_t = 0
        return self._build_state_from_row(self.current_student_df.iloc[self._cur_idx])

    def _pick_first_unconsumed(self) -> Optional[int]:
        for i in range(len(self.current_student_df)):
            if i not in self._consumed:
                return i
        return None

    def step(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        # If all consumed already except the initial one
        if len(self._consumed) >= len(self.current_student_df):
            return None, 0.0, True, {}

        valid = False
        chosen_idx: Optional[int] = None
        # Try to find an unconsumed row that matches the action
        idxs = self._rows_by_action.get(int(action), [])
        for i in idxs:
            if i not in self._consumed:
                chosen_idx = i
                valid = True
                break

        if chosen_idx is None:
            # Invalid action: auto-advance to next unconsumed row; give optional penalty
            chosen_idx = self._pick_first_unconsumed()
            if chosen_idx is None:
                return None, 0.0, True, {}
            row = self.current_student_df.iloc[chosen_idx]
            cat_id = int(row["category_id"])
            norm_score = float(row.get("normalized_score", 0.0))
            # Update trackers to reflect the consumed exercise (even though invalid choice)
            prev_ema = float(self._ema_scores.get(cat_id, self.category_mean_score.get(cat_id, 0.5)))
            new_ema = (1.0 - self.ema_alpha) * prev_ema + self.ema_alpha * norm_score
            self._ema_scores[cat_id] = new_ema
            self._last_step_seen[cat_id] = self._step_t
            try:
                self._recent_choices.append(cat_id)
            except Exception:
                pass
            base_reward = 0.0
            improve = 0.0
            deficit = 0.0
            spacing = 0.0
            diversity = 0.0
            challenge = 0.0
            shaping_reward = 0.0
            reward = float(self.invalid_penalty)
        else:
            # Reward equals (weighted) normalized score of chosen exercise + shaping terms
            row = self.current_student_df.iloc[chosen_idx]
            cat_id = int(row["category_id"])
            norm_score = float(row.get("normalized_score", 0.0))
            base_reward = self.rw_score * norm_score  # ignore correctness term in interactive mode

            # Shaping terms (all in [0,1] approx)
            ema_prev = float(self._ema_scores.get(cat_id, self.category_mean_score.get(cat_id, 0.5)))
            # Improvement over running baseline (positive-only to avoid double-penalizing)
            improve = max(0.0, norm_score - ema_prev)
            # Practice where weak (deficit relative to target mastery)
            deficit = max(0.0, self.need_threshold - ema_prev)
            # Spacing: higher if category not practiced recently
            last_seen = self._last_step_seen.get(cat_id)
            if last_seen is None:
                spacing = 1.0
            else:
                gap = max(0, int(self._step_t) - int(last_seen))
                spacing = min(1.0, (gap / max(1, self.spacing_window)))
            # Diversity: bonus if not in recent window
            try:
                diversity = 1.0 if (self.diversity_recent_k > 0 and (cat_id not in self._recent_choices)) else 0.0
            except Exception:
                diversity = 0.0
            # Challenge: encourage moderate difficulty near target score
            band = max(1e-6, self.challenge_band)
            challenge = max(0.0, 1.0 - abs(norm_score - self.challenge_target) / band)

            shaping_reward = (
                self.rew_improve_w * improve
                + self.rew_deficit_w * deficit
                + self.rew_spacing_w * spacing
                + self.rew_diversity_w * diversity
                + self.rew_challenge_w * challenge
            )
            reward = float(base_reward + shaping_reward)

            # Update trackers after applying reward
            new_ema = (1.0 - self.ema_alpha) * ema_prev + self.ema_alpha * norm_score
            self._ema_scores[cat_id] = new_ema
            self._last_step_seen[cat_id] = self._step_t
            try:
                self._recent_choices.append(cat_id)
            except Exception:
                pass

        # Build info and advance state
        den = self._reward_den
        reward_norm = float(reward / den) if den > 1e-9 else float("nan")
        info = {
            "valid_action": bool(valid),
            "base_reward": float(base_reward) if 'base_reward' in locals() else 0.0,
            "improve": float(improve) if 'improve' in locals() else 0.0,
            "deficit": float(deficit) if 'deficit' in locals() else 0.0,
            "spacing": float(spacing) if 'spacing' in locals() else 0.0,
            "diversity": float(diversity) if 'diversity' in locals() else 0.0,
            "challenge": float(challenge) if 'challenge' in locals() else 0.0,
            "shaping_reward": float(shaping_reward) if 'shaping_reward' in locals() else 0.0,
            "reward_norm": reward_norm,
        }

        # Advance state
        self._cur_idx = chosen_idx
        self._consumed.add(chosen_idx)
        done = len(self._consumed) >= len(self.current_student_df)
        self._step_t += 1
        next_state = None if done else self._build_state_from_row(self.current_student_df.iloc[self._cur_idx])
        return next_state, float(reward), bool(done), info

    def estimate_immediate_reward(self, action: int) -> float:
        """Estimate the immediate reward if `action` were taken now, without mutating state.

        Used by diagnostics (e.g., regret). Falls back to base score-only estimate if shaping cannot be computed.
        """
        try:
            idxs = self._rows_by_action.get(int(action), [])
            chosen_idx = None
            for i in idxs:
                if i not in self._consumed:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                return float(self.invalid_penalty)
            row = self.current_student_df.iloc[chosen_idx]
            cat_id = int(row["category_id"])
            norm_score = float(row.get("normalized_score", 0.0))
            base = self.rw_score * norm_score

            # Compute shaping components using current trackers (no mutation)
            ema_prev = float(self._ema_scores.get(cat_id, self.category_mean_score.get(cat_id, 0.5)))
            improve = max(0.0, norm_score - ema_prev)
            deficit = max(0.0, self.need_threshold - ema_prev)
            last_seen = self._last_step_seen.get(cat_id)
            if last_seen is None:
                spacing = 1.0
            else:
                gap = max(0, int(self._step_t) - int(last_seen))
                spacing = min(1.0, (gap / max(1, self.spacing_window)))
            try:
                diversity = 1.0 if (self.diversity_recent_k > 0 and (cat_id not in self._recent_choices)) else 0.0
            except Exception:
                diversity = 0.0
            band = max(1e-6, self.challenge_band)
            challenge = max(0.0, 1.0 - abs(norm_score - self.challenge_target) / band)

            return float(
                base
                + self.rew_improve_w * improve
                + self.rew_deficit_w * deficit
                + self.rew_spacing_w * spacing
                + self.rew_diversity_w * diversity
                + self.rew_challenge_w * challenge
            )
        except Exception:
            # Fallback: score-only
            try:
                idxs = self._rows_by_action.get(int(action), [])
                for i in idxs:
                    if i not in self._consumed:
                        row = self.current_student_df.iloc[i]
                        norm_score = float(row.get("normalized_score", 0.0))
                        return float(self.rw_score * norm_score)
            except Exception:
                pass
            return 0.0
