from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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


class CurriculumEnvV2:
    """
    Offline environment built from preprocessed CSV.

    State (training-safe):
    - one-hot current category
    - order_norm (global 0-1)
    - normalized_score (current)
    - grade_encoded (label-encoded)
    - sex_binary (Girl=1, Boy=0, unknown=0.5)
    - home_school_lang_match (0/1, NaN->0.5)
    - missing_all, missing_beginning30, missing_last50 (NaN->0)

    Action space: category id to present next.

    Reward (shaped): rw_correct * 1{action == target_next_category} + rw_score * next_normalized_score
    """

    def __init__(self, data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42,
                 reward_correct_w: float = 0.5, reward_score_w: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.df = pd.read_csv(data_path)

        # Reward weighting (normalized to sum to 1 when possible)
        total_w = float(reward_correct_w) + float(reward_score_w)
        if total_w <= 0:
            self.rw_correct, self.rw_score = 1.0, 0.0
        else:
            self.rw_correct = float(reward_correct_w) / total_w
            self.rw_score = float(reward_score_w) / total_w

        # Required columns check
        required = [
            "student_id", "exercise_id", "category", "order", "normalized_score",
            "grade", "sex", "home_school_lang_match",
            "missing_all", "missing_beginning30", "missing_last50"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Category encoding and action space (use fine-grained 'category')
        # Always use the descriptive per-exercise 'category' field for sequencing.
        self.category_col = "category"
        self.df[self.category_col] = self.df[self.category_col].fillna("Unknown")
        self.categories: List[str] = sorted(self.df[self.category_col].unique())
        self.cat2id: Dict[str, int] = {c: i for i, c in enumerate(self.categories)}
        self.action_size = len(self.categories)

        # Global order normalization (consistent across students)
        order_min = self.df["order"].min()
        order_max = self.df["order"].max()
        self.df["order_norm"] = (self.df["order"] - order_min) / (order_max - order_min + 1e-9)

        # Sex
        self.df["sex_bin"] = self.df["sex"].apply(_standardize_sex)

        # Grade encoding
        self.df["grade"] = self.df["grade"].astype(str).fillna("Unknown")
        grades = sorted(self.df["grade"].unique())
        self.grade2id = {g: i for i, g in enumerate(grades)}
        self.df["grade_enc"] = self.df["grade"].map(self.grade2id).astype(float)
        if len(grades) > 1:
            self.df["grade_enc"] = self.df["grade_enc"] / (len(grades) - 1)

        # Language match and missingness
        self.df["home_school_lang_match"] = self.df["home_school_lang_match"].fillna(0.5).astype(float)
        for col in ["missing_all", "missing_beginning30", "missing_last50"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)

        # One-hot category for state
        self.df["category_id"] = self.df[self.category_col].map(self.cat2id).astype(int)

        # Build per-student ordered data
        self.df.sort_values(["student_id", "order"], inplace=True)

        # Split by students
        self.splits = self._split_students(train_ratio, val_ratio, seed)

        # State dimension = one-hot(category) + [order_norm, norm_score, grade_enc, sex_bin, lang_match, missing_all, missing_beginning30, missing_last50]
        self.state_dim = self.action_size + 8

        # Episode internals
        self.current_student_id = None
        self.current_student_df = None
        self.ptr = 0

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
        self.current_student_df = self.df[self.df["student_id"] == self.current_student_id]
        self.ptr = 0
        return self._build_state_from_row(self.current_student_df.iloc[self.ptr])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # If at the last row for this student, episode ends
        if self.ptr >= len(self.current_student_df) - 1:
            return None, 0.0, True, {}

        next_row = self.current_student_df.iloc[self.ptr + 1]
        target_cat = int(next_row["category_id"])
        corr = 1.0 if int(action) == target_cat else 0.0
        next_norm_score = float(next_row.get("normalized_score", 0.0))
        reward = self.rw_correct * corr + self.rw_score * next_norm_score

        self.ptr += 1
        done = self.ptr >= len(self.current_student_df) - 1
        next_state = self._build_state_from_row(self.current_student_df.iloc[self.ptr])
        return next_state, float(reward), bool(done), {"correct": corr, "target": target_cat}


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
    - State format matches `CurriculumEnvV2`: one-hot(category) + 8 numeric feats,
      so existing models can train without modification.
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

        # Episode vars
        self.current_student_id: Optional[int] = None
        self.current_student_df: Optional[pd.DataFrame] = None
        self._rows_by_action: Dict[int, List[int]] = {}
        self._consumed: set = set()
        self._cur_idx: int = 0

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
            # Invalid action: auto-advance to next unconsumed row with zero reward
            chosen_idx = self._pick_first_unconsumed()
            if chosen_idx is None:
                return None, 0.0, True, {}
            reward = 0.0
        else:
            # Reward equals (weighted) normalized score of chosen exercise
            row = self.current_student_df.iloc[chosen_idx]
            norm_score = float(row.get("normalized_score", 0.0))
            reward = self.rw_score * norm_score  # ignore correctness term in interactive mode

        # Advance state
        self._cur_idx = chosen_idx
        self._consumed.add(chosen_idx)
        done = len(self._consumed) >= len(self.current_student_df)
        next_state = None if done else self._build_state_from_row(self.current_student_df.iloc[self._cur_idx])
        return next_state, float(reward), bool(done), {"valid_action": bool(valid)}
