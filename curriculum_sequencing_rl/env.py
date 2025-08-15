from dataclasses import dataclass
from typing import Dict, List, Tuple

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

        # Category encoding and action space
        self.df["category"] = self.df["category"].fillna("Unknown")
        self.categories: List[str] = sorted(self.df["category"].unique())
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
        self.df["category_id"] = self.df["category"].map(self.cat2id).astype(int)

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
