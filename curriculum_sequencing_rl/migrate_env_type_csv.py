import argparse
import csv
import os
import shutil
from typing import List


def detect_env_type(row: List[str], header: List[str]) -> str:
    # Heuristics:
    # 1) If the field after seed (reward_correct_w) accidentally contains 'interactive'/'passive', use that.
    # 2) Else, if acc field is empty -> interactive. Otherwise passive.
    try:
        acc_idx = header.index("acc")
        reward_correct_idx = header.index("reward_correct_w")
    except ValueError:
        # Unexpected header, fallback: if acc present use it, else default passive
        acc_idx = 2 if len(header) > 2 and header[2] == "acc" else None
        reward_correct_idx = 5 if len(header) > 5 else None

    env = None
    if reward_correct_idx is not None and len(row) > reward_correct_idx:
        val = (row[reward_correct_idx] or "").strip().lower()
        if val in ("interactive", "passive"):
            env = val
    if env is None:
        if acc_idx is not None and len(row) > acc_idx and (row[acc_idx] or "").strip() == "":
            env = "interactive"
        else:
            env = "passive"
    return env


def migrate(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read existing
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            # Empty file: just add header with env_type and exit
            header = []
            rows = []
        else:
            rows = [r for r in reader]

    if not header:
        # Create minimal header if empty
        header = ["timestamp", "model", "acc", "reward", "seed", "reward_correct_w", "reward_score_w"]

    # If 'env_type' already present, we still try to fix misaligned rows where reward_correct_w has 'interactive'
    has_env_type = "env_type" in header
    if not has_env_type:
        new_header = header + ["env_type"]
    else:
        new_header = header[:]

    # Ensure indices
    try:
        reward_correct_idx = header.index("reward_correct_w")
    except ValueError:
        reward_correct_idx = 5 if len(header) > 5 else None

    new_rows: List[List[str]] = []
    for row in rows:
        if not row:
            continue
        # Normalize row size to header length (trim or pad) before migration logic
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]

        env_type = detect_env_type(row, header)

        # Fix misalignment if reward_correct_w contains 'interactive'/'passive'
        if reward_correct_idx is not None and (row[reward_correct_idx] or "").strip().lower() in ("interactive", "passive"):
            # Shift left values from reward_correct_idx+1 onward
            fixed = row[:]
            for i in range(reward_correct_idx, len(header) - 1):
                fixed[i] = row[i + 1] if i + 1 < len(row) else ""
            # Last old-header column becomes empty after shift
            fixed[-1] = ""
            row = fixed

        # Append env_type if header lacked it
        if not has_env_type:
            row = row + [env_type]
        else:
            # If env_type column exists, fill it when empty
            env_idx = new_header.index("env_type")
            if env_idx >= len(row):
                row = row + [""] * (env_idx - len(row) + 1)
            if not row[env_idx]:
                row[env_idx] = env_type

        new_rows.append(row)

    # Backup original
    backup_path = csv_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copyfile(csv_path, backup_path)

    # Write migrated
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f"Migration complete. Wrote {len(new_rows)} rows. Backup at: {backup_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate experiment_metrics.csv to include env_type and fix misaligned rows.")
    parser.add_argument("--csv", default=os.path.join("curriculum_sequencing_rl", "experiment_metrics.csv"), help="Path to the metrics CSV")
    args = parser.parse_args()
    migrate(args.csv)
