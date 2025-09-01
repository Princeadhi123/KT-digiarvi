"""Poster-ready plotting for curriculum sequencing RL results.

Usage:
  python -m curriculum_sequencing_rl.plot_poster \
    --csv "path/to/metrics1.csv" --csv "path/to/metrics2.csv" \
    --outdir "c:/Users/you/Desktop/KT digiarvi/curriculum_sequencing_rl/poster" \
    --models ql,dqn,a2c,a3c,ppo,sarl --select latest

It expects CSV(s) produced by ExperimentRunner._save_results_to_csv(),
which contain keys like: model, reward, vpr, regret_ratio, reward_base, reward_norm,
reward_base_contrib, reward_mastery, reward_motivation, term_*.

If some fields are missing, the script will gracefully skip plots requiring them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Global style for poster-quality figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

MODEL_ORDER = ["QL", "DQN", "A2C", "A3C", "PPO", "SARL"]
MODEL_COLORS: Dict[str, str] = {
    "QL": "#4c78a8",
    "DQN": "#f58518",
    "A2C": "#54a24b",
    "A3C": "#e45756",
    "PPO": "#72b7b2",
    "SARL": "#b279a2",
}


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _read_csvs(csv_paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        if not p.exists():
            print(f"[WARN] CSV not found: {p}")
            continue
        try:
            df = pd.read_csv(p)
            df["__source_csv"] = str(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not frames:
        raise FileNotFoundError("No readable CSVs provided.")
    df_all = pd.concat(frames, ignore_index=True, sort=False)
    # Normalize model labels
    if "model" in df_all.columns:
        df_all["model"] = df_all["model"].astype(str).str.upper()
    return df_all


def _coerce_timestamp(ts: str) -> float:
    # Return sortable key; if parse fails, return 0
    try:
        # ISO8601 sorts lexicographically, but be safe and parse via pandas
        return pd.to_datetime(ts, utc=True, errors="coerce").value
    except Exception:
        return 0.0


def _select_per_model(df: pd.DataFrame, models: List[str], strategy: str = "latest") -> pd.DataFrame:
    if "model" not in df.columns:
        raise ValueError("CSV missing 'model' column.")
    df = df[df["model"].isin([m.upper() for m in models])].copy()
    if df.empty:
        raise ValueError("No rows for requested models in provided CSVs.")

    # Handle missing timestamp gracefully
    if "timestamp" not in df.columns:
        # Take last occurrence per model by index order
        return df.sort_index().groupby("model", as_index=False).tail(1)

    df["__ts_key"] = df["timestamp"].astype(str).map(_coerce_timestamp)
    if strategy == "latest":
        idx = df.groupby("model")["__ts_key"].idxmax()
        return df.loc[idx].copy()
    elif strategy == "max_reward":
        if "reward" not in df.columns:
            return df.sort_index().groupby("model", as_index=False).tail(1)
        idx = df.groupby("model")["reward"].idxmax()
        return df.loc[idx].copy()
    elif strategy == "min_regret":
        if "regret_ratio" not in df.columns:
            return df.sort_index().groupby("model", as_index=False).tail(1)
        idx = df.groupby("model")["regret_ratio"].idxmin()
        return df.loc[idx].copy()
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def _compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Percent fields if absent
    if "vpr" in df.columns and "vpr_pct" not in df.columns:
        df["vpr_pct"] = df["vpr"].astype(float) * 100.0
    if "regret_ratio" in df.columns and "regret_ratio_pct" not in df.columns:
        df["regret_ratio_pct"] = df["regret_ratio"].astype(float) * 100.0
    if "reward_norm" in df.columns and "reward_norm_pct" not in df.columns:
        df["reward_norm_pct"] = df["reward_norm"].astype(float) * 100.0
    if "reward_base" in df.columns and "reward_base_pct" not in df.columns:
        df["reward_base_pct"] = df["reward_base"].astype(float) * 100.0

    # Hybrid share pct, compute if missing and parts exist
    has_parts = all(c in df.columns for c in [
        "reward_base_contrib", "reward_mastery", "reward_motivation"
    ])
    if has_parts and not all(c in df.columns for c in [
        "hybrid_base_share_pct", "hybrid_mastery_share_pct", "hybrid_motivation_share_pct"
    ]):
        total = (
            df["reward_base_contrib"].astype(float)
            + df["reward_mastery"].astype(float)
            + df["reward_motivation"].astype(float)
        )
        # Avoid division by zero
        total = total.replace(0, np.nan)
        df["hybrid_base_share_pct"] = (df["reward_base_contrib"].astype(float) / total) * 100.0
        df["hybrid_mastery_share_pct"] = (df["reward_mastery"].astype(float) / total) * 100.0
        df["hybrid_motivation_share_pct"] = (df["reward_motivation"].astype(float) / total) * 100.0
    return df


def _ordered_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure consistent model order
    order = [m for m in MODEL_ORDER if m in df["model"].unique().tolist()]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    return df.sort_values("model")


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", dy: float = 0.01) -> None:
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, max(dy, 0.005)),
            textcoords="offset points",
            clip_on=False,
        )


def plot_reward_bar(df: pd.DataFrame, outdir: Path) -> None:
    if "reward" not in df.columns:
        print("[INFO] Skipping reward bar (no 'reward' column)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="model",
        y="reward",
        hue="model",
        dodge=False,
        legend=False,
        palette=[MODEL_COLORS.get(m, "#999999") for m in df["model"].astype(str)],
        ax=ax,
    )
    ax.set_title("Average Shaped Reward (Test)")
    ax.set_xlabel("")
    ax.set_ylabel("Avg reward")
    ax.margins(y=0.15)
    _annotate_bars(ax, fmt="{:.3f}")
    sns.despine()
    fig.tight_layout()
    fig.savefig(outdir / "poster_reward_bar.png", dpi=300)
    plt.close(fig)


def plot_vpr_bar(df: pd.DataFrame, outdir: Path) -> None:
    col = "vpr_pct" if "vpr_pct" in df.columns else ("vpr" if "vpr" in df.columns else None)
    if col is None:
        print("[INFO] Skipping VPR bar (no 'vpr' column)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = df[col]
    sns.barplot(
        data=df,
        x="model",
        y=col,
        hue="model",
        dodge=False,
        legend=False,
        palette=[MODEL_COLORS.get(m, "#999999") for m in df["model"].astype(str)],
        ax=ax,
    )
    ax.set_title("Valid Pick Rate (VPR)")
    ax.set_xlabel("")
    ax.set_ylabel("VPR (%)" if col.endswith("pct") else "VPR")
    fmt = "{:.1f}%" if col.endswith("pct") else "{:.3f}"
    ax.margins(y=0.15)
    _annotate_bars(ax, fmt=fmt)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outdir / "poster_vpr_bar.png", dpi=300)
    plt.close(fig)


def plot_regret_ratio_bar(df: pd.DataFrame, outdir: Path) -> None:
    col = "regret_ratio_pct" if "regret_ratio_pct" in df.columns else ("regret_ratio" if "regret_ratio" in df.columns else None)
    if col is None:
        print("[INFO] Skipping regret ratio bar (no 'regret_ratio' column)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="model",
        y=col,
        hue="model",
        dodge=False,
        legend=False,
        palette=[MODEL_COLORS.get(m, "#999999") for m in df["model"].astype(str)],
        ax=ax,
    )
    ax.set_title("Instantaneous Regret Ratio (Lower is better)")
    ax.set_xlabel("")
    ax.set_ylabel("Regret ratio (%)" if col.endswith("pct") else "Regret ratio")
    # Two-decimal formatting for both annotations and y-axis ticks
    fmt = "{:.2f}%" if col.endswith("pct") else "{:.2f}"
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.margins(y=0.15)
    _annotate_bars(ax, fmt=fmt)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outdir / "poster_regret_ratio_bar.png", dpi=300)
    plt.close(fig)


def plot_hybrid_shares(df: pd.DataFrame, outdir: Path) -> None:
    needed = ["hybrid_base_share_pct", "hybrid_mastery_share_pct", "hybrid_motivation_share_pct"]
    if not all(c in df.columns for c in needed):
        print("[INFO] Skipping hybrid shares (missing columns)")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = df["model"].astype(str).tolist()
    base = df["hybrid_base_share_pct"].astype(float).to_numpy()
    mast = df["hybrid_mastery_share_pct"].astype(float).to_numpy()
    motiv = df["hybrid_motivation_share_pct"].astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.6
    p1 = ax.bar(x, base, width, label="Base", color="#4e79a7")
    p2 = ax.bar(x, mast, width, bottom=base, label="Mastery", color="#59a14f")
    p3 = ax.bar(x, motiv, width, bottom=base+mast, label="Motivation", color="#f28e2b")

    ax.set_title("Hybrid Contribution Shares (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share (%)")
    ax.margins(y=0.12)
    ax.legend(
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    # Annotate totals to 100%
    for i in range(len(labels)):
        total = base[i] + mast[i] + motiv[i]
        ax.annotate(
            f"{total:.0f}%",
            (x[i], total),
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
            clip_on=False,
        )

    sns.despine()
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(outdir / "poster_hybrid_shares_stacked.png", dpi=300)
    plt.close(fig)


def plot_shaping_terms(df: pd.DataFrame, outdir: Path) -> None:
    cols = ["term_improve", "term_deficit", "term_spacing", "term_diversity", "term_challenge"]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        print("[INFO] Skipping shaping terms (no term_* columns)")
        return
    # Melt for grouped bar plot
    dd = df.melt(id_vars=["model"], value_vars=avail, var_name="term", value_name="avg_value")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=dd, x="model", y="avg_value", hue="term", ax=ax)
    ax.set_title("Shaping Components (average per step)")
    ax.set_xlabel("")
    ax.set_ylabel("Avg component value")
    ax.margins(y=0.12)
    ax.legend(
        title="Term",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    sns.despine()
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(outdir / "poster_shaping_terms.png", dpi=300)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, outdir: Path) -> List[Path]:
    out_paths: List[Path] = []
    plot_reward_bar(df, outdir)
    out_paths.append(outdir / "poster_reward_bar.png")
    plot_vpr_bar(df, outdir)
    out_paths.append(outdir / "poster_vpr_bar.png")
    plot_regret_ratio_bar(df, outdir)
    out_paths.append(outdir / "poster_regret_ratio_bar.png")
    plot_hybrid_shares(df, outdir)
    out_paths.append(outdir / "poster_hybrid_shares_stacked.png")
    plot_shaping_terms(df, outdir)
    out_paths.append(outdir / "poster_shaping_terms.png")
    return out_paths


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    # Default to a 'poster' folder inside curriculum_sequencing_rl
    default_outdir = here.parent / "poster"
    p = argparse.ArgumentParser(description="Generate poster-ready plots for RL models.")
    p.add_argument("--csv", dest="csvs", action="append", required=True,
                   help="Path to a metrics CSV. Provide multiple --csv to merge.")
    p.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory for images.")
    p.add_argument("--models", type=str, default="ql,dqn,a2c,a3c,ppo,sarl",
                   help="Comma-separated models to include.")
    p.add_argument("--select", type=str, default="latest", choices=["latest", "max_reward", "min_regret"],
                   help="How to select one row per model when multiple exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    csv_paths = [Path(s) for s in args.csvs]
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]

    df_all = _read_csvs(csv_paths)
    df_all = _compute_derived_fields(df_all)
    df_sel = _select_per_model(df_all, models=models, strategy=args.select)
    df_sel = _compute_derived_fields(df_sel)
    df_sel = _ordered_df(df_sel)

    print("Selected rows for models:")
    print(df_sel[[c for c in ["model", "timestamp", "reward", "vpr", "regret_ratio", "__source_csv"] if c in df_sel.columns]].to_string(index=False))

    paths = generate_all_plots(df_sel, outdir)
    print("Saved plots:")
    for p in paths:
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
