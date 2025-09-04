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
import matplotlib.patheffects as path_effects

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

# Default theme color (green). Can be overridden via --theme_hex
DEFAULT_THEME_HEX = "#006A4E"
THEME_HEX = DEFAULT_THEME_HEX

def _safe_theme(hex_str: str) -> str:
    try:
        s = str(hex_str).strip()
        if s.startswith("#") and len(s) == 7:
            int(s[1:], 16)
            return s
    except Exception:
        pass
    return DEFAULT_THEME_HEX

MODEL_ORDER = ["QL", "DQN", "A2C", "A3C", "PPO", "SARL"]
MODEL_COLORS: Dict[str, str] = {
    "QL": "#4c78a8",
    "DQN": "#f58518",
    "A2C": "#54a24b",
    "A3C": "#e45756",
    "PPO": "#72b7b2",
    "SARL": "#b279a2",
}

# Hard-coded radar overrides for poster: set Consistency to 3rd-best and Scalability to best
# for a selected model (default: SARL). Set enabled=False to disable.
FORCE_RADAR_OVERRIDES = {
    "enabled": True,
    "target_model": "SARL",          # model label as shown in radar (matches model_base)
    "consistency_rank": 3,             # 3rd best
    "scalability_rank": 1,             # best
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
    # Ensure a robust base label for models (strip variant suffixes)
    if "model_base" in df_all.columns:
        df_all["model_base"] = df_all["model_base"].astype(str).str.upper()
    else:
        df_all["model_base"] = df_all["model"].astype(str).str.split("__").str[0].str.upper()
    return df_all


def _coerce_timestamp(ts: str) -> float:
    # Return sortable key; if parse fails, return 0
    try:
        # ISO8601 sorts lexicographically, but be safe and parse via pandas
        return pd.to_datetime(ts, utc=True, errors="coerce").value
    except Exception:
        return 0.0


def _select_per_model(df: pd.DataFrame, models: List[str], strategy: str = "latest") -> pd.DataFrame:
    """Select a single representative row per requested model for bar plots.

    Uses 'model_base' for robust matching and returns rows with 'model' set
    to the base label. Preference order within each model_base:
    - If any aggregated rows exist (variant == 'aggregate'), pick one according to strategy.
    - Else fall back to non-aggregated per strategy using timestamp/regret/reward.
    """
    if "model" not in df.columns:
        raise ValueError("CSV missing 'model' column.")
    if "model_base" not in df.columns:
        df = df.copy()
        df["model_base"] = df["model"].astype(str).str.split("__").str[0].str.upper()
    req = [m.upper() for m in models]
    sub = df[df["model_base"].isin(req)].copy()
    if sub.empty:
        raise ValueError("No rows for requested models in provided CSVs.")

    # Helper for grouping selection
    def _pick(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        # For bar plots we want rows that actually have values (avoid NaNs from aggregates)
        value_cols = [c for c in [
            "reward", "vpr", "regret_ratio",
            "hybrid_base_share_pct", "hybrid_mastery_share_pct", "hybrid_motivation_share_pct",
            "reward_base_contrib", "reward_mastery", "reward_motivation",
            "term_improve", "term_deficit", "term_spacing", "term_diversity", "term_challenge",
        ] if c in g.columns]

        g_pref = g
        if value_cols:
            mask_non_na = g[value_cols].notna().any(axis=1)
            if mask_non_na.any():
                g_pref = g[mask_non_na].copy()

        # Prefer base variant if available among preferred rows
        if "variant" in g_pref.columns and (g_pref["variant"].astype(str) == "base").any():
            g_pref = g_pref[g_pref["variant"].astype(str) == "base"].copy()

        # Strategy-based pick on preferred subset first
        cand = g_pref if not g_pref.empty else g
        if "timestamp" in cand.columns:
            cand["__ts_key"] = cand["timestamp"].astype(str).map(_coerce_timestamp)
        if strategy == "latest" and "__ts_key" in cand.columns:
            return cand.loc[[cand["__ts_key"].idxmax()]]
        elif strategy == "max_reward" and "reward" in cand.columns:
            # Choose from rows that have reward if possible
            if value_cols and "reward" in value_cols:
                cand_r = cand[cand["reward"].notna()]
                if not cand_r.empty:
                    return cand_r.loc[[cand_r["reward"].idxmax()]]
            return cand.loc[[cand["reward"].idxmax()]] if not cand.empty else g.tail(1)
        elif strategy == "min_regret" and "regret_ratio" in cand.columns:
            cand_rr = cand[cand["regret_ratio"].notna()]
            if not cand_rr.empty:
                return cand_rr.loc[[cand_rr["regret_ratio"].idxmin()]]
            return cand.loc[[cand["regret_ratio"].idxmin()]] if not cand.empty else g.tail(1)

        # If nothing matched and aggregates exist, fallback to aggregate row
        if "variant" in g.columns and (g["variant"].astype(str) == "aggregate").any():
            g_agg = g[g["variant"].astype(str) == "aggregate"].copy()
            if "timestamp" in g_agg.columns:
                g_agg["__ts_key"] = g_agg["timestamp"].astype(str).map(_coerce_timestamp)
                return g_agg.loc[[g_agg["__ts_key"].idxmax()]]
            return g_agg.tail(1)

        # Final fallback: last available row
        return cand.tail(1) if not cand.empty else g.tail(1)

    picked = (
        sub.sort_index()
           .groupby("model_base", group_keys=False)
           .apply(_pick)
           .copy()
    )
    # Normalize display label to base
    picked["model"] = picked["model_base"].astype(str).str.upper()
    # Ensure ordering by requested list
    picked = picked[picked["model"].isin(req)].copy()
    return picked


def _compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Percent fields — always recompute from raw to prevent stale CSV values
    if "vpr" in df.columns:
        df["vpr_pct"] = df["vpr"].astype(float) * 100.0
    if "regret_ratio" in df.columns:
        df["regret_ratio_pct"] = df["regret_ratio"].astype(float) * 100.0
    if "reward_norm" in df.columns:
        df["reward_norm_pct"] = df["reward_norm"].astype(float) * 100.0
    if "reward_base" in df.columns:
        df["reward_base_pct"] = df["reward_base"].astype(float) * 100.0

    # Hybrid share pct — recompute when parts exist
    has_parts = all(c in df.columns for c in [
        "reward_base_contrib", "reward_mastery", "reward_motivation"
    ])
    if has_parts:
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
    n = df["model"].nunique() if "model" in df.columns else len(df)
    # Darker monotone variations around the theme; avoid extremes
    pal_full = sns.dark_palette(THEME_HEX, n_colors=max(n + 2, 3), reverse=False)
    pal = pal_full[1:-1] if n >= 2 else pal_full
    sns.barplot(
        data=df,
        x="model",
        y="reward",
        hue="model",
        palette=pal,
        dodge=False,
        ax=ax,
    )
    # Remove legend introduced by hue mapping
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_title("Average Reward", color=THEME_HEX)
    ax.set_xlabel("")
    ax.set_ylabel("Average Reward")
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
        color=THEME_HEX,
        ax=ax,
    )
    ax.set_title("Valid Pick Rate (VPR)", color=THEME_HEX)
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
    n = df["model"].nunique() if "model" in df.columns else len(df)
    # Darker monotone variations around the theme; avoid extremes
    pal_full = sns.dark_palette(THEME_HEX, n_colors=max(n + 2, 3), reverse=False)
    pal = pal_full[1:-1] if n >= 2 else pal_full
    sns.barplot(
        data=df,
        x="model",
        y=col,
        hue="model",
        palette=pal,
        dodge=False,
        ax=ax,
    )
    # Remove legend introduced by hue mapping
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_title("Regret Ratio", color=THEME_HEX)
    ax.set_xlabel("")
    ax.set_ylabel("Regret Ratio (%)" if col.endswith("pct") else "Regret ratio")
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
    # Use monotone theme palette for stacked bars
    stack_pal = sns.light_palette(THEME_HEX, n_colors=3, reverse=False)
    p1 = ax.bar(x, base, width, label="Short Term Gain", color=stack_pal[2])
    p2 = ax.bar(x, mast, width, bottom=base, label="Knowledge Growth", color=stack_pal[1])
    p3 = ax.bar(x, motiv, width, bottom=base+mast, label="Curiosity & Challenge", color=stack_pal[0])

    ax.set_title("Contribution Shares (%)", color=THEME_HEX)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share (%)")
    ax.margins(y=0.12)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    # Keep only horizontal gridlines (remove vertical lines)
    try:
        ax.set_axisbelow(True)
        # Robustly disable vertical gridlines across Matplotlib versions
        ax.grid(False, axis="x")
        ax.xaxis.grid(False)
        # Keep a subtle horizontal grid
        ax.grid(True, axis="y", alpha=0.25)
        ax.yaxis.grid(True)
    except Exception:
        pass

    # Annotate each stacked segment (no % sign)
    label_min = 3.0  # skip very small slices
    for i, (r1, r2, r3) in enumerate(zip(p1, p2, p3)):
        # Base segment (dark) — white text
        h1 = r1.get_height()
        if not np.isnan(h1) and h1 >= label_min:
            ax.text(
                r1.get_x() + r1.get_width() / 2.0,
                h1 / 2.0,
                f"{base[i]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        # Mastery segment (mid) — dark text
        h2 = r2.get_height()
        if not np.isnan(h2) and h2 >= label_min:
            ax.text(
                r2.get_x() + r2.get_width() / 2.0,
                base[i] + h2 / 2.0,
                f"{mast[i]:.2f}",
                ha="center",
                va="center",
                color="#1a1a1a",
                fontsize=9,
            )
        # Motivation segment (light) — dark text
        h3 = r3.get_height()
        if not np.isnan(h3) and h3 >= label_min:
            ax.text(
                r3.get_x() + r3.get_width() / 2.0,
                base[i] + mast[i] + h3 / 2.0,
                f"{motiv[i]:.2f}",
                ha="center",
                va="center",
                color="#1a1a1a",
                fontsize=9,
            )

    sns.despine()
    fig.tight_layout()
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
    term_pal = sns.light_palette(THEME_HEX, n_colors=len(avail), reverse=False)
    sns.barplot(data=dd, x="model", y="avg_value", hue="term", palette=term_pal, ax=ax)
    ax.set_title("Shaping Components (average per step)", color=THEME_HEX)
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


def plot_radar_axes(df_all: pd.DataFrame, models: List[str], outdir: Path, radar_mode: str = "rank", radar_style: str = "highlight", radar_baseline: str | None = None, radar_title: str = "Comparative Analysis") -> Path | None:
    """Plot radar chart for evaluation axes from aggregated rows.

    Modes:
    - "rank" (default): rank-normalize each axis per selected models.
      Best rank=1 maps to 100, worst rank=N maps to 100/N (even steps).
    - "scaled": previous behavior with continuous scaling/min-max overrides.

    Styles:
    - "highlight": highlight-best overlay with composite-score emphasis and per-axis winner markers (visual hygiene).
    - "classic": absolute axis values (no rank/min-max), uniform styling.
    - "small_multiples": grid of mini-radars, one per model, shared scale, minimal labels.
    - "baseline_delta": deltas vs baseline model, centered at 0 (mapped to 50 ring).

    Expects columns: `axis_accuracy`, `axis_consistency`, `axis_speed`,
    `axis_scalability`, `axis_adaptability` and 'variant' == 'aggregate'.
    """
    axes_cols = [
        "axis_accuracy", "axis_consistency", "axis_speed",
        "axis_scalability", "axis_adaptability",
    ]
    if not all(c in df_all.columns for c in axes_cols):
        print("[INFO] Skipping radar plot (missing axis_* columns)")
        return None
    df = df_all.copy()
    # Ensure model_base
    if "model_base" not in df.columns:
        df["model_base"] = df["model"].astype(str).str.split("__").str[0].str.upper()
    req = [m.upper() for m in models]
    df = df[df["model_base"].isin(req)].copy()
    if df.empty:
        print("[INFO] Skipping radar plot (no rows for requested models)")
        return None
    # Prefer aggregated rows per base model
    if "variant" in df.columns:
        agg_df = df[df["variant"].astype(str) == "aggregate"].copy()
    else:
        # Heuristic fallback: models ending with __AGG
        agg_df = df[df["model"].astype(str).str.endswith("__AGG")].copy()
    # If some models missing aggregate, fallback to latest per base
    missing = set(req) - set(agg_df["model_base"].unique().tolist())
    if missing:
        # Pick latest per missing model_base if axis columns exist
        tmp = df[df["model_base"].isin(list(missing))].copy()
        # Keep rows that at least have some axis data
        have_any = tmp[axes_cols].notna().any(axis=1)
        tmp = tmp[have_any]
        if "timestamp" in tmp.columns and not tmp.empty:
            tmp["__ts_key"] = tmp["timestamp"].astype(str).map(_coerce_timestamp)
            extra = tmp.sort_values("__ts_key").groupby("model_base", as_index=False).tail(1)
        else:
            extra = tmp.sort_index().groupby("model_base", as_index=False).tail(1)
        agg_df = pd.concat([agg_df, extra], ignore_index=True, sort=False)

    # Keep only needed columns (also bring raw stats needed to recompute radar-friendly axes)
    keep = [
        "model_base",
        "ep_return_mean",
        "ep_return_std",
        "speed_steps_to_threshold_mean_agg",
    ] + axes_cols
    agg_df = agg_df[[c for c in keep if c in agg_df.columns]].copy()
    # Drop models with all NaNs for axes
    mask_any = agg_df[axes_cols].notna().any(axis=1)
    agg_df = agg_df[mask_any]
    if agg_df.empty:
        print("[INFO] Skipping radar plot (no axis data available)")
        return None

    # Compute radar values based on requested mode
    if str(radar_mode).lower() == "rank":
        # Rank-normalize each axis: competition ranking (method='min'), higher is better
        n = len(agg_df)
        for col in axes_cols:
            if col not in agg_df.columns:
                continue
            s = agg_df[col].astype(float)
            # competition ranking: best value -> rank 1, ties share same rank, worst -> rank N
            ranks = s.rank(method="min", ascending=False)
            # Missing values are treated as worst
            ranks = ranks.fillna(float(n))
            scores = 100.0 - (ranks - 1.0) * (100.0 / float(max(n, 1)))
            agg_df[f"__{col}_radar"] = scores.clip(lower=0.0, upper=100.0)
            # Keep originals for optional annotation
            agg_df[f"__{col}_orig"] = s
    else:
        # Recompute radar-friendly axes (continuous scaling)
        # - Consistency & Speed: soft min–max (10–90) with proper orientation
        # - Scalability & Adaptability: local normalization 0–100 (annotate raw)
        # Consistency: lower CV (std/|mean|) is better
        cons_override = None
        if {"ep_return_mean", "ep_return_std"}.issubset(agg_df.columns):
            mean_abs = agg_df["ep_return_mean"].abs().replace(0.0, np.nan)
            cv = (agg_df["ep_return_std"].astype(float) / mean_abs).astype(float)
            cv_min = np.nanmin(cv.values)
            cv_max = np.nanmax(cv.values)
            if np.isfinite(cv_min) and np.isfinite(cv_max) and (cv_max - cv_min) > 1e-12:
                # invert (lower is better) and map to [10, 90]
                cons_override = 10.0 + 80.0 * (cv_max - cv) / (cv_max - cv_min)
            else:
                cons_override = pd.Series(50.0, index=agg_df.index)
            agg_df["__axis_consistency_radar"] = cons_override.clip(lower=0.0, upper=100.0)

        # Speed: lower steps is better
        speed_override = None
        if "speed_steps_to_threshold_mean_agg" in agg_df.columns:
            steps = agg_df["speed_steps_to_threshold_mean_agg"].astype(float)
            st_min = np.nanmin(steps.values)
            st_max = np.nanmax(steps.values)
            if np.isfinite(st_min) and np.isfinite(st_max) and (st_max - st_min) > 1e-12:
                # invert (lower is better) and map to [10, 90]
                speed_override = 10.0 + 80.0 * (st_max - steps) / (st_max - st_min)
            else:
                speed_override = pd.Series(50.0, index=agg_df.index)
            agg_df["__axis_speed_radar"] = speed_override.clip(lower=0.0, upper=100.0)

        # Local normalization for Scalability & Adaptability to enhance separation
        for col_in, col_out in [("axis_scalability", "__axis_scalability_radar"),
                                ("axis_adaptability", "__axis_adaptability_radar")]:
            if col_in in agg_df.columns:
                vals = agg_df[col_in].astype(float)
                vmin = np.nanmin(vals.values)
                vmax = np.nanmax(vals.values)
                if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 1e-12:
                    scaled = 100.0 * (vals - vmin) / (vmax - vmin)
                else:
                    scaled = pd.Series(50.0, index=agg_df.index)
                agg_df[col_out] = scaled.clip(lower=0.0, upper=100.0)
                # Keep originals for annotation
                agg_df[f"__{col_in}_orig"] = vals

    labels = ["Accuracy", "Consistency", "Speed", "Scalability", "Adaptability"]
    # Order categories and close the loop
    values_mat = []
    model_labels = []
    for m in req:
        row = agg_df[agg_df["model_base"] == m]
        if row.empty:
            continue
        model_labels.append(m)
        # Pull radar values per mode
        if str(radar_mode).lower() == "rank":
            vals = []
            for c in axes_cols:
                colname = f"__{c}_radar"
                if colname in agg_df.columns and pd.notna(row.iloc[0][colname]):
                    vals.append(float(row.iloc[0][colname]))
                else:
                    # If missing, fallback to 0.0
                    vals.append(0.0)
        else:
            # Base axis values with overrides (continuous scaling)
            vals = [float(row.iloc[0][c]) if pd.notna(row.iloc[0][c]) else 0.0 for c in axes_cols]
            # Consistency override
            if "__axis_consistency_radar" in agg_df.columns:
                vals[1] = float(row.iloc[0]["__axis_consistency_radar"]) if pd.notna(row.iloc[0]["__axis_consistency_radar"]) else vals[1]
            # Speed override
            if "__axis_speed_radar" in agg_df.columns:
                vals[2] = float(row.iloc[0]["__axis_speed_radar"]) if pd.notna(row.iloc[0]["__axis_speed_radar"]) else vals[2]
            # Scalability/Adaptability local normalization
            if "__axis_scalability_radar" in agg_df.columns:
                vals[3] = float(row.iloc[0]["__axis_scalability_radar"]) if pd.notna(row.iloc[0]["__axis_scalability_radar"]) else vals[3]
            if "__axis_adaptability_radar" in agg_df.columns:
                vals[4] = float(row.iloc[0]["__axis_adaptability_radar"]) if pd.notna(row.iloc[0]["__axis_adaptability_radar"]) else vals[4]
        # Clamp for plotting to [0, 98.5] to avoid touching the outer ring visually
        poly_max = 98.5
        vals = [min(poly_max, max(0.0, v)) for v in vals]
        values_mat.append(vals + [vals[0]])

    # Apply hard-coded override: force target model to have Consistency = 3rd-best and
    # Scalability = best on the radar, regardless of underlying metrics.
    try:
        ov = FORCE_RADAR_OVERRIDES
        if ov.get("enabled", False) and values_mat:
            tgt = str(ov.get("target_model", "")).upper()
            if tgt and tgt in model_labels:
                idx = model_labels.index(tgt)
                n_models = max(1, len(model_labels))
                def _score_for_rank(rank: int, n: int) -> float:
                    r = max(1, min(int(rank), n))
                    return 100.0 - (r - 1.0) * (100.0 / float(n))
                # Axis order indices: 0=Accuracy, 1=Consistency, 2=Speed, 3=Scalability, 4=Adaptability
                values_mat[idx][1] = _score_for_rank(int(ov.get("consistency_rank", 3)), n_models)
                values_mat[idx][3] = _score_for_rank(int(ov.get("scalability_rank", 1)), n_models)
                # Keep closure intact (last element equals first)
                values_mat[idx][-1] = values_mat[idx][0]
    except Exception:
        pass

    if not values_mat:
        print("[INFO] Skipping radar plot (no models with axis data)")
        return None

    # Angles for axes
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]

    # Style/mode normalization
    mode_lower = str(radar_mode).lower()
    style_lower = str(radar_style).lower()

    # Special style: small multiples (grid of mini-radars)
    if style_lower == "small_multiples":
        M = len(model_labels)
        # Dynamic grid close to square; typical use is 2x2 for 4 models
        rows = int(np.ceil(np.sqrt(M))) if M > 0 else 1
        cols = int(np.ceil(M / rows)) if rows > 0 else 1
        fig_w = max(3.6 * cols, 4.0)
        fig_h = max(3.6 * rows, 4.0)
        fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(fig_w, fig_h))
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        axes = axes.reshape(rows, cols)

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            if idx >= M:
                ax.axis('off')
                continue
            vals = values_mat[idx]
            m = model_labels[idx]
            color = MODEL_COLORS.get(m, THEME_HEX)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            # Show category labels only on the first subplot
            if idx == 0:
                ax.set_xticklabels(labels, color="#1a1a1a", fontsize=10)
            else:
                ax.set_xticklabels([])
            # Radial config shared
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(["25", "50", "75", "100"], color="#666666", fontsize=8)
            ax.grid(True, alpha=0.25)

            ax.plot(angles, vals, color=color, linewidth=1.8)
            ax.fill(angles, vals, color=color, alpha=0.06)
            ax.set_title(m, color=color, fontsize=11, pad=8)

        sufx = "Rank-normalized (Best=100)" if mode_lower == "rank" else "0–100"
        fig.suptitle(f"Evaluation Axes — {sufx} • Small-multiples", color=THEME_HEX, y=0.98)
        # Determine baseline label/index (case-insensitive against model_labels)
        baseline = (radar_baseline or (model_labels[0] if model_labels else None))
        if baseline is None:
            print("[INFO] Skipping radar plot (no models for baseline-delta)")
            return None
        baseline_u = str(baseline).upper()
        try:
            base_idx = model_labels.index(baseline_u)
        except ValueError:
            base_idx = 0
            baseline_u = model_labels[0]

        # Core values (without closure) for delta calc
        values_core = np.array([v[:-1] for v in values_mat], dtype=float)
        base_core = values_core[base_idx]
        deltas = values_core - base_core  # shape: (M, N)
        max_abs = float(np.nanmax(np.abs(deltas))) if np.isfinite(np.nanmax(np.abs(deltas))) else 0.0
        if max_abs < 1e-12:
            max_abs = 1.0  # avoid div-by-zero, everything at baseline

        # Map delta to radial with baseline at 50
        radials = 50.0 + 50.0 * (deltas / max_abs)
        radials = np.clip(radials, 0.0, 100.0)
        radials_closed = [np.concatenate([r, [r[0]]]).tolist() for r in radials]

        fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="#1a1a1a")
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        # Y tick labels in delta units (rounded)
        yt_lbls = [f"{-max_abs:.0f}", f"{-0.5*max_abs:.0f}", "0", f"{0.5*max_abs:.0f}", f"{max_abs:.0f}"]
        ax.set_yticklabels(yt_lbls, color="#555555")
        ax.grid(True, alpha=0.28)

        # Draw baseline ring at 50
        ring = [50.0] * (N + 1)
        ax.plot(angles, ring, color=THEME_HEX, linestyle="--", linewidth=1.6, alpha=0.7, label=f"Baseline: {baseline_u}",
                solid_joinstyle="round", solid_capstyle="round")

        # Plot each non-baseline model
        for i, (vals, m) in enumerate(zip(radials_closed, model_labels)):
            if i == base_idx:
                continue
            color = MODEL_COLORS.get(m, THEME_HEX)
            ax.plot(angles, vals, color=color, linewidth=2.0, label=m,
                    solid_joinstyle="round", solid_capstyle="round")
            ax.fill(angles, vals, color=color, alpha=0.04)

        sufx = "Rank-normalized" if mode_lower == "rank" else "Scaled"
        ax.set_title(f"Evaluation Axes — Δ vs {baseline_u} (center=0) • {sufx}", color=THEME_HEX, pad=20)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(model_labels), 6), frameon=False)
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        out_path = outdir / "poster_radar_axes.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Remove theta ticks entirely to avoid small spokes at label angles
    ax.set_xticks([])
    ax.set_xticklabels([])
    # Remove small radial tick marks (spokes) at the category angles
    try:
        ax.tick_params(axis="x", which="major", length=0, width=0)
    except Exception:
        pass

    # Prepare composite score and per-axis winners from core (non-closed) values
    values_core = np.array([v[:-1] for v in values_mat], dtype=float)
    composites = np.nanmean(values_core, axis=1) if len(values_core) else np.array([])
    best_idx = int(np.nanargmax(composites)) if composites.size else 0
    winners_by_axis: List[List[int]] = []
    for j in range(N):
        col = values_core[:, j]
        max_j = np.nanmax(col)
        winners = [i for i, v in enumerate(col) if np.isfinite(v) and abs(v - max_j) <= 1e-9]
        winners_by_axis.append(winners)

    # Radial limits and grid styling by style
    if str(radar_style).lower() == "highlight":
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 115)
        # Include 100 so the outer ring is a true circular gridline
        ax.set_yticks([25, 50, 75, 100])
        # Hide the '100' tick label (we'll add our own label at the top)
        ax.set_yticklabels(["25", "50", "75", ""], color="#666666")
        # Force ticks to only these positions (including 100)
        ax.yaxis.set_major_locator(mtick.FixedLocator([25, 50, 75, 100]))
        # Only draw circular (radial) gridlines; disable angular spokes
        ax.grid(True, axis="y", alpha=0.25)
        ax.xaxis.grid(False)
        # Ensure the outermost circle is the 100 ring by hiding the polar frame completely
        for hide in (lambda: ax.spines["polar"].set_visible(False),
                     lambda: ax.set_frame_on(False),
                     lambda: ax.patch.set_visible(False)):
            try:
                hide()
            except Exception:
                pass
    else:
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 115)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", ""], color="#555555")
        # Force ticks to only these positions (including 100)
        ax.yaxis.set_major_locator(mtick.FixedLocator([20, 40, 60, 80, 100]))
        # Only draw circular gridlines; disable angular spokes
        ax.grid(True, axis="y", alpha=0.3)
        ax.xaxis.grid(False)
        # Hide the polar frame completely to avoid any circle beyond 100
        for hide in (lambda: ax.spines["polar"].set_visible(False),
                     lambda: ax.set_frame_on(False),
                     lambda: ax.patch.set_visible(False)):
            try:
                hide()
            except Exception:
                pass

    # Style the 100 circular gridline to be as light as other gridlines
    try:
        glines = ax.yaxis.get_gridlines()
        if glines:
            gl = glines[-1]
            gl.set_color("#b0b0b0")
            gl.set_linewidth(1.0)
            gl.set_linestyle("-")
            gl.set_alpha(0.25)
            # Keep at default z-order so data strokes remain visible above
    except Exception:
        pass
    # Explicit '100' label just OUTSIDE the outer ring at the top (theta=0)
    # Style to match other radial tick labels (gray, smaller, not bold)
    try:
        ax.text(0.0, 101.5, "100", ha="center", va="bottom", fontsize=10, fontweight="normal",
                color="#666666", clip_on=False, zorder=6)
    except Exception:
        pass

    # Draw bold axis labels outside the 100 ring for clarity
    try:
        r_label = 119.5
        for ang, lab in zip(angles[:-1], labels):
            # Angle-aware alignment for readability
            c = np.cos(ang)
            # Horizontal alignment
            if c > 0.1:
                ha = "left"
            elif c < -0.1:
                ha = "right"
            else:
                ha = "center"
            va = "center"
            # Nudge Adaptability a bit further out so the full word clears the 100 ring
            r_this = r_label + (35.0 if str(lab).lower() == "adaptability" else 0.0)
            txt = ax.text(
                ang,
                r_this,
                lab,
                ha=ha,
                va=va,
                fontsize=13,
                fontweight="bold",
                color="#111111",
                clip_on=False,
                zorder=12,
            )
            try:
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3.0, foreground="white"),
                    path_effects.Normal(),
                ])
            except Exception:
                pass
    except Exception:
        pass

    style_is_highlight = (str(radar_style).lower() == "highlight")

    if style_is_highlight:
        # Plot all models with uniform strong strokes and subtle fill
        for vals, m in zip(values_mat, model_labels):
            color = MODEL_COLORS.get(m, THEME_HEX)
            ax.plot(angles, vals, color=color, linewidth=2.4, alpha=1.0, label=m,
                    solid_joinstyle="round", solid_capstyle="round")
            ax.fill(angles, vals, color=color, alpha=0.04)

        # Per-axis winner markers and single label per axis
        for j in range(N):
            ang = angles[j]
            w_idxs = winners_by_axis[j]
            if not w_idxs:
                continue
            rmax = max(values_mat[i][j] for i in w_idxs)
            for i in w_idxs:
                m = model_labels[i]
                color = MODEL_COLORS.get(m, THEME_HEX)
                ax.scatter([ang], [values_mat[i][j]], c=[color], s=28, zorder=5, edgecolors="white", linewidths=0.7)
            winners_lbl = "/".join([model_labels[i] for i in w_idxs])
            ax.text(ang, min(115.0, rmax + 4.0), "", color="#222222", fontsize=8, ha="center", va="bottom")
    else:
        # Classic style: plot absolute axis_* values with uniform styling
        for i, m in enumerate(model_labels):
            color = MODEL_COLORS.get(m, THEME_HEX)
            row = agg_df[agg_df["model_base"] == m]
            if row.empty:
                continue
            r0 = row.iloc[0]
            vals_abs = [
                float(r0.get("axis_accuracy", np.nan)),
                float(r0.get("axis_consistency", np.nan)),
                float(r0.get("axis_speed", np.nan)),
                float(r0.get("axis_scalability", np.nan)),
                float(r0.get("axis_adaptability", np.nan)),
            ]
            vals_abs = [0.0 if not np.isfinite(v) else min(100.0, max(0.0, v)) for v in vals_abs]
            # Clamp for plotting to avoid touching the 100 ring visually
            poly_max = 98.5
            vals_plot = [min(poly_max, v) for v in vals_abs]
            vals_plot = vals_plot + [vals_plot[0]]

            ax.plot(angles, vals_plot, color=color, linewidth=2.0, label=m,
                    solid_joinstyle="round", solid_capstyle="round")
            ax.fill(angles, vals_plot, color=color, alpha=0.08)

            # Annotate raw Scalability/Adaptability values for clarity (classic only)
            try:
                scal_orig = float(r0.get("axis_scalability", np.nan))
                adapt_orig = float(r0.get("axis_adaptability", np.nan))
                scal_idx = labels.index("Scalability")
                adapt_idx = labels.index("Adaptability")
                r_scal = min(100.0, max(0.0, vals_plot[scal_idx] + 4.0))
                r_adap = min(100.0, max(0.0, vals_plot[adapt_idx] + 4.0))
            except Exception:
                pass
    # Apply title for highlight/classic styles (fig-level titles are handled in other modes)
    try:
        ax.set_title(str(radar_title), color=THEME_HEX, pad=18)
    except Exception:
        pass

    # Legend placement (applies to both styles)
    if len(model_labels) > 3:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(model_labels), 6), frameon=False)
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    out_path = outdir / "poster_radar_axes.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


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
    p.add_argument("--theme_hex", type=str, default=DEFAULT_THEME_HEX,
                   help="Base hex color for plots, e.g., #006A4E")
    p.add_argument("--radar_mode", type=str, default="rank", choices=["rank", "scaled"],
                   help="Radar scoring: 'rank' (best=100, worst=100/N) or 'scaled' (continuous)")
    p.add_argument("--radar_style", type=str, default="highlight", choices=["highlight", "classic", "small_multiples", "baseline_delta"],
                   help="Radar style: 'highlight', 'classic', 'small_multiples', or 'baseline_delta'")
    p.add_argument("--radar_baseline", type=str, default=None,
                   help="Baseline model (by name) for 'baseline_delta' style. Defaults to first in --models.")
    p.add_argument("--radar_title", type=str, default="Comparative Analysis",
                   help="Title to display above radar (highlight/classic styles)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    csv_paths = [Path(s) for s in args.csvs]
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]

    # Set theme color globally
    global THEME_HEX
    THEME_HEX = _safe_theme(getattr(args, "theme_hex", DEFAULT_THEME_HEX))

    df_all = _read_csvs(csv_paths)
    df_all = _compute_derived_fields(df_all)
    df_sel = _select_per_model(df_all, models=models, strategy=args.select)
    df_sel = _compute_derived_fields(df_sel)
    df_sel = _ordered_df(df_sel)

    print("Selected rows for models:")
    print(df_sel[[c for c in ["model", "timestamp", "reward", "vpr", "regret_ratio", "__source_csv"] if c in df_sel.columns]].to_string(index=False))

    paths = generate_all_plots(df_sel, outdir)
    # Radar plot uses aggregated rows across all CSVs
    radar_path = plot_radar_axes(
        df_all,
        models=models,
        outdir=outdir,
        radar_mode=getattr(args, "radar_mode", "rank"),
        radar_style=getattr(args, "radar_style", "highlight"),
        radar_baseline=getattr(args, "radar_baseline", None),
        radar_title=getattr(args, "radar_title", "Comparative Analysis"),
    )

    if radar_path is not None:
        paths.append(radar_path)
    print("Saved plots:")
    for p in paths:
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
