#!/usr/bin/env python3
"""Create final LightGBM plots from evaluate_lightgbm_final.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from lightgbm_results_utils import (
        ID_COLUMN,
        LABEL_COLUMN,
        L1_COLUMN,
        LANGUAGE_SPECS,
        LEVENSHTEIN_COLUMN,
        WINDOW_CENTERS,
        WINDOW_LOWS,
        WINDOW_WIDTH,
        center_mask,
        parse_center_range,
    )
except ModuleNotFoundError:
    from results.lightgbm_results_utils import (
        ID_COLUMN,
        LABEL_COLUMN,
        L1_COLUMN,
        LANGUAGE_SPECS,
        LEVENSHTEIN_COLUMN,
        WINDOW_CENTERS,
        WINDOW_LOWS,
        WINDOW_WIDTH,
        center_mask,
        parse_center_range,
    )

DEFAULT_METRICS_JSON = PROJECT_ROOT / "results" / "lightgbm_final_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_GLMM_DATA_FILE = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"

MODEL_COLUMN = "model_name"
PREDICTION_COLUMN = "prediction"
SPLIT_COLUMN = "split"
OUTER_TEST_SPLIT = "outer_test"

DEFAULT_CENTER_RANGES = {
    "de": "0.1-0.9",
    "es": "0.1-0.9",
    "cn": "0.4-0.9",
}

SLIDING_GROUP_MODELS = [
    ("raw_roberta", "B"),
    ("BL1", r"B$^{\mathrm{L1}}$"),
    ("BE", r"B$^{\mathrm{L1}}$C"),
    ("BS", r"B$^{\mathrm{L1}}$S"),
    ("BR", r"B$^{\mathrm{L1}}$E"),
    ("BR_no_L1", "BE"),
    ("BES", r"B$^{\mathrm{L1}}$CS"),
    ("BER", r"B$^{\mathrm{L1}}$CE"),
    ("BSR", r"B$^{\mathrm{L1}}$SE"),
    ("BESR", r"B$^{\mathrm{L1}}$CSE"),
]

SLIDING_GROUP_COLORS = {
    "B": "black",
    r"B$^{\mathrm{L1}}$": "gray",
    r"B$^{\mathrm{L1}}$C": "dodgerblue",
    r"B$^{\mathrm{L1}}$S": "gold",
    r"B$^{\mathrm{L1}}$E": "red",
    "BE": "firebrick",
    r"B$^{\mathrm{L1}}$CS": "limegreen",
    r"B$^{\mathrm{L1}}$SE": "darkorange",
    r"B$^{\mathrm{L1}}$CE": "darkviolet",
    r"B$^{\mathrm{L1}}$CSE": "brown",
}

SLIDING_GROUP_STYLES = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    r"B$^{\mathrm{L1}}$": {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.0,
        "alpha": 1.0,
    },
    r"B$^{\mathrm{L1}}$C": {
        "linestyle": "--",
        "marker": "s",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$S": {
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$E": {
        "linestyle": ":",
        "marker": "D",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    "BE": {
        "linestyle": ":",
        "marker": "h",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$CS": {
        "linestyle": "--",
        "marker": "v",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$SE": {
        "linestyle": "-.",
        "marker": "P",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$CE": {
        "linestyle": ":",
        "marker": "X",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"B$^{\mathrm{L1}}$CSE": {
        "linestyle": "-",
        "marker": "*",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
}

L1_GROUP_MODELS = [
    ("raw_roberta", "B"),
    ("L1_only", r"$^{\mathrm{L1}}$"),
    ("R_only", "E"),
    ("E", r"$^{\mathrm{L1}}$C"),
    ("S", r"$^{\mathrm{L1}}$S"),
    ("R", r"$^{\mathrm{L1}}$E"),
    ("ES", r"$^{\mathrm{L1}}$CS"),
    ("ER", r"$^{\mathrm{L1}}$CE"),
    ("SR", r"$^{\mathrm{L1}}$SE"),
    ("ESR", r"$^{\mathrm{L1}}$CSE"),
]

L1_GROUP_COLORS = {
    "B": "black",
    r"$^{\mathrm{L1}}$": "gray",
    "E": "firebrick",
    r"$^{\mathrm{L1}}$C": "dodgerblue",
    r"$^{\mathrm{L1}}$S": "gold",
    r"$^{\mathrm{L1}}$E": "red",
    r"$^{\mathrm{L1}}$CS": "limegreen",
    r"$^{\mathrm{L1}}$SE": "darkorange",
    r"$^{\mathrm{L1}}$CE": "darkviolet",
    r"$^{\mathrm{L1}}$CSE": "brown",
}

L1_GROUP_STYLES = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    r"$^{\mathrm{L1}}$": {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.0,
        "alpha": 1.0,
    },
    "E": {
        "linestyle": ":",
        "marker": "h",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$C": {
        "linestyle": "--",
        "marker": "s",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$S": {
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$E": {
        "linestyle": ":",
        "marker": "D",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$CS": {
        "linestyle": "--",
        "marker": "v",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$SE": {
        "linestyle": "-.",
        "marker": "P",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$CE": {
        "linestyle": ":",
        "marker": "X",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
    r"$^{\mathrm{L1}}$CSE": {
        "linestyle": "-",
        "marker": "*",
        "linewidth": 4.5,
        "alpha": 0.60,
    },
}

RELATEDNESS_FEATURES = [
    "lca_language",
    "borrowed_from_source",
    "borrowed_from_target",
    "borrowed_from_third",
    "english_word_origin",
    "origin_matches_source_family",
    "same_proto_word",
    "same_proto_root",
    "transmission_path",
]

FEATURE_COLOR_MAP = {
    "lca_language": "deeppink",
    "borrowed_from_source": "forestgreen",
    "borrowed_from_target": "seagreen",
    "borrowed_from_third": "mediumseagreen",
    "english_word_origin": "gold",
    "origin_matches_source_family": "darkorange",
    "same_proto_word": "royalblue",
    "same_proto_root": "dodgerblue",
    "transmission_path": "mediumpurple",
}

LINESTYLES = ["--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

WINDOW_BAR_COLOR = "gray"
WINDOW_BAR_ALPHA = 0.1
WINDOW_BAR_WIDTH = 0.075

UPSET_MODEL_LABELS = {
    "BL1": r"B$^{\mathrm{L1}}$",
    "E": r"$^{\mathrm{L1}}$C",
    "S": r"$^{\mathrm{L1}}$S",
    "R": r"$^{\mathrm{L1}}$E",
}

UPSET_SET_COLORS = {
    r"B$^{\mathrm{L1}}$": "gray",
    r"$^{\mathrm{L1}}$C": "dodgerblue",
    r"$^{\mathrm{L1}}$S": "gold",
    r"$^{\mathrm{L1}}$E": "red",
}

UNIQUE_CORRECT_STYLE_BY_LABEL = {
    r"B$^{\mathrm{L1}}$": SLIDING_GROUP_STYLES[r"B$^{\mathrm{L1}}$"],
    r"$^{\mathrm{L1}}$C": L1_GROUP_STYLES[r"$^{\mathrm{L1}}$C"],
    r"$^{\mathrm{L1}}$S": L1_GROUP_STYLES[r"$^{\mathrm{L1}}$S"],
    r"$^{\mathrm{L1}}$E": L1_GROUP_STYLES[r"$^{\mathrm{L1}}$E"],
}

EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final LightGBM evaluation outputs."
    )
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glmm-data-file", type=Path, default=DEFAULT_GLMM_DATA_FILE)
    parser.add_argument("--de-centers", type=str, default=DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", type=str, default=DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", type=str, default=DEFAULT_CENTER_RANGES["cn"])
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without calling plt.show().",
    )
    return parser.parse_args()


def _k_token(k: float) -> str:
    return f"{k:.6g}"


def model_windows(
    summary: Dict[str, Any],
    model_name: str,
    language_code: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    model_payload = summary.get("models", {}).get(model_name)
    if not isinstance(model_payload, dict):
        return None

    language_payload = (
        model_payload.get("outer_test", {})
        .get("per_language", {})
        .get(language_code)
    )
    if not isinstance(language_payload, dict):
        return None

    window_metrics = language_payload.get("windows", [])
    if not isinstance(window_metrics, list):
        return None

    rmses = np.full(len(WINDOW_CENTERS), np.nan, dtype=float)
    counts = np.zeros(len(WINDOW_CENTERS), dtype=int)

    for idx, metric in enumerate(window_metrics):
        if idx >= len(WINDOW_CENTERS):
            break
        value = metric.get("rmse")
        rmses[idx] = np.nan if value is None else float(value)
        counts[idx] = int(metric.get("n_rows", 0))

    return rmses, counts


def add_window_count_axis(
    ax: plt.Axes,
    ticks: np.ndarray,
    counts: np.ndarray,
    *,
    ylabel: str = "Window set size",
    show_ylabel: bool = True,
) -> plt.Axes:
    """Overlay translucent candidate/window-size bars on a secondary y-axis."""
    ax2 = ax.twinx()
    ax2.bar(
        ticks,
        counts,
        width=WINDOW_BAR_WIDTH,
        color=WINDOW_BAR_COLOR,
        alpha=WINDOW_BAR_ALPHA,
        edgecolor="none",
        zorder=0,
    )
    ax2.set_ylim(0, max(1.0, float(np.max(counts)) * 1.1))
    ax2.grid(False)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)

    if show_ylabel:
        ax2.set_ylabel(ylabel, fontsize=12, color="dimgray")
        ax2.tick_params(axis="y", colors="dimgray", labelcolor="dimgray")
    else:
        ax2.set_yticks([])
        ax2.tick_params(axis="y", right=False, labelright=False)

    return ax2


def compute_glmm_window_means(
    glmm_df: pd.DataFrame,
    language_code: str,
) -> np.ndarray:
    lang_df = glmm_df[glmm_df[L1_COLUMN] == language_code]
    means: List[float] = []

    for low in WINDOW_LOWS.tolist():
        high = float(low + WINDOW_WIDTH)
        window_df = lang_df[
            (lang_df[LEVENSHTEIN_COLUMN] >= float(low))
            & (lang_df[LEVENSHTEIN_COLUMN] <= high)
        ]
        means.append(
            float(window_df[LABEL_COLUMN].mean())
            if not window_df.empty
            else np.nan
        )

    return np.asarray(means, dtype=float)


def load_glmm_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GLMM data file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {L1_COLUMN, LEVENSHTEIN_COLUMN, LABEL_COLUMN}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"GLMM data file is missing required columns: {sorted(missing_cols)}"
        )

    return df


def plot_baseline_group_curves(
    summary: Dict[str, Any],
    output_path: Path,
    range_by_language: Dict[str, tuple[float, float]],
    no_show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        low, high = range_by_language[language_code]
        mask = center_mask(low, high)
        ticks = WINDOW_CENTERS[mask]

        if len(ticks) == 0:
            raise ValueError(f"No centers selected for {language_code}: {low}-{high}")

        reference_counts: np.ndarray | None = None

        for model_name, label in SLIDING_GROUP_MODELS:
            arrays = model_windows(summary, model_name, language_code)
            if arrays is None:
                continue

            rmses, counts = arrays
            if reference_counts is None:
                reference_counts = counts

            style = SLIDING_GROUP_STYLES[label]
            ax.plot(
                ticks,
                rmses[mask],
                color=SLIDING_GROUP_COLORS[label],
                label=label,
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )

        if reference_counts is not None:
            add_window_count_axis(ax, ticks, reference_counts[mask])

        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{center:.1f}" for center in ticks])
        ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)

        if language_code == "cn":
            ax.legend(
                fontsize=10,
                title="Configuration",
                title_fontsize=12,
                loc="lower right",
            )

    axes[-1].set_xlabel(
        "Normalized Levenshtein Distance (Window Center)",
        fontsize=14,
    )
    fig.suptitle(
        "BL1 Group Models by Levenshtein Window",
        fontsize=13,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if not no_show:
        plt.show()

    plt.close(fig)


def plot_l1_group_curves_with_glmm(
    summary: Dict[str, Any],
    glmm_df: pd.DataFrame,
    output_path: Path,
    range_by_language: Dict[str, tuple[float, float]],
    no_show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        low, high = range_by_language[language_code]
        mask = center_mask(low, high)
        ticks = WINDOW_CENTERS[mask]

        if len(ticks) == 0:
            raise ValueError(f"No centers selected for {language_code}: {low}-{high}")

        for model_name, label in L1_GROUP_MODELS:
            arrays = model_windows(summary, model_name, language_code)
            if arrays is None:
                continue

            rmses, _ = arrays
            style = L1_GROUP_STYLES[label]

            ax.plot(
                ticks,
                rmses[mask],
                color=L1_GROUP_COLORS[label],
                label=label,
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )

        glmm_means = compute_glmm_window_means(glmm_df, language_code)
        ax2 = ax.twinx()
        ax2.plot(
            ticks,
            glmm_means[mask],
            color="crimson",
            linestyle="-",
            marker="o",
            linewidth=2.5,
            alpha=0.9,
            label="GLMM",
        )
        ax2.set_ylabel("Avg GLMM", fontsize=12, color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")

        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{center:.1f}" for center in ticks])
        ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)

        if language_code == "cn":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(
                h1 + h2,
                l1 + l2,
                fontsize=10,
                title="Components",
                title_fontsize=12,
                loc="upper right",
            )

    axes[-1].set_xlabel(
        "Normalized Levenshtein Distance (Window Center)",
        fontsize=14,
    )
    fig.suptitle(
        "L1 Group Models vs GLMM by Levenshtein Window",
        fontsize=13,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if not no_show:
        plt.show()

    plt.close(fig)


def plot_relatedness_individual_curves_with_window_sizes(
    summary: Dict[str, Any],
    output_path: Path,
    range_by_language: Dict[str, tuple[float, float]],
    no_show: bool,
) -> None:
    model_specs: List[Tuple[str, str, str]] = [
        ("raw_roberta", "B", "black"),
        ("BL1", r"B$^{\mathrm{L1}}$", "gray"),
    ]

    for feature_name in RELATEDNESS_FEATURES:
        model_name = f"BL1_plus_{feature_name}"
        if model_name in summary.get("models", {}):
            model_specs.append(
                (
                    model_name,
                    f"+ {feature_name}",
                    FEATURE_COLOR_MAP.get(feature_name, "tab:gray"),
                )
            )

    if len(model_specs) == 1:
        raise ValueError("No BL1 relatedness individual-feature models were found.")

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        low, high = range_by_language[language_code]
        mask = center_mask(low, high)
        ticks = WINDOW_CENTERS[mask]

        if len(ticks) == 0:
            raise ValueError(f"No centers selected for {language_code}: {low}-{high}")

        style_idx = 0
        reference_counts: np.ndarray | None = None

        for model_name, label, color in model_specs:
            arrays = model_windows(summary, model_name, language_code)
            if arrays is None:
                continue

            rmses, counts = arrays
            if reference_counts is None:
                reference_counts = counts

            if model_name in {"raw_roberta", "BL1"}:
                style = {
                    "linestyle": "-",
                    "marker": "o",
                    "linewidth": 3.0,
                    "alpha": 1.0 if model_name == "raw_roberta" else 0.9,
                }
            else:
                style = {
                    "linestyle": LINESTYLES[style_idx % len(LINESTYLES)],
                    "marker": MARKERS[style_idx % len(MARKERS)],
                    "linewidth": 4.5,
                    "alpha": 0.60,
                }
                style_idx += 1

            ax.plot(
                ticks,
                rmses[mask],
                color=color,
                label=label,
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )

        ax2: plt.Axes | None = None
        if reference_counts is not None:
            ax2 = add_window_count_axis(ax, ticks, reference_counts[mask])

        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{center:.1f}" for center in ticks])
        ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)

        if language_code == "cn":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ([], []) if ax2 is None else ax2.get_legend_handles_labels()
            ax.legend(
                h1 + h2,
                l1 + l2,
                fontsize=10,
                title="Configuration",
                title_fontsize=12,
                loc="lower right",
            )

    axes[-1].set_xlabel(
        "Normalized Levenshtein Distance (Window Center)",
        fontsize=14,
    )
    fig.suptitle(
        "BL1 Individual Relatedness Features by Levenshtein Window",
        fontsize=13,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if not no_show:
        plt.show()

    plt.close(fig)


def load_predictions_from_summary(summary: Dict[str, Any]) -> pd.DataFrame:
    predictions_file = summary.get("predictions_file")
    if not predictions_file:
        raise ValueError("Metrics JSON does not contain 'predictions_file'.")

    path = Path(str(predictions_file))
    if not path.exists():
        raise FileNotFoundError(
            "Predictions file referenced in metrics JSON does not exist: "
            f"{path}"
        )

    df = pd.read_csv(path, keep_default_na=False)
    required_cols = {
        ID_COLUMN,
        L1_COLUMN,
        LEVENSHTEIN_COLUMN,
        MODEL_COLUMN,
        SPLIT_COLUMN,
        LABEL_COLUMN,
        PREDICTION_COLUMN,
    }
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            "Predictions file is missing required columns for the "
            f"UpSet/window analysis: {sorted(missing_cols)}"
        )

    df = df[df[SPLIT_COLUMN] == OUTER_TEST_SPLIT].copy()
    if df.empty:
        raise ValueError("No outer_test rows were found in the predictions file.")

    return df


def combo_color(active_labels: List[str]) -> str:
    if not active_labels:
        return "lightgray"
    if len(active_labels) == 1:
        return UPSET_SET_COLORS[active_labels[0]]

    rgb = np.array(
        [mcolors.to_rgb(UPSET_SET_COLORS[label]) for label in active_labels],
        dtype=float,
    )
    return mcolors.to_hex(tuple(rgb.mean(axis=0)))


def intersection_summary(
    correct_flags: pd.DataFrame,
    labels: List[str],
) -> List[Dict[str, Any]]:
    grouped = correct_flags.groupby(labels).size().rename("count").reset_index()
    rows: List[Dict[str, Any]] = []

    for _, rec in grouped.iterrows():
        active = [label for label in labels if bool(rec[label])]
        if active:
            rows.append(
                {
                    "active_labels": active,
                    "active_key": "∩".join(active),
                    "count": int(rec["count"]),
                    "degree": len(active),
                }
            )

    rows.sort(key=lambda row: (-row["count"], -row["degree"], row["active_key"]))
    return rows


def unique_correct_masks(
    flags: pd.DataFrame,
    labels: List[str],
) -> Dict[str, pd.Series]:
    masks: Dict[str, pd.Series] = {}

    for label in labels:
        other_labels = [other for other in labels if other != label]
        masks[label] = flags[label] & ~flags[other_labels].any(axis=1)

    return masks


def window_candidate_counts(distances: pd.Series) -> List[int]:
    counts: List[int] = []

    for low in WINDOW_LOWS.tolist():
        low_f = float(low)
        high_f = low_f + WINDOW_WIDTH
        in_window = (distances >= low_f) & (distances <= high_f)
        counts.append(int(in_window.sum()))

    return counts


def unique_correct_window_counts(
    distances: pd.Series,
    unique_masks: Dict[str, pd.Series],
    labels: List[str],
) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}

    for label in labels:
        selected_distances = distances.loc[unique_masks[label]]
        counts: List[int] = []

        for low in WINDOW_LOWS.tolist():
            low_f = float(low)
            high_f = low_f + WINDOW_WIDTH
            in_window = (
                (selected_distances >= low_f)
                & (selected_distances <= high_f)
            )
            counts.append(int(in_window.sum()))

        result[label] = counts

    return result


def print_only_l1e_correct_rows(
    subset_df: pd.DataFrame,
    flags: pd.DataFrame,
    model_to_label: Dict[str, str],
    tau: float,
    k: float,
) -> None:
    l1e_label = model_to_label["R"]
    labels = list(model_to_label.values())
    other_labels = [label for label in labels if label != l1e_label]

    only_l1e_mask = flags[l1e_label] & ~flags[other_labels].any(axis=1)
    selected_index = flags.index[only_l1e_mask]
    language_code = str(subset_df[L1_COLUMN].iloc[0])

    print("\n" + "=" * 110)
    print(
        f"ONLY L1E CORRECT | language={language_code} | "
        f"k={k:.3f} | tau={tau:.12g} | matched items={len(selected_index)}"
    )
    print("=" * 110)

    if len(selected_index) == 0:
        print("No rows found.")
        return

    selected_keys = pd.DataFrame(
        selected_index.tolist(),
        columns=[ID_COLUMN, L1_COLUMN],
    )
    output = subset_df.merge(
        selected_keys,
        on=[ID_COLUMN, L1_COLUMN],
        how="inner",
    ).copy()

    output["abs_error"] = (
        output[PREDICTION_COLUMN].astype(float)
        - output[LABEL_COLUMN].astype(float)
    ).abs()
    output["is_correct"] = output["abs_error"].le(tau + EPSILON)

    model_display_names = {
        "BL1": "BL1",
        "E": "L1C",
        "S": "L1S",
        "R": "L1E",
    }
    model_order = {"BL1": 0, "E": 1, "S": 2, "R": 3}

    output["upset_model"] = output[MODEL_COLUMN].map(model_display_names)
    output["_model_order"] = output[MODEL_COLUMN].map(model_order)

    output = output.sort_values(
        by=[LEVENSHTEIN_COLUMN, ID_COLUMN, "_model_order"],
        ascending=[False, True, True],
        kind="stable",
    )

    preferred_columns = [
        ID_COLUMN,
        L1_COLUMN,
        LEVENSHTEIN_COLUMN,
        MODEL_COLUMN,
        "upset_model",
        LABEL_COLUMN,
        PREDICTION_COLUMN,
        "abs_error",
        "is_correct",
    ]
    remaining_columns = [
        column
        for column in output.columns
        if column not in preferred_columns + ["_model_order"]
    ]
    output = output[preferred_columns + remaining_columns]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(output.to_string(index=False))
    print(f"\nPrinted {len(output)} raw rows for {len(selected_index)} items.")
    print("=" * 110)


def analyze_upset_language(
    subset_df: pd.DataFrame,
    model_to_label: Dict[str, str],
    k: float,
) -> Dict[str, Any]:
    x = subset_df.copy()
    x["abs_error"] = (
        x[PREDICTION_COLUMN].astype(float)
        - x[LABEL_COLUMN].astype(float)
    ).abs()

    index_columns = [ID_COLUMN, L1_COLUMN]
    err = x.pivot(
        index=index_columns,
        columns=MODEL_COLUMN,
        values="abs_error",
    )

    complete = err.notna().all(axis=1)
    if not bool(complete.any()):
        raise ValueError("No complete aligned rows across BL1/E/S/R models for UpSet.")

    err = err.loc[complete]

    gold = (
        x.groupby(index_columns)[LABEL_COLUMN]
        .first()
        .astype(float)
        .loc[complete]
    )
    distances = (
        x.groupby(index_columns)[LEVENSHTEIN_COLUMN]
        .first()
        .astype(float)
        .loc[complete]
    )

    sigma = float(np.std(gold.to_numpy(dtype=float), ddof=0))
    tau = float(k * sigma)
    labels = [model_to_label[model_name] for model_name in model_to_label]

    flags_payload: Dict[str, pd.Series] = {}
    for model_name, label in model_to_label.items():
        flags_payload[label] = err[model_name].le(tau + EPSILON)

    flags = pd.DataFrame(flags_payload)
    exclusive_masks = unique_correct_masks(flags, labels)

    print_only_l1e_correct_rows(
        subset_df=x,
        flags=flags,
        model_to_label=model_to_label,
        tau=tau,
        k=k,
    )

    return {
        "tau": tau,
        "label_std": sigma,
        "n_items_considered": int(len(err)),
        "set_sizes": {label: int(flags[label].sum()) for label in labels},
        "total_unique_correct_counts": {
            label: int(exclusive_masks[label].sum())
            for label in labels
        },
        "intersections": intersection_summary(flags, labels),
        "unique_correct_window_counts": unique_correct_window_counts(
            distances=distances,
            unique_masks=exclusive_masks,
            labels=labels,
        ),
        "window_candidate_counts": window_candidate_counts(distances),
    }


def draw_upset_panel(
    subfig: Any,
    language_name: str,
    subtitle: str,
    payload: Dict[str, Any],
    labels: List[str],
) -> None:
    gs = subfig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 4.0],
        height_ratios=[3.0, 1.7],
        wspace=0.08,
        hspace=0.06,
    )

    ax_blank = subfig.add_subplot(gs[0, 0])
    ax_inter = subfig.add_subplot(gs[0, 1])
    ax_sets = subfig.add_subplot(gs[1, 0])
    ax_matrix = subfig.add_subplot(gs[1, 1], sharex=ax_inter)
    ax_blank.axis("off")

    intersections = payload["intersections"]

    if not intersections:
        ax_inter.text(
            0.5,
            0.5,
            "No intersections to plot",
            ha="center",
            va="center",
        )
        ax_inter.axis("off")
        ax_matrix.axis("off")
    else:
        x = np.arange(len(intersections))
        y = np.asarray([row["count"] for row in intersections], dtype=int)
        colors = [combo_color(row["active_labels"]) for row in intersections]

        ax_inter.bar(x, y, color=colors, alpha=0.65)
        ax_inter.set_ylim(0, max(int(y.max()), 1) * 1.10)
        ax_inter.set_ylabel("Count", fontsize=10)
        ax_inter.grid(axis="y", linestyle="--", alpha=0.35)
        ax_inter.set_xticks([])
        ax_inter.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        for x_pos, y_val in zip(x, y):
            ax_inter.text(
                x_pos,
                y_val,
                str(y_val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax_matrix.set_yticks(np.arange(len(labels)))
        ax_matrix.set_yticklabels([])
        ax_matrix.tick_params(axis="y", left=False, labelleft=False)
        ax_matrix.grid(axis="x", linestyle="--", alpha=0.2)

        for idx, row in enumerate(intersections):
            active = set(row["active_labels"])
            active_positions = [
                pos for pos, label in enumerate(labels) if label in active
            ]
            color = combo_color(list(active))

            if active_positions:
                ax_matrix.plot(
                    [idx, idx],
                    [min(active_positions), max(active_positions)],
                    color=color,
                    linewidth=1.8,
                )

            for pos, label in enumerate(labels):
                ax_matrix.scatter(
                    idx,
                    pos,
                    s=42,
                    c=color if label in active else "lightgray",
                    zorder=3,
                )

        # ax_matrix.set_xticks(x)
        # ax_matrix.set_xticklabels(
        #     [row["active_key"] for row in intersections],
        #     rotation=55,
        #     ha="right",
        #     fontsize=7,
        # )
        # Hide intersection labels to keep more vertical space for the bars/matrix.
        ax_matrix.set_xticks([])
        ax_matrix.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_matrix.tick_params(axis="x", length=0)
        ax_matrix.set_ylim(-0.6, len(labels) - 0.4)
        ax_matrix.invert_yaxis()

    set_sizes = payload["set_sizes"]
    values = [set_sizes[label] for label in labels]
    y_positions = np.arange(len(labels))

    ax_sets.barh(
        y_positions,
        values,
        color=[UPSET_SET_COLORS[label] for label in labels],
        alpha=0.65,
    )

    for y_pos, value in zip(y_positions, values):
        ax_sets.text(
            value / 2 if value > 0 else 0.5,
            y_pos,
            str(int(value)),
            ha="center",
            va="center",
            fontsize=9,
        )

    ax_sets.set_yticks(y_positions)
    ax_sets.set_yticklabels(labels, fontsize=10)
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Set size", fontsize=10)
    ax_sets.grid(axis="x", linestyle="--", alpha=0.35)

    subfig.suptitle(language_name, fontsize=11, y=0.97)
    subfig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=10)


def make_upset_figure(
    payload_by_language: Dict[str, Dict[str, Any]],
    k: float,
) -> plt.Figure:
    ordered_codes = [
        code for code, _ in LANGUAGE_SPECS if code in payload_by_language
    ]
    n = len(ordered_codes)

    if n == 0:
        raise ValueError("No language payload available for UpSet plot.")

    if n == 1:
        fig = plt.figure(figsize=(10, 6.8)) ## change size here
        subfigs = [fig]
    else:
        fig = plt.figure(figsize=(10, 7 * n)) ## change size here
        subfigs = fig.subfigures(nrows=n, ncols=1)

        if not isinstance(subfigs, np.ndarray):
            subfigs = np.array([subfigs])

    labels = [UPSET_MODEL_LABELS[name] for name in UPSET_MODEL_LABELS]

    for subfig, code in zip(subfigs, ordered_codes):
        payload = payload_by_language[code]
        language_name = str(dict(LANGUAGE_SPECS).get(code, code))
        subtitle = (
            f"BL1/L1C/L1S/L1E | k={k:.3f} | "
            f"τ={float(payload['tau']):.4f} | "
            f"n={int(payload['n_items_considered'])}"
        )
        draw_upset_panel(subfig, language_name, subtitle, payload, labels)

    if n > 1:
        fig.suptitle(
            "Correctness UpSet by Language (BL1/L1C/L1S/L1E)",
            fontsize=14,
            y=0.999,
            fontweight="bold",
        )

    return fig


def make_combined_upset_figure(
    payloads_by_k: Dict[float, Dict[str, Dict[str, Any]]],
    thresholds: List[float],
) -> plt.Figure:
    ordered_codes = [code for code, _ in LANGUAGE_SPECS]
    n_rows = len(ordered_codes)
    n_cols = len(thresholds)

    title_margin_in = 0.4
    fig_height = 5 * n_rows + title_margin_in
    fig = plt.figure(figsize=(10 * n_cols, fig_height))
    all_subfigs = fig.subfigures(
        nrows=n_rows + 1,
        ncols=n_cols,
        height_ratios=[title_margin_in / 5] + [1] * n_rows,
        wspace=0.0,
        hspace=0.0,
    )
    all_subfigs = np.asarray(all_subfigs, dtype=object).reshape(n_rows + 1, n_cols)
    subfigs = all_subfigs[1:]
    # fig.subplots_adjust(
    #     left=0.04,
    #     right=0.98,
    #     top=0.80,
    #     bottom=0.05,
    #     wspace=0.12,  # more horizontal whitespace between upset panels
    #     hspace=0.68,
    # ) # more vertical whitespace between upset panels

    if n_rows == 1 and n_cols == 1:
        subfig_grid = np.array([[subfigs]], dtype=object)
    elif n_rows == 1:
        subfig_grid = np.array([subfigs], dtype=object)
    elif n_cols == 1:
        subfig_grid = np.array([[sub] for sub in subfigs], dtype=object)
    else:
        subfig_grid = np.asarray(subfigs, dtype=object)

    labels = [UPSET_MODEL_LABELS[name] for name in UPSET_MODEL_LABELS]

    for row_idx, code in enumerate(ordered_codes):
        language_name = str(dict(LANGUAGE_SPECS).get(code, code))

        for col_idx, k in enumerate(thresholds):
            payload = payloads_by_k.get(k, {}).get(code)
            subfig = subfig_grid[row_idx, col_idx]

            if payload is None:
                ax = subfig.add_subplot(111)
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"No data available\n{language_name}, k={k:.3f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                continue

            subtitle = (
                f"BL1/L1C/L1S/L1E | k={k:.3f} | "
                f"τ={float(payload['tau']):.4f} | "
                f"n={int(payload['n_items_considered'])}"
            )
            draw_upset_panel(subfig, language_name, subtitle, payload, labels)

    fig.suptitle(
        "Correctness UpSet by Language and Threshold (BL1/L1C/L1S/L1E)",
        fontsize=14,
        y=1.0 - 0.2 / fig_height,
        fontweight="bold",
    )
    return fig


def plot_exclusive_correct_enrichment_by_window(
    upset_payloads_by_k: Dict[float, Dict[str, Dict[str, Any]]],
    range_by_language: Dict[str, tuple[float, float]],
    output_path: Path,
    no_show: bool,
) -> None:
    """
    Plot enrichment of each model's exclusive successes by Levenshtein window.

    Translucent gray bars show the number of aligned candidate items in each
    overlapping window. Their scale is the secondary right-hand y-axis.

    For model m, language l, and window w:

        local exclusive-success rate
        ----------------------------
        model's language-level exclusive-success rate

    More explicitly:

        (exclusive successes in window w / candidates in window w)
        -----------------------------------------------------------
        (total exclusive successes in language l /
         total aligned candidates in language l)

    The language-level baseline is calculated over unique aligned outer-test
    items, not by summing sliding-window counts. Therefore, it is unaffected
    by overlapping windows or by the subset of windows displayed in the plot.
    """
    thresholds = [0.05, 0.5]
    labels = [UPSET_MODEL_LABELS[name] for name in UPSET_MODEL_LABELS]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(LANGUAGE_SPECS),
        figsize=(16, 12),
        sharex="col",
        squeeze=False,
    )

    for row_idx, k in enumerate(thresholds):
        payload_by_language = upset_payloads_by_k.get(k, {})

        for col_idx, (language_code, language_name) in enumerate(LANGUAGE_SPECS):
            ax = axes[row_idx, col_idx]
            payload = payload_by_language.get(language_code)

            low, high = range_by_language[language_code]
            mask = center_mask(low, high)
            ticks = WINDOW_CENTERS[mask]

            if len(ticks) == 0:
                raise ValueError(
                    f"No centers selected for {language_code}: {low}-{high}"
                )

            if payload is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_title(language_name)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{center:.1f}" for center in ticks])
                ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)
                continue

            counts_by_model = payload.get("unique_correct_window_counts", {})
            candidate_counts = np.asarray(
                payload.get(
                    "window_candidate_counts",
                    [0] * len(WINDOW_CENTERS),
                ),
                dtype=float,
            )
            print(language_code,candidate_counts)  ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! debugging print
            if len(candidate_counts) != len(WINDOW_CENTERS):
                raise ValueError(
                    "Unexpected number of candidate-window counts for "
                    f"{language_code}, k={k}."
                )

            displayed_candidate_counts = candidate_counts[mask]

            # Candidate counts are independent of model and threshold. Use a
            # secondary axis so the bars provide context without changing the
            # enrichment scale. Show the right-axis label only in the final
            # column to avoid repeating it across every panel.
            add_window_count_axis(
                ax,
                ticks,
                displayed_candidate_counts,
                ylabel="Candidate set size",
                show_ylabel=col_idx == len(LANGUAGE_SPECS) - 1,
            )

            total_candidates = float(payload["n_items_considered"])
            total_unique_correct_counts = payload["total_unique_correct_counts"]

            if total_candidates <= 0:
                raise ValueError(
                    "The language-level aligned candidate count must be "
                    f"positive for {language_code}, k={k}."
                )

            for label in labels:
                unique_counts = np.asarray(
                    counts_by_model.get(label, [0] * len(WINDOW_CENTERS)),
                    dtype=float,
                )

                if len(unique_counts) != len(WINDOW_CENTERS):
                    raise ValueError(
                        "Unexpected number of exclusive-success counts for "
                        f"{label}, {language_code}, k={k}."
                    )

                if label not in total_unique_correct_counts:
                    raise ValueError(
                        "Missing language-level exclusive-success count for "
                        f"{label}, {language_code}, k={k}."
                    )

                # Local rate: exclusive successes among candidates in this
                # particular sliding window.
                local_rate = np.divide(
                    unique_counts,
                    candidate_counts,
                    out=np.full(len(WINDOW_CENTERS), np.nan, dtype=float),
                    where=candidate_counts > 0,
                )

                # Baseline rate: total item-level exclusive successes for this
                # model and language, divided by all aligned test candidates
                # for that language. This intentionally does not use sums of
                # overlapping sliding-window counts.
                total_unique_correct = float(total_unique_correct_counts[label])
                baseline_rate = total_unique_correct / total_candidates

                enrichment = np.divide(
                    local_rate,
                    baseline_rate,
                    out=np.full(len(WINDOW_CENTERS), np.nan, dtype=float),
                    where=(
                        np.isfinite(local_rate)
                        & np.isfinite(baseline_rate)
                        & (baseline_rate > 0)
                    ),
                )

                style = UNIQUE_CORRECT_STYLE_BY_LABEL[label]
                ax.plot(
                    ticks,
                    enrichment[mask],
                    color=UPSET_SET_COLORS[label],
                    marker=str(style["marker"]),
                    linestyle=str(style["linestyle"]),
                    markersize=7,
                    linewidth=float(style["linewidth"]),
                    alpha=float(style["alpha"]),
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    label=label,
                    zorder=3,
                )

            ax.axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.70,
                zorder=1,
            )

            ax.set_title(
                f"{language_name}\nk={k:.2f}, τ={float(payload['tau']):.4f}",
                fontsize=11,
            )
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{center:.1f}" for center in ticks])
            ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)
            ax.set_ylim(bottom=0)

            if col_idx == 0:
                ax.set_ylabel(
                    "Exclusive-success enrichment\n"
                    "(window rate / language baseline)",
                    fontsize=11,
                )

            if row_idx == 1:
                ax.set_xlabel(
                    "Normalized Levenshtein Distance\n(Window Center)",
                    fontsize=11,
                )

            if row_idx == 0 and col_idx == len(LANGUAGE_SPECS) - 1:
                ax.legend(
                    title="Components",
                    title_fontsize=11,
                    fontsize=10,
                    loc="lower right",
                )

    fig.suptitle(
        "Enrichment of Exclusive Correct Predictions by Levenshtein Window",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.985))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if not no_show:
        plt.show()

    plt.close(fig)


def plot_upset_thresholds(
    summary: Dict[str, Any],
    output_dir: Path,
    no_show: bool,
) -> Dict[float, Dict[str, Dict[str, Any]]]:
    predictions_df = load_predictions_from_summary(summary)

    missing_models = [
        model_name
        for model_name in UPSET_MODEL_LABELS
        if model_name not in set(predictions_df[MODEL_COLUMN].astype(str).unique())
    ]

    if missing_models:
        raise ValueError(
            "Predictions file does not contain all required UpSet models "
            f"(BL1/E/S/R): {missing_models}"
        )

    model_names = list(UPSET_MODEL_LABELS)
    model_to_label = {
        model_name: UPSET_MODEL_LABELS[model_name]
        for model_name in model_names
    }

    payloads_by_k: Dict[float, Dict[str, Dict[str, Any]]] = {}

    thresholds = [0.05, 0.5]

    for k in thresholds:
        payload_by_language: Dict[str, Dict[str, Any]] = {}

        for language_code, _ in LANGUAGE_SPECS:
            subset_df = predictions_df.loc[
                predictions_df[L1_COLUMN].eq(language_code)
                & predictions_df[MODEL_COLUMN].isin(model_names)
            ].copy()

            if subset_df.empty:
                continue

            payload_by_language[language_code] = analyze_upset_language(
                subset_df=subset_df,
                model_to_label=model_to_label,
                k=k,
            )

        payloads_by_k[k] = payload_by_language
        json_path = output_dir / f"lightgbm_final_upset_k{_k_token(k)}.json"

        json_path.write_text(
            json.dumps(
                {
                    "k": k,
                    "predictions_file": summary.get("predictions_file"),
                    "models": model_names,
                    "languages": payload_by_language,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved UpSet summary: {json_path}")

    combined_figure = make_combined_upset_figure(payloads_by_k, thresholds)
    combined_figure_path = output_dir / "lightgbm_final_upset_k0.05_k0.5.png"
    combined_figure_path.parent.mkdir(parents=True, exist_ok=True)
    combined_figure.savefig(combined_figure_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {combined_figure_path}")

    if not no_show:
        plt.show()

    plt.close(combined_figure)

    return payloads_by_k


def main() -> None:
    args = parse_args()

    if not args.metrics_json.exists():
        raise FileNotFoundError(
            f"Metrics JSON not found: {args.metrics_json}. "
            "Run evaluate_lightgbm_final.py first."
        )

    summary = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    glmm_df = load_glmm_df(args.glmm_data_file)

    range_by_language = {
        "de": parse_center_range(args.de_centers, "German"),
        "es": parse_center_range(args.es_centers, "Spanish"),
        "cn": parse_center_range(args.cn_centers, "Chinese"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline_group_curves(
        summary=summary,
        output_path=args.output_dir / "lightgbm_final_sliding_baseline_groups.png",
        range_by_language=range_by_language,
        no_show=args.no_show,
    )

    plot_l1_group_curves_with_glmm(
        summary=summary,
        glmm_df=glmm_df,
        output_path=args.output_dir / "lightgbm_final_sliding_l1_groups_glmm.png",
        range_by_language=range_by_language,
        no_show=args.no_show,
    )

    plot_relatedness_individual_curves_with_window_sizes(
        summary=summary,
        output_path=(
            args.output_dir
            / "lightgbm_final_sliding_relatedness_individual_window_sizes.png"
        ),
        range_by_language=range_by_language,
        no_show=args.no_show,
    )

    upset_payloads_by_k = plot_upset_thresholds(
        summary=summary,
        output_dir=args.output_dir,
        no_show=args.no_show,
    )

    plot_exclusive_correct_enrichment_by_window(
        upset_payloads_by_k=upset_payloads_by_k,
        range_by_language=range_by_language,
        output_path=(
            args.output_dir
            / "lightgbm_final_exclusive_correct_enrichment_by_window.png"
        ),
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()