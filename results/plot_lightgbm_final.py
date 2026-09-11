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
    ("BE", "BE"),
    ("BS", "BS"),
    ("BR", "BR"),
    ("BES", "BES"),
    ("BER", "BER"),
    ("BSR", "BSR"),
    ("BESR", "BESR"),
]

SLIDING_GROUP_COLORS = {
    "B": "black",
    "BE": "dodgerblue",
    "BS": "gold",
    "BR": "red",
    "BES": "limegreen",
    "BSR": "darkorange",
    "BER": "darkviolet",
    "BESR": "brown",
}

SLIDING_GROUP_STYLES = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    "BE": {"linestyle": "--", "marker": "s", "linewidth": 4.5, "alpha": 0.60},
    "BS": {"linestyle": "-.", "marker": "^", "linewidth": 4.5, "alpha": 0.60},
    "BR": {"linestyle": ":", "marker": "D", "linewidth": 4.5, "alpha": 0.60},
    "BES": {"linestyle": "--", "marker": "v", "linewidth": 4.5, "alpha": 0.60},
    "BSR": {"linestyle": "-.", "marker": "P", "linewidth": 4.5, "alpha": 0.60},
    "BER": {"linestyle": ":", "marker": "X", "linewidth": 4.5, "alpha": 0.60},
    "BESR": {"linestyle": "-", "marker": "*", "linewidth": 4.5, "alpha": 0.60},
}

L1_GROUP_MODELS = [
    ("raw_roberta", "B"),
    ("E", "E"),
    ("S", "S"),
    ("R", "R"),
    ("ES", "ES"),
    ("ER", "ER"),
    ("SR", "SR"),
    ("ESR", "ESR"),
]

L1_GROUP_COLORS = {
    "B": "black",
    "E": "dodgerblue",
    "S": "gold",
    "R": "red",
    "ES": "limegreen",
    "SR": "darkorange",
    "ER": "darkviolet",
    "ESR": "brown",
}

L1_GROUP_STYLES = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    "E": {"linestyle": "--", "marker": "s", "linewidth": 4.5, "alpha": 0.60},
    "S": {"linestyle": "-.", "marker": "^", "linewidth": 4.5, "alpha": 0.60},
    "R": {"linestyle": ":", "marker": "D", "linewidth": 4.5, "alpha": 0.60},
    "ES": {"linestyle": "--", "marker": "v", "linewidth": 4.5, "alpha": 0.60},
    "SR": {"linestyle": "-.", "marker": "P", "linewidth": 4.5, "alpha": 0.60},
    "ER": {"linestyle": ":", "marker": "X", "linewidth": 4.5, "alpha": 0.60},
    "ESR": {"linestyle": "-", "marker": "*", "linewidth": 4.5, "alpha": 0.60},
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
    "raw_roberta": "B",
    "E": "E",
    "S": "S",
    "R": "R",
}

UPSET_SET_COLORS = {
    "B": "black",
    "E": "tab:blue",
    "S": "gold",
    "R": "tab:red",
}

UPSET_LABEL_ORDER = ["B", "E", "S", "R"]
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final LightGBM evaluation outputs.")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glmm-data-file", type=Path, default=DEFAULT_GLMM_DATA_FILE)
    parser.add_argument("--de-centers", type=str, default=DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", type=str, default=DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", type=str, default=DEFAULT_CENTER_RANGES["cn"])
    parser.add_argument("--no-show", action="store_true", help="Save figures without calling plt.show().")
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

    outer_test = model_payload.get("outer_test", {})
    per_language = outer_test.get("per_language", {})
    language_payload = per_language.get(language_code)
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
        rmse_value = metric.get("rmse")
        rmses[idx] = np.nan if rmse_value is None else float(rmse_value)
        counts[idx] = int(metric.get("n_rows", 0))

    return rmses, counts


def add_window_count_axis(ax: plt.Axes, ticks: np.ndarray, counts: np.ndarray) -> plt.Axes:
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
    ax2.set_ylabel("Window set size", fontsize=12, color="dimgray")
    ax2.tick_params(axis="y", colors="dimgray", labelcolor="dimgray")
    ax2.set_ylim(0, max(1.0, float(np.max(counts)) * 1.1))
    ax2.grid(False)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    return ax2


def compute_glmm_window_means(glmm_df: pd.DataFrame, language_code: str) -> np.ndarray:
    lang_df = glmm_df[glmm_df[L1_COLUMN] == language_code]
    means: List[float] = []
    for low in WINDOW_LOWS.tolist():
        low_f = float(low)
        high_f = float(low_f + WINDOW_WIDTH)
        window_df = lang_df[(lang_df[LEVENSHTEIN_COLUMN] >= low_f) & (lang_df[LEVENSHTEIN_COLUMN] <= high_f)]
        means.append(float(window_df[LABEL_COLUMN].mean()) if not window_df.empty else np.nan)
    return np.array(means, dtype=float)


def load_glmm_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GLMM data file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {L1_COLUMN, LEVENSHTEIN_COLUMN, LABEL_COLUMN}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"GLMM data file is missing required columns: {sorted(missing_cols)}")
    return df


def plot_baseline_group_curves(
    summary: Dict[str, Any],
    output_path: Path,
    range_by_language: Dict[str, tuple[float, float]],
    no_show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        range_low, range_high = range_by_language[language_code]
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(f"No window centers in selected range for {language_code}: {range_low}-{range_high}")

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
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=label,
                color=SLIDING_GROUP_COLORS[label],
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
            ax.legend(fontsize=12, title="Configuration", title_fontsize=14, loc="lower right")

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("BL1 Group Models by Levenshtein Window", fontsize=13, y=0.99, fontweight="bold")
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
        range_low, range_high = range_by_language[language_code]
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(f"No window centers in selected range for {language_code}: {range_low}-{range_high}")

        for model_name, label in L1_GROUP_MODELS:
            arrays = model_windows(summary, model_name, language_code)
            if arrays is None:
                continue
            rmses, _counts = arrays
            style = L1_GROUP_STYLES[label]
            ax.plot(
                ticks,
                rmses[mask],
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=label,
                color=L1_GROUP_COLORS[label],
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
            handles_1, labels_1 = ax.get_legend_handles_labels()
            handles_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(
                handles_1 + handles_2,
                labels_1 + labels_2,
                fontsize=11,
                title="Series",
                title_fontsize=13,
                loc="lower right",
            )

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("L1 Group Models vs GLMM by Levenshtein Window", fontsize=13, y=0.99, fontweight="bold")
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
    model_specs: List[Tuple[str, str, str]] = [("raw_roberta", "B", "black")]
    for feature_name in RELATEDNESS_FEATURES:
        model_name = f"BL1_plus_{feature_name}"
        if model_name not in summary.get("models", {}):
            continue
        color = FEATURE_COLOR_MAP.get(feature_name, "tab:gray")
        model_specs.append((model_name, f"+ {feature_name}", color))

    if len(model_specs) == 1:
        raise ValueError("No BL1 relatedness individual-feature models were found in metrics JSON.")

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        range_low, range_high = range_by_language[language_code]
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(f"No window centers in selected range for {language_code}: {range_low}-{range_high}")

        style_idx = 0
        reference_counts: np.ndarray | None = None
        for model_name, label, color in model_specs:
            arrays = model_windows(summary, model_name, language_code)
            if arrays is None:
                continue
            rmses, counts = arrays
            if reference_counts is None:
                reference_counts = counts

            if model_name == "raw_roberta":
                style = {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0}
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
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=label,
                color=color,
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
            handles_1, labels_1 = ax.get_legend_handles_labels()
            handles_2, labels_2 = ([], []) if ax2 is None else ax2.get_legend_handles_labels()
            ax.legend(
                handles_1 + handles_2,
                labels_1 + labels_2,
                fontsize=10,
                title="Series",
                title_fontsize=12,
                loc="lower right",
            )

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
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
        raise FileNotFoundError(f"Predictions file referenced in metrics JSON does not exist: {path}")

    df = pd.read_csv(path, keep_default_na=False)
    required_cols = {ID_COLUMN, L1_COLUMN, MODEL_COLUMN, SPLIT_COLUMN, LABEL_COLUMN, PREDICTION_COLUMN}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Predictions file is missing required columns for UpSet: {sorted(missing_cols)}")

    df = df[df[SPLIT_COLUMN] == OUTER_TEST_SPLIT].copy()
    if df.empty:
        raise ValueError("No outer_test rows were found in the predictions file.")
    return df


def ordered_upset_labels() -> List[str]:
    return [label for label in UPSET_LABEL_ORDER if label in set(UPSET_SET_COLORS)]


def combo_color(active_labels: List[str]) -> str:
    if not active_labels:
        return "lightgray"
    if len(active_labels) == 1:
        return UPSET_SET_COLORS[active_labels[0]]

    rgb = np.array([mcolors.to_rgb(UPSET_SET_COLORS[label]) for label in active_labels], dtype=float)
    mixed = rgb.mean(axis=0)
    return mcolors.to_hex((float(mixed[0]), float(mixed[1]), float(mixed[2])))


def intersection_summary(correct_flags: pd.DataFrame, labels: List[str]) -> List[Dict[str, Any]]:
    grouped = correct_flags.groupby(labels).size().rename("count").reset_index()
    rows: List[Dict[str, Any]] = []

    for _, rec in grouped.iterrows():
        active = [label for label in labels if bool(rec[label])]
        if not active:
            continue
        rows.append(
            {
                "active_labels": active,
                "active_key": "&".join(active),
                "count": int(rec["count"]),
                "degree": int(len(active)),
            }
        )

    rows.sort(key=lambda r: (-r["count"], -r["degree"], r["active_key"]))
    return rows


def analyze_upset_language(subset_df: pd.DataFrame, model_to_label: Dict[str, str], k: float) -> Dict[str, Any]:
    x = subset_df.copy()
    x["abs_error"] = (x[PREDICTION_COLUMN].astype(float) - x[LABEL_COLUMN].astype(float)).abs()

    err = x.pivot(index=[ID_COLUMN, L1_COLUMN], columns=MODEL_COLUMN, values="abs_error")
    complete = err.notna().all(axis=1)
    if not bool(complete.any()):
        raise ValueError("No complete aligned rows across B/E/S/R models for UpSet.")

    err = err.loc[complete]
    gold = x.groupby([ID_COLUMN, L1_COLUMN])[LABEL_COLUMN].first().astype(float).loc[complete]

    sigma = float(np.std(gold.to_numpy(dtype=float), ddof=0))
    tau = float(k * sigma)

    labels = [model_to_label[m] for m in model_to_label]
    flags_payload: Dict[str, pd.Series] = {}
    for model_name, label in model_to_label.items():
        flags_payload[label] = err[model_name].le(tau + EPSILON)
    flags = pd.DataFrame(flags_payload)

    intersections = intersection_summary(flags, labels)
    set_sizes = {label: int(flags[label].sum()) for label in labels}

    return {
        "tau": tau,
        "label_std": sigma,
        "n_items_considered": int(len(err)),
        "set_sizes": set_sizes,
        "intersections": intersections,
    }


def draw_upset_panel(
    subfig: Any,
    language_name: str,
    subtitle: str,
    payload: Dict[str, Any],
    labels: List[str],
) -> None:
    gs = subfig.add_gridspec(2, 2, width_ratios=[1.25, 4.0], height_ratios=[3.0, 1.7], wspace=0.08, hspace=0.06)
    ax_blank = subfig.add_subplot(gs[0, 0])
    ax_inter = subfig.add_subplot(gs[0, 1])
    ax_sets = subfig.add_subplot(gs[1, 0])
    ax_matrix = subfig.add_subplot(gs[1, 1], sharex=ax_inter)

    ax_blank.axis("off")
    intersections = payload["intersections"]

    if not intersections:
        ax_inter.text(0.5, 0.5, "No intersections to plot", ha="center", va="center")
        ax_inter.axis("off")
        ax_matrix.axis("off")
    else:
        x = np.arange(len(intersections))
        y = np.array([row["count"] for row in intersections], dtype=int)
        colors = [combo_color(row["active_labels"]) for row in intersections]

        ax_inter.bar(x, y, color=colors, alpha=0.65)
        ax_inter.set_ylabel("Count", fontsize=10)
        ax_inter.grid(axis="y", linestyle="--", alpha=0.35)
        ax_inter.set_xticks([])
        ax_inter.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        for x_pos, y_val in zip(x, y):
            ax_inter.text(x_pos, y_val, str(int(y_val)), ha="center", va="bottom", fontsize=9)

        y_positions = np.arange(len(labels))
        ax_matrix.set_yticks(y_positions)
        ax_matrix.set_yticklabels(labels, fontsize=10)
        ax_matrix.grid(axis="x", linestyle="--", alpha=0.2)

        for idx, row in enumerate(intersections):
            active = set(row["active_labels"])
            active_positions = [pos for pos, lbl in enumerate(labels) if lbl in active]
            inter_color = combo_color(list(active))
            if active_positions:
                ax_matrix.plot([idx, idx], [min(active_positions), max(active_positions)], color=inter_color, linewidth=1.8)
            for pos, label in enumerate(labels):
                marker_color = inter_color if label in active else "lightgray"
                ax_matrix.scatter(idx, pos, s=42, c=marker_color, zorder=3)

        ax_matrix.set_xticks(x)
        ax_matrix.set_xticklabels([row["active_key"] for row in intersections], rotation=55, ha="right", fontsize=9)
        ax_matrix.tick_params(axis="x", length=0)
        ax_matrix.set_ylim(-0.6, len(labels) - 0.4)
        ax_matrix.invert_yaxis()

    set_sizes = payload["set_sizes"]
    vals = [set_sizes[label] for label in labels]
    y = np.arange(len(labels))
    set_colors = [UPSET_SET_COLORS[label] for label in labels]

    ax_sets.barh(y, vals, color=set_colors, alpha=0.65)
    for y_pos, label, val in zip(y, labels, vals):
        text_color = "white" if label == "B" else "black"
        ax_sets.text(val / 2 if val > 0 else 0.5, y_pos, str(int(val)), va="center", ha="center", fontsize=9, color=text_color)

    ax_sets.set_yticks(y)
    ax_sets.set_yticklabels(labels, fontsize=10)
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Set size", fontsize=10)
    ax_sets.grid(axis="x", linestyle="--", alpha=0.35)

    subfig.suptitle(language_name, fontsize=13, y=0.96)
    subfig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=10)


def make_upset_figure(payload_by_language: Dict[str, Dict[str, Any]], k: float) -> plt.Figure:
    ordered_codes = [code for code, _name in LANGUAGE_SPECS if code in payload_by_language]
    n = len(ordered_codes)
    if n == 0:
        raise ValueError("No language payload available for UpSet plot.")

    if n == 1:
        fig = plt.figure(figsize=(10, 6.8))
        subfigs = [fig]
    else:
        fig = plt.figure(figsize=(10, 7 * n))
        subfigs = fig.subfigures(nrows=n, ncols=1)
        if not isinstance(subfigs, np.ndarray):
            subfigs = np.array([subfigs])

    labels = [UPSET_MODEL_LABELS[m] for m in UPSET_MODEL_LABELS]

    for subfig, code in zip(subfigs, ordered_codes):
        payload = payload_by_language[code]
        language_name = str(dict(LANGUAGE_SPECS).get(code, code))
        tau = float(payload.get("tau", np.nan))
        n_items = int(payload.get("n_items_considered", 0))
        subtitle = (
            f"B/E/S/R | k={k:.3f} | tau={tau:.4f} | "
            f"n={n_items}"
        )
        draw_upset_panel(subfig, language_name, subtitle, payload, labels)

    if n > 1:
        fig.suptitle("Correctness UpSet by Language (B/E/S/R)", fontsize=14, y=0.999, fontweight="bold")

    return fig


def plot_upset_thresholds(summary: Dict[str, Any], output_dir: Path, no_show: bool) -> None:
    predictions_df = load_predictions_from_summary(summary)

    missing_models = [name for name in UPSET_MODEL_LABELS if name not in set(predictions_df[MODEL_COLUMN].astype(str).unique())]
    if missing_models:
        raise ValueError(
            "Predictions file does not contain all required UpSet models (raw_roberta/E/S/R): "
            f"{missing_models}"
        )

    model_names = list(UPSET_MODEL_LABELS.keys())
    model_to_label = {name: UPSET_MODEL_LABELS[name] for name in model_names}

    for k in [0.05, 0.5]:
        payload_by_language: Dict[str, Dict[str, Any]] = {}
        for language_code, _language_name in LANGUAGE_SPECS:
            subset_mask = (predictions_df[L1_COLUMN] == language_code) & (
                predictions_df[MODEL_COLUMN].isin(model_names)
            )
            subset_df = pd.DataFrame(predictions_df.loc[subset_mask]).copy()
            if subset_df.empty:
                continue
            payload_by_language[language_code] = analyze_upset_language(subset_df, model_to_label, k)

        fig = make_upset_figure(payload_by_language, k)
        figure_path = output_dir / f"lightgbm_final_upset_k{_k_token(k)}.png"
        json_path = output_dir / f"lightgbm_final_upset_k{_k_token(k)}.json"

        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {figure_path}")

        summary_payload = {
            "k": k,
            "predictions_file": summary.get("predictions_file"),
            "models": model_names,
            "languages": payload_by_language,
        }
        json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"Saved UpSet summary: {json_path}")

        if not no_show:
            plt.show()
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.metrics_json.exists():
        raise FileNotFoundError(
            f"Metrics JSON not found: {args.metrics_json}. Run evaluate_lightgbm_final.py first."
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
        output_path=args.output_dir / "lightgbm_final_sliding_relatedness_individual_window_sizes.png",
        range_by_language=range_by_language,
        no_show=args.no_show,
    )

    plot_upset_thresholds(summary=summary, output_dir=args.output_dir, no_show=args.no_show)


if __name__ == "__main__":
    main()

