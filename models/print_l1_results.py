#!/usr/bin/env python3
"""Print human-readable console summaries for the L1-inclusive LightGBM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_lightgbm import print_console_report
from evaluate_lightgbm_extra import (
    CONFIG_MODEL_LABELS,
    FEATURE_COLOR_MAP,
    L1_COL,
    LINESTYLES,
    MARKERS,
    MODEL_COL,
    OUTER_TEST_SPLIT,
    WINDOW_CENTERS,
    WINDOW_LOWS,
    WINDOW_WIDTH,
    parse_added_feature,
    print_window_report,
    select_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "lightgbm_l1_experiment"
DEFAULT_STANDARD_SUMMARY = DEFAULT_OUTPUT_ROOT / "lightgbm_models_l1" / "lightgbm_metrics_summary_l1.json"
DEFAULT_EXTRA_SUMMARY = DEFAULT_OUTPUT_ROOT / "lightgbm_extra_models_l1" / "lightgbm_extra_window_metrics_l1.json"
DEFAULT_STANDARD_PLOT = PROJECT_ROOT / "results" / "figures" / "levenshtein_window_analysis_l1.png"
DEFAULT_EXTRA_PLOT = PROJECT_ROOT / "results" / "figures" / "levenshtein_window_extra_features_l1.png"
DEFAULT_EXTRA_CONFIG_PLOT = PROJECT_ROOT / "results" / "figures" / "levenshtein_window_extra_configs_l1.png"
DEFAULT_GLMM_DATA_FILE = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"

CONFIG_ONLY_MODEL_LABELS = {
    "extra__roberta_only": "B",
    **CONFIG_MODEL_LABELS,
}

CONFIG_ONLY_PLOT_COLORS = {
    "extra__roberta_only": "black",
    "extra__config__established_only": "dodgerblue",
    "extra__config__similarity_only": "gold",
    "extra__config__relatedness_only": "red",
    "extra__config__established_similarity": "limegreen",
    "extra__config__established_relatedness": "darkviolet",
    "extra__config__similarity_relatedness": "darkorange",
    "extra__config__established_similarity_relatedness": "brown",
}

CONFIG_ONLY_PLOT_STYLES = {
    "extra__roberta_only": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    "extra__config__established_only": {"linestyle": "--", "marker": "s", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__similarity_only": {"linestyle": "-.", "marker": "^", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__relatedness_only": {"linestyle": ":", "marker": "D", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__established_similarity": {"linestyle": "--", "marker": "v", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__established_relatedness": {"linestyle": ":", "marker": "X", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__similarity_relatedness": {"linestyle": "-.", "marker": "P", "linewidth": 4.5, "alpha": 0.60},
    "extra__config__established_similarity_relatedness": {"linestyle": "-", "marker": "*", "linewidth": 4.5, "alpha": 0.60},
}

CONFIG_ONLY_MODEL_NAMES = set(CONFIG_MODEL_LABELS)

GLMM_FEATURE_COL = "char_edit_dist_norm"
GLMM_TARGET_COL = "GLMM_score"

LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]

STANDARD_FOLDS_DIR = PROJECT_ROOT / "data" / "feature_enriched" / "folds"
STANDARD_ROBERTA_PREDICTIONS_DIR = PROJECT_ROOT / "models" / "roberta_predictions"
STANDARD_BASELINE_CONFIG = "B"
STANDARD_CONFIG_MODEL_MAP = {
    "B": "ablation__roberta_only",
    "BC": "group__established",
    "BS": "group__similarity",
    "BE": "group__relatedness",
    "BCS": "group__established_similarity",
    "BCE": "group__established_relatedness",
    "BSE": "group__relatedness_similarity",
    "BCSE": "group__established_relatedness_similarity",
}
STANDARD_PLOT_COLORS = {
    "B": "black",
    "BC": "dodgerblue",
    "BS": "gold",
    "BE": "red",
    "BCS": "limegreen",
    "BSE": "darkorange",
    "BCE": "darkviolet",
    "BCSE": "brown",
}
STANDARD_PLOT_STYLES = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    "BC": {"linestyle": "--", "marker": "s", "linewidth": 4.5, "alpha": 0.60},
    "BS": {"linestyle": "-.", "marker": "^", "linewidth": 4.5, "alpha": 0.60},
    "BE": {"linestyle": ":", "marker": "D", "linewidth": 4.5, "alpha": 0.60},
    "BCS": {"linestyle": "--", "marker": "v", "linewidth": 4.5, "alpha": 0.60},
    "BSE": {"linestyle": "-.", "marker": "P", "linewidth": 4.5, "alpha": 0.60},
    "BCE": {"linestyle": ":", "marker": "X", "linewidth": 4.5, "alpha": 0.60},
    "BCSE": {"linestyle": "-", "marker": "*", "linewidth": 4.5, "alpha": 0.60},
}
DEFAULT_CENTER_RANGES = {
    "de": (0.1, 0.9),
    "es": (0.1, 0.9),
    "cn": (0.4, 0.9),
}
STANDARD_ROW_KEY_COLUMNS = ["item_id", "L1"]
STANDARD_ROBERTA_SCORE_COLUMN = "roberta_base_score"
PREDICTION_COL = "prediction"
WINDOW_BAR_COLOR = "gray"
WINDOW_BAR_ALPHA = 0.1
WINDOW_BAR_WIDTH = 0.075


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print console summaries for the LightGBM L1 experiment from saved metrics JSON files."
    )
    parser.add_argument("--standard-summary", type=Path, default=DEFAULT_STANDARD_SUMMARY)
    parser.add_argument("--extra-summary", type=Path, default=DEFAULT_EXTRA_SUMMARY)
    parser.add_argument("--standard-plot", type=Path, default=DEFAULT_STANDARD_PLOT)
    parser.add_argument("--extra-plot", type=Path, default=DEFAULT_EXTRA_PLOT)
    parser.add_argument("--extra-config-plot", type=Path, default=DEFAULT_EXTRA_CONFIG_PLOT)
    parser.add_argument("--glmm-data-file", type=Path, default=DEFAULT_GLMM_DATA_FILE)
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip printing the standard LightGBM summary JSON.",
    )
    parser.add_argument(
        "--skip-extra",
        action="store_true",
        help="Skip printing the extra LightGBM summary JSON.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip regenerating the standard and extra sliding-window plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create plots without opening matplotlib windows.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_predictions_path(summary: Dict[str, Any], summary_path: Path, label: str) -> Path:
    predictions_file = summary.get("predictions_file")
    if not predictions_file:
        raise ValueError(f"{label} summary does not contain 'predictions_file': {summary_path}")

    predictions_path = Path(str(predictions_file))
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{label} predictions file referenced by summary does not exist: {predictions_path}"
        )
    return predictions_path


def require_glmm_data_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"GLMM data file not found: {path}")
    return path


def extract_extra_model_specs(
    extra_summary: Dict[str, Any],
    predictions_df: pd.DataFrame | None,
) -> List[Tuple[str, str]]:
    baseline_model = str(extra_summary.get("baseline_model", "extra__roberta_only"))

    if predictions_df is not None and not predictions_df.empty:
        return select_models(predictions_df, baseline_model)

    ordered_specs: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for language_code, _language_name in LANGUAGE_SPECS:
        language_payload = extra_summary.get("languages", {}).get(language_code, {})
        for model_name, model_payload in language_payload.get("models", {}).items():
            if model_name in seen:
                continue
            label = str(model_payload.get("label", model_name))
            ordered_specs.append((model_name, label))
            seen.add(model_name)
    return ordered_specs


def load_extra_predictions(extra_summary: Dict[str, Any]) -> pd.DataFrame | None:
    predictions_file = extra_summary.get("predictions_file")
    if not predictions_file:
        return None

    path = Path(str(predictions_file))
    if not path.exists():
        print(f"\n[warning] Extra predictions file referenced in summary does not exist: {path}")
        return None

    df = pd.read_csv(path, keep_default_na=False)
    required_cols = {MODEL_COL, L1_COL, "prediction", "GLMM_score", "split"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            f"\n[warning] Extra predictions file is missing required columns for overall report: {sorted(missing_cols)}"
        )
        return None

    return df[df["split"] == OUTER_TEST_SPLIT].copy()


def filter_addition_only_predictions(extra_predictions_df: pd.DataFrame) -> pd.DataFrame:
    return extra_predictions_df[~extra_predictions_df[MODEL_COL].isin(CONFIG_ONLY_MODEL_NAMES)].copy()


def metric_arrays_from_payload(model_payload: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    window_metrics = model_payload.get("window_metrics", [])
    rmse_values = []
    counts = []

    for metric in window_metrics:
        rmse_value = metric.get("rmse")
        rmse_values.append(np.nan if rmse_value is None else float(rmse_value))
        counts.append(int(metric.get("n_rows", 0)))

    return np.array(rmse_values, dtype=float), np.array(counts, dtype=int)


def validate_window_structure(model_payload: Dict[str, Any], model_name: str, language_code: str) -> None:
    window_metrics = model_payload.get("window_metrics", [])
    if len(window_metrics) != len(WINDOW_CENTERS):
        raise ValueError(
            f"Unexpected number of window metrics for model '{model_name}' in language '{language_code}': "
            f"expected {len(WINDOW_CENTERS)}, found {len(window_metrics)}"
        )

    for expected_low, expected_center, metric in zip(WINDOW_LOWS, WINDOW_CENTERS, window_metrics):
        low = float(metric.get("low"))
        high = float(metric.get("high"))
        center = float(metric.get("center"))
        if not np.isclose(low, float(expected_low)):
            raise ValueError(
                f"Unexpected window low for model '{model_name}' in language '{language_code}': {low} != {expected_low}"
            )
        if not np.isclose(high, float(expected_low + WINDOW_WIDTH)):
            raise ValueError(
                f"Unexpected window high for model '{model_name}' in language '{language_code}': {high} != {expected_low + WINDOW_WIDTH}"
            )
        if not np.isclose(center, float(expected_center)):
            raise ValueError(
                f"Unexpected window center for model '{model_name}' in language '{language_code}': {center} != {expected_center}"
            )


def center_mask(range_low: float, range_high: float) -> np.ndarray:
    return (WINDOW_CENTERS >= range_low) & (WINDOW_CENTERS <= range_high)


def range_from_language_payload(language_payload: Dict[str, Any], language_code: str) -> tuple[float, float]:
    center_range = language_payload.get("center_range")
    if not isinstance(center_range, dict):
        raise ValueError(f"Missing center_range for language '{language_code}' in extra summary")

    return float(center_range["low"]), float(center_range["high"])


def compute_glmm_window_means(glmm_df: pd.DataFrame, language_code: str) -> tuple[np.ndarray, np.ndarray]:
    lang_df = glmm_df[glmm_df[L1_COL] == language_code]
    means = []
    counts = []

    for low in WINDOW_LOWS:
        high = low + WINDOW_WIDTH
        window_df = lang_df[(lang_df[GLMM_FEATURE_COL] >= low) & (lang_df[GLMM_FEATURE_COL] <= high)]
        counts.append(len(window_df))
        means.append(window_df[GLMM_TARGET_COL].mean() if len(window_df) > 0 else np.nan)

    return np.array(means, dtype=float), np.array(counts, dtype=int)


def load_glmm_df(path: Path) -> pd.DataFrame:
    path = require_glmm_data_file(path)
    df = pd.read_csv(path)
    required_cols = {L1_COL, GLMM_FEATURE_COL, GLMM_TARGET_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"GLMM data file is missing required columns: {sorted(missing_cols)}")
    return df


def compute_window_rmse_and_counts(series_df: pd.DataFrame, min_window_rows: int) -> tuple[np.ndarray, np.ndarray]:
    rmses = []
    counts = []

    for low in WINDOW_LOWS:
        high = low + WINDOW_WIDTH
        window_df = series_df[(series_df[GLMM_FEATURE_COL] >= low) & (series_df[GLMM_FEATURE_COL] <= high)]
        count = len(window_df)
        counts.append(count)

        if count < min_window_rows:
            rmses.append(np.nan)
            continue

        gold = window_df[GLMM_TARGET_COL].to_numpy(dtype=float)
        pred = window_df[PREDICTION_COL].to_numpy(dtype=float)
        rmses.append(float(np.sqrt(np.mean((gold - pred) ** 2))))

    return np.array(rmses, dtype=float), np.array(counts, dtype=int)


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


def resolve_language_center_range(
    language_payload: Dict[str, Any] | None,
    language_code: str,
) -> tuple[float, float]:
    if isinstance(language_payload, dict) and isinstance(language_payload.get("center_range"), dict):
        return range_from_language_payload(language_payload, language_code)
    return DEFAULT_CENTER_RANGES[language_code]


def build_standard_roberta_baseline_rows(
    folds_dir: Path = STANDARD_FOLDS_DIR,
    roberta_predictions_dir: Path = STANDARD_ROBERTA_PREDICTIONS_DIR,
) -> pd.DataFrame:
    baseline_parts = []

    for outer_fold in [1, 2, 3, 4, 5]:
        outer_dir = folds_dir / f"outer_fold_{outer_fold}"
        outer_test_path = outer_dir / "outer_test.csv"
        roberta_test_path = roberta_predictions_dir / f"outer_fold_{outer_fold}" / "outer_test_predictions.csv"

        if not outer_test_path.exists() or not roberta_test_path.exists():
            continue

        outer_test_df = pd.read_csv(outer_test_path)
        roberta_df = pd.read_csv(roberta_test_path)

        required_roberta_cols = set(STANDARD_ROW_KEY_COLUMNS + [STANDARD_ROBERTA_SCORE_COLUMN])
        if required_roberta_cols - set(roberta_df.columns):
            continue

        merged = outer_test_df.merge(
            roberta_df[STANDARD_ROW_KEY_COLUMNS + [STANDARD_ROBERTA_SCORE_COLUMN]],
            on=STANDARD_ROW_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

        if merged[STANDARD_ROBERTA_SCORE_COLUMN].isna().any():
            continue

        required_outer_cols = {L1_COL, GLMM_TARGET_COL, GLMM_FEATURE_COL}
        if required_outer_cols - set(merged.columns):
            continue

        baseline_parts.append(
            pd.DataFrame(
                {
                    MODEL_COL: STANDARD_CONFIG_MODEL_MAP[STANDARD_BASELINE_CONFIG],
                    "split": OUTER_TEST_SPLIT,
                    L1_COL: merged[L1_COL].values,
                    GLMM_TARGET_COL: merged[GLMM_TARGET_COL].values,
                    PREDICTION_COL: merged[STANDARD_ROBERTA_SCORE_COLUMN].values,
                    GLMM_FEATURE_COL: merged[GLMM_FEATURE_COL].values,
                }
            )
        )

    if not baseline_parts:
        return pd.DataFrame(columns=[MODEL_COL, "split", L1_COL, GLMM_TARGET_COL, PREDICTION_COL, GLMM_FEATURE_COL])

    return pd.concat(baseline_parts, ignore_index=True)


def build_extra_plot_styles(
    model_specs: List[Tuple[str, str]],
    baseline_model: str,
) -> tuple[dict[str, str | tuple[float, float, float, float]], dict[str, Dict[str, str | float]]]:
    color_cycle = plt.get_cmap("tab20", len(model_specs))
    colors: dict[str, str | tuple[float, float, float, float]] = {}
    styles: dict[str, Dict[str, str | float]] = {}
    style_idx = 0

    for idx, (model_name, _label) in enumerate(model_specs):
        if model_name == baseline_model:
            colors[model_name] = (0.0, 0.0, 0.0, 1.0)
            styles[model_name] = {
                "linestyle": "-",
                "marker": "o",
                "linewidth": 3.0,
                "alpha": 1.0,
            }
            continue

        added_feature = parse_added_feature(model_name)
        if added_feature is None:
            colors[model_name] = color_cycle(idx)
        else:
            colors[model_name] = FEATURE_COLOR_MAP.get(added_feature, color_cycle(idx))
        styles[model_name] = {
            "linestyle": LINESTYLES[style_idx % len(LINESTYLES)],
            "marker": MARKERS[style_idx % len(MARKERS)],
            "linewidth": 4.5,
            "alpha": 0.60,
        }
        style_idx += 1

    return colors, styles


def print_extra_summary(extra_summary: Dict[str, Any]) -> None:
    print("\n=== LightGBM extra L1 window summary ===")

    predictions_df = load_extra_predictions(extra_summary)
    model_specs = extract_extra_model_specs(extra_summary, predictions_df)
    if not model_specs:
        print("No extra models found in the summary JSON.")
        return

    languages_payload = extra_summary.get("languages", {})
    for language_code, language_name in LANGUAGE_SPECS:
        language_payload = languages_payload.get(language_code)
        if language_payload is None:
            print(f"\n{language_name} ({language_code}): no summary found.")
            continue

        if not language_payload.get("models"):
            print(f"\n{language_name} ({language_code}): no models found.")
            continue

        for model_name, fallback_label in model_specs:
            model_payload = language_payload.get("models", {}).get(model_name)
            if model_payload is None:
                continue

            validate_window_structure(model_payload, model_name, language_code)
            label = str(model_payload.get("label", fallback_label))
            rmses, counts = metric_arrays_from_payload(model_payload)
            print_window_report(language_name, label, rmses, counts)

    if predictions_df is not None and not predictions_df.empty:
        print_overall_report_with_pearson(predictions_df, model_specs)
    else:
        print("\n[warning] Skipping extra overall RMSE report because predictions CSV is unavailable.")


def print_overall_report_with_pearson(df: pd.DataFrame, model_specs: List[Tuple[str, str]]) -> None:
    languages = sorted(df[L1_COL].astype(str).unique())

    header_cols: List[str] = ["model_label", "overall_rmse", "overall_pearson"]
    for lang in languages:
        header_cols.extend([f"{lang}_rmse", f"{lang}_pearson"])

    header = "  ".join(f"{c:<30}" if i == 0 else f"{c:>12}" for i, c in enumerate(header_cols))
    separator = "-" * len(header)

    print("\n=== Outer-test RMSE / Pearson by model and L1 (no Levenshtein filtering) ===")
    print(header)
    print(separator)

    for model_name, label in model_specs:
        series_df = df[df[MODEL_COL] == model_name]
        if series_df.empty:
            continue

        overall_rmse = float(np.sqrt(np.mean(
            (series_df[GLMM_TARGET_COL].to_numpy(dtype=float) - series_df["prediction"].to_numpy(dtype=float)) ** 2
        )))
        overall_pearson = float(series_df[GLMM_TARGET_COL].corr(series_df["prediction"]))

        row_values: List[str] = [f"{overall_rmse:.4f}", f"{overall_pearson:.4f}"]
        for lang in languages:
            lang_df = series_df[series_df[L1_COL] == lang]
            if lang_df.empty:
                row_values.extend(["n/a", "n/a"])
                continue

            lang_rmse = float(np.sqrt(np.mean(
                (lang_df[GLMM_TARGET_COL].to_numpy(dtype=float) - lang_df["prediction"].to_numpy(dtype=float)) ** 2
            )))
            lang_pearson = float(lang_df[GLMM_TARGET_COL].corr(lang_df["prediction"]))
            row_values.extend([f"{lang_rmse:.4f}", f"{lang_pearson:.4f}"])

        row = f"  {label:<30}" + "".join(f"{v:>12}" for v in row_values)
        print(row)

    print(separator)


def print_standard_summary(standard_summary: Dict[str, Any]) -> None:
    print_console_report(standard_summary)


def create_standard_plot(standard_summary: Dict[str, Any], standard_summary_path: Path, output_path: Path, no_show: bool) -> None:
    predictions_path = require_predictions_path(standard_summary, standard_summary_path, "Standard")
    df = pd.read_csv(predictions_path, keep_default_na=False)
    required_cols = {"split", MODEL_COL, L1_COL, GLMM_TARGET_COL, PREDICTION_COL, GLMM_FEATURE_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns for standard plotting: {sorted(missing_cols)}")

    df = df[df["split"] == OUTER_TEST_SPLIT].copy()
    available_models = set(df[MODEL_COL].astype(str).unique())
    baseline_model_name = STANDARD_CONFIG_MODEL_MAP[STANDARD_BASELINE_CONFIG]
    if baseline_model_name not in available_models:
        baseline_df = build_standard_roberta_baseline_rows()
        if baseline_df.empty:
            raise ValueError(
                "Missing raw-RoBERTa baseline predictions in the standard CSV and fallback reconstruction failed."
            )
        df = pd.concat([df, baseline_df], ignore_index=True)
        available_models = set(df[MODEL_COL].astype(str).unique())

    config_specs = [(STANDARD_BASELINE_CONFIG, baseline_model_name)]
    config_specs.extend(
        (config_label, model_name)
        for config_label, model_name in STANDARD_CONFIG_MODEL_MAP.items()
        if config_label != STANDARD_BASELINE_CONFIG and model_name in available_models
    )
    if not config_specs:
        raise ValueError(f"No expected standard models found. Expected one of: {sorted(STANDARD_CONFIG_MODEL_MAP.values())}")

    min_window_rows = int(standard_summary.get("min_window_rows", 50))
    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    print(f"\nRegenerating standard L1 plot -> {output_path}")
    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        lang_df = df[df[L1_COL] == language_code]
        if lang_df.empty:
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no outer_test rows found for standard plot.")
            continue

        range_low, range_high = DEFAULT_CENTER_RANGES[language_code]
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_name}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )

        reference_counts: np.ndarray | None = None
        for config_label, model_name in config_specs:
            series_df = lang_df[lang_df[MODEL_COL] == model_name]
            if series_df.empty:
                continue

            window_rmse, window_counts = compute_window_rmse_and_counts(series_df, min_window_rows)
            if reference_counts is None:
                reference_counts = window_counts

            style = STANDARD_PLOT_STYLES.get(config_label, {"linestyle": "-", "marker": "o", "linewidth": 2.2, "alpha": 0.95})
            ax.plot(
                ticks,
                window_rmse[mask],
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=config_label,
                color=STANDARD_PLOT_COLORS[config_label],
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
                fontsize=12,
                title="Configuration",
                title_fontsize=14,
                loc="lower right",
            )

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("LCP Performance by Levenshtein Similarity Window", fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved standard plot to: {output_path}")

    if not no_show:
        plt.show()
    plt.close(fig)


def create_extra_plot(extra_summary: Dict[str, Any], extra_summary_path: Path, output_path: Path, no_show: bool) -> None:
    predictions_path = require_predictions_path(extra_summary, extra_summary_path, "Extra")
    full_predictions_df = pd.read_csv(predictions_path, keep_default_na=False)
    required_cols = {"split", MODEL_COL, L1_COL, GLMM_TARGET_COL, PREDICTION_COL, GLMM_FEATURE_COL}
    missing_cols = required_cols - set(full_predictions_df.columns)
    if missing_cols:
        raise ValueError(
            f"Extra predictions file is missing required columns for plotting: {sorted(missing_cols)}"
        )

    filtered_df = filter_addition_only_predictions(full_predictions_df)
    filtered_df = filtered_df[filtered_df["split"] == OUTER_TEST_SPLIT].copy()
    if filtered_df.empty:
        raise ValueError("No extra addition-model rows remain after excluding config-only E/S/R/ES/ER/SR/ESR models.")

    baseline_model = str(extra_summary.get("baseline_model", "extra__roberta_only"))
    model_specs = select_models(filtered_df, baseline_model)
    if not model_specs:
        raise ValueError("No extra baseline/addition models found after filtering config-only rows.")

    colors, styles = build_extra_plot_styles(model_specs, baseline_model)
    min_window_rows = int(extra_summary.get("min_window_rows", 50))
    languages_payload = extra_summary.get("languages", {})

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    print(f"\nRegenerating extra L1 additions plot -> {output_path}")
    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        lang_df = filtered_df[filtered_df[L1_COL] == language_code]
        if lang_df.empty:
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no outer_test rows found for extra additions plot.")
            continue

        range_low, range_high = resolve_language_center_range(languages_payload.get(language_code), language_code)
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_code}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )

        reference_counts: np.ndarray | None = None
        for model_name, label in model_specs:
            series_df = lang_df[lang_df[MODEL_COL] == model_name]
            if series_df.empty:
                continue

            window_rmse, window_counts = compute_window_rmse_and_counts(series_df, min_window_rows)
            if reference_counts is None:
                reference_counts = window_counts

            style = styles[model_name]
            ax.plot(
                ticks,
                window_rmse[mask],
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=label,
                color=colors[model_name],
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
            ax.legend(fontsize=11, title="Model", title_fontsize=13, loc="lower right")

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("Etymology Feature Addition vs RoBERTa Baseline by Levenshtein Window", fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved extra additions plot to: {output_path}")

    if not no_show:
        plt.show()
    plt.close(fig)


def create_extra_config_plot(extra_summary: Dict[str, Any], output_path: Path, glmm_data_file: Path, no_show: bool) -> None:
    languages_payload = extra_summary.get("languages", {})
    glmm_df = load_glmm_df(glmm_data_file)

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        language_payload = languages_payload.get(language_code)
        if not isinstance(language_payload, dict):
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no config summary found for separate config plot.")
            continue

        model_payloads = language_payload.get("models", {})
        if not isinstance(model_payloads, dict):
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no config models found for separate config plot.")
            continue

        range_low, range_high = range_from_language_payload(language_payload, language_code)
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_code}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )

        ax.invert_yaxis()

        plotted_any_model = False
        for model_name, label in CONFIG_ONLY_MODEL_LABELS.items():
            model_payload = model_payloads.get(model_name)
            if not isinstance(model_payload, dict):
                continue

            validate_window_structure(model_payload, model_name, language_code)
            rmses, _counts = metric_arrays_from_payload(model_payload)
            style = CONFIG_ONLY_PLOT_STYLES[model_name]

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
                color=CONFIG_ONLY_PLOT_COLORS[model_name],
            )
            plotted_any_model = True

        glmm_means, _glmm_counts = compute_glmm_window_means(glmm_df, language_code)
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

        if not plotted_any_model:
            print(f"\n{language_name} ({language_code}): no baseline/config lines were available for config plot.")

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle(
        "Extra Configuration Models vs GLMM by Levenshtein Window",
        fontsize=13,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved extra config+GLMM plot to: {output_path}")

    if not no_show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.skip_standard and args.skip_extra:
        raise ValueError("Nothing to print: both --skip-standard and --skip-extra were provided.")

    standard_summary: Dict[str, Any] | None = None
    extra_summary: Dict[str, Any] | None = None

    if not args.skip_standard:
        loaded_standard_summary = load_json(args.standard_summary)
        standard_summary = loaded_standard_summary
        print_standard_summary(loaded_standard_summary)

    if not args.skip_extra:
        loaded_extra_summary = load_json(args.extra_summary)
        extra_summary = loaded_extra_summary
        print_extra_summary(loaded_extra_summary)

    if args.skip_plots:
        return

    if standard_summary is not None:
        create_standard_plot(standard_summary, args.standard_summary, args.standard_plot, args.no_show)

    if extra_summary is not None:
        create_extra_plot(extra_summary, args.extra_summary, args.extra_plot, args.no_show)
        create_extra_config_plot(extra_summary, args.extra_config_plot, args.glmm_data_file, args.no_show)


if __name__ == "__main__":
    main()

