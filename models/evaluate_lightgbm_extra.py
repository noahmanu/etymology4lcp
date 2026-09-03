#!/usr/bin/env python3
"""Evaluate LightGBM extra-experiment models with sliding-window RMSE plots."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = Path(__file__).resolve().parent / "lightgbm_extra_models" / "lightgbm_extra_all_predictions.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "results" / "figures" / "levenshtein_window_extra_features.png"
OUTPUT_JSON = Path(__file__).resolve().parent / "lightgbm_extra_models" / "lightgbm_extra_window_metrics.json"

SPLIT_COL = "split"
MODEL_COL = "model_name"
L1_COL = "L1"
LABEL_COL = "GLMM_score"
PRED_COL = "prediction"
LEV_COL = "char_edit_dist_norm"
OUTER_TEST_SPLIT = "outer_test"
BASELINE_MODEL = "extra__roberta_only"
ADDED_PREFIX = "extra__add__"
CONFIG_MODEL_LABELS = {
    "extra__config__established_only": "E",
    "extra__config__similarity_only": "S",
    "extra__config__relatedness_only": "R",
}

RELATEDNESS_FEATURES = {
    "lca_language",
    "borrowed_from_source",
    "borrowed_from_target",
    "borrowed_from_third",
    "english_word_origin",
    "origin_matches_source_family",
    "same_proto_word",
    "same_proto_root",
    "transmission_path",
}

LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]

WINDOW_LOWS = np.arange(0.0, 0.9, 0.1)
WINDOW_WIDTH = 0.2
WINDOW_CENTERS = np.round(WINDOW_LOWS + WINDOW_WIDTH / 2, 1)
MIN_WINDOW_ROWS = 50

DEFAULT_CENTER_RANGES = {
    "de": "0.1-0.9",
    "es": "0.1-0.9",
    "cn": "0.4-0.9",
}

LINESTYLES = ["--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Requested palette: repeated color families use distinct named shades.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and summarize sliding-window RMSE for RoBERTa-only baseline, "
            "E/S/R configuration models, and relatedness-feature addition models."
        )
    )
    parser.add_argument("--predictions-file", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--output-figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--baseline-model", type=str, default=BASELINE_MODEL)
    parser.add_argument("--min-window-rows", type=int, default=MIN_WINDOW_ROWS)
    parser.add_argument(
        "--de-centers",
        type=str,
        default=DEFAULT_CENTER_RANGES["de"],
        help="German window-center range as low-high (example: 0.2-0.7).",
    )
    parser.add_argument(
        "--es-centers",
        type=str,
        default=DEFAULT_CENTER_RANGES["es"],
        help="Spanish window-center range as low-high (example: 0.3-0.7).",
    )
    parser.add_argument(
        "--cn-centers",
        type=str,
        default=DEFAULT_CENTER_RANGES["cn"],
        help="Chinese window-center range as low-high (example: 0.5-0.7).",
    )
    parser.add_argument("--no-show", action="store_true", help="Save figure without calling plt.show().")
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def parse_added_feature(model_name: str) -> str | None:
    if not model_name.startswith(ADDED_PREFIX):
        return None

    added_feature = model_name[len(ADDED_PREFIX):]
    if added_feature not in RELATEDNESS_FEATURES:
        return None

    return added_feature


def select_models(df: pd.DataFrame, baseline_model: str) -> list[tuple[str, str]]:
    model_specs: list[tuple[str, str]] = []
    available_models = set(df[MODEL_COL].astype(str).unique())

    if baseline_model in available_models:
        model_specs.append((baseline_model, "B"))

    for model_name, label in CONFIG_MODEL_LABELS.items():
        if model_name in available_models:
            model_specs.append((model_name, label))

    single_feature_models: list[tuple[str, str]] = []
    for model_name in sorted(available_models):
        added_feature = parse_added_feature(model_name)
        if added_feature is None:
            continue
        single_feature_models.append((model_name, f"+ {added_feature}"))

    return model_specs + single_feature_models


def compute_window_rmse(series_df: pd.DataFrame, min_window_rows: int) -> tuple[np.ndarray, np.ndarray]:
    rmses = []
    counts = []

    for low in WINDOW_LOWS:
        high = low + WINDOW_WIDTH
        window_df = series_df[(series_df[LEV_COL] >= low) & (series_df[LEV_COL] <= high)]
        count = len(window_df)
        counts.append(count)

        if count < min_window_rows:
            rmses.append(np.nan)
            continue

        rmses.append(rmse(window_df[LABEL_COL].to_numpy(dtype=float), window_df[PRED_COL].to_numpy(dtype=float)))

    return np.array(rmses, dtype=float), np.array(counts, dtype=int)


def print_window_report(language_name: str, label: str, rmses: np.ndarray, counts: np.ndarray) -> None:
    print(f"\n{language_name} | {label}")
    for low, center, value, count in zip(WINDOW_LOWS, WINDOW_CENTERS, rmses, counts):
        low_f = float(low)
        center_f = float(center)
        high = low_f + WINDOW_WIDTH
        value_text = "nan" if np.isnan(value) else f"{value:.4f}"
        print(f"  [{low_f:.1f}, {high:.1f}] center={center_f:.1f} -> rmse={value_text}, n={int(count)}")


def parse_center_range(raw: str, label: str) -> tuple[float, float]:
    parts = raw.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} center range '{raw}'. Use low-high, e.g. 0.3-0.7")

    low = float(parts[0])
    high = float(parts[1])
    if low > high:
        raise ValueError(f"Invalid {label} center range '{raw}': low must be <= high")
    return low, high


def center_mask(range_low: float, range_high: float) -> np.ndarray:
    return (WINDOW_CENTERS >= range_low) & (WINDOW_CENTERS <= range_high)


def main() -> None:
    args = parse_args()

    range_by_language = {
        "de": parse_center_range(args.de_centers, "German"),
        "es": parse_center_range(args.es_centers, "Spanish"),
        "cn": parse_center_range(args.cn_centers, "Chinese"),
    }

    if not args.predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")

    df = pd.read_csv(args.predictions_file, keep_default_na=False)

    required_cols = {SPLIT_COL, MODEL_COL, L1_COL, LABEL_COL, PRED_COL, LEV_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing_cols)}")

    df = df[df[SPLIT_COL] == OUTER_TEST_SPLIT].copy()

    model_specs = select_models(df, args.baseline_model)
    if not model_specs:
        raise ValueError(
            "No matching models found. Expected baseline and relatedness-feature extra models in predictions file."
        )

    color_cycle = plt.get_cmap("tab20", len(model_specs))
    colors: dict[str, tuple[float, float, float, float]] = {}
    styles: dict[str, dict[str, object]] = {}
    style_idx = 0
    for idx, (model_name, _) in enumerate(model_specs):
        if model_name == args.baseline_model:
            colors[model_name] = (0.0, 0.0, 0.0, 1.0)
            styles[model_name] = {
                "linestyle": "-",
                "marker": "o",
                "linewidth": 3.0,
                "alpha": 1.0,
            }
            continue

        if model_name in CONFIG_MODEL_LABELS:
            config_color_map = {
                "extra__config__established_only": "dimgray",
                "extra__config__similarity_only": "slategray",
                "extra__config__relatedness_only": "darkslategray",
            }
            config_style_map = {
                "extra__config__established_only": {"linestyle": "-", "marker": "s"},
                "extra__config__similarity_only": {"linestyle": "-", "marker": "^"},
                "extra__config__relatedness_only": {"linestyle": "-", "marker": "D"},
            }
            colors[model_name] = config_color_map[model_name]
            styles[model_name] = {
                "linestyle": config_style_map[model_name]["linestyle"],
                "marker": config_style_map[model_name]["marker"],
                "linewidth": 3.0,
                "alpha": 0.95,
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

    metrics_payload: Dict[str, Any] = {
        "predictions_file": str(args.predictions_file),
        "baseline_model": args.baseline_model,
        "min_window_rows": args.min_window_rows,
        "window_width": WINDOW_WIDTH,
        "window_centers": WINDOW_CENTERS.tolist(),
        "languages": {},
    }

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        lang_df = df[df[L1_COL] == language_code]
        ax.invert_yaxis()

        language_payload: Dict[str, Any] = {
            "language_code": language_code,
            "language_name": language_name,
            "center_range": {
                "low": range_by_language[language_code][0],
                "high": range_by_language[language_code][1],
            },
            "models": {},
        }

        if lang_df.empty:
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no outer_test rows found.")
            metrics_payload["languages"][language_code] = language_payload
            continue

        for model_name, label in model_specs:
            series_df = lang_df[lang_df[MODEL_COL] == model_name]
            if series_df.empty:
                continue

            window_rmse, window_counts = compute_window_rmse(series_df, args.min_window_rows)
            print_window_report(language_name, label, window_rmse, window_counts)

            range_low, range_high = range_by_language[language_code]
            mask = center_mask(range_low, range_high)
            x_values = WINDOW_CENTERS[mask]
            y_values = window_rmse[mask]
            style = styles[model_name]

            ax.plot(
                x_values,
                y_values,
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=label,
                color=colors[model_name],
            )

            language_payload["models"][model_name] = {
                "label": label,
                "window_metrics": [
                    {
                        "low": float(low),
                        "high": float(low + WINDOW_WIDTH),
                        "center": float(center),
                        "rmse": None if np.isnan(rmse_value) else float(rmse_value),
                        "n_rows": int(count),
                    }
                    for low, center, rmse_value, count in zip(WINDOW_LOWS, WINDOW_CENTERS, window_rmse, window_counts)
                ],
            }

        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        range_low, range_high = range_by_language[language_code]
        ticks = WINDOW_CENTERS[center_mask(range_low, range_high)]
        if len(ticks) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_code}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{center:.1f}" for center in ticks])
        ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)

        if language_code == "cn":
            ax.legend(fontsize=11, title="Model", title_fontsize=13, loc="lower right")

        metrics_payload["languages"][language_code] = language_payload

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("Etymology Feature Addition vs RoBERTa Baseline by Levenshtein Window", fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_figure, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {args.output_figure}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved sliding-window metrics: {args.output_json}")

    print_overall_report(df, model_specs)

    if not args.no_show:
        plt.show()


def print_overall_report(df: pd.DataFrame, model_specs: list[tuple[str, str]]) -> None:
    languages = sorted(df[L1_COL].astype(str).unique())

    header_cols = ["model_label"] + ["overall"] + languages
    header = "  ".join(f"{c:<30}" if i == 0 else f"{c:>10}" for i, c in enumerate(header_cols))
    separator = "-" * len(header)

    print("\n=== Outer-test RMSE by model and L1 (no Levenshtein filtering) ===")
    print(header)
    print(separator)

    for model_name, label in model_specs:
        series_df = df[df[MODEL_COL] == model_name]
        if series_df.empty:
            continue

        overall_rmse = float(np.sqrt(np.mean(
            (series_df[LABEL_COL].to_numpy(dtype=float) - series_df[PRED_COL].to_numpy(dtype=float)) ** 2
        )))

        per_l1_rmse: dict[str, str] = {}
        for lang in languages:
            lang_df = series_df[series_df[L1_COL] == lang]
            if lang_df.empty:
                per_l1_rmse[lang] = "n/a"
            else:
                val = float(np.sqrt(np.mean(
                    (lang_df[LABEL_COL].to_numpy(dtype=float) - lang_df[PRED_COL].to_numpy(dtype=float)) ** 2
                )))
                per_l1_rmse[lang] = f"{val:.4f}"

        row_values = [f"{overall_rmse:.4f}"] + [per_l1_rmse[lang] for lang in languages]
        row = f"  {label:<30}" + "".join(f"{v:>10}" for v in row_values)
        print(row)

    print(separator)


if __name__ == "__main__":
    main()
