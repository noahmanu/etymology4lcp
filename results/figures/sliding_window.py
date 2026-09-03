import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREDICTIONS_FILE = Path(__file__).resolve().parents[2] / "models" / "lightgbm_models" / "lightgbm_all_predictions.csv"
FOLDS_DIR = Path(__file__).resolve().parents[2] / "data" / "feature_enriched" / "folds"
ROBERTA_PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "models" / "roberta_predictions"

# Window centers for 0.2-wide overlapping windows: [0.0-0.2, ..., 0.8-1.0]
WINDOW_CENTERS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
WINDOW_LOWS = np.arange(0.0, 0.9, 0.1)
WINDOW_WIDTH = 0.2

SPLIT_COL = "split"
MODEL_COL = "model_name"
L1_COL = "L1"
LABEL_COL = "GLMM_score"
PRED_COL = "prediction"
LEV_COL = "char_edit_dist_norm"
OUTER_TEST_SPLIT = "outer_test"
OUTER_FOLDS = [1, 2, 3, 4, 5]
ROW_KEY_COLUMNS = ["item_id", "L1"]
ROBERTA_SCORE_COLUMN = "roberta_base_score"

DEFAULT_CENTER_RANGES = {
    "German": "0.1-0.9",
    "Spanish": "0.1-0.9",
    "Mandarin": "0.4-0.9",
}

LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]

# "B" is raw RoBERTa predictions (no additional LightGBM training).
CONFIG_MODEL_MAP = {
    "B": "ablation__roberta_only",
    "BE": "group__established",
    "BS": "group__similarity",
    "BR": "group__relatedness",
    "BES": "group__established_similarity",
    "BSR": "group__relatedness_similarity",
    "BER": "group__established_relatedness",
    "BESR": "group__established_relatedness_similarity",
}
BASELINE_CONFIG = "B"

# Colors for each configuration
colors = {
    "B":    "black",
    "BE":   "dodgerblue",
    "BS":   "gold",
    "BR":   "red",
    # Mixed tones for combined configurations
    "BES":  "limegreen",
    "BSR":  "darkorange",
    "BER":  "darkviolet",
    "BESR": "brown",
}

STYLE_BY_CONFIG = {
    "B": {"linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 1.0},
    "BE": {"linestyle": "--", "marker": "s", "linewidth": 4.5, "alpha": 0.60},
    "BS": {"linestyle": "-.", "marker": "^", "linewidth": 4.5, "alpha": 0.60},
    "BR": {"linestyle": ":", "marker": "D", "linewidth": 4.5, "alpha": 0.60},
    "BES": {"linestyle": "--", "marker": "v", "linewidth": 4.5, "alpha": 0.60},
    "BSR": {"linestyle": "-.", "marker": "P", "linewidth": 4.5, "alpha": 0.60},
    "BER": {"linestyle": ":", "marker": "X", "linewidth": 4.5, "alpha": 0.60},
    "BESR": {"linestyle": "-", "marker": "*", "linewidth": 4.5, "alpha": 0.60},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sliding-window RMSE curves by language from model predictions.")
    parser.add_argument("--predictions-file", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--folds-dir", type=Path, default=FOLDS_DIR)
    parser.add_argument("--roberta-predictions-dir", type=Path, default=ROBERTA_PREDICTIONS_DIR)
    parser.add_argument("--de-centers", type=str, default=DEFAULT_CENTER_RANGES["German"])
    parser.add_argument("--es-centers", type=str, default=DEFAULT_CENTER_RANGES["Spanish"])
    parser.add_argument("--cn-centers", type=str, default=DEFAULT_CENTER_RANGES["Mandarin"])
    parser.add_argument("--output-file", type=Path, default=Path("levenshtein_window_analysis.png"))
    parser.add_argument("--min-window-rows", type=int, default=50)
    parser.add_argument("--no-show", action="store_true", help="Save figure without calling plt.show().")
    return parser.parse_args()


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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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


def print_window_report(language_name: str, config_label: str, rmses: np.ndarray, counts: np.ndarray) -> None:
    print(f"\n{language_name} | {config_label}")
    for low, center, value, count in zip(WINDOW_LOWS, WINDOW_CENTERS, rmses, counts):
        low_f = float(low)
        center_f = float(center)
        high = low_f + WINDOW_WIDTH
        value_text = "nan" if np.isnan(value) else f"{value:.4f}"
        print(f"  [{low_f:.1f}, {high:.1f}] center={center_f:.1f} -> rmse={value_text}, n={int(count)}")


def build_roberta_baseline_rows(folds_dir: Path, roberta_predictions_dir: Path) -> pd.DataFrame:
    baseline_parts = []

    for outer_fold in OUTER_FOLDS:
        outer_dir = folds_dir / f"outer_fold_{outer_fold}"
        outer_test_path = outer_dir / "outer_test.csv"
        roberta_test_path = roberta_predictions_dir / f"outer_fold_{outer_fold}" / "outer_test_predictions.csv"

        if not outer_test_path.exists() or not roberta_test_path.exists():
            continue

        outer_test_df = pd.read_csv(outer_test_path)
        roberta_df = pd.read_csv(roberta_test_path)

        required_roberta_cols = set(ROW_KEY_COLUMNS + [ROBERTA_SCORE_COLUMN])
        missing_roberta_cols = required_roberta_cols - set(roberta_df.columns)
        if missing_roberta_cols:
            continue

        merged = outer_test_df.merge(
            roberta_df[ROW_KEY_COLUMNS + [ROBERTA_SCORE_COLUMN]],
            on=ROW_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

        if merged[ROBERTA_SCORE_COLUMN].isna().any():
            continue

        required_outer_cols = {L1_COL, LABEL_COL, LEV_COL}
        missing_outer_cols = required_outer_cols - set(merged.columns)
        if missing_outer_cols:
            continue

        baseline_parts.append(
            pd.DataFrame(
                {
                    MODEL_COL: CONFIG_MODEL_MAP[BASELINE_CONFIG],
                    SPLIT_COL: OUTER_TEST_SPLIT,
                    L1_COL: merged[L1_COL].values,
                    LABEL_COL: merged[LABEL_COL].values,
                    PRED_COL: merged[ROBERTA_SCORE_COLUMN].values,
                    LEV_COL: merged[LEV_COL].values,
                }
            )
        )

    if not baseline_parts:
        return pd.DataFrame(columns=[MODEL_COL, SPLIT_COL, L1_COL, LABEL_COL, PRED_COL, LEV_COL])

    return pd.concat(baseline_parts, ignore_index=True)


def main() -> None:
    args = parse_args()

    if not args.predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")

    df = pd.read_csv(args.predictions_file, keep_default_na=False)
    required_cols = {SPLIT_COL, MODEL_COL, L1_COL, LABEL_COL, PRED_COL, LEV_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing_cols)}")

    df = df[df[SPLIT_COL] == OUTER_TEST_SPLIT].copy()
    available_models = set(df[MODEL_COL].astype(str).unique())

    baseline_model_name = CONFIG_MODEL_MAP[BASELINE_CONFIG]
    if baseline_model_name not in available_models:
        baseline_df = build_roberta_baseline_rows(args.folds_dir, args.roberta_predictions_dir)
        if baseline_df.empty:
            raise ValueError(
                "Missing raw-RoBERTa baseline predictions in the CSV and fallback reconstruction failed. "
                f"Expected model_name='{baseline_model_name}' in {args.predictions_file}, or readable fold/pred files under "
                f"{args.folds_dir} and {args.roberta_predictions_dir}."
            )
        df = pd.concat([df, baseline_df], ignore_index=True)
        available_models = set(df[MODEL_COL].astype(str).unique())
        print("Baseline model rows were reconstructed from RoBERTa predictions.")

    config_specs = [(BASELINE_CONFIG, baseline_model_name)]
    config_specs.extend(
        (config_label, model_name)
        for config_label, model_name in CONFIG_MODEL_MAP.items()
        if config_label != BASELINE_CONFIG and model_name in available_models
    )
    if not config_specs:
        raise ValueError(
            "No expected models found in predictions file. "
            f"Expected one of: {sorted(CONFIG_MODEL_MAP.values())}"
        )

    range_by_language = {
        "German": parse_center_range(args.de_centers, "German"),
        "Spanish": parse_center_range(args.es_centers, "Spanish"),
        "Mandarin": parse_center_range(args.cn_centers, "Mandarin"),
    }

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)

    for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
        lang_df = df[df[L1_COL] == language_code]
        if lang_df.empty:
            ax.set_visible(False)
            print(f"\n{language_name} ({language_code}): no outer_test rows found.")
            continue

        for config_label, model_name in config_specs:
            series_df = lang_df[lang_df[MODEL_COL] == model_name]
            if series_df.empty:
                continue

            window_rmse, window_counts = compute_window_rmse(series_df, args.min_window_rows)
            print_window_report(language_name, config_label, window_rmse, window_counts)

            range_low, range_high = range_by_language[language_name]
            mask = center_mask(range_low, range_high)
            ticks = WINDOW_CENTERS[mask]
            y_values = window_rmse[mask]
            style = STYLE_BY_CONFIG.get(config_label, {"linestyle": "-", "marker": "o", "linewidth": 2.2, "alpha": 0.95})

            ax.plot(
                ticks,
                y_values,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=config_label,
                color=colors[config_label],
            )

        range_low, range_high = range_by_language[language_name]
        mask = center_mask(range_low, range_high)
        ticks = WINDOW_CENTERS[mask]
        if len(ticks) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_name}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )

        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{c:.1f}" for c in ticks])
        ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)

        # Only show legend in Mandarin subplot
        if language_name == "Mandarin":
            ax.legend(
                fontsize=12,
                title="Configuration",
                title_fontsize=14,
                loc="lower right",
            )

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
    fig.suptitle("LCP Performance by Levenshtein Similarity Window", fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.995))

    #plt.tight_layout()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {args.output_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
