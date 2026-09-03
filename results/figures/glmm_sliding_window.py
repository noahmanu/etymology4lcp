import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_LOWS = np.arange(0.0, 0.9, 0.1)
WINDOW_WIDTH = 0.2
WINDOW_CENTERS = WINDOW_LOWS + WINDOW_WIDTH / 2
FEATURE_COL = "char_edit_dist_norm"
TARGET_COL = "GLMM_score"
LANGUAGE_COL = "L1"
LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]

DEFAULT_CENTER_RANGES = {
    "de": "0.1-0.9",
    "es": "0.1-0.9",
    "cn": "0.4-0.9",
}


def build_data_path() -> Path:
    """Resolve the pooled preprocessed dataset path from this script location."""
    return Path(__file__).resolve().parents[2] / "data" / "pooled" / "pooled_preprocessed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average GLMM by Levenshtein sliding windows.")
    parser.add_argument("--data-file", type=Path, default=build_data_path())
    parser.add_argument("--output-file", type=Path, default=Path(__file__).resolve().parent / "glmm_sliding_window_analysis.png")
    parser.add_argument("--de-centers", type=str, default=DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", type=str, default=DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", type=str, default=DEFAULT_CENTER_RANGES["cn"])
    parser.add_argument("--no-show", action="store_true", help="Save figure without calling plt.show().")
    return parser.parse_args()


def compute_window_means(df: pd.DataFrame, language_code: str) -> tuple[np.ndarray, np.ndarray]:
    """Return average GLMM score and row counts for each Levenshtein sliding window."""
    lang_df = df[df[LANGUAGE_COL] == language_code]
    means = []
    counts = []

    for low in WINDOW_LOWS:
        high = low + WINDOW_WIDTH
        window_df = lang_df[(lang_df[FEATURE_COL] >= low) & (lang_df[FEATURE_COL] <= high)]
        counts.append(len(window_df))
        means.append(window_df[TARGET_COL].mean() if len(window_df) > 0 else np.nan)

    return np.array(means), np.array(counts)


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
    df = pd.read_csv(args.data_file)

    range_by_language = {
        "de": parse_center_range(args.de_centers, "German"),
        "es": parse_center_range(args.es_centers, "Spanish"),
        "cn": parse_center_range(args.cn_centers, "Mandarin"),
    }

    required_cols = {FEATURE_COL, TARGET_COL, LANGUAGE_COL}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    available_language_codes = set(df[LANGUAGE_COL].dropna().unique())
    selected_specs = [spec for spec in LANGUAGE_SPECS if spec[0] in available_language_codes]
    if not selected_specs:
        expected = [spec[0] for spec in LANGUAGE_SPECS]
        raise ValueError(f"None of the expected language codes found: {expected}")

    fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=False)

    for ax, (language_code, language_name) in zip(axes, selected_specs):
        means, counts = compute_window_means(df, language_code)

        range_low, range_high = range_by_language[language_code]
        mask = center_mask(range_low, range_high)
        x_values = WINDOW_CENTERS[mask]
        y_values = means[mask]

        if len(x_values) == 0:
            raise ValueError(
                f"No window centers in selected range for {language_code}: {range_low}-{range_high}. "
                f"Available centers: {WINDOW_CENTERS.tolist()}"
            )

        ax.plot(x_values, y_values, marker="o", linewidth=2)
        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("Avg GLMM", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{c:.1f}" for c in x_values])
        ax.set_xlim(float(x_values.min()) - 0.02, float(x_values.max()) + 0.02)

        print(f"\n{language_name} ({language_code}):")
        low_iter = WINDOW_LOWS[mask]
        count_iter = counts[mask]
        for low, center, mean, count in zip(low_iter, x_values, y_values, count_iter):
            low_f = float(low)
            center_f = float(center)
            high = low_f + WINDOW_WIDTH
            mean_text = f"{mean:.4f}" if not np.isnan(mean) else "nan"
            print(f"  [{low_f:.1f}, {high:.1f}) center={center_f:.1f} -> avg_glmm={mean_text}, n={int(count)}")

    for ax in axes[len(selected_specs):]:
        ax.set_visible(False)

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=11)

    fig.suptitle("Average GLMM Score by Levenshtein Similarity Window", fontsize=13)
    plt.tight_layout()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {args.output_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
