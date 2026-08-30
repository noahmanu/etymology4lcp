from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_LOWS = np.arange(0.0, 0.9, 0.1)
WINDOW_WIDTH = 0.2
WINDOW_CENTERS = WINDOW_LOWS + WINDOW_WIDTH / 2
# Mandarin starts from the 0.3-0.5 window (center 0.4).
MANDARIN_WINDOW_CENTERS = WINDOW_CENTERS[3:]
FEATURE_COL = "char_edit_dist_norm"
TARGET_COL = "GLMM_score"
LANGUAGE_COL = "L1"
LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]


def build_data_path() -> Path:
    """Resolve the pooled preprocessed dataset path from this script location."""
    return Path(__file__).resolve().parents[2] / "data" / "pooled" / "pooled_preprocessed.csv"


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


def main() -> None:
    data_path = build_data_path()
    df = pd.read_csv(data_path)

    required_cols = {FEATURE_COL, TARGET_COL, LANGUAGE_COL}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    available_language_codes = set(df[LANGUAGE_COL].dropna().unique())
    selected_specs = [spec for spec in LANGUAGE_SPECS if spec[0] in available_language_codes]
    if not selected_specs:
        expected = [spec[0] for spec in LANGUAGE_SPECS]
        raise ValueError(f"None of the expected language codes found: {expected}")

    fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

    for ax, (language_code, language_name) in zip(axes, selected_specs):
        means, counts = compute_window_means(df, language_code)

        x_values = MANDARIN_WINDOW_CENTERS if language_name == "Mandarin" else WINDOW_CENTERS
        y_values = means[3:] if language_name == "Mandarin" else means

        ax.plot(x_values, y_values, marker="o", linewidth=2)
        ax.set_title(language_name, fontsize=12)
        ax.set_ylabel("Avg GLMM", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(WINDOW_CENTERS)

        print(f"\n{language_name} ({language_code}):")
        low_iter = WINDOW_LOWS[3:] if language_name == "Mandarin" else WINDOW_LOWS
        count_iter = counts[3:] if language_name == "Mandarin" else counts
        for low, center, mean, count in zip(low_iter, x_values, y_values, count_iter):
            high = low + WINDOW_WIDTH
            mean_text = f"{mean:.4f}" if not np.isnan(mean) else "nan"
            print(f"  [{low:.1f}, {high:.1f}) center={center:.1f} -> avg_glmm={mean_text}, n={int(count)}")

    for ax in axes[len(selected_specs):]:
        ax.set_visible(False)

    axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=11)
    axes[-1].set_xticklabels([f"{c:.1f}" for c in WINDOW_CENTERS])

    fig.suptitle("Average GLMM Score by Levenshtein Similarity Window", fontsize=13)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "glmm_sliding_window_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
