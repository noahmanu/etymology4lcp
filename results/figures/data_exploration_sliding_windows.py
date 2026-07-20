"""
Data exploration script for analyzing Levenshtein-based feature distributions by language.
Creates sliding window distributions and bar charts for each L1 language subset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the data
data_path = Path(__file__).parent.parent.parent / "data" / "pooled" / "pooled_preprocessed.csv"
df = pd.read_csv(data_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"L1 values: {df['L1'].unique()}")
print(
    f"levensht_geo_norm range: "
    f"[{df['levensht_geo_norm'].min():.2f}, {df['levensht_geo_norm'].max():.2f}]"
)
print(f"levensht_norm range: [{df['levensht_norm'].min():.2f}, {df['levensht_norm'].max():.2f}]")

# Create subsets by L1 language
languages = sorted(df['L1'].unique())
language_subsets = {lang: df[df['L1'] == lang] for lang in languages}

print(f"\nLanguage subsets:")
for lang, subset in language_subsets.items():
    print(f"  {lang}: {len(subset)} items")

# Shared sliding-window settings
window_width = 0.2
window_step = 0.1
start_min = 0.0


def get_window_config(end_max):
    """Build sliding window centers/ranges for a given feature max value."""
    window_centers = np.arange(start_min + window_width / 2, end_max, window_step)
    window_ranges = [
        (center - window_width / 2, center + window_width / 2) for center in window_centers
    ]
    return window_centers, window_ranges

# Compute counts for each window and language
def compute_window_counts(subset, feature_col, window_ranges):
    """Count items falling into each window for the selected feature."""
    counts = []
    for window_min, window_max in window_ranges:
        count = len(
            subset[
                (subset[feature_col] >= window_min)
                & (subset[feature_col] < window_max)
            ]
        )
        counts.append(count)
    return counts

def plot_feature_distributions(feature_col, output_filename, title_label, end_max):
    """Create the language-wise sliding window plot for one Levenshtein feature."""
    window_centers, window_ranges = get_window_config(end_max)

    print(f"\nWindow configuration ({feature_col}):")
    print(f"  Window width: {window_width}")
    print(f"  Window step: {window_step}")
    print(f"  Range: [{start_min:.1f}, {end_max:.1f}]")
    print(f"  Number of windows: {len(window_centers)}")
    print(f"  Window centers: {window_centers}")

    fig, axes = plt.subplots(1, len(languages), figsize=(15, 5))
    if len(languages) == 1:
        axes = [axes]

    for idx, (lang, subset) in enumerate(sorted(language_subsets.items())):
        counts = compute_window_counts(subset, feature_col, window_ranges)

        ax = axes[idx]
        ax.bar(window_centers, counts, width=window_step * 0.8, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"{title_label} (window center)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Language: {lang} (n={len(subset)})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.05, end_max + 0.05)

        print(f"\n{lang} distribution ({feature_col}):")
        print(f"  Total items: {len(subset)}")
        print(f"  {feature_col} mean: {subset[feature_col].mean():.4f}")
        print(f"  {feature_col} std: {subset[feature_col].std():.4f}")
        print(f"  Window counts: {counts}")

    plt.tight_layout()
    output_path = Path(__file__).parent / "results" / "figures" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Figure saved to: {output_path}")
    plt.show()


plot_feature_distributions(
    "levensht_geo_norm",
    "levensht_geo_norm_sliding_windows.png",
    "Levensht geo Norm",
    end_max=3.0,
)
plot_feature_distributions(
    "levensht_norm",
    "levensht_norm_sliding_windows.png",
    "Levensht Norm",
    end_max=1.0,
)


