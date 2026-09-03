"""Explore Levenshtein sliding-window feature distributions by language."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW_WIDTH = 0.2
WINDOW_STEP = 0.1
START_MIN = 0.0

FEATURE_ALIASES = {
    "levensht_geo_norm": ["levensht_geo_norm", "char_edit_dist_geo_norm"],
    "levensht_norm": ["levensht_norm", "char_edit_dist_norm"],
}

DEFAULT_CENTER_RANGES = {
    "de": "0.1-0.9",
    "es": "0.1-0.9",
    "cn": "0.1-0.9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot language-wise sliding-window count distributions.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "pooled" / "pooled_preprocessed.csv",
    )
    parser.add_argument("--de-centers", type=str, default=DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", type=str, default=DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", type=str, default=DEFAULT_CENTER_RANGES["cn"])
    parser.add_argument("--no-show", action="store_true", help="Save figures without calling plt.show().")
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


def resolve_feature_column(df: pd.DataFrame, canonical_name: str) -> str:
    aliases = FEATURE_ALIASES[canonical_name]
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise ValueError(f"Could not resolve feature '{canonical_name}'. Tried aliases: {aliases}")


def get_window_config(end_max: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    window_centers = np.arange(START_MIN + WINDOW_WIDTH / 2, end_max, WINDOW_STEP)
    window_ranges = [(center - WINDOW_WIDTH / 2, center + WINDOW_WIDTH / 2) for center in window_centers]
    return window_centers, window_ranges


def compute_window_counts(subset: pd.DataFrame, feature_col: str, window_ranges: list[tuple[float, float]]) -> np.ndarray:
    counts = []
    for window_min, window_max in window_ranges:
        count = len(subset[(subset[feature_col] >= window_min) & (subset[feature_col] < window_max)])
        counts.append(count)
    return np.array(counts, dtype=int)


def center_mask(centers: np.ndarray, low: float, high: float) -> np.ndarray:
    return (centers >= low) & (centers <= high)


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_col: str,
    output_filename: str,
    title_label: str,
    end_max: float,
    range_by_language: dict[str, tuple[float, float]],
    show_plot: bool,
) -> None:
    window_centers, window_ranges = get_window_config(end_max)

    print(f"\nWindow configuration ({feature_col}):")
    print(f"  Window width: {WINDOW_WIDTH}")
    print(f"  Window step: {WINDOW_STEP}")
    print(f"  Range: [{START_MIN:.1f}, {end_max:.1f}]")
    print(f"  Number of windows: {len(window_centers)}")
    print(f"  Window centers: {window_centers}")

    languages = [str(v) for v in sorted(df["L1"].dropna().unique())]
    language_subsets = {lang: df[df["L1"] == lang] for lang in languages}

    fig, axes = plt.subplots(1, len(languages), figsize=(5 * len(languages), 5))
    if len(languages) == 1:
        axes = [axes]

    for idx, (lang, subset) in enumerate(sorted(language_subsets.items())):
        counts = compute_window_counts(subset, feature_col, window_ranges)
        range_low, range_high = range_by_language.get(lang, (window_centers.min(), window_centers.max()))
        mask = center_mask(window_centers, float(range_low), float(range_high))

        selected_centers = window_centers[mask]
        selected_counts = counts[mask]
        if len(selected_centers) == 0:
            raise ValueError(
                f"No window centers in selected range for {lang}: {range_low}-{range_high}. "
                f"Available centers: {window_centers.tolist()}"
            )

        ax = axes[idx]
        ax.bar(selected_centers, selected_counts, width=WINDOW_STEP * 0.8, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"{title_label} (window center)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Language: {lang} (n={len(subset)})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(selected_centers)
        ax.set_xticklabels([f"{c:.1f}" for c in selected_centers])
        ax.set_xlim(float(selected_centers.min()) - 0.05, float(selected_centers.max()) + 0.05)

        print(f"\n{lang} distribution ({feature_col}):")
        print(f"  Total items: {len(subset)}")
        print(f"  {feature_col} mean: {subset[feature_col].mean():.4f}")
        print(f"  {feature_col} std: {subset[feature_col].std():.4f}")
        print("  Window counts by center:")
        for center, count in zip(selected_centers, selected_counts):
            print(f"    center={float(center):.1f}: n={int(count)}")

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")

    if show_plot:
        plt.show()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_file)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"L1 values: {df['L1'].unique()}")

    required_cols = {"L1"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    geo_col = resolve_feature_column(df, "levensht_geo_norm")
    norm_col = resolve_feature_column(df, "levensht_norm")

    print(
        f"{geo_col} range: "
        f"[{df[geo_col].min():.2f}, {df[geo_col].max():.2f}]"
    )
    print(f"{norm_col} range: [{df[norm_col].min():.2f}, {df[norm_col].max():.2f}]")

    range_by_language = {
        "de": parse_center_range(args.de_centers, "German"),
        "es": parse_center_range(args.es_centers, "Spanish"),
        "cn": parse_center_range(args.cn_centers, "Mandarin"),
    }

    show_plot = not args.no_show
    plot_feature_distributions(
        df,
        geo_col,
        "levensht_geo_norm_sliding_windows.png",
        "Levensht geo Norm",
        end_max=3.0,
        range_by_language=range_by_language,
        show_plot=show_plot,
    )
    plot_feature_distributions(
        df,
        norm_col,
        "levensht_norm_sliding_windows.png",
        "Levensht Norm",
        end_max=1.0,
        range_by_language=range_by_language,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    main()


