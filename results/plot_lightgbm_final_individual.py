#!/usr/bin/env python3
"""Plot individual-feature additions and leave-one-out ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_JSON = PROJECT_ROOT / "results" / "lightgbm_final_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_CENTER_RANGES = {"de": "0.1-0.9", "es": "0.1-0.9", "cn": "0.4-0.9"}
LANGUAGE_SPECS = [("de", "German"), ("es", "Spanish"), ("cn", "Mandarin")]
WINDOW_CENTERS = np.round(np.arange(0.0, 0.9, 0.1) + 0.1, 1)
FEATURE_GROUPS = {
    "established": ["zipf_freq", "target_word_length", "num_syllables", "pos_tag"],
    "similarity": [
        "char_edit_dist", "char_edit_dist_norm", "char_edit_dist_geo_norm",
        "phon_edit_dist", "phon_edit_dist_norm", "phon_edit_dist_geo_norm",
        "jaro_distance", "jaro_winkler_distance", "soundex_dist",
        "metaphone_dist_norm", "char_cosine_sim",
    ],
    "relatedness": [
        "lca_language", "borrowed_from_source", "borrowed_from_target",
        "borrowed_from_third", "english_word_origin", "origin_matches_source_family",
        "same_proto_word", "same_proto_root", "transmission_path",
    ],
}
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


def parse_center_range(raw: str, label: str) -> tuple[float, float]:
    parts = raw.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} center range '{raw}'. Use low-high, e.g. 0.3-0.7")
    low, high = map(float, parts)
    if low > high:
        raise ValueError(f"Invalid {label} center range '{raw}': low must be <= high")
    return low, high


def center_mask(range_low: float, range_high: float) -> np.ndarray:
    return (WINDOW_CENTERS >= range_low) & (WINDOW_CENTERS <= range_high)


def model_windows(
    summary: dict[str, Any], model_name: str, language_code: str
) -> tuple[np.ndarray, np.ndarray] | None:
    model_payload = summary.get("models", {}).get(model_name)
    if not isinstance(model_payload, dict):
        return None
    language_payload = model_payload.get("outer_test", {}).get("per_language", {}).get(language_code)
    if not isinstance(language_payload, dict):
        return None
    window_metrics = language_payload.get("windows", [])
    if not isinstance(window_metrics, list):
        return None
    rmses = np.full(len(WINDOW_CENTERS), np.nan, dtype=float)
    counts = np.zeros(len(WINDOW_CENTERS), dtype=int)
    for idx, metric in enumerate(window_metrics[:len(WINDOW_CENTERS)]):
        value = metric.get("rmse")
        rmses[idx] = np.nan if value is None else float(value)
        counts[idx] = int(metric.get("n_rows", 0))
    return rmses, counts


def add_window_count_axis(ax: plt.Axes, ticks: np.ndarray, counts: np.ndarray) -> None:
    ax2 = ax.twinx()
    ax2.bar(ticks, counts, width=0.075, color="gray", alpha=0.1, edgecolor="none", zorder=0)
    ax2.set_ylim(0, max(1.0, float(np.max(counts)) * 1.1))
    ax2.grid(False)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    ax2.set_ylabel("Window set size", fontsize=12, color="dimgray")
    ax2.tick_params(axis="y", colors="dimgray", labelcolor="dimgray")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final LightGBM individual-feature additions and deletions."
    )
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--de-centers", default=DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", default=DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", default=DEFAULT_CENTER_RANGES["cn"])
    parser.add_argument("--no-show", action="store_true", help="Save figures without displaying them.")
    return parser.parse_args()


def plot_feature_group(
    summary: dict[str, Any],
    group: str,
    features: list[str],
    mode: str,
    output_path: Path,
    ranges: dict[str, tuple[float, float]],
    no_show: bool,
) -> None:
    if mode == "addition":
        prefix, reference, reference_label, sign = "BL1_plus_", "BL1", r"B$^{\mathrm{L1}}$", "+"
    elif mode == "deletion":
        prefix, reference, reference_label, sign = "BL1_all_minus_", "BESR", r"B$^{\mathrm{L1}}$CSE", "-"
    else:
        raise ValueError(f"Unknown individual-feature analysis: {mode}")

    models = summary.get("models", {})
    if reference not in models:
        raise ValueError(f"Required {mode} reference model '{reference}' is missing from the metrics.")
    present = [feature for feature in features if f"{prefix}{feature}" in models]
    missing = [feature for feature in features if feature not in present]
    if missing:
        print(f"[warning] Missing {mode} models for {group}: {', '.join(missing)}")
    if not present:
        print(f"[warning] Skipping {group} {mode} plot: no individual-feature models found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), sharex=False)
    palette = plt.get_cmap("tab20")
    try:
        for ax, (language_code, language_name) in zip(axes, LANGUAGE_SPECS):
            low, high = ranges[language_code]
            mask = center_mask(low, high)
            ticks = WINDOW_CENTERS[mask]
            if len(ticks) == 0:
                raise ValueError(f"No centers selected for {language_code}: {low}-{high}")

            reference_arrays = model_windows(summary, reference, language_code)
            if reference_arrays is None:
                raise ValueError(f"Missing {reference} metrics for {language_name}.")
            reference_rmses, counts = reference_arrays

            if mode == "addition":
                raw_arrays = model_windows(summary, "raw_roberta", language_code)
                if raw_arrays is not None:
                    ax.plot(ticks, raw_arrays[0][mask], color="black", label="B",
                            marker="o", linewidth=3.0, zorder=3)

            ax.plot(ticks, reference_rmses[mask], color="gray" if mode == "addition" else "black",
                    label=reference_label, marker="o", linewidth=3.0, zorder=3)

            for idx, feature in enumerate(present):
                model_name = f"{prefix}{feature}"
                arrays = model_windows(summary, model_name, language_code)
                if arrays is None:
                    print(f"[warning] Missing {model_name} metrics for {language_name}.")
                    continue
                color = FEATURE_COLOR_MAP.get(feature, palette(idx % 20))
                ax.plot(
                    ticks, arrays[0][mask], color=color, label=f"{sign} {feature}",
                    linestyle=LINESTYLES[idx % len(LINESTYLES)],
                    marker=MARKERS[idx % len(MARKERS)],
                    linewidth=4.5, alpha=0.60, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                )

            add_window_count_axis(ax, ticks, counts[mask])
            ax.set_title(language_name, fontsize=12)
            ax.set_ylabel("RMSE", fontsize=12)
            ax.invert_yaxis()
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{center:.1f}" for center in ticks])
            ax.set_xlim(float(ticks.min()) - 0.02, float(ticks.max()) + 0.02)
            if language_code == "cn":
                ax.legend(
                    fontsize=10, title="Configuration", title_fontsize=12,
                    loc="lower right",
                )

        axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=14)
        fig.suptitle(
            f"{group.capitalize()} Individual Feature {mode.capitalize()}s by Levenshtein Window",
            fontsize=13, y=0.99, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.995))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        if not no_show:
            plt.show()
    finally:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.metrics_json.exists():
        raise FileNotFoundError(
            f"Metrics JSON not found: {args.metrics_json}. Run evaluate_lightgbm_final.py first."
        )
    summary = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    ranges = {
        "de": parse_center_range(args.de_centers, "German"),
        "es": parse_center_range(args.es_centers, "Spanish"),
        "cn": parse_center_range(args.cn_centers, "Mandarin"),
    }
    for mode in ("addition", "deletion"):
        for group, features in FEATURE_GROUPS.items():
            filename = f"lightgbm_final_sliding_{group}_individual_{mode}_window_sizes.png"
            plot_feature_group(
                summary, group, features, mode, args.output_dir / filename, ranges, args.no_show
            )


if __name__ == "__main__":
    main()
