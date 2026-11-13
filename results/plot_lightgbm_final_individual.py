#!/usr/bin/env python3
"""Plot individual-feature additions and leave-one-out ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lightgbm_cv_utils import FEATURE_GROUPS

if __package__:
    from . import plot_lightgbm_final as plots
else:
    import plot_lightgbm_final as plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final LightGBM individual-feature additions and deletions."
    )
    parser.add_argument("--metrics-json", type=Path, default=plots.DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=plots.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--de-centers", default=plots.DEFAULT_CENTER_RANGES["de"])
    parser.add_argument("--es-centers", default=plots.DEFAULT_CENTER_RANGES["es"])
    parser.add_argument("--cn-centers", default=plots.DEFAULT_CENTER_RANGES["cn"])
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
        for ax, (language_code, language_name) in zip(axes, plots.LANGUAGE_SPECS):
            low, high = ranges[language_code]
            mask = plots.center_mask(low, high)
            ticks = plots.WINDOW_CENTERS[mask]
            if len(ticks) == 0:
                raise ValueError(f"No centers selected for {language_code}: {low}-{high}")

            reference_arrays = plots.model_windows(summary, reference, language_code)
            if reference_arrays is None:
                raise ValueError(f"Missing {reference} metrics for {language_name}.")
            reference_rmses, counts = reference_arrays

            if mode == "addition":
                raw_arrays = plots.model_windows(summary, "raw_roberta", language_code)
                if raw_arrays is not None:
                    ax.plot(ticks, raw_arrays[0][mask], color="black", label="B",
                            marker="o", linewidth=3.0, zorder=3)

            ax.plot(ticks, reference_rmses[mask], color="gray" if mode == "addition" else "black",
                    label=reference_label, marker="o", linewidth=3.0, zorder=3)

            for idx, feature in enumerate(present):
                model_name = f"{prefix}{feature}"
                arrays = plots.model_windows(summary, model_name, language_code)
                if arrays is None:
                    print(f"[warning] Missing {model_name} metrics for {language_name}.")
                    continue
                color = plots.FEATURE_COLOR_MAP.get(feature, palette(idx % 20))
                ax.plot(
                    ticks, arrays[0][mask], color=color, label=f"{sign} {feature}",
                    linestyle=plots.LINESTYLES[idx % len(plots.LINESTYLES)],
                    marker=plots.MARKERS[idx % len(plots.MARKERS)],
                    linewidth=4.5, alpha=0.60, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                )

            plots.add_window_count_axis(ax, ticks, counts[mask])
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
        "de": plots.parse_center_range(args.de_centers, "German"),
        "es": plots.parse_center_range(args.es_centers, "Spanish"),
        "cn": plots.parse_center_range(args.cn_centers, "Mandarin"),
    }
    for mode in ("addition", "deletion"):
        for group, features in FEATURE_GROUPS.items():
            filename = f"lightgbm_final_sliding_{group}_individual_{mode}_window_sizes.png"
            plot_feature_group(
                summary, group, features, mode, args.output_dir / filename, ranges, args.no_show
            )


if __name__ == "__main__":
    main()
