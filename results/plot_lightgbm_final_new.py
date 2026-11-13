#!/usr/bin/env python3
"""Plot the final LightGBM analyses other than individual-feature curves."""

from __future__ import annotations

import json

if __package__:
    from . import plot_lightgbm_final as plots
else:
    import plot_lightgbm_final as plots


def main() -> None:
    args = plots.parse_args()
    if not args.metrics_json.exists():
        raise FileNotFoundError(
            f"Metrics JSON not found: {args.metrics_json}. "
            "Run evaluate_lightgbm_final.py first."
        )

    summary = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    glmm_df = plots.load_glmm_df(args.glmm_data_file)
    ranges = {
        "de": plots.parse_center_range(args.de_centers, "German"),
        "es": plots.parse_center_range(args.es_centers, "Spanish"),
        "cn": plots.parse_center_range(args.cn_centers, "Mandarin"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plots.plot_baseline_group_curves(
        summary, args.output_dir / "lightgbm_final_sliding_baseline_groups.png", ranges, args.no_show
    )
    plots.plot_l1_group_curves_with_glmm(
        summary, glmm_df, args.output_dir / "lightgbm_final_sliding_l1_groups_glmm.png", ranges, args.no_show
    )
    upset_payloads = plots.plot_upset_thresholds(summary, args.output_dir, args.no_show)
    plots.plot_exclusive_correct_enrichment_by_window(
        upset_payloads,
        ranges,
        args.output_dir / "lightgbm_final_exclusive_correct_enrichment_by_window.png",
        args.no_show,
    )


if __name__ == "__main__":
    main()
