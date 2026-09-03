#!/usr/bin/env python3
"""Print human-readable console summaries for the L1-inclusive LightGBM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from evaluate_lightgbm import print_console_report
from evaluate_lightgbm_extra import (
    L1_COL,
    MODEL_COL,
    OUTER_TEST_SPLIT,
    WINDOW_CENTERS,
    WINDOW_LOWS,
    WINDOW_WIDTH,
    print_overall_report,
    print_window_report,
    select_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "lightgbm_l1_experiment"
DEFAULT_STANDARD_SUMMARY = DEFAULT_OUTPUT_ROOT / "lightgbm_models_l1" / "lightgbm_metrics_summary_l1.json"
DEFAULT_EXTRA_SUMMARY = DEFAULT_OUTPUT_ROOT / "lightgbm_extra_models_l1" / "lightgbm_extra_window_metrics_l1.json"

LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print console summaries for the LightGBM L1 experiment from saved metrics JSON files."
    )
    parser.add_argument("--standard-summary", type=Path, default=DEFAULT_STANDARD_SUMMARY)
    parser.add_argument("--extra-summary", type=Path, default=DEFAULT_EXTRA_SUMMARY)
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
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
        print_overall_report(predictions_df, model_specs)
    else:
        print("\n[warning] Skipping extra overall RMSE report because predictions CSV is unavailable.")


def print_standard_summary(standard_summary: Dict[str, Any]) -> None:
    print_console_report(standard_summary)


def main() -> None:
    args = parse_args()

    if args.skip_standard and args.skip_extra:
        raise ValueError("Nothing to print: both --skip-standard and --skip-extra were provided.")

    if not args.skip_standard:
        standard_summary = load_json(args.standard_summary)
        print_standard_summary(standard_summary)

    if not args.skip_extra:
        extra_summary = load_json(args.extra_summary)
        print_extra_summary(extra_summary)


if __name__ == "__main__":
    main()

