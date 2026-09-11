#!/usr/bin/env python3
"""Evaluate final LightGBM predictions and cache fold/language/window metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from lightgbm_results_utils import (
        ID_COLUMN,
        LABEL_COLUMN,
        L1_COLUMN,
        LEVENSHTEIN_COLUMN,
        MIN_WINDOW_ROWS,
        OUTER_FOLDS,
        ROW_KEY_COLUMNS,
        WINDOW_CENTERS,
        WINDOW_WIDTH,
        compute_sliding_window_metrics,
        metrics_from_frame,
        rmse_pearson,
    )
except ModuleNotFoundError:
    from results.lightgbm_results_utils import (
        ID_COLUMN,
        LABEL_COLUMN,
        L1_COLUMN,
        LEVENSHTEIN_COLUMN,
        MIN_WINDOW_ROWS,
        OUTER_FOLDS,
        ROW_KEY_COLUMNS,
        WINDOW_CENTERS,
        WINDOW_WIDTH,
        compute_sliding_window_metrics,
        metrics_from_frame,
        rmse_pearson,
    )

DEFAULT_PREDICTIONS_FILE = PROJECT_ROOT / "models" / "lightgbm_final_models" / "lightgbm_final_predictions.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "results" / "lightgbm_final_metrics.json"
DEFAULT_FULL_DATASET_FILE = PROJECT_ROOT / "data" / "feature_enriched" / "full.csv"

SPLIT_COLUMN = "split"
MODEL_COLUMN = "model_name"
PREDICTION_COLUMN = "prediction"
OUTER_TEST_SPLIT = "outer_test"
ORIGINAL_SPLIT_COLUMN = "original_split"
ORIGINAL_TEST_SPLIT = "test"

BEST_MODEL_SPECS = {
    "BER_cn": {"model_name": "BER", "languages": ["cn"]},
    "BESR_de": {"model_name": "BESR", "languages": ["de"]},
    "BESR_es": {"model_name": "BESR", "languages": ["es"]},
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final LightGBM model suite.")
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--full-dataset-file", type=Path, default=DEFAULT_FULL_DATASET_FILE)
    parser.add_argument("--min-window-rows", type=int, default=MIN_WINDOW_ROWS)
    parser.add_argument(
        "--force-recompute",
        #action="store_true",
        default=True,
        help="Recompute metrics even if --output-json already exists.",
    )
    return parser.parse_args()


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return f"{float(value):.4f}"


def attach_original_split(pred_df: pd.DataFrame, full_dataset_file: Path) -> pd.DataFrame:
    if ORIGINAL_SPLIT_COLUMN in pred_df.columns:
        return pred_df

    if not full_dataset_file.exists():
        raise FileNotFoundError(f"Missing full dataset file: {full_dataset_file}")

    full_df = pd.read_csv(
        full_dataset_file,
        usecols=ROW_KEY_COLUMNS + [ORIGINAL_SPLIT_COLUMN],
        keep_default_na=False,
    )
    duplicated_keys = int(full_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if duplicated_keys:
        raise ValueError(
            f"Full dataset contains duplicate row keys in {full_dataset_file}: {duplicated_keys} duplicates"
        )

    merged = pred_df.merge(
        full_df,
        on=ROW_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if merged[ORIGINAL_SPLIT_COLUMN].isna().any():
        missing = int(merged[ORIGINAL_SPLIT_COLUMN].isna().to_numpy().sum())
        raise ValueError(
            f"Could not attach '{ORIGINAL_SPLIT_COLUMN}' to {missing} prediction rows from {full_dataset_file}"
        )

    return merged


def evaluate_model(model_df: pd.DataFrame, min_window_rows: int) -> Dict[str, Any]:
    model_df = model_df.copy()

    duplicate_outer_test = int(model_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if duplicate_outer_test:
        raise ValueError(
            "Outer-test predictions contain duplicate (item_id, L1) rows for a model; "
            f"found {duplicate_outer_test} duplicates."
        )

    per_fold: Dict[str, Any] = {}
    for fold in OUTER_FOLDS:
        fold_df = model_df[model_df["outer_fold"] == fold]
        if fold_df.empty:
            continue

        per_language_fold: Dict[str, Any] = {}
        for language, language_df in fold_df.groupby(L1_COLUMN, dropna=False):
            language_key = "NA" if pd.isna(language) else str(language)
            lang_metrics = metrics_from_frame(language_df, LABEL_COLUMN, PREDICTION_COLUMN)
            per_language_fold[language_key] = {
                **lang_metrics,
                "n_rows": int(len(language_df)),
            }

        fold_metrics = metrics_from_frame(fold_df, LABEL_COLUMN, PREDICTION_COLUMN)
        per_fold[f"outer_fold_{fold}"] = {
            **fold_metrics,
            "n_rows": int(len(fold_df)),
            "per_language": per_language_fold,
        }

    per_language_outer: Dict[str, Any] = {}
    for language, language_df in model_df.groupby(L1_COLUMN, dropna=False):
        language_key = "NA" if pd.isna(language) else str(language)
        language_metrics = metrics_from_frame(language_df, LABEL_COLUMN, PREDICTION_COLUMN)
        windows = compute_sliding_window_metrics(
            language_df,
            min_window_rows=min_window_rows,
            feature_col=LEVENSHTEIN_COLUMN,
            label_col=LABEL_COLUMN,
            pred_col=PREDICTION_COLUMN,
        )
        per_language_outer[language_key] = {
            **language_metrics,
            "n_rows": int(len(language_df)),
            "windows": windows,
        }

    overall_metrics = metrics_from_frame(model_df, LABEL_COLUMN, PREDICTION_COLUMN)
    return {
        "outer_test": {
            **overall_metrics,
            "n_rows": int(len(model_df)),
            "per_language": per_language_outer,
        },
        "per_fold_outer_test": per_fold,
    }


def compute_best_model_subset_results(pred_df: pd.DataFrame) -> Dict[str, Any]:
    if ORIGINAL_SPLIT_COLUMN not in pred_df.columns:
        return {}

    payload: Dict[str, Any] = {}
    for key, spec in BEST_MODEL_SPECS.items():
        model_name = str(spec["model_name"])
        languages = [str(v) for v in spec["languages"]]

        subset_df = pred_df[
            (pred_df[MODEL_COLUMN] == model_name)
            & (pred_df[ORIGINAL_SPLIT_COLUMN] == ORIGINAL_TEST_SPLIT)
            & (pred_df[L1_COLUMN].isin(languages))
        ].copy()

        if subset_df.empty:
            payload[key] = {
                "model_name": model_name,
                "languages": languages,
                "original_split": ORIGINAL_TEST_SPLIT,
                "overall": None,
                "per_language": {},
            }
            continue

        per_language: Dict[str, Any] = {}
        for language, language_df in subset_df.groupby(L1_COLUMN, dropna=False):
            language_key = "NA" if pd.isna(language) else str(language)
            per_language[language_key] = {
                **metrics_from_frame(language_df, LABEL_COLUMN, PREDICTION_COLUMN),
                "n_rows": int(len(language_df)),
            }

        payload[key] = {
            "model_name": model_name,
            "languages": languages,
            "original_split": ORIGINAL_TEST_SPLIT,
            "overall": {
                **metrics_from_frame(subset_df, LABEL_COLUMN, PREDICTION_COLUMN),
                "n_rows": int(len(subset_df)),
            },
            "per_language": per_language,
        }

    return payload


def build_leaderboard(models_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, model_metrics in models_payload.items():
        outer_test = model_metrics["outer_test"]
        row: Dict[str, Any] = {
            "model_name": model_name,
            "rmse": outer_test.get("rmse"),
            "pearson": outer_test.get("pearson"),
            "n_rows": outer_test.get("n_rows"),
        }

        per_language = outer_test.get("per_language", {})
        for language_code in ["de", "es", "cn"]:
            lang_metrics = per_language.get(language_code, {})
            row[f"{language_code}_rmse"] = lang_metrics.get("rmse")
            row[f"{language_code}_pearson"] = lang_metrics.get("pearson")
        rows.append(row)

    rows.sort(
        key=lambda r: (
            np.inf if r["rmse"] is None or pd.isna(r["rmse"]) else float(r["rmse"]),
            str(r["model_name"]),
        )
    )
    return rows


def print_summary_tables(summary: Dict[str, Any]) -> None:
    leaderboard = summary.get("leaderboard", [])
    leaderboard_df = pd.DataFrame(leaderboard)

    print("\n=== Final LightGBM outer-test leaderboard ===")
    if leaderboard_df.empty:
        print("No models found in summary.")
        return

    display_cols = [
        "model_name",
        "rmse",
        "pearson",
        "n_rows",
        "de_rmse",
        "de_pearson",
        "es_rmse",
        "es_pearson",
        "cn_rmse",
        "cn_pearson",
    ]
    for col in display_cols:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = np.nan

    printable = leaderboard_df.copy()
    for col in [
        "rmse",
        "pearson",
        "de_rmse",
        "de_pearson",
        "es_rmse",
        "es_pearson",
        "cn_rmse",
        "cn_pearson",
    ]:
        printable[col] = printable[col].map(_fmt_metric)
    printable["n_rows"] = printable["n_rows"].fillna(0).astype(int)

    print(printable[display_cols].to_string(index=False))

    print("\n=== Best-model original test subsets ===")
    best_payload = summary.get("best_model_original_test", {})
    if not best_payload:
        print("Original test subset metrics are not available.")
        return

    subset_rows: List[Dict[str, Any]] = []
    for subset_name, subset_metrics in best_payload.items():
        overall = subset_metrics.get("overall")
        if overall is None:
            subset_rows.append(
                {
                    "subset": subset_name,
                    "model": subset_metrics.get("model_name"),
                    "languages": ",".join(subset_metrics.get("languages", [])),
                    "rmse": "n/a",
                    "pearson": "n/a",
                    "n_rows": 0,
                }
            )
            continue

        subset_rows.append(
            {
                "subset": subset_name,
                "model": subset_metrics.get("model_name"),
                "languages": ",".join(subset_metrics.get("languages", [])),
                "rmse": _fmt_metric(overall.get("rmse")),
                "pearson": _fmt_metric(overall.get("pearson")),
                "n_rows": int(overall.get("n_rows", 0)),
            }
        )
    subset_df = pd.DataFrame(subset_rows)
    # Ensure both metrics are always visible in output, including cached/legacy payloads.
    for col in ["rmse", "pearson"]:
        if col not in subset_df.columns:
            subset_df[col] = "n/a"
    print(subset_df[["subset", "model", "languages", "rmse", "pearson", "n_rows"]].to_string(index=False))


def compute_summary(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.predictions_file.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {args.predictions_file}. Run predict_lightgbm_final.py first."
        )

    pred_df = pd.read_csv(args.predictions_file, keep_default_na=False)
    required_cols = {
        ID_COLUMN,
        L1_COLUMN,
        LABEL_COLUMN,
        PREDICTION_COLUMN,
        "outer_fold",
        SPLIT_COLUMN,
        MODEL_COLUMN,
        LEVENSHTEIN_COLUMN,
    }
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing_cols)}")

    outer_test_df = pred_df[pred_df[SPLIT_COLUMN] == OUTER_TEST_SPLIT].copy()
    if outer_test_df.empty:
        raise ValueError("No outer_test rows found in predictions file.")

    has_original_split = False
    try:
        outer_test_df = attach_original_split(outer_test_df, args.full_dataset_file)
        has_original_split = True
    except FileNotFoundError:
        print(
            f"[warning] Full dataset file not found ({args.full_dataset_file}); "
            "original test subset metrics will be unavailable."
        )
    except ValueError as err:
        print(f"[warning] Could not attach original_split: {err}")

    models_payload: Dict[str, Any] = {}
    for model_name, model_df in outer_test_df.groupby(MODEL_COLUMN, dropna=False):
        models_payload[str(model_name)] = evaluate_model(model_df, int(args.min_window_rows))

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "predictions_file": str(args.predictions_file),
        "full_dataset_file": str(args.full_dataset_file),
        "min_window_rows": int(args.min_window_rows),
        "window_width": float(WINDOW_WIDTH),
        "window_centers": [float(v) for v in WINDOW_CENTERS],
        "models": models_payload,
        "leaderboard": build_leaderboard(models_payload),
        "has_original_split": has_original_split,
        "best_model_original_test": compute_best_model_subset_results(outer_test_df),
    }
    return summary


def main() -> None:
    args = parse_args()

    if args.output_json.exists() and not args.force_recompute:
        summary = json.loads(args.output_json.read_text(encoding="utf-8"))
        print(f"Using cached metrics from: {args.output_json}")
        print_summary_tables(summary)
        return

    summary = compute_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary_tables(summary)
    print(f"\nSaved final LightGBM metrics: {args.output_json}")


if __name__ == "__main__":
    main()

