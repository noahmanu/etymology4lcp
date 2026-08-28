#!/usr/bin/env python3
"""Evaluate LightGBM CV predictions per model, L1, and Levenshtein windows."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from roberta_cv_utils import ID_COLUMN, rmse_pearson

OUTER_FOLDS = [1, 2, 3, 4, 5]
L1_COLUMN = "L1"
LEVENSHTEIN_COLUMN = "char_edit_dist_norm"
ROW_KEY_COLUMNS = [ID_COLUMN, L1_COLUMN]

# Overlapping sliding windows: [0.0-0.2, 0.1-0.3, ..., 0.8-1.0]
LEVENSHTEIN_WINDOWS: List[Tuple[float, float]] = [
    (round(float(lo), 1), round(float(lo) + 0.2, 1))
    for lo in np.arange(0.0, 0.9, 0.1)
]
MIN_WINDOW_ROWS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM CV predictions.")
    parser.add_argument("--predictions-file", type=Path, default=Path("./lightgbm_models/lightgbm_all_predictions.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("./lightgbm_models/lightgbm_metrics_summary.json"))
    return parser.parse_args()


def metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    labels = df["GLMM_score"].to_numpy()
    preds = df["prediction"].to_numpy()
    n = len(df)

    if n == 0:
        return {"rmse": float("nan"), "pearson": float("nan")}
    if n == 1:
        rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
        return {"rmse": rmse, "pearson": float("nan")}

    return rmse_pearson(labels, preds)


def unique_row_key_count(df: pd.DataFrame) -> int:
    return int(df[ROW_KEY_COLUMNS].drop_duplicates().shape[0])


def metrics_for_window(df: pd.DataFrame, lo: float, hi: float) -> Optional[Dict]:
    if LEVENSHTEIN_COLUMN not in df.columns:
        return None

    mask = (df[LEVENSHTEIN_COLUMN] >= lo) & (df[LEVENSHTEIN_COLUMN] <= hi)
    sub = df.loc[mask]
    if sub.empty or len(sub) < MIN_WINDOW_ROWS:
        return None

    return {
        **metrics_from_df(sub),
        "n_rows": int(len(sub)),
    }


def metrics_by_l1_and_windows(df: pd.DataFrame) -> Dict[str, Dict]:
    if L1_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {L1_COLUMN}")

    result = {}
    for l1, group_df in df.groupby(L1_COLUMN, dropna=False):
        l1_key = "NA" if pd.isna(l1) else str(l1)
        windows = {}

        for lo, hi in LEVENSHTEIN_WINDOWS:
            window_key = f"{lo:.1f}-{hi:.1f}"
            window_metrics = metrics_for_window(group_df, lo, hi)
            if window_metrics is not None:
                windows[window_key] = window_metrics

        result[l1_key] = {
            **metrics_from_df(group_df),
            "n_rows": int(len(group_df)),
            "n_unique_item_id": int(group_df[ID_COLUMN].nunique()),
            "levenshtein_windows": windows,
        }

    return result


def evaluate_model(model_df: pd.DataFrame) -> Dict:
    per_fold: Dict[str, Dict] = {}

    for outer_fold in OUTER_FOLDS:
        fold_df = model_df[model_df["outer_fold"] == outer_fold]
        if fold_df.empty:
            continue

        split_payload = {}
        for split_name, split_df in fold_df.groupby("split", dropna=False):
            split_payload[str(split_name)] = {
                **metrics_from_df(split_df),
                "n_rows": int(len(split_df)),
                "n_unique_item_id": int(split_df[ID_COLUMN].nunique()),
                "n_unique_item_l1": unique_row_key_count(split_df),
                "per_l1": metrics_by_l1_and_windows(split_df),
            }

        per_fold[f"outer_fold_{outer_fold}"] = split_payload

    aggregate = {}
    for split_name, split_df in model_df.groupby("split", dropna=False):
        split_name = str(split_name)

        if split_name == "outer_test":
            duplicated_test_row_keys = int(split_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
            if duplicated_test_row_keys:
                raise ValueError(
                    "Outer-test predictions contain duplicate (item_id, L1) pairs across folds. "
                    f"Found {duplicated_test_row_keys} duplicate row keys for model."
                )

        aggregate[split_name] = {
            **metrics_from_df(split_df),
            "n_rows": int(len(split_df)),
            "n_unique_item_id": int(split_df[ID_COLUMN].nunique()),
            "n_unique_item_l1": unique_row_key_count(split_df),
            "per_l1": metrics_by_l1_and_windows(split_df),
        }

    return {
        "per_fold": per_fold,
        "aggregate": aggregate,
    }


def main() -> None:
    args = parse_args()

    if not args.predictions_file.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {args.predictions_file}. Run predict_lightgbm.py first."
        )

    pred_df = pd.read_csv(args.predictions_file)

    required_cols = {
        ID_COLUMN,
        "L1",
        "GLMM_score",
        "prediction",
        "outer_fold",
        "split",
        "model_name",
        LEVENSHTEIN_COLUMN,
    }
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing_cols)}")

    summary = {
        "predictions_file": str(args.predictions_file),
        "models": {},
    }

    for model_name, model_df in pred_df.groupby("model_name", dropna=False):
        model_key = str(model_name)
        summary["models"][model_key] = evaluate_model(model_df)

        outer_test_df = model_df[model_df["split"] == "outer_test"]
        outer_test_metrics = metrics_from_df(outer_test_df)
        print(
            f"{model_key}: outer_test RMSE={outer_test_metrics['rmse']:.4f}, "
            f"Pearson={outer_test_metrics['pearson']:.4f}, n={len(outer_test_df)}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved LightGBM evaluation summary: {args.output_json}")


if __name__ == "__main__":
    main()


