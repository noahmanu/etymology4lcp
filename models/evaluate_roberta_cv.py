#!/usr/bin/env python3
"""Aggregate RoBERTa CV prediction metrics across outer folds."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from roberta_cv_utils import ID_COLUMN, rmse_pearson

OUTER_FOLDS = [1, 2, 3, 4, 5]
L1_COLUMN = "L1"
ROW_KEY_COLUMNS = [ID_COLUMN, L1_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa CV predictions on fold and full-dataset levels.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_fold_predictions(predictions_dir: Path, outer_fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = predictions_dir / f"outer_fold_{outer_fold}"
    oof_path = fold_dir / "outer_train_oof_predictions.csv"
    test_path = fold_dir / "outer_test_predictions.csv"

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF prediction file: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing outer-test prediction file: {test_path}")

    return pd.read_csv(oof_path), pd.read_csv(test_path)


def metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    return rmse_pearson(df["GLMM_score"].to_numpy(), df["roberta_base_score"].to_numpy())


def metrics_by_l1(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if L1_COLUMN not in df.columns:
        raise ValueError(f"Missing required column for per-L1 metrics: {L1_COLUMN}")

    grouped = {}
    for l1, group_df in df.groupby(L1_COLUMN, dropna=False):
        l1_key = "NA" if pd.isna(l1) else str(l1)
        grouped[l1_key] = {
            **metrics_from_df(group_df),
            "n_rows": int(len(group_df)),
            "n_unique_item_id": int(group_df[ID_COLUMN].nunique()),
        }
    return grouped


def unique_row_key_count(df: pd.DataFrame) -> int:
    return int(df[ROW_KEY_COLUMNS].drop_duplicates().shape[0])


def main() -> None:
    args = parse_args()

    if args.output_json is None:
        output_json = args.predictions_dir / "cv_metrics_summary.json"
    else:
        output_json = args.output_json

    per_fold = {}
    all_oof_frames: List[pd.DataFrame] = []
    all_test_frames: List[pd.DataFrame] = []

    for outer_fold in OUTER_FOLDS:
        oof_df, test_df = load_fold_predictions(args.predictions_dir, outer_fold)

        oof_metrics = metrics_from_df(oof_df)
        test_metrics = metrics_from_df(test_df)

        per_fold[f"outer_fold_{outer_fold}"] = {
            "outer_train_oof": {
                **oof_metrics,
                "n_rows": int(len(oof_df)),
                "n_unique_item_id": int(oof_df[ID_COLUMN].nunique()),
                "n_unique_item_l1": unique_row_key_count(oof_df),
                "per_l1": metrics_by_l1(oof_df),
            },
            "outer_test": {
                **test_metrics,
                "n_rows": int(len(test_df)),
                "n_unique_item_id": int(test_df[ID_COLUMN].nunique()),
                "n_unique_item_l1": unique_row_key_count(test_df),
                "per_l1": metrics_by_l1(test_df),
            },
        }

        all_oof_frames.append(oof_df)
        all_test_frames.append(test_df)

    all_oof_df = pd.concat(all_oof_frames, ignore_index=True)
    all_test_df = pd.concat(all_test_frames, ignore_index=True)

    # Outer-test rows should be disjoint across folds and together cover the full dataset once.
    duplicated_test_row_keys = int(all_test_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if duplicated_test_row_keys:
        raise ValueError(
            "Outer-test predictions contain duplicate (item_id, L1) pairs across folds. "
            f"Found {duplicated_test_row_keys} duplicate row keys."
        )

    full_test_metrics = metrics_from_df(all_test_df)
    pooled_oof_metrics = metrics_from_df(all_oof_df)
    full_test_per_l1 = metrics_by_l1(all_test_df)
    pooled_oof_per_l1 = metrics_by_l1(all_oof_df)

    summary = {
        "predictions_dir": str(args.predictions_dir),
        "per_fold": per_fold,
        "aggregate": {
            "outer_test_full_dataset": {
                **full_test_metrics,
                "n_rows": int(len(all_test_df)),
                "n_unique_item_id": int(all_test_df[ID_COLUMN].nunique()),
                "n_unique_item_l1": unique_row_key_count(all_test_df),
                "per_l1": full_test_per_l1,
            },
            "outer_train_oof_pooled": {
                **pooled_oof_metrics,
                "n_rows": int(len(all_oof_df)),
                "n_unique_item_id": int(all_oof_df[ID_COLUMN].nunique()),
                "n_unique_item_l1": unique_row_key_count(all_oof_df),
                "per_l1": pooled_oof_per_l1,
                "note": "Each item appears in multiple outer-train folds; this pooled OOF score is not a one-prediction-per-item full-dataset score.",
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Full-dataset outer-test metrics: "
        f"RMSE={full_test_metrics['rmse']:.4f}, Pearson={full_test_metrics['pearson']:.4f}, "
        f"n={len(all_test_df)}"
    )
    for l1, l1_metrics in full_test_per_l1.items():
        print(
            f"  L1={l1}: RMSE={l1_metrics['rmse']:.4f}, "
            f"Pearson={l1_metrics['pearson']:.4f}, n={l1_metrics['n_rows']}"
        )
    print(f"Saved summary metrics to: {output_json}")


if __name__ == "__main__":
    main()



