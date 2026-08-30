#!/usr/bin/env python3
"""Generate LightGBM extra-experiment CV predictions in one CSV file."""

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from roberta_cv_utils import ID_COLUMN, LABEL_COLUMN, fold_dir

OUTER_FOLDS = [1, 2, 3, 4, 5]
BASELINE_MODEL = "extra__roberta_only"
ROW_KEY_COLUMNS = [ID_COLUMN, "L1"]
ROBERTA_SCORE_COLUMN = "roberta_base_score"
PREDICTION_METADATA_COLUMNS = ["L1", "en_target_word", "L1_source_word", "char_edit_dist_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with trained LightGBM extra-experiment models.")
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--roberta-predictions-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--models-dir", type=Path, default=Path("./lightgbm_extra_models"))
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("./lightgbm_extra_models/lightgbm_extra_all_predictions.csv"),
    )
    return parser.parse_args()


def merge_roberta_scores(base_df: pd.DataFrame, roberta_pred_path: Path) -> pd.DataFrame:
    if not roberta_pred_path.exists():
        raise FileNotFoundError(f"Missing RoBERTa predictions: {roberta_pred_path}")

    pred_df = pd.read_csv(roberta_pred_path)
    if ROBERTA_SCORE_COLUMN not in pred_df.columns:
        raise ValueError(f"Missing '{ROBERTA_SCORE_COLUMN}' in {roberta_pred_path}")

    duplicated_keys = int(pred_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if duplicated_keys:
        raise ValueError(
            f"Duplicate RoBERTa row keys in {roberta_pred_path}: {duplicated_keys} duplicate key pairs"
        )

    merged = base_df.merge(
        pred_df[ROW_KEY_COLUMNS + [ROBERTA_SCORE_COLUMN]],
        on=ROW_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if merged[ROBERTA_SCORE_COLUMN].isna().any():
        missing = int(merged[ROBERTA_SCORE_COLUMN].isna().to_numpy().sum())
        raise ValueError(f"Missing {missing} RoBERTa scores after merge from {roberta_pred_path}")

    return merged


def cast_model_columns(df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()
    for col in categorical_columns:
        x[col] = x[col].fillna("NA").astype(str).astype("category")
    for col in feature_columns:
        if col not in categorical_columns and x[col].dtype == bool:
            x[col] = x[col].astype(np.int8)
    return x


def build_prediction_frame(merged_df: pd.DataFrame, preds: np.ndarray, outer_fold: int, split_name: str, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: merged_df[ID_COLUMN].values,
            "L1": merged_df["L1"].values,
            "en_target_word": merged_df["en_target_word"].values,
            "L1_source_word": merged_df["L1_source_word"].values,
            "char_edit_dist_norm": merged_df["char_edit_dist_norm"].values,
            "prediction": preds,
            LABEL_COLUMN: merged_df[LABEL_COLUMN].values,
            "outer_fold": outer_fold,
            "split": split_name,
            "model_name": model_name,
        }
    )


def main() -> None:
    args = parse_args()
    manifest_path = args.models_dir / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing LightGBM training manifest: {manifest_path}. Run train_lightgbm_extra.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pred_frames = []

    # Baseline is raw RoBERTa predictions (no LightGBM training).
    folds_to_score = sorted({int(run["outer_fold"]) for run in manifest}) if manifest else OUTER_FOLDS
    for outer_fold in folds_to_score:
        outer_dir = fold_dir(args.folds_dir, outer_fold)
        outer_pred_dir = args.roberta_predictions_dir / f"outer_fold_{outer_fold}"

        split_inputs = [
            ("outer_train_fit", outer_dir / "outer_train.csv", outer_pred_dir / "outer_train_oof_predictions.csv"),
            ("outer_test", outer_dir / "outer_test.csv", outer_pred_dir / "outer_test_predictions.csv"),
        ]

        for split_name, csv_path, roberta_pred_path in split_inputs:
            base_df = pd.read_csv(csv_path)
            merged_df = merge_roberta_scores(base_df, roberta_pred_path)
            baseline_preds = merged_df[ROBERTA_SCORE_COLUMN].to_numpy(dtype=float)
            pred_frames.append(
                build_prediction_frame(merged_df, baseline_preds, outer_fold, split_name, BASELINE_MODEL)
            )

    for run in manifest:
        outer_fold = int(run["outer_fold"])
        model_name = str(run["model_name"])
        feature_columns = list(run["feature_columns"])
        categorical_columns = list(run["categorical_columns"])
        model_path = Path(run["model_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = joblib.load(model_path)

        outer_dir = fold_dir(args.folds_dir, outer_fold)
        outer_pred_dir = args.roberta_predictions_dir / f"outer_fold_{outer_fold}"

        split_inputs = [
            ("outer_train_fit", outer_dir / "outer_train.csv", outer_pred_dir / "outer_train_oof_predictions.csv"),
            ("outer_test", outer_dir / "outer_test.csv", outer_pred_dir / "outer_test_predictions.csv"),
        ]

        for split_name, csv_path, roberta_pred_path in split_inputs:
            base_df = pd.read_csv(csv_path)
            merged_df = merge_roberta_scores(base_df, roberta_pred_path)

            x = cast_model_columns(merged_df, feature_columns, categorical_columns)
            preds = model.predict(x)
            pred_frames.append(build_prediction_frame(merged_df, preds, outer_fold, split_name, model_name))

    if not pred_frames:
        raise RuntimeError("No predictions were generated.")

    all_preds: pd.DataFrame = pd.concat(pred_frames, ignore_index=True)
    all_preds = all_preds[
        [
            ID_COLUMN,
            *PREDICTION_METADATA_COLUMNS,
            "prediction",
            LABEL_COLUMN,
            "outer_fold",
            "split",
            "model_name",
        ]
    ]

    all_preds = all_preds.sort_values(["model_name", "outer_fold", "split", ID_COLUMN, "L1"])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    all_preds.to_csv(args.output_file, index=False)

    print(f"Saved combined LightGBM extra predictions: {args.output_file}")
    print(f"Rows: {len(all_preds)} | Models: {all_preds['model_name'].nunique()}")


if __name__ == "__main__":
    main()


