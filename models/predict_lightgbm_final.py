#!/usr/bin/env python3
"""Create CV predictions for all final LightGBM models and the raw RoBERTa baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
from lightgbm_cv_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    OUTER_FOLDS,
    PREDICTION_METADATA_COLUMNS,
    ROBERTA_SCORE_COLUMN,
    cast_model_columns,
    fold_dir,
    merge_roberta_scores,
    parse_outer_folds,
    resolve_feature_tokens,
    build_feature_map,
)


BASELINE_MODEL_NAME = "raw_roberta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with final LightGBM CV models.")
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "feature_enriched" / "folds",
    )
    parser.add_argument(
        "--roberta-predictions-dir",
        type=Path,
        default=SCRIPT_DIR / "roberta_predictions",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=SCRIPT_DIR / "lightgbm_final_models",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=SCRIPT_DIR / "lightgbm_final_models" / "lightgbm_final_predictions.csv",
    )
    parser.add_argument("--outer-folds", type=str, default="1,2,3,4,5")
    parser.add_argument("--baseline-model-name", type=str, default=BASELINE_MODEL_NAME)
    return parser.parse_args()


def build_prediction_frame(
    merged_df: pd.DataFrame,
    preds: np.ndarray,
    outer_fold: int,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
    payload: Dict[str, Any] = {
        ID_COLUMN: merged_df[ID_COLUMN].values,
        "prediction": preds,
        LABEL_COLUMN: merged_df[LABEL_COLUMN].values,
        "outer_fold": outer_fold,
        "split": split_name,
        "model_name": model_name,
    }
    for col in PREDICTION_METADATA_COLUMNS:
        payload[col] = merged_df[col].values if col in merged_df.columns else np.nan

    return pd.DataFrame(payload)


def load_manifest(models_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = models_dir / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing training manifest: {manifest_path}. Run train_lightgbm_final.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Invalid manifest format in {manifest_path}; expected a JSON list.")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.models_dir)

    requested_folds = set(parse_outer_folds(args.outer_folds))
    fold_ids_from_manifest = sorted({int(run["outer_fold"]) for run in manifest}) if manifest else OUTER_FOLDS
    folds_to_score = [fold for fold in fold_ids_from_manifest if fold in requested_folds]
    if not folds_to_score:
        raise ValueError("No overlapping folds between --outer-folds and training manifest folds.")

    prediction_frames: List[pd.DataFrame] = []

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
            prediction_frames.append(
                build_prediction_frame(
                    merged_df=merged_df,
                    preds=baseline_preds,
                    outer_fold=outer_fold,
                    split_name=split_name,
                    model_name=args.baseline_model_name,
                )
            )

    for run in manifest:
        outer_fold = int(run["outer_fold"])
        if outer_fold not in requested_folds:
            continue

        model_name = str(run["model_name"])
        model_path = Path(str(run["model_path"]))
        feature_tokens = [str(v) for v in run.get("feature_tokens", [])]
        categorical_columns = [str(v) for v in run.get("categorical_columns", [])]

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = joblib.load(model_path)

        outer_dir = fold_dir(args.folds_dir, outer_fold)
        outer_pred_dir = args.roberta_predictions_dir / f"outer_fold_{outer_fold}"
        split_inputs = [
            ("outer_train_fit", outer_dir / "outer_train.csv", outer_pred_dir / "outer_train_oof_predictions.csv"),
            ("outer_test", outer_dir / "outer_test.csv", outer_pred_dir / "outer_test_predictions.csv"),
        ]

        uses_roberta = ROBERTA_SCORE_COLUMN in set(feature_tokens)

        for split_name, csv_path, roberta_pred_path in split_inputs:
            base_df = pd.read_csv(csv_path)
            model_df = merge_roberta_scores(base_df, roberta_pred_path) if uses_roberta else base_df

            feature_map = build_feature_map(model_df)
            feature_columns = resolve_feature_tokens(feature_tokens, feature_map)
            x = cast_model_columns(model_df, feature_columns, categorical_columns)
            preds = np.asarray(model.predict(x), dtype=float)

            prediction_frames.append(
                build_prediction_frame(
                    merged_df=model_df,
                    preds=preds,
                    outer_fold=outer_fold,
                    split_name=split_name,
                    model_name=model_name,
                )
            )

    if not prediction_frames:
        raise RuntimeError("No predictions were generated.")

    all_preds = pd.concat(prediction_frames, ignore_index=True)
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
    print(f"Saved final LightGBM predictions: {args.output_file}")
    print(f"Rows: {len(all_preds)} | Models: {all_preds['model_name'].nunique()}")


if __name__ == "__main__":
    main()

