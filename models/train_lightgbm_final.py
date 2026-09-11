#!/usr/bin/env python3
"""Train the final LightGBM model suite on top of RoBERTa-base predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
from lightgbm_cv_utils import (
    ALL_CANONICAL_FEATURES,
    FEATURE_GROUPS,
    LABEL_COLUMN,
    L1_COLUMN,
    ROBERTA_SCORE_COLUMN,
    build_feature_map,
    cast_model_columns,
    categorical_columns_for_features,
    fold_dir,
    merge_roberta_scores,
    parse_outer_folds,
    resolve_feature_tokens,
    rmse_pearson,
    set_seed,
)


BASELINE_L1_MODEL = "BL1"

BL1_GROUP_CONFIGS = {
    "BE": ["established"],
    "BS": ["similarity"],
    "BR": ["relatedness"],
    "BES": ["established", "similarity"],
    "BER": ["established", "relatedness"],
    "BSR": ["similarity", "relatedness"],
    "BESR": ["established", "similarity", "relatedness"],
}

L1_GROUP_CONFIGS = {
    "E": ["established"],
    "S": ["similarity"],
    "R": ["relatedness"],
    "ES": ["established", "similarity"],
    "ER": ["established", "relatedness"],
    "SR": ["similarity", "relatedness"],
    "ESR": ["established", "similarity", "relatedness"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final LightGBM CV models.")
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
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "lightgbm_final_models",
    )
    parser.add_argument("--outer-folds", type=str, default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _group_features(group_names: List[str]) -> List[str]:
    features: List[str] = []
    for group_name in group_names:
        features.extend(FEATURE_GROUPS[group_name])
    return features


def build_model_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    specs.append(
        {
            "model_name": BASELINE_L1_MODEL,
            "model_type": "baseline_l1",
            "feature_tokens": [ROBERTA_SCORE_COLUMN, L1_COLUMN],
            "groups": [],
            "dropped_feature": None,
        }
    )

    for canonical_feature in ALL_CANONICAL_FEATURES:
        specs.append(
            {
                "model_name": f"BL1_plus_{canonical_feature}",
                "model_type": "baseline_l1_individual",
                "feature_tokens": [ROBERTA_SCORE_COLUMN, L1_COLUMN, canonical_feature],
                "groups": [],
                "dropped_feature": None,
            }
        )

    for model_name, group_names in BL1_GROUP_CONFIGS.items():
        specs.append(
            {
                "model_name": model_name,
                "model_type": "baseline_l1_groups",
                "feature_tokens": [ROBERTA_SCORE_COLUMN, L1_COLUMN] + _group_features(group_names),
                "groups": group_names,
                "dropped_feature": None,
            }
        )

    for model_name, group_names in L1_GROUP_CONFIGS.items():
        specs.append(
            {
                "model_name": model_name,
                "model_type": "l1_groups",
                "feature_tokens": [L1_COLUMN] + _group_features(group_names),
                "groups": group_names,
                "dropped_feature": None,
            }
        )

    for dropped_feature in ALL_CANONICAL_FEATURES:
        kept = [feature for feature in ALL_CANONICAL_FEATURES if feature != dropped_feature]
        specs.append(
            {
                "model_name": f"BL1_all_minus_{dropped_feature}",
                "model_type": "baseline_l1_ablation",
                "feature_tokens": [ROBERTA_SCORE_COLUMN, L1_COLUMN] + kept,
                "groups": ["established", "similarity", "relatedness"],
                "dropped_feature": dropped_feature,
            }
        )

    expected_models = 1 + len(ALL_CANONICAL_FEATURES) + 7 + 7 + len(ALL_CANONICAL_FEATURES)
    if len(specs) != expected_models:
        raise RuntimeError(f"Unexpected number of model specs: {len(specs)} != {expected_models}")

    return specs


def fit_one_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    args: argparse.Namespace,
) -> tuple[lgb.LGBMRegressor, Dict[str, float], Dict[str, float]]:
    x_train = cast_model_columns(train_df, feature_columns, categorical_columns)
    y_train = train_df[LABEL_COLUMN].astype(float)

    x_test = cast_model_columns(test_df, feature_columns, categorical_columns)
    y_test = test_df[LABEL_COLUMN].astype(float)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train, categorical_feature=categorical_columns)

    train_pred = np.asarray(model.predict(x_train), dtype=float)
    test_pred = np.asarray(model.predict(x_test), dtype=float)
    train_metrics = rmse_pearson(y_train.to_numpy(dtype=float), train_pred)
    test_metrics = rmse_pearson(y_test.to_numpy(dtype=float), test_pred)

    return model, train_metrics, test_metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_outer_folds = parse_outer_folds(args.outer_folds)
    set_seed(args.seed)
    specs = build_model_specs()

    training_manifest: List[Dict[str, Any]] = []

    for outer_fold in selected_outer_folds:
        outer_dir = fold_dir(args.folds_dir, outer_fold)
        outer_train_df = pd.read_csv(outer_dir / "outer_train.csv")
        outer_test_df = pd.read_csv(outer_dir / "outer_test.csv")

        outer_pred_dir = args.roberta_predictions_dir / f"outer_fold_{outer_fold}"
        train_pred_path = outer_pred_dir / "outer_train_oof_predictions.csv"
        test_pred_path = outer_pred_dir / "outer_test_predictions.csv"

        train_df = merge_roberta_scores(outer_train_df, train_pred_path)
        test_df = merge_roberta_scores(outer_test_df, test_pred_path)

        feature_map = build_feature_map(train_df)

        for spec in specs:
            model_name = str(spec["model_name"])
            feature_tokens = [str(v) for v in spec["feature_tokens"]]
            feature_columns = resolve_feature_tokens(feature_tokens, feature_map)
            categorical_columns = categorical_columns_for_features(feature_columns, feature_map)

            model, train_metrics, test_metrics = fit_one_model(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                args=args,
            )

            run_dir = args.output_dir / f"outer_fold_{outer_fold}" / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = run_dir / "model.joblib"
            joblib.dump(model, model_path)

            metadata: Dict[str, Any] = {
                "outer_fold": outer_fold,
                "model_name": model_name,
                "model_type": spec["model_type"],
                "groups": spec["groups"],
                "dropped_feature": spec["dropped_feature"],
                "feature_tokens": feature_tokens,
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "n_rows_train": int(len(train_df)),
                "n_rows_test": int(len(test_df)),
                "train_metrics": train_metrics,
                "outer_test_metrics": test_metrics,
                "params": {
                    "seed": args.seed,
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "num_leaves": args.num_leaves,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "n_jobs": args.n_jobs,
                },
                "model_path": str(model_path),
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            training_manifest.append(metadata)

            print(
                f"outer_fold_{outer_fold} | {model_name}: "
                f"train RMSE={train_metrics['rmse']:.4f}, Pearson={train_metrics['pearson']:.4f} | "
                f"outer_test RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}"
            )

    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    print(f"Saved final LightGBM training manifest: {manifest_path}")


if __name__ == "__main__":
    main()

