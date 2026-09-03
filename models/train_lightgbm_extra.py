#!/usr/bin/env python3
"""Train LightGBM models for single-feature additions on top of RoBERTa scores."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from roberta_cv_utils import ID_COLUMN, LABEL_COLUMN, fold_dir, rmse_pearson, set_seed

OUTER_FOLDS = [1, 2, 3, 4, 5]
ROW_KEY_COLUMNS = [ID_COLUMN, "L1"]
ROBERTA_SCORE_COLUMN = "roberta_base_score"

FEATURE_ALIASES: Dict[str, List[str]] = {
    "zipf_freq": ["zipf_freq", "zipf_frequency", "log_frequency"],
    "target_word_length": ["target_word_length"],
    "num_syllables": ["num_syllables", "syllable_count"],
    "pos_tag": ["pos_tag", "en_target_pos"],
    "char_edit_dist": ["char_edit_dist"],
    "char_edit_dist_norm": ["char_edit_dist_norm"],
    "char_edit_dist_geo_norm": ["char_edit_dist_geo_norm"],
    "phon_edit_dist": ["phon_edit_dist", "phonetic_edit_dist"],
    "phon_edit_dist_norm": ["phon_edit_dist_norm", "phonetic_edit_dist_norm"],
    "phon_edit_dist_geo_norm": ["phon_edit_dist_geo_norm", "phonetic_edit_dist_geo_norm"],
    "jaro_distance": ["jaro_distance", "jaro_dist"],
    "jaro_winkler_distance": ["jaro_winkler_distance", "jaro_winkler_dist"],
    "soundex_dist": ["soundex_dist"],
    "metaphone_dist_norm": ["metaphone_dist_norm", "metaphone_dist"],
    "char_cosine_sim": ["char_cosine_sim"],
    "lca_language": ["lca_language"],
    "borrowed_from_source": ["borrowed_from_source"],
    "borrowed_from_target": ["borrowed_from_target"],
    "borrowed_from_third": ["borrowed_from_third"],
    "english_word_origin": ["english_word_origin"],
    "origin_matches_source_family": ["origin_matches_source_family"],
    "same_proto_word": ["same_proto_word"],
    "same_proto_root": ["same_proto_root"],
    "transmission_path": ["transmission_path"],
}

FEATURE_GROUPS: Dict[str, List[str]] = {
    "established": ["zipf_freq", "target_word_length", "num_syllables", "pos_tag"],
    "similarity": [
        "char_edit_dist",
        "char_edit_dist_norm",
        "char_edit_dist_geo_norm",
        "phon_edit_dist",
        "phon_edit_dist_norm",
        "phon_edit_dist_geo_norm",
        "jaro_distance",
        "jaro_winkler_distance",
        "soundex_dist",
        "metaphone_dist_norm",
        "char_cosine_sim",
    ],
    "relatedness": [
        "lca_language",
        "borrowed_from_source",
        "borrowed_from_target",
        "borrowed_from_third",
        "english_word_origin",
        "origin_matches_source_family",
        "same_proto_word",
        "same_proto_root",
        "transmission_path",
    ],
}

CONFIG_MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "extra__config__established_only": {
        "config_label": "E",
        "group_config": ["established"],
    },
    "extra__config__similarity_only": {
        "config_label": "S",
        "group_config": ["similarity"],
    },
    "extra__config__relatedness_only": {
        "config_label": "R",
        "group_config": ["relatedness"],
    },
}

ALL_FEATURES = [
    "zipf_freq",
    "target_word_length",
    "num_syllables",
    "pos_tag",
    "char_edit_dist",
    "char_edit_dist_norm",
    "char_edit_dist_geo_norm",
    "phon_edit_dist",
    "phon_edit_dist_norm",
    "phon_edit_dist_geo_norm",
    "jaro_distance",
    "jaro_winkler_distance",
    "soundex_dist",
    "metaphone_dist_norm",
    "char_cosine_sim",
    "lca_language",
    "borrowed_from_source",
    "borrowed_from_target",
    "borrowed_from_third",
    "english_word_origin",
    "origin_matches_source_family",
    "same_proto_word",
    "same_proto_root",
    "transmission_path",
]

CATEGORICAL_CANONICAL = {"pos_tag", "lca_language", "english_word_origin", "transmission_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM models: one added feature per model on top of RoBERTa score."
    )
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--roberta-predictions-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("./lightgbm_extra_models"))
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--outer-folds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated outer folds to train (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=None,
        help="Optional cap on number of extra-feature models (useful for smoke tests).",
    )
    return parser.parse_args()


def unique_preserve_order(values: List[int]) -> List[int]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_outer_folds(raw_value: str) -> List[int]:
    folds: List[int] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        fold = int(part)
        if fold not in OUTER_FOLDS:
            raise ValueError(f"Invalid outer fold: {fold}. Expected values in {OUTER_FOLDS}")
        folds.append(fold)
    if not folds:
        raise ValueError("No outer folds were provided.")
    return unique_preserve_order(folds)


def resolve_feature_column(df: pd.DataFrame, canonical_feature: str) -> str:
    aliases = FEATURE_ALIASES[canonical_feature]
    for col in aliases:
        if col in df.columns:
            return col
    raise ValueError(f"Could not resolve feature '{canonical_feature}'. Tried aliases: {aliases}")


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


def build_feature_map(df: pd.DataFrame) -> Dict[str, str]:
    return {canonical: resolve_feature_column(df, canonical) for canonical in ALL_FEATURES}


def cast_model_columns(df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()

    for col in categorical_columns:
        x[col] = x[col].fillna("NA").astype(str).astype("category")

    for col in feature_columns:
        if col not in categorical_columns and x[col].dtype == bool:
            x[col] = x[col].astype(np.int8)

    return x


def make_feature_specs(feature_map: Dict[str, str], feature_limit: int | None) -> List[Dict[str, Any]]:
    selected_features = list(ALL_FEATURES)
    if feature_limit is not None:
        if feature_limit < 0:
            raise ValueError("--feature-limit must be >= 0")
        selected_features = selected_features[:feature_limit]

    specs: List[Dict[str, Any]] = []

    # Group-only configurations with no RoBERTa score (E/S/R).
    for model_name, spec in CONFIG_MODEL_SPECS.items():
        canonical_features = [
            canonical
            for group_name in spec["group_config"]
            for canonical in FEATURE_GROUPS[str(group_name)]
        ]
        specs.append(
            {
                "model_name": model_name,
                "feature_columns": [feature_map[c] for c in canonical_features],
                "feature_canonical": canonical_features,
                "added_feature_canonical": None,
                "added_feature_column": None,
                "config_label": spec["config_label"],
                "group_config": spec["group_config"],
            }
        )

    for canonical in selected_features:
        actual_col = feature_map[canonical]
        specs.append(
            {
                "model_name": f"extra__add__{canonical}",
                "feature_columns": [ROBERTA_SCORE_COLUMN, actual_col],
                "feature_canonical": [canonical],
                "added_feature_canonical": canonical,
                "added_feature_column": actual_col,
                "config_label": None,
                "group_config": None,
            }
        )

    return specs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_outer_folds = parse_outer_folds(args.outer_folds)

    set_seed(args.seed)
    training_manifest = []

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
        specs = make_feature_specs(feature_map, args.feature_limit)

        for spec in specs:
            model_name = str(spec["model_name"])
            feature_columns = list(spec["feature_columns"])
            feature_canonical = [str(v) for v in list(spec["feature_canonical"])]
            categorical_columns = [
                feature_map[canonical] for canonical in feature_canonical if canonical in CATEGORICAL_CANONICAL
            ]

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

            train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)

            train_metrics = rmse_pearson(y_train.to_numpy(), train_preds)
            test_metrics = rmse_pearson(y_test.to_numpy(), test_preds)

            run_dir = args.output_dir / f"outer_fold_{outer_fold}" / model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            model_path = run_dir / "model.joblib"
            joblib.dump(model, model_path)

            metadata = {
                "outer_fold": outer_fold,
                "model_name": model_name,
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "added_feature_canonical": spec["added_feature_canonical"],
                "added_feature_column": spec["added_feature_column"],
                "config_label": spec["config_label"],
                "group_config": spec["group_config"],
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
    print(f"Saved LightGBM extra training manifest: {manifest_path}")


if __name__ == "__main__":
    main()



