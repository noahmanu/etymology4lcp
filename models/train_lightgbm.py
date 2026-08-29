#!/usr/bin/env python3
"""Train LightGBM models on CV folds using RoBERTa predictions + tabular features."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

GROUP_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "group__relatedness": ("relatedness",),
    "group__similarity": ("similarity",),
    "group__relatedness_similarity": ("relatedness", "similarity"),
    "group__established": ("established",),
    "group__established_similarity": ("established", "similarity"),
    "group__established_relatedness": ("established", "relatedness"),
    "group__established_relatedness_similarity": ("established", "relatedness", "similarity"),
}

CATEGORICAL_CANONICAL = {"pos_tag", "lca_language", "english_word_origin", "transmission_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM fold models with feature-group configs and leave-one-out ablations.")
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--roberta-predictions-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("./lightgbm_models"))
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
        "--ablation-limit",
        type=int,
        default=None,
        help="Optional cap on number of leave-one-out ablation models (useful for smoke tests).",
    )
    return parser.parse_args()


def parse_outer_folds(raw_value: str) -> List[int]:
    folds = []
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
    raise ValueError(
        f"Could not resolve feature '{canonical_feature}'. Tried aliases: {aliases}"
    )


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


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
        missing = int(merged[ROBERTA_SCORE_COLUMN].isna().sum())
        raise ValueError(
            f"Missing {missing} RoBERTa scores after merge from {roberta_pred_path}"
        )

    return merged


def build_full_feature_map(df: pd.DataFrame) -> Dict[str, str]:
    # Maps canonical names to concrete dataframe columns (handles naming differences).
    feature_map: Dict[str, str] = {}
    for group_name in ("established", "similarity", "relatedness"):
        for canonical in FEATURE_GROUPS[group_name]:
            feature_map[canonical] = resolve_feature_column(df, canonical)
    return feature_map


def cast_model_columns(df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()

    for col in categorical_columns:
        x[col] = x[col].fillna("NA").astype(str).astype("category")

    for col in feature_columns:
        if col not in categorical_columns:
            if x[col].dtype == bool:
                x[col] = x[col].astype(np.int8)
    return x


def make_feature_specs(feature_map: Dict[str, str], ablation_limit: int | None) -> List[Dict[str, object]]:
    full_canonical = (
        FEATURE_GROUPS["established"]
        + FEATURE_GROUPS["similarity"]
        + FEATURE_GROUPS["relatedness"]
    )

    full_actual = [feature_map[c] for c in full_canonical]
    all_features_with_roberta = unique_preserve_order([ROBERTA_SCORE_COLUMN] + full_actual)

    specs: List[Dict[str, object]] = []

    # Group-based configurations required in the project overview.
    for model_name, group_tuple in GROUP_CONFIGS.items():
        if model_name == "group__established_relatedness_similarity":
            # Keep exact feature ordering aligned with ablation__all_features.
            selected = list(all_features_with_roberta)
        else:
            selected = [ROBERTA_SCORE_COLUMN]
            for group in group_tuple:
                selected.extend(feature_map[c] for c in FEATURE_GROUPS[group])
            selected = unique_preserve_order(selected)
        specs.append(
            {
                "model_name": model_name,
                "feature_columns": selected,
                "ablated_feature": None,
                "group_config": list(group_tuple),
            }
        )

    # Leave-one-out ablation over the full feature set.
    specs.append(
        {
            "model_name": "ablation__all_features",
            "feature_columns": all_features_with_roberta,
            "ablated_feature": None,
            "group_config": ["established", "similarity", "relatedness"],
        }
    )

    ablation_features = all_features_with_roberta
    if ablation_limit is not None:
        if ablation_limit < 0:
            raise ValueError("--ablation-limit must be >= 0")
        ablation_features = ablation_features[:ablation_limit]

    for dropped in ablation_features:
        kept = [f for f in all_features_with_roberta if f != dropped]
        specs.append(
            {
                "model_name": f"ablation__drop__{dropped}",
                "feature_columns": kept,
                "ablated_feature": dropped,
                "group_config": ["established", "similarity", "relatedness"],
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

        feature_map = build_full_feature_map(train_df)
        specs = make_feature_specs(feature_map, args.ablation_limit)

        for spec in specs:
            model_name = str(spec["model_name"])
            feature_columns = list(spec["feature_columns"])
            categorical_columns = [
                c
                for c in feature_columns
                if any(c == feature_map[canon] for canon in CATEGORICAL_CANONICAL)
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
                "group_config": spec["group_config"],
                "ablated_feature": spec["ablated_feature"],
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
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            training_manifest.append(metadata)
            print(
                f"outer_fold_{outer_fold} | {model_name}: "
                f"train RMSE={train_metrics['rmse']:.4f}, Pearson={train_metrics['pearson']:.4f} | "
                f"outer_test RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}"
            )

    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    print(f"Saved LightGBM training manifest: {manifest_path}")


if __name__ == "__main__":
    main()



