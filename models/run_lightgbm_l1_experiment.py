#!/usr/bin/env python3
"""Run full LightGBM + LightGBM-extra pipeline with L1 always included as a feature.

This script trains, predicts, and evaluates both model families:
- Standard family (`train_lightgbm.py` equivalent): group configs + leave-one-out ablations
- Extra family (`train_lightgbm_extra.py` equivalent): E/S/R/ES/ER/SR/ESR configs + single-feature additions

Behavioral rule in this experiment:
- Baseline model remains raw RoBERTa scores (no LightGBM training)
- All trained LightGBM models always include `L1` as an input feature
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from roberta_cv_utils import ID_COLUMN, LABEL_COLUMN, fold_dir, rmse_pearson, set_seed

OUTER_FOLDS = [1, 2, 3, 4, 5]
ROW_KEY_COLUMNS = [ID_COLUMN, "L1"]
ROBERTA_SCORE_COLUMN = "roberta_base_score"
L1_COLUMN = "L1"

STANDARD_BASELINE_MODEL = "ablation__roberta_only"
EXTRA_BASELINE_MODEL = "extra__roberta_only"

PREDICTION_METADATA_COLUMNS = ["L1", "en_target_word", "L1_source_word", "char_edit_dist_norm"]

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

EXTRA_CONFIG_MODEL_SPECS: Dict[str, Dict[str, object]] = {
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
    "extra__config__established_similarity": {
        "config_label": "ES",
        "group_config": ["established", "similarity"],
    },
    "extra__config__established_relatedness": {
        "config_label": "ER",
        "group_config": ["established", "relatedness"],
    },
    "extra__config__similarity_relatedness": {
        "config_label": "SR",
        "group_config": ["similarity", "relatedness"],
    },
    "extra__config__established_similarity_relatedness": {
        "config_label": "ESR",
        "group_config": ["established", "similarity", "relatedness"],
    },
}

ALL_FEATURES = (
    FEATURE_GROUPS["established"]
    + FEATURE_GROUPS["similarity"]
    + FEATURE_GROUPS["relatedness"]
)

CATEGORICAL_CANONICAL = {"pos_tag", "lca_language", "english_word_origin", "transmission_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/predict/evaluate LightGBM and LightGBM-extra with L1 always included in trained models."
        )
    )
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--roberta-predictions-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--full-dataset-file", type=Path, default=Path("../data/feature_enriched/full.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("./lightgbm_l1_experiment"))

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
        help="Optional cap on number of standard leave-one-out ablation models.",
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=None,
        help="Optional cap on number of extra single-feature models.",
    )

    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluate_lightgbm*.py calls.")
    parser.add_argument(
        "--extra-no-show",
        action="store_true",
        help="Pass --no-show to evaluate_lightgbm_extra.py.",
    )
    return parser.parse_args()


def unique_preserve_order(values: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
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
    return [int(v) for v in unique_preserve_order(folds)]


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


def cast_model_columns(df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()

    for col in categorical_columns:
        x[col] = x[col].fillna("NA").astype(str).astype("category")

    for col in feature_columns:
        if col not in categorical_columns and x[col].dtype == bool:
            x[col] = x[col].astype(np.int8)

    return x


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor()


def build_feature_map(df: pd.DataFrame) -> Dict[str, str]:
    return {canonical: resolve_feature_column(df, canonical) for canonical in ALL_FEATURES}


def make_standard_specs(feature_map: Dict[str, str], ablation_limit: int | None) -> List[Dict[str, object]]:
    full_actual = [feature_map[c] for c in ALL_FEATURES]
    all_features_with_roberta = unique_preserve_order([ROBERTA_SCORE_COLUMN] + full_actual)

    specs: List[Dict[str, object]] = []

    for model_name, group_tuple in GROUP_CONFIGS.items():
        if model_name == "group__established_relatedness_similarity":
            selected = list(all_features_with_roberta)
        else:
            selected = [ROBERTA_SCORE_COLUMN]
            for group in group_tuple:
                selected.extend(feature_map[c] for c in FEATURE_GROUPS[group])
            selected = unique_preserve_order(selected)

        selected_with_l1 = unique_preserve_order(selected + [L1_COLUMN])
        specs.append(
            {
                "model_name": model_name,
                "feature_columns": selected_with_l1,
                "ablated_feature": None,
                "group_config": list(group_tuple),
            }
        )

    full_with_l1 = unique_preserve_order(all_features_with_roberta + [L1_COLUMN])
    specs.append(
        {
            "model_name": "ablation__all_features",
            "feature_columns": full_with_l1,
            "ablated_feature": None,
            "group_config": ["established", "similarity", "relatedness"],
        }
    )

    ablation_features = list(full_with_l1)
    if ablation_limit is not None:
        if ablation_limit < 0:
            raise ValueError("--ablation-limit must be >= 0")
        ablation_features = ablation_features[:ablation_limit]

    for dropped in ablation_features:
        kept = [f for f in full_with_l1 if f != dropped]
        specs.append(
            {
                "model_name": f"ablation__drop__{dropped}",
                "feature_columns": kept,
                "ablated_feature": dropped,
                "group_config": ["established", "similarity", "relatedness"],
            }
        )

    return specs


def make_extra_specs(feature_map: Dict[str, str], feature_limit: int | None) -> List[Dict[str, Any]]:
    selected_features = list(ALL_FEATURES)
    if feature_limit is not None:
        if feature_limit < 0:
            raise ValueError("--feature-limit must be >= 0")
        selected_features = selected_features[:feature_limit]

    specs: List[Dict[str, Any]] = []

    for model_name, spec in EXTRA_CONFIG_MODEL_SPECS.items():
        canonical_features = [
            canonical
            for group_name in spec["group_config"]
            for canonical in FEATURE_GROUPS[str(group_name)]
        ]
        feature_columns = [feature_map[c] for c in canonical_features]
        feature_columns = unique_preserve_order(feature_columns + [L1_COLUMN])

        specs.append(
            {
                "model_name": model_name,
                "feature_columns": feature_columns,
                "feature_canonical": canonical_features + [L1_COLUMN],
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
                "feature_columns": unique_preserve_order([ROBERTA_SCORE_COLUMN, actual_col, L1_COLUMN]),
                "feature_canonical": [canonical, L1_COLUMN],
                "added_feature_canonical": canonical,
                "added_feature_column": actual_col,
                "config_label": None,
                "group_config": None,
            }
        )

    return specs


def categorical_columns_for_features(feature_columns: List[str], feature_map: Dict[str, str]) -> List[str]:
    categorical_actual = {feature_map[c] for c in CATEGORICAL_CANONICAL}
    categorical_actual.add(L1_COLUMN)
    return [col for col in feature_columns if col in categorical_actual]


def fit_and_save_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Dict[str, float], Dict[str, float], Path]:
    n_estimators = int(args.n_estimators)
    learning_rate = float(args.learning_rate)
    num_leaves = int(args.num_leaves)
    subsample = float(args.subsample)
    colsample_bytree = float(args.colsample_bytree)
    random_state = int(args.seed)
    n_jobs = int(args.n_jobs)

    x_train = cast_model_columns(train_df, feature_columns, categorical_columns)
    y_train = train_df[LABEL_COLUMN].astype(float)

    x_test = cast_model_columns(test_df, feature_columns, categorical_columns)
    y_test = test_df[LABEL_COLUMN].astype(float)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train, categorical_feature=categorical_columns)

    train_preds = np.asarray(model.predict(x_train), dtype=float)
    test_preds = np.asarray(model.predict(x_test), dtype=float)

    train_metrics = rmse_pearson(y_train.to_numpy(), train_preds)
    test_metrics = rmse_pearson(y_test.to_numpy(), test_preds)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)

    return train_metrics, test_metrics, model_path


def train_standard_models(args: argparse.Namespace, selected_outer_folds: List[int], standard_dir: Path) -> Path:
    standard_dir.mkdir(parents=True, exist_ok=True)
    training_manifest: List[Dict[str, Any]] = []
    folds_dir = Path(args.folds_dir)
    roberta_predictions_dir = Path(args.roberta_predictions_dir)

    for outer_fold in selected_outer_folds:
        outer_dir = fold_dir(folds_dir, outer_fold)
        outer_train_df = pd.read_csv(outer_dir / "outer_train.csv")
        outer_test_df = pd.read_csv(outer_dir / "outer_test.csv")

        outer_pred_dir = roberta_predictions_dir / f"outer_fold_{outer_fold}"
        train_pred_path = outer_pred_dir / "outer_train_oof_predictions.csv"
        test_pred_path = outer_pred_dir / "outer_test_predictions.csv"

        train_df = merge_roberta_scores(outer_train_df, train_pred_path)
        test_df = merge_roberta_scores(outer_test_df, test_pred_path)

        feature_map = build_feature_map(train_df)
        specs = make_standard_specs(feature_map, args.ablation_limit)

        for spec in specs:
            model_name = str(spec["model_name"])
            feature_columns = list(spec["feature_columns"])
            categorical_columns = categorical_columns_for_features(feature_columns, feature_map)

            run_dir = standard_dir / f"outer_fold_{outer_fold}" / model_name
            train_metrics, test_metrics, model_path = fit_and_save_model(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                args=args,
                run_dir=run_dir,
            )

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
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            training_manifest.append(metadata)

            print(
                f"[standard] outer_fold_{outer_fold} | {model_name}: "
                f"train RMSE={train_metrics['rmse']:.4f}, Pearson={train_metrics['pearson']:.4f} | "
                f"outer_test RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}"
            )

    manifest_path = standard_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    print(f"Saved standard training manifest: {manifest_path}")
    return manifest_path


def train_extra_models(args: argparse.Namespace, selected_outer_folds: List[int], extra_dir: Path) -> Path:
    extra_dir.mkdir(parents=True, exist_ok=True)
    training_manifest: List[Dict[str, Any]] = []
    folds_dir = Path(args.folds_dir)
    roberta_predictions_dir = Path(args.roberta_predictions_dir)

    for outer_fold in selected_outer_folds:
        outer_dir = fold_dir(folds_dir, outer_fold)
        outer_train_df = pd.read_csv(outer_dir / "outer_train.csv")
        outer_test_df = pd.read_csv(outer_dir / "outer_test.csv")

        outer_pred_dir = roberta_predictions_dir / f"outer_fold_{outer_fold}"
        train_pred_path = outer_pred_dir / "outer_train_oof_predictions.csv"
        test_pred_path = outer_pred_dir / "outer_test_predictions.csv"

        train_df = merge_roberta_scores(outer_train_df, train_pred_path)
        test_df = merge_roberta_scores(outer_test_df, test_pred_path)

        feature_map = build_feature_map(train_df)
        specs = make_extra_specs(feature_map, args.feature_limit)

        for spec in specs:
            model_name = str(spec["model_name"])
            feature_columns = list(spec["feature_columns"])
            categorical_columns = categorical_columns_for_features(feature_columns, feature_map)

            run_dir = extra_dir / f"outer_fold_{outer_fold}" / model_name
            train_metrics, test_metrics, model_path = fit_and_save_model(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                args=args,
                run_dir=run_dir,
            )

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
                f"[extra] outer_fold_{outer_fold} | {model_name}: "
                f"train RMSE={train_metrics['rmse']:.4f}, Pearson={train_metrics['pearson']:.4f} | "
                f"outer_test RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}"
            )

    manifest_path = extra_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    print(f"Saved extra training manifest: {manifest_path}")
    return manifest_path


def build_prediction_frame(
    merged_df: pd.DataFrame,
    preds: np.ndarray,
    outer_fold: int,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
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


def predict_from_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    output_file: Path,
    baseline_model_name: str,
) -> Path:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pred_frames: List[pd.DataFrame] = []
    folds_dir = Path(args.folds_dir)
    roberta_predictions_dir = Path(args.roberta_predictions_dir)

    folds_to_score = sorted({int(run["outer_fold"]) for run in manifest}) if manifest else OUTER_FOLDS
    for outer_fold in folds_to_score:
        outer_dir = fold_dir(folds_dir, outer_fold)
        outer_pred_dir = roberta_predictions_dir / f"outer_fold_{outer_fold}"

        split_inputs = [
            ("outer_train_fit", outer_dir / "outer_train.csv", outer_pred_dir / "outer_train_oof_predictions.csv"),
            ("outer_test", outer_dir / "outer_test.csv", outer_pred_dir / "outer_test_predictions.csv"),
        ]

        for split_name, csv_path, roberta_pred_path in split_inputs:
            base_df = pd.read_csv(csv_path)
            merged_df = merge_roberta_scores(base_df, roberta_pred_path)
            baseline_preds = merged_df[ROBERTA_SCORE_COLUMN].to_numpy(dtype=float)
            pred_frames.append(
                build_prediction_frame(merged_df, baseline_preds, outer_fold, split_name, baseline_model_name)
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

        outer_dir = fold_dir(folds_dir, outer_fold)
        outer_pred_dir = roberta_predictions_dir / f"outer_fold_{outer_fold}"
        split_inputs = [
            ("outer_train_fit", outer_dir / "outer_train.csv", outer_pred_dir / "outer_train_oof_predictions.csv"),
            ("outer_test", outer_dir / "outer_test.csv", outer_pred_dir / "outer_test_predictions.csv"),
        ]

        needs_roberta_merge = ROBERTA_SCORE_COLUMN in set(feature_columns)

        for split_name, csv_path, roberta_pred_path in split_inputs:
            base_df = pd.read_csv(csv_path)
            merged_df = merge_roberta_scores(base_df, roberta_pred_path) if needs_roberta_merge else base_df

            x = cast_model_columns(merged_df, feature_columns, categorical_columns)
            preds = model.predict(x)
            pred_frames.append(build_prediction_frame(merged_df, preds, outer_fold, split_name, model_name))

    if not pred_frames:
        raise RuntimeError("No predictions were generated.")

    all_preds = pd.concat(pred_frames, ignore_index=True)
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_preds.to_csv(output_file, index=False)

    print(f"Saved predictions: {output_file}")
    print(f"Rows: {len(all_preds)} | Models: {all_preds['model_name'].nunique()}")
    return output_file


def run_eval_scripts(
    args: argparse.Namespace,
    standard_pred_file: Path,
    extra_pred_file: Path,
    standard_dir: Path,
    extra_dir: Path,
) -> None:
    models_dir = Path(__file__).resolve().parent

    standard_json = standard_dir / "lightgbm_metrics_summary_l1.json"
    extra_json = extra_dir / "lightgbm_extra_window_metrics_l1.json"
    extra_figure = Path(__file__).resolve().parents[1] / "results" / "figures" / "levenshtein_window_extra_features_l1.png"

    cmd_standard = [
        sys.executable,
        str(models_dir / "evaluate_lightgbm.py"),
        "--predictions-file",
        str(standard_pred_file),
        "--full-dataset-file",
        str(args.full_dataset_file),
        "--output-json",
        str(standard_json),
    ]

    cmd_extra = [
        sys.executable,
        str(models_dir / "evaluate_lightgbm_extra.py"),
        "--predictions-file",
        str(extra_pred_file),
        "--output-json",
        str(extra_json),
        "--output-figure",
        str(extra_figure),
        "--baseline-model",
        EXTRA_BASELINE_MODEL,
    ]
    if args.extra_no_show:
        cmd_extra.append("--no-show")

    print("\nRunning evaluate_lightgbm.py...")
    subprocess.run(cmd_standard, check=True)

    print("\nRunning evaluate_lightgbm_extra.py...")
    subprocess.run(cmd_extra, check=True)

    print(f"Saved standard evaluation JSON: {standard_json}")
    print(f"Saved extra evaluation JSON: {extra_json}")
    print(f"Saved extra evaluation figure: {extra_figure}")


def main() -> None:
    args = parse_args()
    selected_outer_folds = parse_outer_folds(args.outer_folds)

    set_seed(args.seed)

    output_root = args.output_root
    standard_dir = output_root / "lightgbm_models_l1"
    extra_dir = output_root / "lightgbm_extra_models_l1"

    print("Training standard LightGBM family with L1 included...")
    standard_manifest = train_standard_models(args, selected_outer_folds, standard_dir)

    print("\nTraining extra LightGBM family with L1 included...")
    extra_manifest = train_extra_models(args, selected_outer_folds, extra_dir)

    print("\nPredicting standard LightGBM family...")
    standard_pred_file = predict_from_manifest(
        args=args,
        manifest_path=standard_manifest,
        output_file=standard_dir / "lightgbm_all_predictions_l1.csv",
        baseline_model_name=STANDARD_BASELINE_MODEL,
    )

    print("\nPredicting extra LightGBM family...")
    extra_pred_file = predict_from_manifest(
        args=args,
        manifest_path=extra_manifest,
        output_file=extra_dir / "lightgbm_extra_all_predictions_l1.csv",
        baseline_model_name=EXTRA_BASELINE_MODEL,
    )

    if args.skip_eval:
        print("\nSkipping evaluation scripts (--skip-eval).")
        return

    print("\nEvaluating both families...")
    run_eval_scripts(args, standard_pred_file, extra_pred_file, standard_dir, extra_dir)


if __name__ == "__main__":
    main()

