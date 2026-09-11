#!/usr/bin/env python3
"""Shared utilities for final LightGBM CV training/prediction/evaluation scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

ID_COLUMN = "item_id"
LABEL_COLUMN = "GLMM_score"
L1_COLUMN = "L1"
LEVENSHTEIN_COLUMN = "char_edit_dist_norm"
ROBERTA_SCORE_COLUMN = "roberta_base_score"

OUTER_FOLDS = [1, 2, 3, 4, 5]
ROW_KEY_COLUMNS = [ID_COLUMN, L1_COLUMN]
PREDICTION_METADATA_COLUMNS = [L1_COLUMN, "en_target_word", "L1_source_word", LEVENSHTEIN_COLUMN]

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

ALL_CANONICAL_FEATURES = (
    FEATURE_GROUPS["established"] + FEATURE_GROUPS["similarity"] + FEATURE_GROUPS["relatedness"]
)

CATEGORICAL_CANONICAL = {"pos_tag", "lca_language", "english_word_origin", "transmission_path"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for value in values:
        if value in seen:
            continue
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


def fold_dir(base_dir: Path, outer_fold: int) -> Path:
    return base_dir / f"outer_fold_{outer_fold}"


def resolve_feature_column(df: pd.DataFrame, canonical_feature: str) -> str:
    aliases = FEATURE_ALIASES[canonical_feature]
    for col in aliases:
        if col in df.columns:
            return col
    raise ValueError(f"Could not resolve feature '{canonical_feature}'. Tried aliases: {aliases}")


def build_feature_map(df: pd.DataFrame) -> Dict[str, str]:
    return {
        canonical: resolve_feature_column(df, canonical)
        for canonical in ALL_CANONICAL_FEATURES
    }


def resolve_feature_tokens(feature_tokens: Iterable[str], feature_map: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for token in feature_tokens:
        if token in (ROBERTA_SCORE_COLUMN, L1_COLUMN):
            out.append(token)
        else:
            if token not in feature_map:
                raise ValueError(f"Unknown feature token '{token}'.")
            out.append(feature_map[token])
    return [str(v) for v in unique_preserve_order(out)]


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


def categorical_columns_for_features(feature_columns: Iterable[str], feature_map: Dict[str, str]) -> List[str]:
    categorical_actual = {feature_map[c] for c in CATEGORICAL_CANONICAL}
    categorical_actual.add(L1_COLUMN)
    return [col for col in feature_columns if col in categorical_actual]


def cast_model_columns(df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()

    for col in categorical_columns:
        if col not in x.columns:
            raise ValueError(f"Categorical column '{col}' is missing from model frame.")
        x[col] = x[col].fillna("NA").astype(str).astype("category")

    for col in feature_columns:
        if col in categorical_columns:
            continue
        if x[col].dtype == bool:
            x[col] = x[col].astype(np.int8)

    return x


def rmse_pearson(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    n = int(y_true.shape[0])

    if n == 0:
        return {"rmse": float("nan"), "pearson": float("nan")}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if n < 2:
        return {"rmse": rmse, "pearson": float("nan")}

    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    if y_true_std == 0.0 or y_pred_std == 0.0:
        return {"rmse": rmse, "pearson": float("nan")}

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"rmse": rmse, "pearson": pearson}




