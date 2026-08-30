#!/usr/bin/env python3
"""Evaluate LightGBM CV predictions per model, L1, and optionally original test subsets."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from roberta_cv_utils import ID_COLUMN, rmse_pearson

OUTER_FOLDS = [1, 2, 3, 4, 5]
L1_COLUMN = "L1"
LEVENSHTEIN_COLUMN = "char_edit_dist_norm"
ORIGINAL_SPLIT_COLUMN = "original_split"
ORIGINAL_TEST_SPLIT = "test"
ROW_KEY_COLUMNS = [ID_COLUMN, L1_COLUMN]
FEATURE_GROUP_MODEL_PREFIX = "group__"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent
FULL_DATASET_FILE = PROJECT_ROOT / "data" / "feature_enriched" / "full.csv"
DEFAULT_PREDICTIONS_FILE = MODELS_DIR / "lightgbm_models" / "lightgbm_all_predictions.csv"
DEFAULT_OUTPUT_JSON = MODELS_DIR / "lightgbm_models" / "lightgbm_metrics_summary.json"

ORIGINAL_TEST_CONFIGS = {
    "group__established_relatedness_similarity": {
        "label": "BESR",
        "languages": ["es", "de"],
    },
    "group__established_relatedness": {
        "label": "BER",
        "languages": ["cn"],
    },
}

# Overlapping sliding windows: [0.0-0.2, 0.1-0.3, ..., 0.8-1.0]
LEVENSHTEIN_WINDOWS: List[Tuple[float, float]] = [
    (round(float(lo), 1), round(float(lo) + 0.2, 1))
    for lo in np.arange(0.0, 0.9, 0.1)
]
MIN_WINDOW_ROWS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM CV predictions.")
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--full-dataset-file", type=Path, default=FULL_DATASET_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
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


def is_feature_group_model(model_name: str) -> bool:
    return str(model_name).startswith(FEATURE_GROUP_MODEL_PREFIX)


def attach_original_split(pred_df: pd.DataFrame, full_dataset_file: Path) -> pd.DataFrame:
    if ORIGINAL_SPLIT_COLUMN in pred_df.columns:
        return pred_df

    if not full_dataset_file.exists():
        raise FileNotFoundError(
            f"Missing full dataset file: {full_dataset_file}. Run data/feature_enriched/generate_full_dataset.py first."
        )

    full_df = pd.read_csv(full_dataset_file, usecols=ROW_KEY_COLUMNS + [ORIGINAL_SPLIT_COLUMN], keep_default_na=False)
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

    original_split_series = merged[ORIGINAL_SPLIT_COLUMN]
    if original_split_series.isna().any():
        missing = int(np.count_nonzero(original_split_series.isna().to_numpy()))
        raise ValueError(
            f"Could not attach '{ORIGINAL_SPLIT_COLUMN}' to {missing} prediction rows from {full_dataset_file}"
        )

    return merged


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


def original_test_metrics(model_name: str, model_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    config = ORIGINAL_TEST_CONFIGS.get(model_name)
    if config is None:
        return None

    if ORIGINAL_SPLIT_COLUMN not in model_df.columns:
        raise ValueError(
            f"Missing required column: {ORIGINAL_SPLIT_COLUMN}. Merge predictions with full.csv first."
        )

    subset = model_df[(model_df["split"] == "outer_test") & (model_df[ORIGINAL_SPLIT_COLUMN] == ORIGINAL_TEST_SPLIT)]
    subset = subset[subset[L1_COLUMN].isin(config["languages"])]
    if subset.empty:
        return None

    return {
        "label": config["label"],
        "model_name": model_name,
        "original_split": ORIGINAL_TEST_SPLIT,
        "languages": list(config["languages"]),
        "per_l1": metrics_by_l1(subset, include_windows=False),
    }


def metrics_by_l1(df: pd.DataFrame, include_windows: bool) -> Dict[str, Dict]:
    if L1_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {L1_COLUMN}")

    result: Dict[str, Dict[str, Any]] = {}
    for l1, group_df in df.groupby(L1_COLUMN, dropna=False):
        l1_key = "NA" if pd.isna(l1) else str(l1)
        payload: Dict[str, Any] = {
            **metrics_from_df(group_df),
            "n_rows": int(len(group_df)),
            "n_unique_item_id": int(group_df[ID_COLUMN].nunique()),
        }

        if include_windows:
            windows = {}
            for lo, hi in LEVENSHTEIN_WINDOWS:
                window_key = f"{lo:.1f}-{hi:.1f}"
                window_metrics = metrics_for_window(group_df, lo, hi)
                if window_metrics is not None:
                    windows[window_key] = window_metrics

            payload["levenshtein_windows"] = windows

        result[l1_key] = payload

    return result


def evaluate_model(model_name: str, model_df: pd.DataFrame) -> Dict:
    include_windows = is_feature_group_model(model_name)
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
                "per_l1": metrics_by_l1(split_df, include_windows=include_windows),
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
            "per_l1": metrics_by_l1(split_df, include_windows=include_windows),
        }

    return {
        "per_fold": per_fold,
        "aggregate": aggregate,
    }


def _fmt_metric(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.4f}"


def build_console_summary(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for model_name, model_summary in summary["models"].items():
        aggregate = cast(Dict[str, Any], model_summary.get("aggregate", {}))
        outer_test = cast(Dict[str, Any], aggregate.get("outer_test", {}))

        row: Dict[str, Any] = {
            "model": model_name,
            "outer_test_rmse": outer_test.get("rmse"),
            "outer_test_pearson": outer_test.get("pearson"),
            "outer_test_n": outer_test.get("n_rows"),
        }

        rows.append(row)

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values("model").reset_index(drop=True)
    return report_df


def print_console_report(summary: Dict[str, Any]) -> None:
    report_df = build_console_summary(summary)

    print("\n=== LightGBM evaluation summary ===")
    if report_df.empty:
        print("No models found in the predictions file.")
        return

    display_df = report_df.assign(
        outer_test_rmse=report_df["outer_test_rmse"].map(_fmt_metric),
        outer_test_pearson=report_df["outer_test_pearson"].map(_fmt_metric),
        outer_test_n=report_df["outer_test_n"].fillna(0).astype(int),
    )

    cols = ["model", "outer_test_rmse", "outer_test_pearson", "outer_test_n"]
    print(display_df[cols].to_string(index=False))

    print("\n=== BER / BESR language-subset results ===")
    any_original = False
    for model_name, model_summary in summary["models"].items():
        original = cast(Optional[Dict[str, Any]], model_summary.get("original_shared_task_test"))
        if original is None:
            continue

        any_original = True
        for l1, l1_metrics in original.get("per_l1", {}).items():
            print(
                f"{original['label']} ({model_name}) L1={l1}: RMSE={_fmt_metric(l1_metrics.get('rmse'))}, "
                f"Pearson={_fmt_metric(l1_metrics.get('pearson'))}, n={int(l1_metrics.get('n_rows', 0))}"
            )

    if not any_original:
        print("No BESR/BER original-test subsets were found in the predictions.")

    ablation_models = [
        model_name
        for model_name in summary["models"].keys()
        if str(model_name).startswith("ablation__")
    ]
    print(
        f"\n=== Leave-one-out ablation outer-test L1 subset results ({len(ablation_models)} models) ==="
    )
    any_ablation = False
    for model_name in sorted(ablation_models):
        model_summary = summary["models"][model_name]
        aggregate = cast(Dict[str, Any], model_summary.get("aggregate", {}))
        outer_test = cast(Dict[str, Any], aggregate.get("outer_test", {}))
        per_l1 = cast(Dict[str, Dict[str, Any]], outer_test.get("per_l1", {}))

        if not per_l1:
            continue

        any_ablation = True
        print(model_name)
        for l1, l1_metrics in per_l1.items():
            print(
                f"  L1={l1}: RMSE={_fmt_metric(l1_metrics.get('rmse'))}, "
                f"Pearson={_fmt_metric(l1_metrics.get('pearson'))}, n={int(l1_metrics.get('n_rows', 0))}"
            )

    if not any_ablation:
        print("No leave-one-out ablation models were found in the predictions file.")

    print("\n=== Levenshtein window analysis by feature-group config ===")
    any_windows = False
    for model_name, model_summary in summary["models"].items():
        aggregate = cast(Dict[str, Any], model_summary.get("aggregate", {}))
        outer_test = cast(Dict[str, Any], aggregate.get("outer_test", {}))
        per_l1 = cast(Dict[str, Dict[str, Any]], outer_test.get("per_l1", {}))

        has_windows = False
        for l1_metrics in per_l1.values():
            if l1_metrics.get("levenshtein_windows"):
                has_windows = True
                break

        if not has_windows:
            continue

        any_windows = True
        print(model_name)
        for l1, l1_metrics in per_l1.items():
            windows = cast(Dict[str, Dict[str, Any]], l1_metrics.get("levenshtein_windows", {}))
            if not windows:
                continue

            print(
                f"  L1={l1}: RMSE={_fmt_metric(l1_metrics.get('rmse'))}, "
                f"Pearson={_fmt_metric(l1_metrics.get('pearson'))}, n={int(l1_metrics.get('n_rows', 0))}"
            )
            for window_key, wm in windows.items():
                print(
                    f"    levenshtein [{window_key}]: RMSE={_fmt_metric(wm.get('rmse'))}, "
                    f"Pearson={_fmt_metric(wm.get('pearson'))}, n={int(wm.get('n_rows', 0))}"
                )

    if not any_windows:
        print("No Levenshtein window analysis was available in the current predictions file.")


def main() -> None:
    args = parse_args()

    if not args.predictions_file.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {args.predictions_file}. Run predict_lightgbm.py first."
        )

    pred_df = pd.read_csv(args.predictions_file, keep_default_na=False)
    pred_df = attach_original_split(pred_df, args.full_dataset_file)

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
        "full_dataset_file": str(args.full_dataset_file),
        "models": {},
    }

    for model_name, model_df in pred_df.groupby("model_name", dropna=False):
        model_key = str(model_name)
        model_summary = evaluate_model(model_key, model_df)

        original_test_summary = original_test_metrics(model_key, model_df)
        if original_test_summary is not None:
            model_summary["original_shared_task_test"] = original_test_summary

        summary["models"][model_key] = model_summary

    print_console_report(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved LightGBM evaluation summary: {args.output_json}")


if __name__ == "__main__":
    main()


