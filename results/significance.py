#!/usr/bin/env python3
"""Significance tests for key final LightGBM model comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from lightgbm_results_utils import ID_COLUMN, L1_COLUMN, LABEL_COLUMN, LEVENSHTEIN_COLUMN, ROW_KEY_COLUMNS
except ModuleNotFoundError:
    from results.lightgbm_results_utils import ID_COLUMN, L1_COLUMN, LABEL_COLUMN, LEVENSHTEIN_COLUMN, ROW_KEY_COLUMNS

DEFAULT_PREDICTIONS_FILE = PROJECT_ROOT / "models" / "lightgbm_final_models" / "lightgbm_final_predictions.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "results" / "lightgbm_significance.json"
DEFAULT_ROBERTA_PREDICTIONS_DIR = PROJECT_ROOT / "models" / "roberta_predictions"

SPLIT_COLUMN = "split"
MODEL_COLUMN = "model_name"
PREDICTION_COLUMN = "prediction"
OUTER_TEST_SPLIT = "outer_test"
ROBERTA_SCORE_COLUMN = "roberta_base_score"
CHAR_EDIT_RANGE_LOW = 0.6
CHAR_EDIT_RANGE_HIGH = 1.0
HOLM_EXCLUDED_BASE_MODELS = {"B"}

# Pairwise comparisons evaluated with paired permutation/bootstrap tests.
COMPARISONS: List[Tuple[str, str]] = [
    ("B", "BR"),
    ("B", "BR_no_L1"),
    ("L1_only", "E"),
    ("BL1", "BR"),
    ("E", "ER"),
    ("S", "SR"),
    ("ES", "ESR"),
    ("BE", "BER"),
    ("BS", "BSR"),
    ("BES", "BESR"),
]

SUBSETS: List[Tuple[str, Optional[str]]] = [
    ("overall", None),
    ("de", "de"),
    ("es", "es"),
    ("cn", "cn"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Significance tests for final LightGBM model comparisons.")
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--roberta-predictions-dir", type=Path, default=DEFAULT_ROBERTA_PREDICTIONS_DIR)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--chunk-size", type=int, default=2000)
    return parser.parse_args()


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return f"{float(value):.4f}"


def _validate_predictions_frame(pred_df: pd.DataFrame) -> None:
    required = {ID_COLUMN, L1_COLUMN, LABEL_COLUMN, PREDICTION_COLUMN, SPLIT_COLUMN, MODEL_COLUMN}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")


def _load_roberta_baseline_rows(roberta_predictions_dir: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fold_dir in sorted(roberta_predictions_dir.glob("outer_fold_*")):
        file_path = fold_dir / "outer_test_predictions.csv"
        if not file_path.exists():
            continue
        fold_df = pd.read_csv(file_path, keep_default_na=False)
        required_cols = set(ROW_KEY_COLUMNS + [LABEL_COLUMN, ROBERTA_SCORE_COLUMN, LEVENSHTEIN_COLUMN])
        missing = required_cols - set(fold_df.columns)
        if missing:
            raise ValueError(f"RoBERTa predictions file is missing required columns {sorted(missing)}: {file_path}")
        fold_df[PREDICTION_COLUMN] = fold_df[ROBERTA_SCORE_COLUMN].astype(float)
        fold_df[SPLIT_COLUMN] = OUTER_TEST_SPLIT
        fold_df[MODEL_COLUMN] = "B"
        parts.append(
            fold_df[
                ROW_KEY_COLUMNS
                + [LEVENSHTEIN_COLUMN, LABEL_COLUMN, PREDICTION_COLUMN, SPLIT_COLUMN, MODEL_COLUMN]
            ]
        )

    if not parts:
        return pd.DataFrame(
            columns=ROW_KEY_COLUMNS + [LEVENSHTEIN_COLUMN, LABEL_COLUMN, PREDICTION_COLUMN, SPLIT_COLUMN, MODEL_COLUMN]
        )

    combined = pd.concat(parts, ignore_index=True)
    dup = int(combined.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if dup:
        raise ValueError(f"RoBERTa baseline rows contain duplicate keys: {dup}")
    return combined


def _paired_rows(
    pred_df: pd.DataFrame,
    base_model: str,
    r_model: str,
    language: Optional[str],
    char_edit_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    subset = pred_df[pred_df[SPLIT_COLUMN] == OUTER_TEST_SPLIT].copy()
    if language is not None:
        subset = subset[subset[L1_COLUMN] == language].copy()
    if char_edit_range is not None:
        low, high = char_edit_range
        if LEVENSHTEIN_COLUMN not in subset.columns:
            raise ValueError(
                f"Missing column '{LEVENSHTEIN_COLUMN}' required for range-restricted significance tests"
            )
        # Convert once here so both model slices are filtered consistently.
        subset[LEVENSHTEIN_COLUMN] = pd.to_numeric(subset[LEVENSHTEIN_COLUMN], errors="coerce")
        subset = subset[subset[LEVENSHTEIN_COLUMN].between(float(low), float(high), inclusive="both")].copy()

    base_df = subset[subset[MODEL_COLUMN] == base_model][ROW_KEY_COLUMNS + [LABEL_COLUMN, PREDICTION_COLUMN]].copy()
    r_df = subset[subset[MODEL_COLUMN] == r_model][ROW_KEY_COLUMNS + [LABEL_COLUMN, PREDICTION_COLUMN]].copy()

    dup_base = int(base_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    dup_r = int(r_df.duplicated(ROW_KEY_COLUMNS).to_numpy().sum())
    if dup_base or dup_r:
        raise ValueError(
            f"Duplicate prediction rows found for comparison {base_model} vs {r_model} "
            f"(base={dup_base}, with_r={dup_r})"
        )

    merged = base_df.merge(
        r_df,
        on=ROW_KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_with_r"),
    )

    if merged.empty:
        return merged

    y_base = merged[f"{LABEL_COLUMN}_base"].to_numpy(dtype=float)
    y_with_r = merged[f"{LABEL_COLUMN}_with_r"].to_numpy(dtype=float)
    if not np.allclose(y_base, y_with_r, rtol=0.0, atol=1e-10, equal_nan=True):
        raise ValueError(f"Label mismatch after pairing rows for {base_model} vs {r_model}")

    merged["_base_model_used"] = base_model
    merged["_with_r_model_used"] = r_model
    return merged


def _sign_flip_pvalues(
    deltas: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> Tuple[float, float]:
    obs = float(np.mean(deltas))
    obs_abs = abs(obs)

    ge_one_sided = 0
    ge_two_sided = 0
    done = 0
    n = int(deltas.size)

    while done < n_permutations:
        k = min(chunk_size, n_permutations - done)
        signs = rng.integers(0, 2, size=(k, n), dtype=np.int8)
        signs = (signs * 2) - 1
        perm_means = (signs * deltas).mean(axis=1)
        ge_one_sided += int(np.count_nonzero(perm_means >= obs))
        ge_two_sided += int(np.count_nonzero(np.abs(perm_means) >= obs_abs))
        done += k

    p_one_sided = float((ge_one_sided + 1) / (n_permutations + 1))
    p_two_sided = float((ge_two_sided + 1) / (n_permutations + 1))
    return p_one_sided, p_two_sided


def _bootstrap_ci_mean(
    deltas: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> Tuple[float, float]:
    n = int(deltas.size)
    draws: List[np.ndarray] = []
    done = 0

    while done < n_bootstrap:
        k = min(chunk_size, n_bootstrap - done)
        idx = rng.integers(0, n, size=(k, n))
        means = deltas[idx].mean(axis=1)
        draws.append(means)
        done += k

    sampled_means = np.concatenate(draws)
    low, high = np.percentile(sampled_means, [2.5, 97.5])
    return float(low), float(high)


def _holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    m = int(p.size)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)
    for i, val in enumerate(sorted_p):
        adjusted_sorted[i] = (m - i) * val
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def _bh_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    m = int(p.size)
    order = np.argsort(p)
    sorted_p = p[order]

    ranks = np.arange(1, m + 1, dtype=float)
    adjusted_sorted = (m / ranks) * sorted_p
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def _compute_one_test(
    pred_df: pd.DataFrame,
    base_model: str,
    r_model: str,
    subset_name: str,
    language: Optional[str],
    n_permutations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk_size: int,
    char_edit_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    paired = _paired_rows(
        pred_df,
        base_model=base_model,
        r_model=r_model,
        language=language,
        char_edit_range=char_edit_range,
    )

    payload: Dict[str, Any] = {
        "subset": subset_name,
        "language": language,
        "base_model": base_model,
        "with_r_model": r_model,
        "n_rows": int(len(paired)),
    }

    payload["base_model_used"] = str(paired["_base_model_used"].iloc[0]) if not paired.empty else str(base_model)
    payload["with_r_model_used"] = str(paired["_with_r_model_used"].iloc[0]) if not paired.empty else str(r_model)

    if paired.empty:
        payload.update(
            {
                "status": "missing",
                "rmse_base": None,
                "rmse_with_r": None,
                "rmse_gain": None,
                "mean_delta_mse": None,
                "delta_mse_ci95": [None, None],
                "p_value_one_sided": None,
                "p_value_two_sided": None,
            }
        )
        return payload

    y = paired[f"{LABEL_COLUMN}_base"].to_numpy(dtype=float)
    pred_base = paired[f"{PREDICTION_COLUMN}_base"].to_numpy(dtype=float)
    pred_with_r = paired[f"{PREDICTION_COLUMN}_with_r"].to_numpy(dtype=float)

    se_base = (y - pred_base) ** 2
    se_with_r = (y - pred_with_r) ** 2
    deltas = se_base - se_with_r

    rmse_base = float(np.sqrt(np.mean(se_base)))
    rmse_with_r = float(np.sqrt(np.mean(se_with_r)))
    rmse_gain = float(rmse_base - rmse_with_r)
    mean_delta = float(np.mean(deltas))

    p_one, p_two = _sign_flip_pvalues(
        deltas=deltas,
        n_permutations=n_permutations,
        rng=rng,
        chunk_size=chunk_size,
    )
    ci_low, ci_high = _bootstrap_ci_mean(
        deltas=deltas,
        n_bootstrap=n_bootstrap,
        rng=rng,
        chunk_size=chunk_size,
    )

    payload.update(
        {
            "status": "ok",
            "rmse_base": rmse_base,
            "rmse_with_r": rmse_with_r,
            "rmse_gain": rmse_gain,
            "mean_delta_mse": mean_delta,
            "delta_mse_ci95": [ci_low, ci_high],
            "p_value_one_sided": p_one,
            "p_value_two_sided": p_two,
        }
    )
    return payload


def _apply_multiple_testing_corrections(tests: List[Dict[str, Any]], alpha: float) -> None:
    valid_indices = [i for i, row in enumerate(tests) if row["status"] == "ok" and row["p_value_one_sided"] is not None]
    holm_indices = [i for i in valid_indices if str(tests[i].get("base_model")) not in HOLM_EXCLUDED_BASE_MODELS]
    raw_p = [float(tests[i]["p_value_one_sided"]) for i in holm_indices]

    holm = _holm_adjust(raw_p) if raw_p else []
    bh = _bh_adjust(raw_p) if raw_p else []

    for pos, idx in enumerate(holm_indices):
        tests[idx]["p_value_one_sided_holm"] = holm[pos]
        tests[idx]["p_value_one_sided_bh"] = bh[pos]

    valid_index_set = set(valid_indices)
    holm_index_set = set(holm_indices)
    for i, row in enumerate(tests):
        base_model = str(row.get("base_model"))
        holm_eligible = base_model not in HOLM_EXCLUDED_BASE_MODELS
        row["holm_eligible"] = bool(holm_eligible)
        row["correction_scope"] = "holm_over_non_B" if holm_eligible else "raw_only"

        if i not in holm_index_set:
            row["p_value_one_sided_holm"] = None
            row["p_value_one_sided_bh"] = None
        row["significant_holm"] = bool(
            i in holm_index_set
            and row["status"] == "ok"
            and row["p_value_one_sided_holm"] is not None
            and float(row["p_value_one_sided_holm"]) < float(alpha)
            and float(row["mean_delta_mse"]) > 0.0
        )
        row["significant_raw"] = bool(
            i in valid_index_set
            and row["p_value_one_sided"] is not None
            and float(row["p_value_one_sided"]) < float(alpha)
            and float(row["mean_delta_mse"]) > 0.0
        )


def _run_experiment(
    pred_df: pd.DataFrame,
    n_permutations: int,
    n_bootstrap: int,
    alpha: float,
    chunk_size: int,
    seed: int,
    char_edit_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    tests: List[Dict[str, Any]] = []
    for base_model, r_model in COMPARISONS:
        for subset_name, language in SUBSETS:
            tests.append(
                _compute_one_test(
                    pred_df=pred_df,
                    base_model=base_model,
                    r_model=r_model,
                    subset_name=subset_name,
                    language=language,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                    chunk_size=chunk_size,
                    char_edit_range=char_edit_range,
                )
            )
    _apply_multiple_testing_corrections(tests=tests, alpha=alpha)
    return tests


def compute_significance(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.predictions_file.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {args.predictions_file}. Run predict_lightgbm_final.py first."
        )

    pred_df = pd.read_csv(args.predictions_file, keep_default_na=False)

    # Add raw RoBERTa baseline as model B when available.
    present_models = set(pred_df[MODEL_COLUMN].astype(str).unique().tolist()) if MODEL_COLUMN in pred_df.columns else set()
    if "B" not in present_models:
        roberta_b_df = _load_roberta_baseline_rows(args.roberta_predictions_dir)
        if not roberta_b_df.empty:
            pred_df = pd.concat([pred_df, roberta_b_df], ignore_index=True, sort=False)

    _validate_predictions_frame(pred_df)

    tests = _run_experiment(
        pred_df=pred_df,
        n_permutations=int(args.n_permutations),
        n_bootstrap=int(args.n_bootstrap),
        alpha=float(args.alpha),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
        char_edit_range=None,
    )
    range_tests = _run_experiment(
        pred_df=pred_df,
        n_permutations=int(args.n_permutations),
        n_bootstrap=int(args.n_bootstrap),
        alpha=float(args.alpha),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
        char_edit_range=(CHAR_EDIT_RANGE_LOW, CHAR_EDIT_RANGE_HIGH),
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "predictions_file": str(args.predictions_file),
        "n_permutations": int(args.n_permutations),
        "n_bootstrap": int(args.n_bootstrap),
        "alpha": float(args.alpha),
        "correction": "holm over non-B contrasts only (primary), bh/fdr over non-B contrasts only (secondary)",
        "holm_excluded_base_models": sorted(HOLM_EXCLUDED_BASE_MODELS),
        "comparisons": [{"base": a, "with_r": b} for a, b in COMPARISONS],
        "subsets": [{"name": name, "language": language} for name, language in SUBSETS],
        "tests": tests,
        "char_edit_dist_norm_range_experiment": {
            "range": [CHAR_EDIT_RANGE_LOW, CHAR_EDIT_RANGE_HIGH],
            "correction": "holm over non-B contrasts only (primary), bh/fdr over non-B contrasts only (secondary); computed independently from the full-data experiment",
            "holm_excluded_base_models": sorted(HOLM_EXCLUDED_BASE_MODELS),
            "tests": range_tests,
        },
    }


def _print_results_block(tests: List[Dict[str, Any]], title: str) -> None:
    df = pd.DataFrame(tests)
    print(f"\n=== {title} ===")
    if df.empty:
        print("No tests available.")
        return

    display_cols = [
        "subset",
        "base_model",
        "with_r_model",
        "n_rows",
        "rmse_base",
        "rmse_with_r",
        "rmse_gain",
        "mean_delta_mse",
        "p_value_one_sided",
        "p_value_one_sided_holm",
        "significant_holm",
    ]

    for col in display_cols:
        if col not in df.columns:
            df[col] = np.nan

    printable = df.copy()
    metric_cols = [
        "rmse_base",
        "rmse_with_r",
        "rmse_gain",
        "mean_delta_mse",
        "p_value_one_sided",
        "p_value_one_sided_holm",
    ]
    for col in metric_cols:
        printable[col] = printable[col].map(_fmt_metric)
    printable["n_rows"] = printable["n_rows"].fillna(0).astype(int)

    print(printable[display_cols].to_string(index=False))


def print_results(payload: Dict[str, Any]) -> None:
    _print_results_block(payload.get("tests", []), "Significance tests (all outer-test rows)")

    range_payload = payload.get("char_edit_dist_norm_range_experiment", {})
    range_vals = range_payload.get("range", [CHAR_EDIT_RANGE_LOW, CHAR_EDIT_RANGE_HIGH])
    low = float(range_vals[0])
    high = float(range_vals[1])
    _print_results_block(
        range_payload.get("tests", []),
        f"Significance tests (char_edit_dist_norm in [{low:.1f}, {high:.1f}])",
    )


def main() -> None:
    args = parse_args()
    payload = compute_significance(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_results(payload)
    print(f"\nSaved significance results: {args.output_json}")


if __name__ == "__main__":
    main()

