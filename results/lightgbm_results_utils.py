#!/usr/bin/env python3
"""Shared constants and helpers for final LightGBM evaluation/plotting scripts."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

ID_COLUMN = "item_id"
LABEL_COLUMN = "GLMM_score"
L1_COLUMN = "L1"
LEVENSHTEIN_COLUMN = "char_edit_dist_norm"
ROW_KEY_COLUMNS = [ID_COLUMN, L1_COLUMN]

OUTER_FOLDS = [1, 2, 3, 4, 5]
LANGUAGE_SPECS = [
    ("de", "German"),
    ("es", "Spanish"),
    ("cn", "Mandarin"),
]

WINDOW_LOWS = np.arange(0.0, 0.9, 0.1)
WINDOW_WIDTH = 0.2
WINDOW_CENTERS = np.round(WINDOW_LOWS + (WINDOW_WIDTH / 2.0), 1)
MIN_WINDOW_ROWS = 50


def rmse_pearson(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    if y_true.size == 0:
        return {"rmse": float("nan"), "pearson": float("nan")}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if y_true.size < 2:
        return {"rmse": rmse, "pearson": float("nan")}

    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return {"rmse": rmse, "pearson": float("nan")}

    return {"rmse": rmse, "pearson": float(np.corrcoef(y_true, y_pred)[0, 1])}


def metrics_from_frame(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict[str, float]:
    if df.empty:
        return {"rmse": float("nan"), "pearson": float("nan")}
    return rmse_pearson(df[label_col].to_numpy(dtype=float), df[pred_col].to_numpy(dtype=float))


def compute_sliding_window_metrics(
    df: pd.DataFrame,
    min_window_rows: int,
    feature_col: str,
    label_col: str,
    pred_col: str,
) -> List[Dict[str, float | int | None]]:
    rows: List[Dict[str, float | int | None]] = []
    for low in WINDOW_LOWS:
        low_f = float(low)
        high_f = float(low_f + WINDOW_WIDTH)
        center_f = float(low_f + (WINDOW_WIDTH / 2.0))

        window_df = df[(df[feature_col] >= low_f) & (df[feature_col] <= high_f)]
        count = int(len(window_df))
        if count < min_window_rows:
            rows.append(
                {
                    "low": low_f,
                    "high": high_f,
                    "center": center_f,
                    "rmse": None,
                    "pearson": None,
                    "n_rows": count,
                }
            )
            continue

        metrics = rmse_pearson(
            window_df[label_col].to_numpy(dtype=float),
            window_df[pred_col].to_numpy(dtype=float),
        )
        rows.append(
            {
                "low": low_f,
                "high": high_f,
                "center": center_f,
                "rmse": float(metrics["rmse"]),
                "pearson": float(metrics["pearson"]),
                "n_rows": count,
            }
        )
    return rows


def parse_center_range(raw: str, label: str) -> tuple[float, float]:
    parts = raw.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} center range '{raw}'. Use low-high, e.g. 0.3-0.7")

    low = float(parts[0])
    high = float(parts[1])
    if low > high:
        raise ValueError(f"Invalid {label} center range '{raw}': low must be <= high")
    return low, high


def center_mask(range_low: float, range_high: float) -> np.ndarray:
    return (WINDOW_CENTERS >= range_low) & (WINDOW_CENTERS <= range_high)

