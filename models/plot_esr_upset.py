#!/usr/bin/env python3
"""Create UpSet-style overlap plots from regression predictions.

Correctness is defined item-wise as:
    abs(prediction - GLMM_score) <= tau
with tau computed per language subset as:
    tau = k * std(GLMM_score)

By default, this compares E/S/R config models. Pass --include-baseline to add B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from evaluate_lightgbm_extra import LABEL_COL, L1_COL, LANGUAGE_SPECS, MODEL_COL, OUTER_TEST_SPLIT, PRED_COL, SPLIT_COL
from roberta_cv_utils import ID_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_FILE = (
    Path(__file__).resolve().parent
    / "lightgbm_l1_experiment"
    / "lightgbm_extra_models_l1"
    / "lightgbm_extra_all_predictions_l1.csv"
)
DEFAULT_OUTPUT_FIGURE = PROJECT_ROOT / "results" / "figures" / "esr_correctness_upset.png"
DEFAULT_OUTPUT_JSON = (
    Path(__file__).resolve().parent
    / "lightgbm_l1_experiment"
    / "lightgbm_extra_models_l1"
    / "esr_correctness_upset_summary.json"
)

MODEL_E = "extra__config__established_only"
MODEL_S = "extra__config__similarity_only"
MODEL_R = "extra__config__relatedness_only"
MODEL_B = "extra__roberta_only"

MODEL_LABELS = {
    MODEL_E: "E",
    MODEL_S: "S",
    MODEL_R: "R",
    MODEL_B: "B",
}

SET_COLORS = {
    "B": "black",
    "E": "tab:blue",
    "S": "gold",
    "R": "tab:red",
}

LABEL_ORDER = ["B", "E", "S", "R"]

LANGUAGE_NAME_BY_CODE = {code: name for code, name in LANGUAGE_SPECS}
EPSILON = 1e-12


def _k_token(k: float) -> str:
    # Compact, filename-friendly numeric token (e.g., 0.033 -> 0.033).
    return f"{k:.6g}"


def output_path_with_k(path: Path, k: float) -> Path:
    """Return output path that includes k.

    If path contains '{k}', replace it. Otherwise append '_k<k>' before suffix.
    """
    k_token = _k_token(k)
    raw = str(path)
    if "{k}" in raw:
        return Path(raw.format(k=k_token))
    return path.with_name(f"{path.stem}_k{k_token}{path.suffix}")


def _combo_key(active_labels: list[str]) -> str:
    ordered = [label for label in LABEL_ORDER if label in set(active_labels)]
    return "".join(ordered)


def _blend_combo_color(active_labels: list[str]) -> str:
    rgb = np.array([mcolors.to_rgb(SET_COLORS[label]) for label in active_labels], dtype=float)
    mixed = rgb.mean(axis=0)
    rgb_tuple = (float(mixed[0]), float(mixed[1]), float(mixed[2]))
    return mcolors.to_hex(rgb_tuple)


def ordered_model_labels(models: list[str]) -> list[str]:
    present = {MODEL_LABELS[m] for m in models}
    return [label for label in LABEL_ORDER if label in present]


def combo_color(active_labels: list[str]) -> str:
    if not active_labels:
        return "lightgray"

    if len(active_labels) == 1:
        return SET_COLORS[active_labels[0]]

    # All multi-set intersections are blended from their active set colors.
    return _blend_combo_color(active_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create UpSet-style overlap plots for correctness of E/S/R models "
            "(optionally including baseline B)."
        )
    )
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_OUTPUT_FIGURE,
        help="Output figure path. '{k}' is supported; otherwise '_k<k>' is appended automatically.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path. '{k}' is supported; otherwise '_k<k>' is appended automatically.",
    )
    parser.add_argument("--k", type=float, default=0.50, help="Scale factor in tau = k * std(GLMM_score).")
    parser.add_argument(
        "--language",
        type=str,
        default="all",
        help="Use 'all' for one panel per language, or a specific L1 code (de, es, cn).",
    )
    parser.add_argument(
        "--include-baseline",
        default=True,
        help="Include baseline RoBERTa model (B) as a fourth set.",
    )
    parser.add_argument(
        "--include-none",
        default=False,
        help="Also show the 'None correct' intersection in UpSet bars.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save figure without opening a matplotlib window.")
    return parser.parse_args()


def selected_models(include_baseline: bool) -> list[str]:
    models = [MODEL_E, MODEL_S, MODEL_R]
    if include_baseline:
        models.append(MODEL_B)
    return models


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path, keep_default_na=False)
    required_cols = {ID_COLUMN, L1_COL, MODEL_COL, SPLIT_COL, LABEL_COL, PRED_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Predictions file is missing required columns: {sorted(missing_cols)}")

    df = df[df[SPLIT_COL] == OUTER_TEST_SPLIT].copy()
    if df.empty:
        raise ValueError("No outer_test rows found in predictions file.")
    return df


def filter_rows(df: pd.DataFrame, language: str, models: list[str]) -> pd.DataFrame:
    valid_languages = {"all", "de", "es", "cn"}
    if language not in valid_languages:
        raise ValueError(f"Invalid --language '{language}'. Choose one of: {sorted(valid_languages)}")

    out = df[df[MODEL_COL].isin(models)].copy()
    if language != "all":
        out = out[out[L1_COL] == language].copy()

    if out.empty:
        raise ValueError("No matching rows found after model/language filtering.")

    missing_models = [m for m in models if m not in set(out[MODEL_COL].astype(str).unique())]
    if missing_models:
        raise ValueError(f"Missing required models in predictions: {missing_models}")

    dup_count = int(out.duplicated([ID_COLUMN, L1_COL, MODEL_COL]).sum())
    if dup_count:
        raise ValueError(
            f"Found {dup_count} duplicate item-language-model rows; expected one row per item per model in outer_test."
        )
    return out


def build_subset_error_frame(subset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = subset_df.copy()
    x["abs_error"] = (x[PRED_COL].astype(float) - x[LABEL_COL].astype(float)).abs()

    err = x.pivot(index=[ID_COLUMN, L1_COL], columns=MODEL_COL, values="abs_error")
    complete = err.notna().all(axis=1)
    if not bool(complete.any()):
        raise ValueError("No complete aligned items across selected models.")

    err = err.loc[complete]

    label_nunique = x.groupby([ID_COLUMN, L1_COL])[LABEL_COL].nunique()
    bad = label_nunique[label_nunique > 1]
    if not bad.empty:
        raise ValueError("Found inconsistent gold labels for same item/L1 across models.")

    gold = x.groupby([ID_COLUMN, L1_COL])[LABEL_COL].first().astype(float).loc[complete]
    return err, gold


def intersection_summary(correct_flags: pd.DataFrame, models: list[str], include_none: bool) -> list[dict[str, Any]]:
    grouped = correct_flags.groupby(models).size().rename("count").reset_index()
    display_labels = ordered_model_labels(models)

    rows: list[dict[str, Any]] = []
    for _, rec in grouped.iterrows():
        active_set = {MODEL_LABELS[m] for m in models if bool(rec[m])}
        active = [label for label in display_labels if label in active_set]
        if not active and not include_none:
            continue
        rows.append(
            {
                "active_labels": active,
                "active_key": "∩".join(active) if active else "None",
                "count": int(rec["count"]),
                "degree": int(len(active)),
            }
        )

    rows.sort(key=lambda r: (-r["count"], -r["degree"], r["active_key"]))
    return rows


def analyze_language(subset_df: pd.DataFrame, models: list[str], k: float, include_none: bool) -> dict[str, Any]:
    err, gold = build_subset_error_frame(subset_df)
    sigma = float(np.std(gold.to_numpy(dtype=float), ddof=0))
    tau = float(k * sigma)

    flags = err[models].le(tau + EPSILON)
    set_sizes = {MODEL_LABELS[m]: int(flags[m].sum()) for m in models}
    intersections = intersection_summary(flags, models, include_none)

    return {
        "tau": tau,
        "label_std": sigma,
        "n_items_considered": int(len(err)),
        "set_sizes": set_sizes,
        "intersections": intersections,
    }


def draw_upset_panel(
    fig: plt.Figure,
    subfig: Any,
    language_title: str,
    subtitle: str,
    payload: dict[str, Any],
    models: list[str],
) -> None:
    gs = subfig.add_gridspec(2, 2, width_ratios=[1.25, 4.0], height_ratios=[3.0, 1.7], wspace=0.08, hspace=0.06)
    ax_blank = subfig.add_subplot(gs[0, 0])
    ax_inter = subfig.add_subplot(gs[0, 1])
    ax_sets = subfig.add_subplot(gs[1, 0])
    ax_matrix = subfig.add_subplot(gs[1, 1], sharex=ax_inter)

    ax_blank.axis("off")

    intersections = payload["intersections"]
    if not intersections:
        ax_inter.text(0.5, 0.5, "No intersections to plot", ha="center", va="center")
        ax_inter.axis("off")
        ax_matrix.axis("off")
    else:
        x = np.arange(len(intersections))
        y = np.array([row["count"] for row in intersections], dtype=int)
        colors = [combo_color(row["active_labels"]) for row in intersections]
        ax_inter.bar(x, y, color=colors, alpha=0.65)
        ax_inter.set_ylabel("Count", fontsize=10)
        ax_inter.grid(axis="y", linestyle="--", alpha=0.35)
        ax_inter.set_xticks([])
        # Shared x-axis with matrix: keep top labels hidden so intersections are labeled only once below.
        ax_inter.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for x_pos, y_val in zip(x, y):
            ax_inter.text(x_pos, y_val, str(int(y_val)), ha="center", va="bottom", fontsize=9)

        model_labels = ordered_model_labels(models)
        y_positions = np.arange(len(model_labels))
        ax_matrix.set_yticks(y_positions)
        ax_matrix.set_yticklabels(model_labels, fontsize=10)
        ax_matrix.grid(axis="x", linestyle="--", alpha=0.2)

        for i, row in enumerate(intersections):
            active = set(row["active_labels"])
            active_positions = [j for j, lbl in enumerate(model_labels) if lbl in active]
            inter_color = combo_color(list(active))
            if active_positions:
                ax_matrix.plot([i, i], [min(active_positions), max(active_positions)], color=inter_color, linewidth=1.8)
            for j, lbl in enumerate(model_labels):
                filled = lbl in active
                marker_color = inter_color if filled else "lightgray"
                ax_matrix.scatter(i, j, s=42, c=marker_color, zorder=3)

        ax_matrix.set_xticks(x)
        ax_matrix.set_xticklabels([row["active_key"] for row in intersections], rotation=55, ha="right", fontsize=9)
        ax_matrix.tick_params(axis="x", length=0)
        ax_matrix.set_ylim(-0.6, len(model_labels) - 0.4)
        ax_matrix.invert_yaxis()

    set_sizes = payload["set_sizes"]
    labels = ordered_model_labels(models)
    vals = [set_sizes[label] for label in labels]
    y = np.arange(len(labels))
    set_colors = [SET_COLORS[label] for label in labels]
    ax_sets.barh(y, vals, color=set_colors, alpha=0.65)
    for y_pos, label, val in zip(y, labels, vals):
        text_color = "white" if label == "B" else "black"
        ax_sets.text(val / 2, y_pos, str(int(val)), va="center", ha="center", fontsize=9, color=text_color)
    ax_sets.set_yticks(y)
    ax_sets.set_yticklabels(labels, fontsize=10)
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Set size", fontsize=10)
    ax_sets.grid(axis="x", linestyle="--", alpha=0.35)

    subfig.suptitle(language_title, fontsize=13, y=0.96)
    subfig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=10)


def make_figure(language_payload: dict[str, dict[str, Any]], models: list[str], k: float, include_baseline: bool, include_none: bool) -> plt.Figure:
    ordered_codes = [code for code, _ in LANGUAGE_SPECS if code in language_payload]
    n = len(ordered_codes)
    if n == 1:
        fig = plt.figure(figsize=(10, 6.8))
        subfigs = [fig]
    else:
        fig = plt.figure(figsize=(10, 7 * n))
        subfigs = fig.subfigures(nrows=n, ncols=1)
        if not isinstance(subfigs, np.ndarray):
            subfigs = np.array([subfigs])

    for idx, (subfig, code) in enumerate(zip(subfigs, ordered_codes)):
        payload = language_payload[code]
        language_title = str(LANGUAGE_NAME_BY_CODE.get(code, code))
        models_text = "E/S/R/B" if include_baseline else "E/S/R"
        none_text = "including None" if include_none else "excluding None"
        subtitle = (
            f"{models_text} | k={k:.3f} | tau={payload['tau']:.4f} | "
            f"n={payload['n_items_considered']} | {none_text}"
        )
        draw_upset_panel(
            fig,
            subfig,
            language_title,
            subtitle,
            payload,
            models,
        )

    if n > 1:
        title = "Correctness UpSet by Language (E/S/R/B)" if include_baseline else "Correctness UpSet by Language (E/S/R)"
        fig.suptitle(title, fontsize=14, y=0.999, fontweight="bold")
    return fig


def main() -> None:
    args = parse_args()
    if args.k <= 0.0:
        raise ValueError("--k must be > 0")

    output_figure = output_path_with_k(args.output_figure, float(args.k))
    output_json = output_path_with_k(args.output_json, float(args.k))

    models = selected_models(bool(args.include_baseline))

    df = load_predictions(args.predictions_file)
    df = filter_rows(df, args.language, models)

    codes: list[str] = [code for code, _ in LANGUAGE_SPECS] if args.language == "all" else [str(args.language)]
    language_payload: dict[str, dict[str, Any]] = {}

    for code in codes:
        subset: pd.DataFrame = df.loc[df[L1_COL] == code].copy()
        if subset.empty:
            continue
        language_payload[code] = analyze_language(subset, models, float(args.k), bool(args.include_none))

    if not language_payload:
        raise ValueError("No language subsets remain after filtering.")

    fig = make_figure(
        language_payload=language_payload,
        models=models,
        k=float(args.k),
        include_baseline=bool(args.include_baseline),
        include_none=bool(args.include_none),
    )

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=160, bbox_inches="tight")
    print(f"Saved UpSet figure: {output_figure}")

    if not args.no_show:
        plt.show()
    plt.close(fig)

    summary = {
        "predictions_file": str(args.predictions_file),
        "output_figure": str(output_figure),
        "k": float(args.k),
        "language": str(args.language),
        "include_baseline": bool(args.include_baseline),
        "include_none": bool(args.include_none),
        "selected_models": models,
        "languages": language_payload,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved UpSet summary JSON: {output_json}")


if __name__ == "__main__":
    main()




