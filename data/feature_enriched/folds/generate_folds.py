"""
Generate grouped outer and inner cross-validation folds for the enriched dataset.

This script reads ``data/feature_enriched/full.csv`` and creates:

    data/feature_enriched/folds/
        outer_fold_1/
            outer_train.csv
            outer_test.csv
            inner_fold_1/
                inner_train.csv
                inner_test.csv
            ...
        ...
        outer_fold_5/

All rows with the same ``item_id`` are kept in the same fold. This prevents
the same target word—which may occur once per L1—from appearing in both the
training and test portions of an outer or inner split.

Splits are stratified on ``L1`` while grouping on ``item_id``.
"""

from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "feature_enriched" / "full.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

RANDOM_SEED = 667
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
STRATIFY_COLUMN = "L1"
GROUP_COLUMN = "item_id"


def _validate_input_columns(df: pd.DataFrame) -> None:
    """Ensure required split columns exist and contain usable group IDs."""
    required_columns = {STRATIFY_COLUMN, GROUP_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}."
        )

    empty_group_ids = df[GROUP_COLUMN].astype(str).str.strip().eq("")
    if empty_group_ids.any():
        raise ValueError(
            f"Column '{GROUP_COLUMN}' contains {empty_group_ids.sum():,} empty value(s). "
            "Every row must have an item_id so groups cannot leak across folds."
        )


def _validate_stratification_capacity(df: pd.DataFrame, n_splits: int) -> None:
    """Ensure every L1 class occurs in enough distinct item groups."""
    groups_per_label = df.groupby(STRATIFY_COLUMN)[GROUP_COLUMN].nunique()
    insufficient_labels = groups_per_label[groups_per_label < n_splits]

    if not insufficient_labels.empty:
        details = ", ".join(
            f"{label!r}: {count}" for label, count in insufficient_labels.items()
        )
        raise ValueError(
            f"Cannot create {n_splits} stratified grouped folds. "
            "Each L1 value must occur in at least "
            f"{n_splits} distinct {GROUP_COLUMN} groups. Found: {details}."
        )


def _build_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    stratify_column: str = STRATIFY_COLUMN,
    group_column: str = GROUP_COLUMN,
) -> list[list[int]]:
    """Create deterministic L1-stratified folds without splitting item_ids."""
    _validate_stratification_capacity(df, n_splits)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []

    # The returned indices are positional indices, which match this script's
    # reset RangeIndex dataframes.
    for _, test_indices in splitter.split(
        X=df,
        y=df[stratify_column],
        groups=df[group_column],
    ):
        folds.append(sorted(test_indices.tolist()))

    return folds


def _validate_partition(partitions: list[list[int]], expected_size: int) -> None:
    """Ensure a fold partition covers each row exactly once."""
    flattened = [row_index for partition in partitions for row_index in partition]

    if len(flattened) != expected_size:
        raise ValueError(
            f"Partition size mismatch: expected {expected_size}, got {len(flattened)}."
        )

    if len(set(flattened)) != expected_size:
        raise ValueError("Partitions overlap or omit rows.")


def _validate_group_partition(
    df: pd.DataFrame,
    partitions: list[list[int]],
    group_column: str = GROUP_COLUMN,
) -> None:
    """Ensure every item_id appears in exactly one fold."""
    group_to_fold: dict[str, int] = {}

    for fold_number, row_indices in enumerate(partitions, start=1):
        fold_groups = df.iloc[row_indices][group_column].unique()

        for group_id in fold_groups:
            if group_id in group_to_fold:
                raise ValueError(
                    f"Group leakage detected: {group_column}={group_id!r} appears in "
                    f"both fold {group_to_fold[group_id]} and fold {fold_number}."
                )

            group_to_fold[group_id] = fold_number

    if len(group_to_fold) != df[group_column].nunique():
        raise ValueError(
            "Not all item_id groups were assigned to a fold."
        )


def _write_split(df: pd.DataFrame, indices: list[int], output_path: Path) -> None:
    """Write a dataframe slice to disk without the pandas index."""
    split_df = df.iloc[indices].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)


def _remove_existing_fold_directories(output_dir: Path) -> None:
    """Delete previously generated outer fold directories before regenerating."""
    for outer_dir in output_dir.glob("outer_fold_*"):
        if outer_dir.is_dir():
            shutil.rmtree(outer_dir)


def generate_folds(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> None:
    """Generate and save grouped outer and inner fold slices."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    # Use a predictable positional index because splitters return row positions.
    df = pd.read_csv(input_path, keep_default_na=False).reset_index(drop=True)

    _validate_input_columns(df)

    if verbose:
        print("=" * 60)
        print("Generating Grouped Outer and Inner Folds")
        print("=" * 60)
        print(f"Input:       {input_path}")
        print(f"Output:      {output_dir}")
        print(f"Rows:        {len(df):,}")
        print(f"Unique IDs:  {df[GROUP_COLUMN].nunique():,}")
        print(f"Group by:    {GROUP_COLUMN}")
        print(f"Stratify by: {STRATIFY_COLUMN}")
        print(f"Seed:        {random_seed}")

    _remove_existing_fold_directories(output_dir)

    outer_folds = _build_stratified_group_folds(
        df,
        n_splits=OUTER_N_SPLITS,
        seed=random_seed,
    )
    _validate_partition(outer_folds, len(df))
    _validate_group_partition(df, outer_folds)

    all_outer_indices = set(range(len(df)))

    for outer_fold_number, outer_test_indices in enumerate(outer_folds, start=1):
        outer_dir = output_dir / f"outer_fold_{outer_fold_number}"
        outer_train_indices = sorted(all_outer_indices - set(outer_test_indices))

        _write_split(df, outer_train_indices, outer_dir / "outer_train.csv")
        _write_split(df, outer_test_indices, outer_dir / "outer_test.csv")

        outer_train_df = df.iloc[outer_train_indices].reset_index(drop=True)

        inner_folds = _build_stratified_group_folds(
            outer_train_df,
            n_splits=INNER_N_SPLITS,
            seed=random_seed,
        )
        _validate_partition(inner_folds, len(outer_train_df))
        _validate_group_partition(outer_train_df, inner_folds)

        all_inner_indices = set(range(len(outer_train_df)))

        for inner_fold_number, inner_test_indices in enumerate(inner_folds, start=1):
            inner_dir = outer_dir / f"inner_fold_{inner_fold_number}"
            inner_train_indices = sorted(all_inner_indices - set(inner_test_indices))

            _write_split(
                outer_train_df,
                inner_train_indices,
                inner_dir / "inner_train.csv",
            )
            _write_split(
                outer_train_df,
                inner_test_indices,
                inner_dir / "inner_test.csv",
            )

        if verbose:
            outer_test_size = len(outer_test_indices)
            outer_train_size = len(outer_train_indices)
            outer_test_groups = df.iloc[outer_test_indices][GROUP_COLUMN].nunique()

            print(
                f"✓ outer_fold_{outer_fold_number}: "
                f"train={outer_train_size:,}, "
                f"test={outer_test_size:,}, "
                f"test_item_ids={outer_test_groups:,}"
            )

    if verbose:
        print("=" * 60)
        print("✓ Grouped fold generation complete")
        print("=" * 60)


if __name__ == "__main__":
    generate_folds()