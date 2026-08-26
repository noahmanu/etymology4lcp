"""
Generate outer and inner cross-validation folds for the enriched dataset.

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

The outer split follows the 5-fold structure described in the project overview.
For each outer train split, 4 inner folds are created so that model training can
load the required slices directly from disk.
"""

from pathlib import Path
import random
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "feature_enriched" / "full.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

RANDOM_SEED = 667
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
STRATIFY_COLUMN = "L1"


def _build_stratified_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    stratify_column: str = STRATIFY_COLUMN,
) -> list[list[int]]:
    """Create deterministic, language-balanced fold indices.

    The starting fold is rotated between stratified groups so that remainder rows
    are spread across folds instead of always accumulating in the first fold.
    """
    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    start_fold = 0

    for label in sorted(df[stratify_column].unique()):
        label_indices = df.index[df[stratify_column] == label].tolist()
        rng.shuffle(label_indices)

        for position, row_index in enumerate(label_indices):
            fold_index = (start_fold + position) % n_splits
            folds[fold_index].append(row_index)

        start_fold = (start_fold + (len(label_indices) % n_splits)) % n_splits

    return [sorted(fold) for fold in folds]


def _validate_partition(partitions: list[list[int]], expected_size: int) -> None:
    """Ensure a fold partition covers each row exactly once."""
    flattened = [row_index for partition in partitions for row_index in partition]

    if len(flattened) != expected_size:
        raise ValueError(
            f"Partition size mismatch: expected {expected_size}, got {len(flattened)}."
        )

    if len(set(flattened)) != expected_size:
        raise ValueError("Partitions overlap or omit rows.")


def _write_split(df: pd.DataFrame, indices: list[int], output_path: Path) -> None:
    """Write a dataframe slice to disk without the pandas index."""
    split_df = df.loc[indices].reset_index(drop=True)
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
    """Generate and save all outer and inner fold slices."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path, keep_default_na=False)

    if verbose:
        print("=" * 60)
        print("Generating Outer and Inner Folds")
        print("=" * 60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_dir}")
        print(f"Rows:   {len(df):,}")
        print(f"Seed:   {random_seed}")

    _remove_existing_fold_directories(output_dir)

    outer_folds = _build_stratified_folds(df, OUTER_N_SPLITS, random_seed)
    _validate_partition(outer_folds, len(df))
    all_outer_indices = set(df.index)

    for outer_fold_number, outer_test_indices in enumerate(outer_folds, start=1):
        outer_dir = output_dir / f"outer_fold_{outer_fold_number}"
        outer_train_indices = sorted(all_outer_indices - set(outer_test_indices))

        _write_split(df, outer_train_indices, outer_dir / "outer_train.csv")
        _write_split(df, outer_test_indices, outer_dir / "outer_test.csv")

        outer_train_df = df.loc[outer_train_indices].reset_index(drop=True)
        inner_folds = _build_stratified_folds(outer_train_df, INNER_N_SPLITS, random_seed)
        _validate_partition(inner_folds, len(outer_train_df))
        all_inner_indices = set(outer_train_df.index)

        for inner_fold_number, inner_test_indices in enumerate(inner_folds, start=1):
            inner_dir = outer_dir / f"inner_fold_{inner_fold_number}"
            inner_train_indices = sorted(all_inner_indices - set(inner_test_indices))

            _write_split(outer_train_df, inner_train_indices, inner_dir / "inner_train.csv")
            _write_split(outer_train_df, inner_test_indices, inner_dir / "inner_test.csv")

        if verbose:
            outer_test_size = len(outer_test_indices)
            outer_train_size = len(outer_train_indices)
            print(
                f"✓ outer_fold_{outer_fold_number}: "
                f"train={outer_train_size:,}, test={outer_test_size:,}"
            )

    if verbose:
        print("=" * 60)
        print("✓ Fold generation complete")
        print("=" * 60)


if __name__ == "__main__":
    generate_folds()

