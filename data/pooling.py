"""
Data pooling module for BEA-ST multilingual LCP dataset.

This module combines train/dev/test splits from all source languages (cn, de, es)
into a single pooled dataset for cross-validation.

Functions
---------
pool_data()
    Reads all 9 CSV files and combines them into a single shuffled dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


# Project root and data paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ORIGINAL = PROJECT_ROOT / "data" / "original"
DATA_POOLED = PROJECT_ROOT / "data" / "pooled"

# Random seed for reproducibility
RANDOM_SEED = 667

# Languages and splits
LANGUAGES = ["cn", "de", "es"]
SPLITS = ["train", "dev", "test"]


def pool_data(
    original_dir: Optional[Path] = None,
    pooled_dir: Optional[Path] = None,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pool all train/dev/test splits from all languages into a single dataset.

    Reads 9 CSV files (3 languages × 3 splits), combines them, shuffles,
    and saves to data/pooled/pooled_dataset.csv.

    Parameters
    ----------
    original_dir : Path, optional
        Directory containing original data. Default: data/original/
    pooled_dir : Path, optional
        Directory to save pooled data. Default: data/pooled/
    random_seed : int, optional
        Random seed for shuffling. Default: 667
    verbose : bool, optional
        Print progress information. Default: True

    Returns
    -------
    pd.DataFrame
        Pooled and shuffled dataset with all instances.

    Notes
    -----
    - Creates pooled_dir if it doesn't exist
    - Validates that all CSV files have the same columns
    - Adds a 'original_split' column to track source (train/dev/test)
    - Shuffles data with random_seed for reproducibility
    - Saves to pooled_dataset.csv

    Examples
    --------
    >>> df = pool_data()
    >>> print(df.shape)
    (22548, 9)  # 3 languages × 7516 instances per language
    """
    # Set default paths
    if original_dir is None:
        original_dir = DATA_ORIGINAL
    if pooled_dir is None:
        pooled_dir = DATA_POOLED

    # Create output directory
    pooled_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Data Pooling - BEA-ST Multilingual LCP Dataset")
        print("=" * 60)

    # Collect all dataframes
    dfs = []
    expected_columns = None
    total_instances = 0

    for language in LANGUAGES:
        for split in SPLITS:
            # Construct file path
            csv_path = original_dir / split / language / f"kvl_shared_task_{language}_{split}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}\n"
                    f"Expected structure: data/original/{split}/{language}/"
                )

            # Read CSV
            df = pd.read_csv(csv_path)

            # Validate columns
            if expected_columns is None:
                expected_columns = list(df.columns)
            elif list(df.columns) != expected_columns:
                raise ValueError(
                    f"Column mismatch in {csv_path}\n"
                    f"Expected: {expected_columns}\n"
                    f"Got: {list(df.columns)}"
                )

            # Add metadata column to track original split
            df["original_split"] = split

            dfs.append(df)
            total_instances += len(df)

            if verbose:
                print(f"✓ Loaded {language}/{split}: {len(df):,} instances")

    if verbose:
        print("-" * 60)
        print(f"Total instances collected: {total_instances:,}")

    # Combine all dataframes
    pooled_df = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"Combined dataset shape: {pooled_df.shape}")
        print(f"\nLanguage distribution:")
        print(pooled_df["L1"].value_counts().sort_index())
        print(f"\nOriginal split distribution:")
        print(pooled_df["original_split"].value_counts())

    # Shuffle the dataset
    pooled_df = pooled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    if verbose:
        print(f"\n✓ Dataset shuffled (random_seed={random_seed})")

    # Save to CSV
    output_path = pooled_dir / "pooled_dataset.csv"
    pooled_df.to_csv(output_path, index=False)

    if verbose:
        print(f"✓ Saved to: {output_path}")
        print(f"\nColumns: {list(pooled_df.columns)}")
        print("=" * 60)
        print(f"✓ Pooling complete! Shape: {pooled_df.shape}")
        print("=" * 60)

    return pooled_df


def validate_pooled_data(pooled_df: pd.DataFrame) -> dict:
    """
    Validate the pooled dataset for completeness and correctness.

    Parameters
    ----------
    pooled_df : pd.DataFrame
        Pooled dataset to validate.

    Returns
    -------
    dict
        Validation report with statistics and checks.
    """
    report = {
        "total_instances": len(pooled_df),
        "languages": pooled_df["L1"].value_counts().to_dict(),
        "original_splits": pooled_df["original_split"].value_counts().to_dict(),
        "missing_values": pooled_df.isnull().sum().to_dict(),
        "expected_columns": [
            "item_id", "L1", "en_target_word", "en_target_pos",
            "en_target_clue", "L1_source_word", "L1_context",
            "GLMM_score", "original_split"
        ],
        "actual_columns": list(pooled_df.columns),
    }

    # Check expected total (3 languages × [6091 train + 677 dev + 748 test])
    expected_per_lang = 6091 + 677 + 748  # 7516
    expected_total = 3 * expected_per_lang  # 22548
    report["expected_total"] = expected_total
    report["count_matches"] = (report["total_instances"] == expected_total)

    # Check language balance
    lang_counts = pooled_df["L1"].value_counts()
    report["language_balanced"] = (
        lang_counts["cn"] == lang_counts["de"] == lang_counts["es"]
    )

    # Check for duplicate item_ids within languages
    duplicates = pooled_df.groupby("L1")["item_id"].apply(
        lambda x: x.duplicated().sum()
    )
    report["duplicates_per_language"] = duplicates.to_dict()

    return report


if __name__ == "__main__":
    # Run pooling when script is executed directly
    print("\n🚀 Starting data pooling...\n")

    pooled_df = pool_data(verbose=True)

    print("\n📊 Validation Report:")
    print("=" * 60)
    report = validate_pooled_data(pooled_df)

    print(f"Total instances: {report['total_instances']:,}")
    print(f"Expected total: {report['expected_total']:,}")
    print(f"Count matches: {'✓' if report['count_matches'] else '✗'}")
    print(f"Language balanced: {'✓' if report['language_balanced'] else '✗'}")

    print(f"\nLanguage counts:")
    for lang, count in sorted(report['languages'].items()):
        print(f"  {lang}: {count:,}")

    print(f"\nOriginal split counts:")
    for split, count in sorted(report['original_splits'].items()):
        print(f"  {split}: {count:,}")

    print(f"\nDuplicates per language:")
    for lang, dup_count in sorted(report['duplicates_per_language'].items()):
        print(f"  {lang}: {dup_count}")

    missing_any = any(v > 0 for v in report['missing_values'].values())
    if missing_any:
        print(f"\n⚠️  Missing values detected:")
        for col, count in report['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print(f"\n✓ No missing values detected")

    print("=" * 60)
    print("✓ Pooling and validation complete!")
    print("=" * 60)

