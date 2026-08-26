"""Generate fold-specific character n-gram TF-IDF cosine similarity.

This script iterates through outer folds created by
`data/feature_enriched/folds/generate_folds.py`, fits one TF-IDF vectorizer per
outer fold on outer-train strings only, computes `char_cosine_sim` for both the
outer-train and outer-test splits, and saves the fitted vectorizer for
reproducibility.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS_DIR = PROJECT_ROOT / "data" / "feature_enriched" / "folds"
VECTORIZER_FILENAME = "char_cosine_sim_vectorizer.pkl"
CHAR_NGRAM_RANGE = (2, 4)
EN_COLUMN = "en_norm"
L1_COLUMN = "l1_closest"


def _require_columns(df: pd.DataFrame, file_path: Path, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _load_split(csv_path: Path, en_column: str, l1_column: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=False)
    _require_columns(df, csv_path, [en_column, l1_column])
    return df


def _fit_vectorizer(train_df: pd.DataFrame, en_column: str, l1_column: str) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=1,
        sublinear_tf=True,
    )

    fit_corpus = pd.concat(
        [train_df[en_column].astype(str), train_df[l1_column].astype(str)],
        ignore_index=True,
    )
    vectorizer.fit(fit_corpus)
    return vectorizer


def _add_char_cosine_sim(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    en_column: str,
    l1_column: str,
) -> pd.DataFrame:
    df = df.copy()
    en_vectors = vectorizer.transform(df[en_column].astype(str))
    l1_vectors = vectorizer.transform(df[l1_column].astype(str))

    # Vectors are L2-normalized by TF-IDF, so row-wise dot product is cosine similarity.
    cosine_values = en_vectors.multiply(l1_vectors).sum(axis=1).A1
    cosine_values = np.clip(cosine_values, 0.0, 1.0)
    df["char_cosine_sim"] = cosine_values
    return df


def _save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(vectorizer, handle)


def _outer_fold_dirs(folds_dir: Path) -> list[Path]:
    dirs = [path for path in folds_dir.glob("outer_fold_*") if path.is_dir()]

    def _fold_number(path: Path) -> int:
        try:
            return int(path.name.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(dirs, key=_fold_number)


def generate_cosine_similarity_feature(
    folds_dir: Path = DEFAULT_FOLDS_DIR,
    en_column: str = EN_COLUMN,
    l1_column: str = L1_COLUMN,
    verbose: bool = True,
) -> None:
    if not folds_dir.exists():
        raise FileNotFoundError(f"Folds directory not found: {folds_dir}")

    fold_dirs = _outer_fold_dirs(folds_dir)
    if not fold_dirs:
        raise FileNotFoundError(f"No outer fold directories found in: {folds_dir}")

    if verbose:
        print("=" * 60)
        print("Generating fold-specific char_cosine_sim")
        print("=" * 60)
        print(f"Folds directory: {folds_dir}")
        print(f"Using columns: en='{en_column}', l1='{l1_column}'")

    for fold_dir in fold_dirs:
        outer_train_path = fold_dir / "outer_train.csv"
        outer_test_path = fold_dir / "outer_test.csv"
        vectorizer_path = fold_dir / VECTORIZER_FILENAME

        outer_train_df = _load_split(outer_train_path, en_column, l1_column)
        outer_test_df = _load_split(outer_test_path, en_column, l1_column)

        vectorizer = _fit_vectorizer(outer_train_df, en_column, l1_column)

        outer_train_df = _add_char_cosine_sim(outer_train_df, vectorizer, en_column, l1_column)
        outer_test_df = _add_char_cosine_sim(outer_test_df, vectorizer, en_column, l1_column)

        outer_train_df.to_csv(outer_train_path, index=False)
        outer_test_df.to_csv(outer_test_path, index=False)
        _save_vectorizer(vectorizer, vectorizer_path)

        if verbose:
            print(
                f"- {fold_dir.name}: "
                f"train={len(outer_train_df):,}, test={len(outer_test_df):,}, "
                f"vectorizer={vectorizer_path.name}"
            )

    if verbose:
        print("=" * 60)
        print("Done")
        print("=" * 60)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate outer-fold char n-gram TF-IDF cosine similarity features."
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=DEFAULT_FOLDS_DIR,
        help="Directory containing outer_fold_* directories.",
    )
    parser.add_argument(
        "--en-column",
        type=str,
        default=EN_COLUMN,
        help="English word-form column used for cosine similarity.",
    )
    parser.add_argument(
        "--l1-column",
        type=str,
        default=L1_COLUMN,
        help="L1 word-form column used for cosine similarity.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    generate_cosine_similarity_feature(
        folds_dir=args.folds_dir,
        en_column=args.en_column,
        l1_column=args.l1_column,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

