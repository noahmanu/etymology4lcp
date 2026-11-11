# Noah-Manuel Michael
# Created: 2026-07-01
# Updated: 2026-11-11
#
# Text normalization and preprocessing for the pooled dataset.
#
# This module normalizes English target words and L1 source words, preparing
# them for feature computation. Comma-separated alternatives in L1 are preserved
# for later processing (e.g., taking minimum edit distance across alternatives).
#
# Columns added:
#     en_norm  : lowercased, stripped English target word
#     l1_norm  : stripped L1 with brackets/markers/slashes/hyphens removed,
#                but preserving diacritics and capitalization, comma-separated alternatives preserved
#     l1_clean : L1 with bracketed notes removed, diacritics stripped,
#                German markers removed, comma-separated alternatives preserved
#     l1_closest : comma-split L1 alternative (whitespace removed) with minimum geo
#                  Levenshtein-based distance to en_norm
#     char_edit_dist   : raw Levenshtein distance for l1_closest vs en_norm
#     char_edit_dist_geo_norm : Levenshtein distance normalized by sqrt(len(a))*sqrt(len(b))
#     char_edit_dist_norm : Levenshtein distance normalized by max(len(a), len(b))

import unicodedata
import math
import pandas as pd
import regex as re
from pathlib import Path
from typing import Optional
import pinyin
from rapidfuzz.distance import Levenshtein


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_POOLED = PROJECT_ROOT / "data" / "pooled"

# Pattern to match all types of brackets and their contents
# Matches: (note), （注）, [note]
_BRACKET_PATTERN = re.compile(r"\s*[(（\[].*?[)）\]]")

# German grammatical markers to remove (etw = etwas, jdn = jemanden, jdm = jemandem, sich = reflexive)
# Also handles combinations like jdn/etw, jdm/etw
_GERMAN_MARKERS = re.compile(r'\b(jdn/etw|jdm/etw|etw|jdn|jdm|sich|etw/jdn|jdn/sich|jdm /sich|sich/etw)\b', re.IGNORECASE)
_GERMAN_GENDER_SUFFIX = re.compile(r"/in\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Lowercase and strip whitespace from a string."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks (accents) from a Unicode string."""
    if not isinstance(text, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def chinese_to_pinyin(text: str) -> str:
    """Convert Chinese characters to pinyin (no tone marks)."""
    if not isinstance(text, str):
        return ""
    return pinyin.get(text, format="strip", delimiter=" ")


def _normalize_l1_common(text: str, *, lowercase: bool) -> str:
    """Apply shared L1 cleanup steps used by both l1_norm and l1_clean."""
    text = text.strip()
    if lowercase:
        text = text.lower()

    text = text.replace("…", "")
    text = text.replace("，", ", ")
    text = text.replace("、", ", ")
    text = text.replace("；", ", ")
    text = _BRACKET_PATTERN.sub("", text)
    text = _GERMAN_MARKERS.sub("", text)
    text = _GERMAN_GENDER_SUFFIX.sub("", text)
    return text


def _finalize_l1_alternatives(text: str) -> str:
    """Normalize whitespace/punctuation and preserve clean comma-separated alternatives."""
    parts = []
    for part in text.split(","):
        part = part.strip()
        if "/" in part:
            slash_parts = [p.strip() for p in part.split("/") if p.strip()]
            part = slash_parts[0] if slash_parts else ""
        if part:
            parts.append(part)

    text = ", ".join(parts)
    text = text.replace("-", "")
    text = re.sub(r"\s+", " ", text)

    parts = [part.strip() for part in text.split(",")]
    parts = [part for part in parts if part]
    return ", ".join(parts)


def normalize_l1_with_alternatives(text: str, _l1: str) -> str:
    """
    Normalize L1 source word while preserving comma-separated alternatives.

    Keeps diacritics and capitalization but removes brackets, German markers,
    slashes, and hyphens.

    Steps:
        1. Strip whitespace (preserve capitalization)
        2. Remove Chinese placeholder ellipsis (…)
        3. Replace Chinese separators (、, ；) with regular comma
        4. Remove all brackets and contents: (), （）, []
        5. Remove German grammatical markers (etw, jdn, jdm, jdn/etw, jdm/etw, sich)
        6. Remove German gender suffix "/in"
        7. Remove "/" and everything after it (for all languages)
        8. Remove hyphens
        9. Remove double spaces
        10. Keep all comma-separated alternatives
        11. Strip whitespace from each alternative
    """
    if not isinstance(text, str):
        return ""

    text = _normalize_l1_common(text, lowercase=False)
    return _finalize_l1_alternatives(text)


def clean_l1_translation(text: str, l1: str) -> str:
    """
    Clean L1 source word while preserving comma-separated alternatives.

    Steps:
        1. Lowercase and strip
        2. Remove Chinese placeholder ellipsis (…)
        3. Replace Chinese separators (、, ；) with regular comma
        4. Remove all brackets and contents: (), （）, []
        5. Convert Chinese characters to pinyin
        6. Remove diacritics/accents (including from pinyin)
        7. Remove German grammatical markers (etw, jdn, jdm, jdn/etw, jdm/etw, sich)
        8. Remove German gender suffix "/in"
        9. Remove "/" and everything after it (for all languages)
        10. Remove hyphens
        11. Remove double spaces
        12. Keep all comma-separated alternatives
        13. Strip whitespace from each alternative
    """
    if not isinstance(text, str):
        return ""

    text = _normalize_l1_common(text, lowercase=True)

    # Convert Chinese characters to pinyin (only for Chinese entries)
    if l1 == "cn":
        text = chinese_to_pinyin(text)
        # Pinyin conversion may introduce fullwidth comma ，; normalize back to regular comma
        text = text.replace("，", ", ")

    # Remove diacritics
    text = strip_accents(text)

    return _finalize_l1_alternatives(text)


def char_edit_dist_geo_normalized(a: str, b: str) -> float:
    """Levenshtein distance normalized by sqrt(len(a))*sqrt(len(b))."""
    dist = Levenshtein.distance(str(a), str(b))
    len_a = max(len(str(a)), 1)
    len_b = max(len(str(b)), 1)
    return dist / math.sqrt(len_a) / math.sqrt(len_b)


def char_edit_dist_maxlen_normalized(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer of the two word lengths."""
    dist = Levenshtein.distance(str(a), str(b))
    len_a = len(str(a))
    len_b = len(str(b))
    return dist / max(max(len_a, len_b), 1)


def find_best_l1_match(en_text: str, l1_clean_text: str) -> tuple[str, float, float, float]:
    """Return best comma-split L1 alternative and its raw/georatic/maxlen edit distances."""
    en = normalize_text(en_text)
    en_for_dist = str(en)
    if not isinstance(l1_clean_text, str) or not l1_clean_text.strip():
        return "", float("nan"), float("nan"), float("nan")

    alternatives = [re.sub(r"\s+", "", alt) for alt in l1_clean_text.split(",")]
    alternatives = [alt for alt in alternatives if alt]
    if not alternatives:
        return "", float("nan"), float("nan"), float("nan")

    best_alt = min(alternatives, key=lambda alt: char_edit_dist_geo_normalized(en_for_dist, alt))
    raw_dist = float(Levenshtein.distance(en_for_dist, best_alt))
    geo_norm_dist = float(char_edit_dist_geo_normalized(en_for_dist, best_alt))
    norm_dist = float(char_edit_dist_maxlen_normalized(en_for_dist, best_alt))

    return best_alt, raw_dist, geo_norm_dist, norm_dist


def preprocess_pooled(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Preprocess the pooled dataset.

    Reads pooled_dataset.csv, adds normalized columns, and saves to
    pooled_preprocessed.csv.

    Parameters
    ----------
    input_path : Path, optional
        Path to pooled dataset. Default: data/pooled/pooled_dataset.csv
    output_path : Path, optional
        Path for output. Default: data/pooled/pooled_preprocessed.csv
    verbose : bool, optional
        Print progress information. Default: True

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset with en_norm, l1_norm, l1_clean, l1_closest,
        char_edit_dist, char_edit_dist_geo_norm, and char_edit_dist_norm columns added.
    """
    # Set default paths
    if input_path is None:
        input_path = DATA_POOLED / "pooled_dataset.csv"
    if output_path is None:
        output_path = DATA_POOLED / "pooled_preprocessed.csv"

    if verbose:
        print("=" * 60)
        print("Preprocessing Pooled Dataset")
        print("=" * 60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")

    # Load pooled data
    df = pd.read_csv(input_path)

    if verbose:
        print(f"\nLoaded {len(df):,} instances")
        print(f"Columns: {list(df.columns)}")

    # Add normalized columns
    df["en_norm"] = df["en_target_word"].apply(normalize_text)

    l1_source_words = df["L1_source_word"].tolist()
    l1_languages = df["L1"].tolist()
    df["l1_norm"] = [
        normalize_l1_with_alternatives(source_word, l1_lang)
        for source_word, l1_lang in zip(l1_source_words, l1_languages)
    ]
    df["l1_clean"] = [
        clean_l1_translation(source_word, l1_lang)
        for source_word, l1_lang in zip(l1_source_words, l1_languages)
    ]

    best_matches = [
        find_best_l1_match(en_word, l1_word)
        for en_word, l1_word in zip(df["en_norm"].tolist(), df["l1_clean"].tolist())
    ]
    df[["l1_closest", "char_edit_dist", "char_edit_dist_geo_norm", "char_edit_dist_norm"]] = pd.DataFrame(
        best_matches,
        columns=["l1_closest", "char_edit_dist", "char_edit_dist_geo_norm", "char_edit_dist_norm"],
        index=df.index,
    )

    if verbose:
        print("\n✓ Added preprocessing columns:")
        print("  - en_norm:  normalized English target word")
        print("  - l1_norm:  L1 with brackets/markers/slashes removed, preserving case and diacritics")
        print("  - l1_clean: cleaned L1 (brackets removed, alternatives preserved, lowercased)")
        print("  - l1_closest: best L1 alternative by normalized edit distance")
        print("  - char_edit_dist: raw edit distance for l1_closest")
        print("  - char_edit_dist_geo_norm: edit distance normalized by sqrt(len(a))*sqrt(len(b))")
        print("  - char_edit_dist_norm: edit distance normalized by max(len(a), len(b))")

        # Show examples
        print("\nExample preprocessed instances:")
        print("-" * 60)
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\nRow {i+1}:")
            print(f"  Original EN: {row['en_target_word']}")
            print(f"  Normalized:  {row['en_norm']}")
            print(f"  Original L1: {row['L1_source_word']}")
            print(f"  Cleaned:     {row['l1_clean']}")
            print(f"  Closest:     {row['l1_closest']}")
            print(f"  char_edit_dist:    {row['char_edit_dist']}")
            print(f"  geo norm:   {row['char_edit_dist_geo_norm']}")
            print(f"  norm:        {row['char_edit_dist_norm']}")

    # Save preprocessed data
    df.to_csv(output_path, index=False)

    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ Saved preprocessed dataset: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Total columns: {len(df.columns)}")
        print("=" * 60)

    return df


if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    df = preprocess_pooled(verbose=True)

    # Quick validation
    print("Validation:")
    print(f"  Total instances: {len(df):,}")
    print(f"  Non-empty en_norm: {df['en_norm'].notna().sum():,}")
    print(f"  Non-empty l1_clean: {df['l1_clean'].notna().sum():,}")
    print(f"  Non-empty l1_closest: {df['l1_closest'].notna().sum():,}")
    print(f"  Non-empty char_edit_dist: {df['char_edit_dist'].notna().sum():,}")
    print(f"  Non-empty char_edit_dist_geo_norm: {df['char_edit_dist_geo_norm'].notna().sum():,}")
    print(f"  Non-empty char_edit_dist_norm: {df['char_edit_dist_norm'].notna().sum():,}")

    # Count instances with comma-separated alternatives
    has_alternatives = df['l1_clean'].str.contains(',', na=False).sum()
    print(f"  L1 entries with alternatives (contain comma): {has_alternatives:,}")

    if has_alternatives > 0:
        print("\n  Example entries with alternatives:")
        examples = df[df['l1_clean'].str.contains(',', na=False)].head(3)
        for _, row in examples.iterrows():
            print(f"    {row['L1']}: {row['l1_clean']}")



