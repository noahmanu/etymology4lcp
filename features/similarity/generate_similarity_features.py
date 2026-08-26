"""Generate the fold-independent similarity features.

Phonetic distances use English pronunciations from CMUdict and Epitran IPA
transcriptions for the source language. Character edit distances are already
provided by preprocessing. Feature F7 (character n-gram TF-IDF cosine
similarity) is intentionally not generated here because it is fitted
separately within each outer fold. Jaro and Jaro-Winkler are computed with
the ``jellyfish`` library. Additionally, Soundex- and Metaphone-based
Levenshtein distances are computed with ``jellyfish``.
"""

import math
import re
import cmudict
import epitran
import jellyfish
import pandas as pd
from pathlib import Path
from typing import Iterable
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"
CEDICT_PATH = PROJECT_ROOT / "features" / "similarity" / "cedict_ts.u8"

_EPITRAN_CODES = {
    "de": "deu-Latn",
    "es": "spa-Latn",
    "cn": "cmn-Hans",
}

# CMUdict uses ARPABET; stress digits are intentionally omitted from IPA.
_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ",
    "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
    "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_CMUDICT = cmudict.dict()


def _initialize_epitran_objects() -> dict:
    """Initialize Epitran objects, with special handling for cmn-Hans."""
    objects = {}
    for language, code in _EPITRAN_CODES.items():
        if code == "cmn-Hans":
            objects[code] = epitran.Epitran(code, cedict_file=str(CEDICT_PATH))
        else:
            objects[code] = epitran.Epitran(code)
    return objects


# Initialize Epitran objects once at module load
_EPITRAN_OBJECTS = _initialize_epitran_objects()


def _arpabet_to_ipa(pronunciation: Iterable[str]) -> str:
    """Convert one CMUdict pronunciation to an IPA character string."""
    return "".join(
        _ARPABET_TO_IPA.get(re.sub(r"\d", "", phone), "") for phone in pronunciation
    )


def _english_ipa(word: str) -> str:
    """Return the first CMUdict IPA pronunciation, or an empty string."""
    pronunciations = _CMUDICT.get(word, ())
    return _arpabet_to_ipa(pronunciations[0]) if pronunciations else ""


def _source_ipa(word: str, language: str) -> str:
    """Return an Epitran IPA transcription for a source word."""
    code = _EPITRAN_CODES.get(language)
    if not code or not word:
        return ""
    epitran_obj = _EPITRAN_OBJECTS.get(code)
    if not epitran_obj:
        return ""
    return epitran_obj.transliterate(word).replace(" ", "")


def _normalized_edit_distance(first: str, second: str) -> float:
    """Normalize edit distance by the geometric mean of nonzero lengths."""
    denominator = math.sqrt(max(len(first), 1) * max(len(second), 1))
    return Levenshtein.distance(first, second) / denominator


def _max_length_normalized_edit_distance(first: str, second: str) -> float:
    """Normalize edit distance by the longer of the two strings."""
    denominator = max(len(first), len(second), 1)
    return Levenshtein.distance(first, second) / denominator


def _source_alternatives(row: pd.Series) -> list[str]:
    """Return source alternatives from l1_norm, split by comma."""
    source = row["l1_norm"]
    return [part.strip() for part in source.split(",") if part.strip()]


def _compute_row(row: pd.Series) -> tuple[float, float, float, float, float, float, float]:
    """Compute phonetic, Jaro, and phonetic-encoding distances for one row."""
    english = row["en_norm"]
    source_clean = row["l1_closest"]
    language = row["L1"].lower()

    alternatives = _source_alternatives(row)

    english_ipa = _english_ipa(english)
    source_ipa_list = [_source_ipa(alternative, language) for alternative in alternatives]

    # Compute phonetic distances across all alternatives, take minimum
    phonetic_distances = [
        Levenshtein.distance(english_ipa, ipa) for ipa in source_ipa_list
    ]
    phonetic_distance = float(min(phonetic_distances))

    phonetic_norm = min(
        _max_length_normalized_edit_distance(english_ipa, ipa)
        for ipa in source_ipa_list
    )

    phonetic_geo_norm = min(
        _normalized_edit_distance(english_ipa, ipa) for ipa in source_ipa_list
    )

    # Compute Jaro distances
    jaro_distance = 1.0 - jellyfish.jaro_similarity(english, source_clean)
    jaro_winkler_distance = 1.0 - jellyfish.jaro_winkler_similarity(english, source_clean)

    # Soundex-based Levenshtein distance (jellyfish)
    en_soundex = jellyfish.soundex(english)
    src_soundex = jellyfish.soundex(source_clean)
    soundex_dist = float(Levenshtein.distance(en_soundex, src_soundex))

    # Metaphone-based normalized Levenshtein distance (jellyfish)
    en_metaphone = jellyfish.metaphone(english)
    src_metaphone = jellyfish.metaphone(source_clean)
    meta_den = max(len(en_metaphone), len(src_metaphone), 1)
    metaphone_dist_norm = Levenshtein.distance(en_metaphone, src_metaphone) / meta_den

    return (
        phonetic_distance,
        phonetic_norm,
        phonetic_geo_norm,
        jaro_distance,
        jaro_winkler_distance,
        soundex_dist,
        metaphone_dist_norm,
    )


def add_similarity_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """Add phonetic, Jaro, and phonetic-encoding features to a preprocessed DataFrame."""
    total_rows = len(data_df)
    progress_interval = 1_000
    print(f"Computing similarity features for {total_rows:,} rows...", flush=True)

    values = []
    for row_number, (_, row) in enumerate(data_df.iterrows(), start=1):
        values.append(_compute_row(row))
        if row_number % progress_interval == 0 or row_number == total_rows:
            print(f"  Processed {row_number:,}/{total_rows:,} rows", flush=True)

    feature_values = pd.DataFrame(values, index=data_df.index)
    feature_values.columns = [
        "phon_edit_dist",
        "phon_edit_dist_norm",
        "phon_edit_dist_geo_norm",
        "jaro_distance",
        "jaro_winkler_distance",
        "soundex_dist",
        "metaphone_dist_norm",
    ]
    data_df[feature_values.columns] = feature_values
    return data_df


def main() -> None:
    """Load the preprocessed pooled data, add features, and save it in place."""
    print(f"Loading data from {DATA_PATH}", flush=True)
    data_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    data_df = add_similarity_features(data_df)
    print(f"Saving data to {DATA_PATH}", flush=True)
    data_df.to_csv(DATA_PATH, index=False)
    print("Similarity feature generation complete.", flush=True)


if __name__ == "__main__":
    main()