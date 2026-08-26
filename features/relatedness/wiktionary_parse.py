"""
Wiktionary Etymology Parser
Extracts and stores etymology text data from Wiktionary JSONL extracts for the pooled dataset.
Etymology fields captured per entry:
- etymology_text: Main etymology text content only

# For Chinese source alternatives, the script first attempts an etymology lookup for
# the full alternative. If no etymology is found, it falls back to jieba search-mode
# subtokens and aggregates their etymology entries.

The script ensures no duplicate etymology entries are saved by using set-based
deduplication before writing to the output CSV.
"""

import argparse
import csv
import gzip
import json
import unicodedata

# import jieba

from pathlib import Path
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELATEDNESS_DIR = Path(__file__).resolve().parent

INPUT_CSV_PATH = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"
OUTPUT_PATH = RELATEDNESS_DIR / "output" / "wiktionary_etymologies.csv"

EXTRACT_PATHS = {
    "en": RELATEDNESS_DIR / "en-extract.jsonl.gz",
    "de": RELATEDNESS_DIR / "de-extract.jsonl.gz",
    "es": RELATEDNESS_DIR / "es-extract.jsonl.gz",
    "zh": RELATEDNESS_DIR / "zh-extract.jsonl.gz",
}

L1_TO_WIKTIONARY = {
    "cn": "zh",
    "de": "de",
    "es": "es",
}

POS_ALIASES = {
    "noun": {"noun"},
    "verb": {"verb"},
    "adjective": {"adj", "adjective"},
    "adverb": {"adv", "adverb"},
    "number": {"num", "number"},
    "preposition": {"prep", "preposition", "prep_phrase"},
    "determiner": {"det", "determiner"},
    "not-no": {"particle"},
    "misc": {"unknown"},
}

OUTPUT_FIELDNAMES = [
    "item_id",
    "L1",
    "en_target_word",
    "en_norm",
    "en_target_pos",
    "L1_source_word",
    "l1_norm",
    "source_alternatives",
    "target_etym_en",
    "target_etym_de",
    "target_etym_es",
    "target_etym_cn",
    "source_etym_en",
    "source_etym_de",
    "source_etym_es",
    "source_etym_cn",
]

# Raw field names differ by Wiktionary edition; map them into canonical output keys.
ETYMOLOGY_FIELDS_BY_EDITION = {
    "en": {
        "etymology_text": ["etymology_text"],
    },
    "de": {
        "etymology_text": ["etymology_texts"],
    },
    "es": {
        "etymology_text": ["etymology_texts"],
    },
    "zh": {
        "etymology_text": ["etymology_texts"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-item Wiktionary etymology texts for pooled dataset rows."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing/debugging.",
    )
    return parser.parse_args()


def smart_open(filepath: Path):
    """Open a file whether it is actually gzipped or plain text."""
    with open(filepath, "rb") as handle:
        magic = handle.read(2)

    if magic == b"\x1f\x8b":
        return gzip.open(filepath, "rt", encoding="utf-8")

    return open(filepath, "r", encoding="utf-8")


def normalize_lookup_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFC", text).strip().casefold()


def split_alternatives(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    alternatives = []
    seen = set()

    for part in text.split(","):
        candidate = part.strip()
        if not candidate:
            continue

        key = normalize_lookup_text(candidate)
        if key and key not in seen:
            seen.add(key)
            alternatives.append(candidate)

    return alternatives


# def split_chinese_subtokens(text: str) -> list[str]:
#     """
#     Return unique, non-empty Chinese search-mode subtokens from jieba.
#
#     Search mode may return both multi-character words and smaller useful
#     components, allowing the parser to find etymology data when no entry exists
#     for the original complete source alternative.
#     """
#     if not isinstance(text, str) or not text.strip():
#         return []
#
#     subtokens = []
#     seen = set()
#
#     for token in jieba.cut_for_search(text):
#         candidate = token.strip()
#         key = normalize_lookup_text(candidate)
#
#         if key and key not in seen:
#             seen.add(key)
#             subtokens.append(candidate)
#
#     return subtokens


def normalize_pos_label(pos: str) -> str:
    if not isinstance(pos, str):
        return ""

    return pos.strip().lower().replace(" ", "_")


def get_target_pos_aliases(raw_pos: str) -> set[str]:
    normalized = normalize_pos_label(raw_pos).replace("_", "-")

    if normalized in POS_ALIASES:
        return POS_ALIASES[normalized]
    if normalized == "adj":
        return POS_ALIASES["adjective"]
    if normalized == "adv":
        return POS_ALIASES["adverb"]
    if normalized in {"prep", "prep_phrase"}:
        return POS_ALIASES["preposition"]
    if normalized == "det":
        return POS_ALIASES["determiner"]
    if normalized == "num":
        return POS_ALIASES["number"]

    return set()


def pick_first_present(entry: dict, field_names: list[str]):
    for field_name in field_names:
        value = entry.get(field_name)
        if value is not None:
            return value

    return None


def extract_etymology_payload(entry: dict, *, edition_lang: str) -> dict[str, object]:
    field_map = ETYMOLOGY_FIELDS_BY_EDITION.get(edition_lang, {})
    payload = {}

    value = pick_first_present(entry, field_map.get("etymology_text", []))
    payload["etymology_text"] = "" if value is None else value

    return payload


def has_useful_etymology(entry: dict, *, edition_lang: str) -> bool:
    payload = extract_etymology_payload(entry, edition_lang=edition_lang)
    value = payload.get("etymology_text")

    if isinstance(value, str) and value.strip():
        return True

    if isinstance(value, (list, dict)) and value:
        return True

    if value not in ("", None) and not isinstance(value, (str, list, dict)):
        return True

    return False


def serialize_value(value):
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def simplify_entry(entry: dict, *, lookup_word: str, edition_lang: str) -> dict:
    etymology_payload = extract_etymology_payload(entry, edition_lang=edition_lang)

    return {
        "lookup_word": lookup_word,
        "matched_word": entry.get("word", ""),
        "edition_lang": "cn" if edition_lang == "zh" else edition_lang,
        "lang_code": (
            "cn"
            if entry.get("lang_code", "") == "zh"
            else entry.get("lang_code", "")
        ),
        "pos": entry.get("pos", ""),
        "etymology_text": etymology_payload["etymology_text"],
    }


def load_dataset_rows(input_csv: Path, limit: int | None = None) -> list[dict]:
    rows = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for index, row in enumerate(reader, start=1):
            rows.append(row)

            if limit is not None and index >= limit:
                break

    return rows


def build_query_sets(rows: list[dict]) -> dict[str, set[tuple[str, str]]]:
    # """
    # Build the set of word/language pairs needed from each Wiktionary edition.
    #
    # Chinese source alternatives include their jieba subtokens so that a later
    # fallback lookup does not require another pass through the extract files.
    # """
    query_sets = {edition: set() for edition in EXTRACT_PATHS}

    for row in rows:
        l1_raw = row.get("L1", "").strip().lower()
        l1_code = L1_TO_WIKTIONARY.get(l1_raw)

        if not l1_code:
            continue

        en_norm = row.get("en_norm", "")
        en_key = normalize_lookup_text(en_norm)

        if en_key:
            # Only consult English edition and the row's involved L1 edition.
            query_sets["en"].add((en_key, "en"))
            query_sets[l1_code].add((en_key, "en"))

        for alternative in split_alternatives(row.get("l1_norm", "")):
            alternative_key = normalize_lookup_text(alternative)

            if not alternative_key:
                continue

            query_sets["en"].add((alternative_key, l1_code))
            query_sets[l1_code].add((alternative_key, l1_code))

            # Pre-load Chinese subtoken entries for fallback etymology lookups.
            # if l1_code == "zh":
            #     for subtoken in split_chinese_subtokens(alternative):
            #         subtoken_key = normalize_lookup_text(subtoken)
            #
            #         if not subtoken_key:
            #             continue
            #
            #         query_sets["en"].add((subtoken_key, "zh"))
            #         query_sets["zh"].add((subtoken_key, "zh"))

    return query_sets


def load_matching_entries(
    extract_path: Path,
    *,
    edition_lang: str,
    queries: set[tuple[str, str]],
) -> dict[tuple[str, str], list[dict]]:
    matches: dict[tuple[str, str], list[dict]] = {}

    if not queries:
        return matches

    with smart_open(extract_path) as handle:
        for line in handle:
            entry = json.loads(line)
            word = entry.get("word")
            lang_code = entry.get("lang_code")

            if not word or not lang_code:
                continue

            lookup_key = (normalize_lookup_text(word), str(lang_code))

            if lookup_key not in queries:
                continue

            if not has_useful_etymology(entry, edition_lang=edition_lang):
                continue

            payload = simplify_entry(
                entry,
                lookup_word=word,
                edition_lang=edition_lang,
            )
            matches.setdefault(lookup_key, []).append(payload)

    return matches


def deduplicate_entries(entries: list[dict]) -> list[dict]:
    """
    Remove duplicate etymology entries by comparing their full JSON representation.
    """
    deduped = []
    seen = set()

    for entry in entries:
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(entry)

    return deduped


def filter_target_entries_by_pos(entries: list[dict], raw_pos: str) -> list[dict]:
    aliases = {
        normalize_pos_label(alias)
        for alias in get_target_pos_aliases(raw_pos)
    }

    if not aliases:
        return deduplicate_entries(entries)

    filtered = [
        entry
        for entry in entries
        if normalize_pos_label(entry.get("pos", "")) in aliases
    ]

    return deduplicate_entries(filtered)


def serialize_entries(entries: list[dict]) -> str:
    if not entries:
        return ""

    normalized_entries = []

    for entry in entries:
        normalized_entries.append(
            {
                key: serialize_value(value)
                for key, value in entry.items()
            }
        )

    return json.dumps(normalized_entries, ensure_ascii=False)


def collect_entries(
    index: dict[tuple[str, str], list[dict]],
    *,
    words: list[str],
    lang_code: str,
) -> list[dict]:
    collected = []

    for word in words:
        key = (normalize_lookup_text(word), lang_code)

        for entry in index.get(key, []):
            entry_copy = cast(dict[str, object], cast(object, dict(entry)))
            entry_copy["lookup_word"] = word
            collected.append(entry_copy)

    return deduplicate_entries(collected)


# def collect_source_entries(
#     index: dict[tuple[str, str], list[dict]],
#     *,
#     alternatives: list[str],
#     lang_code: str,
# ) -> list[dict]:
#     """
#     Collect source etymology entries for each source alternative.
#
#     Every full source alternative is attempted first. For Chinese alternatives,
#     when the complete alternative has no etymology entries in this Wiktionary
#     edition, use jieba search-mode subtokens as a fallback and aggregate all
#     matching subtoken entries.
#
#     This helper is used for both source_etym_en and source_etym_cn.
#     """
#     collected = []
#
#     for alternative in alternatives:
#         full_alternative_entries = collect_entries(
#             index,
#             words=[alternative],
#             lang_code=lang_code,
#         )
#
#         # Full alternative found: retain it and do not subtokenize.
#         if full_alternative_entries:
#             collected.extend(full_alternative_entries)
#             continue
#
#         # Only Chinese source alternatives receive jieba fallback behavior.
#         if lang_code == "zh":
#             subtoken_entries = collect_entries(
#                 index,
#                 words=split_chinese_subtokens(alternative),
#                 lang_code=lang_code,
#             )
#             collected.extend(subtoken_entries)
#
#     return deduplicate_entries(collected)


def build_output_rows(
    rows: list[dict],
    indexes: dict[str, dict[tuple[str, str], list[dict]]],
) -> tuple[list[dict], list[dict], int]:
    output_rows = []
    missing_rows = []
    no_etymology_count = 0

    for row in rows:
        l1_raw = row.get("L1", "").strip().lower()
        l1_code = L1_TO_WIKTIONARY.get(l1_raw)

        if not l1_code:
            continue

        l1_output_code = "cn" if l1_code == "zh" else l1_code
        target_word = row.get("en_norm", "").strip()
        source_alternatives = split_alternatives(row.get("l1_norm", ""))

        target_etym_en_entries = filter_target_entries_by_pos(
            collect_entries(
                indexes["en"],
                words=[target_word],
                lang_code="en",
            ),
            row.get("en_target_pos", ""),
        )

        target_etym_l1_entries = filter_target_entries_by_pos(
            collect_entries(
                indexes[l1_code],
                words=[target_word],
                lang_code="en",
            ),
            row.get("en_target_pos", ""),
        )

        source_etym_en_entries = collect_entries(
            indexes["en"],
            words=source_alternatives,
            lang_code=l1_code,
        )

        # source_etym_en_entries = collect_source_entries(
        #     indexes["en"],
        #     alternatives=source_alternatives,
        #     lang_code=l1_code,
        # )

        source_etym_l1_entries = collect_entries(
            indexes[l1_code],
            words=source_alternatives,
            lang_code=l1_code,
        )

        # source_etym_l1_entries = collect_source_entries(
        #     indexes[l1_code],
        #     alternatives=source_alternatives,
        #     lang_code=l1_code,
        # )

        output_row = {
            "item_id": row.get("item_id", ""),
            "L1": row.get("L1", ""),
            "en_target_word": row.get("en_target_word", ""),
            "en_norm": row.get("en_norm", ""),
            "en_target_pos": row.get("en_target_pos", ""),
            "L1_source_word": row.get("L1_source_word", ""),
            "l1_norm": row.get("l1_norm", ""),
            "source_alternatives": json.dumps(
                source_alternatives,
                ensure_ascii=False,
            ),
            "target_etym_en": serialize_entries(target_etym_en_entries),
            "target_etym_de": "",
            "target_etym_es": "",
            "target_etym_cn": "",
            "source_etym_en": serialize_entries(source_etym_en_entries),
            "source_etym_de": "",
            "source_etym_es": "",
            "source_etym_cn": "",
        }

        output_row[f"target_etym_{l1_output_code}"] = serialize_entries(
            target_etym_l1_entries
        )
        output_row[f"source_etym_{l1_output_code}"] = serialize_entries(
            source_etym_l1_entries
        )

        output_rows.append(output_row)

        missing_fields = [
            field_name
            for field_name in [
                "target_etym_en",
                f"target_etym_{l1_output_code}",
                "source_etym_en",
                f"source_etym_{l1_output_code}",
            ]
            if not output_row.get(field_name)
        ]

        if missing_fields:
            missing_rows.append(
                {
                    "item_id": row.get("item_id", ""),
                    "L1": row.get("L1", ""),
                    "en_norm": row.get("en_norm", ""),
                    "l1_norm": row.get("l1_norm", ""),
                    "missing_fields": ",".join(missing_fields),
                }
            )

        if len(missing_fields) == 4:
            no_etymology_count += 1

    return output_rows, missing_rows, no_etymology_count


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    rows = load_dataset_rows(args.input_csv, limit=args.limit)
    query_sets = build_query_sets(rows)

    indexes = {
        edition_lang: load_matching_entries(
            extract_path,
            edition_lang=edition_lang,
            queries=query_sets[edition_lang],
        )
        for edition_lang, extract_path in EXTRACT_PATHS.items()
    }

    output_rows, missing_rows, no_etymology_count = build_output_rows(rows, indexes)

    write_csv(args.output_csv, OUTPUT_FIELDNAMES, output_rows)

    print(f"Dataset rows processed: {len(rows)}")
    print(f"Output rows written: {len(output_rows)}")
    print(f"Rows with at least one missing lookup: {len(missing_rows)}")
    print(f"Rows with no etymology data whatsoever: {no_etymology_count}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()