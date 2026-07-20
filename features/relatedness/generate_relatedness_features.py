"""Generate LLM-based word relatedness features for the pooled LCP data."""
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELATEDNESS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"
DEFAULT_ETYMOLOGY_PATH = RELATEDNESS_DIR / "output" / "wiktionary_etymologies.csv"
DEFAULT_OUTPUT_DIR = RELATEDNESS_DIR / "output"
DEFAULT_API_KEY_PATH = RELATEDNESS_DIR / "open_api_key.txt"

LANGUAGE_METADATA = {
    "de": {"name": "German", "suffix": "de"},
    "es": {"name": "Spanish", "suffix": "es"},
    "cn": {"name": "Mandarin Chinese", "suffix": "cn"},
}

FEATURE_KEYS = [
    "F1: Lowest Shared Genetic Proto-Language",
    "F2: Borrowed from Source Language",
    "F3: Borrowed from Target Language",
    "F4: Immediate Taxonomic Branch of English Word",
    "F5: Origin Matches Source Language Family",
    "F6: Descendant of the Same Proto-Word",
    "F7: Descendant of the Same Proto-Root",
    "F8: English Transmission Mechanism",
]

PROMPT_TEMPLATE = """You are an expert historical linguist.
You will be presented with a source-target word pair ({source_lang} is the source language, English is the target language).
You must return a JSON object (dictionary) corresponding to the features described below.
You receive the Wiktionary etymology entries for the source and target words for additional context, if available.
If there is no Wiktionary information provided, please answer to the best of your ability.
Since the POS tag of the source word is not given, we also present you with the source context.
If there are multiple etymology entries for a source word, choose the one that matches the POS you infer from the source context as your reference.
## Word Relatedness Features
---
### Feature 1: Lowest Shared Genetic Proto-Language
**Description:** Identifies the youngest reconstructed ancestral language that both words naturally evolved from. This strictly measures genetic lineage, ignoring contact, borrowing, or donor languages.
**Type:** Categorical
**Values:**
*   `Proto-West-Germanic`: Shared genetic ancestor at the West Germanic level.
*   `Proto-Germanic`: Shared genetic ancestor at the broader Germanic level.
*   `Proto-Indo-European`: Shared genetic ancestor at the PIE level.
*   `None`: Used if the words do not share a reconstructed genetic proto-language, OR if their only shared history is via borrowing (e.g., both borrowed from Arabic or Latin).
**Further Comments:** None.
---
### Feature 2: Borrowed from Source Language
**Description:** Indicates whether the English target word was directly borrowed from the specified source language.
**Type:** Binary
**Values:**
*   `0`: English target word was not borrowed from the source language.
*   `1`: English target word was directly borrowed from the source language (e.g., English *kindergarten* from German).
**Further Comments:** None.
---
### Feature 3: Borrowed from Target Language
**Description:** Indicates whether the source word was directly borrowed from English.
**Type:** Binary
**Values:**
*   `0`: Source word was not borrowed from English.
*   `1`: Source word was directly borrowed from English (e.g., German *Computer*, Spanish *fútbol*).
**Further Comments:** None.
---
### Feature 4: Immediate Taxonomic Branch of English Word
**Description:** Classifies the broad language branch or family of the immediate predecessor from which English acquired the word. This does not ask for the specific language, but rather the linguistic category of the immediate donor or ancestor.
**Type:** Categorical
**Values:**
*   `Germanic`: Entered from Old English, Old Norse, Dutch, German, etc.
*   `Romance`: Entered from French, Latin, Spanish, Italian, etc.
*   `Hellenic`: Entered from Greek.
*   `Semitic`: Entered from Arabic, Hebrew, etc.
*   `Sinitic`: Entered from Mandarin, Cantonese, etc.
*   `Other`: Any other taxonomic branch not covered above.
**Further Comments:** None.
---
### Feature 5: Origin Matches Source Language Family?
**Description:** Determines if the immediate linguistic origin of the English target word aligns with the broader language family of the source word.
**Type:** Binary
**Values:**
*   `0`: English target word origin does not match source language family.
*   `1`: English target word origin matches source language family.
**Further Comments:** Assign `1` if the English target word's immediate taxonomic branch (assigned in Feature 4) matches the broader language family of the source language. Specifically: if the source language is German, assign `1` if Feature 4 is 'Germanic'; if the source language is Spanish, assign `1` if Feature 4 is 'Romance'; if the source language is Mandarin Chinese, assign `1` if Feature 4 is 'Sinitic'. Otherwise, assign `0`.
---
### Feature 6: Descendant of the Same Proto-Word?
**Description:** Evaluates if both words descend from the exact same reconstructed ancestral lexical item at the youngest shared proto-language stage.
**Type:** Binary
**Values:**
*   `0`: Source and target words do not descend from the same reconstructed ancestral lexical item. They may share a broader root, but are not descendants of the same proto-word.
*   `1`: Source and target words are descendants of the same reconstructed ancestral lexical item at the youngest relevant shared proto-language stage.
**Further Comments:** "Proto-word" means a reconstructed lexical form, not merely a root or semantic concept. Two words count as sharing the same proto-word even if they diverged into different language branches. Direct cognacy does not require the modern forms to look similar. If a shared reconstructed lexical item exists, assign `1`. Examples: Spanish *padre* and English *father* both descend from PIE *\\*ph₂tḗr* (Assign 1). Spanish *pie* and English *foot* both descend from the exact same PIE proto-word *\\*pṓds* (Assign 1).
---
### Feature 7: Descendant of the Same Proto-Root?
**Description:** Evaluates if the words share a broader ancestral root or word family, specifically when they do not descend from the exact same reconstructed proto-word.
**Type:** Binary
**Values:**
*   `0`: Source and target words do not share any reconstructable ancestral root or word family.
*   `1`: Source and target words share a broader ancestral root or word family but are NOT descendants of the same reconstructed proto-word.
**Further Comments:** Assign `1` if the words share a historical root or word family but do NOT descend from the exact same reconstructed proto-word. F6 and F7 are mutually exclusive based on the youngest shared ancestor: if they share a proto-word, assign F6=1 and F7=0. If they only share a broader root, assign F6=0 and F7=1. Example: Spanish *juventud* and English *youth* both stem from the shared Proto-Indo-European root for "young", but they do not descend from the exact same proto-word because the Italic and Germanic branches added completely different suffixes to that base root to form their respective nouns (F6=0, F7=1).
---
### Feature 8: English Transmission Mechanism
**Description:** Describes the mechanical path of how the word reached Modern English, ignoring which languages were involved and focusing entirely on the complexity of the transmission history.
**Type:** Categorical
**Values:**
*   `Inherited`: Passed down continuously into Modern English from its deepest ancestral roots with no borrowing events (0 hops).
*   `Direct`: Borrowed directly into English from the ultimate source language with no intermediary languages (1 hop).
*   `Indirect`: Passed through at least one intermediary language before reaching English (2+ hops).
**Further Comments:** None.
---
## One-Shot Example
Input:
Spanish source word: padre
English target word: father
Source context:
Mi padre siempre me da buenos consejos, y mi madre también.
Wiktionary Info:
Source word in Spanish Wiktionary:
Inherited from Old Spanish padre ("father"), from Latin patrem, pater, from Proto-Italic patēr, from Proto-Indo-European ph₂tḗr.
Target word in English Wiktionary:
Inherited from Middle English fader, from Old English fæder, from Proto-West Germanic fader, from Proto-Germanic fadēr, from Proto-Indo-European ph₂tḗr, possibly from peh₂- + *-tḗr. Doublet of ayr, faeder, athair, padre, pater and père.
Output:
{{
  "F1: Lowest Shared Genetic Proto-Language": "Proto-Indo-European",
  "F2: Borrowed from Source Language": 0,
  "F3: Borrowed from Target Language": 0,
  "F4: Immediate Taxonomic Branch of English Word": "Germanic",
  "F5: Origin Matches Source Language Family": 0,
  "F6: Descendant of the Same Proto-Word": 1,
  "F7: Descendant of the Same Proto-Root": 0,
  "F8: English Transmission Mechanism": "Inherited"
}}
## Your Task
Now, process the following word pair and return ONLY the JSON object. Do not include any markdown formatting, explanations, or text outside the JSON object.
Input:
Input:
{source_lang} source word: {l1_source}
English target word: {en_target}
Source context:
{source_context}
Wiktionary Info:
Source word in {source_lang} Wiktionary:
{source_etymology_l1}
Source word in English Wiktionary:
{source_etymology_en}
Target word in {source_lang} Wiktionary:
{target_etymology_l1}
Target word in English Wiktionary:
{target_etymology_en}
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate word relatedness features for the pooled LCP data."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5",
        help="Model name (e.g. 'gpt-4o', 'gpt-5').",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to pooled_preprocessed.csv (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "-e",
        "--etymology-path",
        type=Path,
        default=DEFAULT_ETYMOLOGY_PATH,
        help=f"Path to wiktionary_etymologies.csv (default: {DEFAULT_ETYMOLOGY_PATH}).",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path (default: features/relatedness/output/relatedness_features_<model>.csv).",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help="Text file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Optional row limit for testing/debugging.",
    )
    return parser.parse_args()


def normalize_key(value: Any) -> str:
    """Normalize values for dictionary lookup and matching."""
    if value is None:
        return ""
    return str(value).strip().casefold()


def load_dataframe(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with stable string values."""
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=limit)
    return dataframe.fillna("")


def add_lookup_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create normalized lookup columns used for matching rows across files."""
    keyed = dataframe.copy()
    keyed["_l1_key"] = keyed["L1"].map(normalize_key)
    keyed["_en_target_key"] = keyed["en_target_word"].map(normalize_key)
    keyed["_source_key"] = keyed["L1_source_word"].map(normalize_key)
    return keyed


def build_etymology_lookup(
        etymology_df: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Index etymology rows by normalized (L1, en_target_word)."""
    lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in etymology_df.groupby(["_l1_key", "_en_target_key"], sort=False):
        lookup[(str(key[0]), str(key[1]))] = group
    return lookup


def select_etymology_row(
        data_row: dict[str, Any],
        lookup: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    """Select the best etymology row for a data row."""
    key = (
        normalize_key(data_row.get("L1")),
        normalize_key(data_row.get("en_target_word")),
    )
    candidates = lookup.get(key)
    if candidates is None or candidates.empty:
        return {}
    if len(candidates) == 1:
        return cast(dict[str, Any], candidates.iloc[0].to_dict())
    source_word = normalize_key(data_row.get("L1_source_word"))
    source_matches = candidates[candidates["_source_key"] == source_word]
    if not source_matches.empty:
        return cast(dict[str, Any], source_matches.iloc[0].to_dict())
    return cast(dict[str, Any], candidates.iloc[0].to_dict())


def parse_serialized_value(value: str) -> Any:
    """Parse a CSV cell that may contain JSON-serialized content."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def stringify_nested_value(value: Any) -> str:
    """Convert nested etymology payload content to readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [stringify_nested_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [
            f"{key}: {stringify_nested_value(nested_value)}"
            for key, nested_value in value.items()
            if stringify_nested_value(nested_value)
        ]
        return "; ".join(parts)
    return str(value)


def format_etymology_entries(value: str) -> str:
    """Format raw etymology cell content into prompt-friendly text."""
    parsed = parse_serialized_value(value)
    if parsed == "":
        return "Not available."
    if isinstance(parsed, list):
        formatted_entries = []
        for entry in parsed:
            if isinstance(entry, dict):
                etymology_text = stringify_nested_value(entry.get("etymology_text"))
                details = []
                matched_word = stringify_nested_value(entry.get("matched_word"))
                edition_lang = stringify_nested_value(entry.get("edition_lang"))
                lang_code = stringify_nested_value(entry.get("lang_code"))
                pos = stringify_nested_value(entry.get("pos"))
                if matched_word:
                    details.append(f"matched_word={matched_word}")
                if edition_lang:
                    details.append(f"edition={edition_lang}")
                if lang_code:
                    details.append(f"lang_code={lang_code}")
                if pos:
                    details.append(f"pos={pos}")
                if etymology_text:
                    if details:
                        formatted_entries.append(
                            f"- {'; '.join(details)} | etymology: {etymology_text}"
                        )
                    else:
                        formatted_entries.append(f"- {etymology_text}")
                elif details:
                    formatted_entries.append(f"- {'; '.join(details)}")
            else:
                text = stringify_nested_value(entry)
                if text:
                    formatted_entries.append(f"- {text}")
        return "\n".join(formatted_entries) if formatted_entries else "Not available."
    if isinstance(parsed, dict):
        text = stringify_nested_value(parsed)
        return text or "Not available."
    text = stringify_nested_value(parsed)
    return text or "Not available."


def build_prompt(data_row: dict[str, Any], etymology_row: dict[str, Any], l1_code: str) -> str:
    """Build the prompt with just the variable data."""
    metadata = LANGUAGE_METADATA[l1_code]
    suffix = metadata["suffix"]
    source_lang_name = metadata["name"]

    return PROMPT_TEMPLATE.format(
        source_lang=source_lang_name,
        l1_source=data_row.get("L1_source_word", ""),
        en_target=data_row.get("en_target_word", ""),
        source_context=data_row.get("L1_context", ""),
        source_etymology_l1=format_etymology_entries(etymology_row.get(f"source_etym_{suffix}", "")),
        source_etymology_en=format_etymology_entries(etymology_row.get("source_etym_en", "")),
        target_etymology_l1=format_etymology_entries(etymology_row.get(f"target_etym_{suffix}", "")),
        target_etymology_en=format_etymology_entries(etymology_row.get("target_etym_en", "")),
    )


def load_api_key(api_key_path: Path | None) -> str:
    """Load the OpenAI API key from an env var or text file."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    candidate_paths = []
    if api_key_path is not None:
        candidate_paths.append(api_key_path)
    candidate_paths.extend(
        [
            PROJECT_ROOT / "API-key.txt",
            RELATEDNESS_DIR.parent / "API-key.txt",
        ]
    )
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        "No OpenAI API key found. Set OPENAI_API_KEY or pass --api-key-path."
    )


def build_client(api_key: str):
    """Create an OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required to run this script. Install it first."
        ) from exc
    return OpenAI(api_key=api_key)


def extract_response_text(response: Any) -> str:
    """Extract text from OpenAI response."""
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise ValueError("Could not extract text from the OpenAI response.")


def request_prediction(client: Any, model: str, prompt: str) -> str:
    """Send the prompt to the model and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert historical linguist."},
            {"role": "user", "content": prompt},
        ],
    )
    return extract_response_text(response)


def parse_prediction(raw_text: str) -> dict[str, Any]:
    """Parse the model response as JSON, tolerating extra surrounding text."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start: end + 1])
        raise


def main() -> None:
    """Run the relatedness feature generation pipeline."""
    args = parse_args()
    output_path = args.output_path or (
            DEFAULT_OUTPUT_DIR / f"relatedness_features_{args.model.replace('/', '_')}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model}")
    print(f"Reading data from: {args.data_path}")
    print(f"Reading etymology data from: {args.etymology_path}")
    print(f"Writing output to: {output_path}")

    data_df = add_lookup_keys(load_dataframe(args.data_path, limit=args.limit))
    etymology_df = add_lookup_keys(load_dataframe(args.etymology_path))
    etymology_lookup = build_etymology_lookup(etymology_df)

    if data_df.empty:
        raise ValueError(f"No rows found in {args.data_path}")

    api_key = load_api_key(args.api_key_path)
    client = build_client(api_key)

    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(data_df.to_dict(orient="records"), start=1):
        data_row = cast(dict[str, Any], row)
        l1_code = normalize_key(data_row.get("L1"))

        if l1_code not in LANGUAGE_METADATA:
            raise ValueError(
                f"Unsupported source language code on row {index}: {data_row.get('L1')!r}"
            )

        etymology_row = select_etymology_row(data_row, etymology_lookup)
        prompt = build_prompt(data_row, etymology_row, l1_code)
        raw_response = request_prediction(client, args.model, prompt)
        prediction = parse_prediction(raw_response)

        output_row = {
            "item_id": data_row.get("item_id", ""),
            "L1": data_row.get("L1", ""),
            "en_target_word": data_row.get("en_target_word", ""),
        }
        output_row.update({key: prediction.get(key, "") for key in FEATURE_KEYS})
        output_rows.append(output_row)

        print(
            f"[{index}/{len(data_df)}] {data_row.get('L1_source_word', '')} -> "
            f"{data_row.get('en_target_word', '')}"
        )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Done! Output saved to {output_path}")


if __name__ == "__main__":
    main()