from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POOLED_PATH = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"
RELATEDNESS_PATH = (
    PROJECT_ROOT
    / "features"
    / "relatedness"
    / "output"
    / "relatedness_features_gpt-5.6-luna_nafilled.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "feature_enriched" / "full.csv"

relatedness_columns = [
    "item_id",
    "L1",
    "lca_language",
    "borrowed_from_source",
    "borrowed_from_target",
    "borrowed_from_third",
    "english_word_origin",
    "origin_matches_source_family",
    "same_proto_word",
    "same_proto_root",
    "transmission_path",
]

pooled_df = pd.read_csv(POOLED_PATH, keep_default_na=False)
relatedness_df = pd.read_csv(RELATEDNESS_PATH, keep_default_na=False)

full_df = pooled_df.merge(
    relatedness_df[relatedness_columns],
    on=["item_id", "L1"],
    how="left",
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
full_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved merged dataset to {OUTPUT_PATH}")




