## Noah-Manuel Michael
## Created: 2026-07-20
## Updated: 2026-07-20
## Script to read in the SUBTLEX-UK frequencies and save them to the feature enriched dataset.

import pandas as pd
import syllables
from pathlib import Path

pd.set_option('display.max_columns', None)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREQ_PATH = PROJECT_ROOT / "data" / "original" / "SUBTLEX-UK.xlsx"
DATA_PATH = PROJECT_ROOT / "data" / "pooled" / "pooled_preprocessed.csv"

_US_TO_UK_SPELLING = {
    "industrialization": "industrialisation",
    "mobilization": "mobilisation",
    "modernization": "modernisation",
    "visualization": "visualisation",
}


def load_subtlex_frequencies(freq_path: Path) -> pd.DataFrame:
    """
    Load SUBTLEX-UK frequency data and return a DataFrame with lowercase spelling and log frequency.
    """
    freq_df = pd.read_excel(freq_path, usecols=["Spelling", "LogFreq(Zipf)"])
    freq_df["spelling_lower"] = freq_df["Spelling"].str.lower()
    freq_df = freq_df.drop(columns=["Spelling"])
    # Drop duplicates keeping the first occurrence
    freq_df = freq_df.drop_duplicates(subset="spelling_lower", keep="first")
    return freq_df


def add_log_frequency(data_df: pd.DataFrame, freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``log_frequency`` column by looking up *en_norm* in SUBTLEX-UK.
    """
    data_df["spelling_lower"] = data_df["en_norm"].str.lower().replace(_US_TO_UK_SPELLING)
    data_df = data_df.merge(freq_df, on="spelling_lower", how="left")
    data_df = data_df.rename(columns={"LogFreq(Zipf)": "log_frequency"})
    data_df = data_df.drop(columns=["spelling_lower"])

    # Fill remaining missing frequencies with the lowest observed value
    if data_df["log_frequency"].isna().any():
        min_freq = data_df["log_frequency"].min()
        data_df["log_frequency"] = data_df["log_frequency"].fillna(min_freq)

    return data_df


def add_target_word_length(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``target_word_length`` column (character count of *en_norm*).
    """
    data_df["target_word_length"] = data_df["en_norm"].str.len()
    return data_df


def get_syllable_count(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the estimated syllable count as a feature.
    :param data_df:
    :return:
    """
    data_df["syllable_count"] = data_df["en_norm"].apply(syllables.estimate)
    return data_df


def main():
    freq_df = load_subtlex_frequencies(FREQ_PATH)
    data_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    data_df = add_log_frequency(data_df, freq_df)
    data_df = add_target_word_length(data_df)
    data_df = get_syllable_count(data_df)
    data_df.to_csv(DATA_PATH, index=False)


if __name__ == "__main__":
    main()