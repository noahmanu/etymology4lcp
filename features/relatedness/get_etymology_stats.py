## Noah-Manuel Michael
## Created: 2026-07-21
## Updated: 2026-07-21
## Script to print etymology statistics per language.

import pandas as pd

pd.options.display.max_columns = None


def main():
    l1s = ["cn", "de", "es"]
    df_etymology = pd.read_csv('output/wiktionary_etymologies.csv')
    print(df_etymology.head())
    for l1 in l1s:
        target_counter = 0
        source_counter = 0
        for index, row in df_etymology.iterrows():
            if row["L1"] == l1:
                if any([type(row["target_etym_en"]) == str, type(row[f"target_etym_{l1}"]) == str]):
                    target_counter += 1
                if any([type(row["source_etym_en"]) == str, type(row[f"source_etym_{l1}"]) == str]):
                    source_counter += 1
        print(l1, target_counter, source_counter)


if __name__ == "__main__":
    main()