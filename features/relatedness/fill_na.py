import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

csv_path = "output/relatedness_features_gpt-5.6-luna.csv"
csv_path_out = "output/relatedness_features_gpt-5.6-luna_nafilled.csv"

# Keep pandas NA detection, but treat literal "None" as a real string value.
na_values_without_none = set(STR_NA_VALUES) - {"None"}
df = pd.read_csv(csv_path, keep_default_na=False, na_values=na_values_without_none)
df[["same_proto_word", "same_proto_root"]] = df[
    ["same_proto_word", "same_proto_root"]
].fillna(0).astype("int64")

df.to_csv(csv_path_out, index=False)
print(f"Updated file written to {csv_path_out}")
