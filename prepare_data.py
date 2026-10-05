''' import pandas as pd
import yaml

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["data"]

df = pd.read_csv(cfg["raw_csv"], usecols=["ProductId", "Score", "Text"])
print(f"Loaded {len(df)} reviews")

df = df.rename(columns={"Text": "review", "Score": "original_rating",
                        "ProductId": "product_id"})
df["review"] = (
    df["review"].fillna("")
    .str.replace(r"<[^>]+>", " ", regex=True)   # remove HTML tags
    .str.replace(r"s+", " ", regex=True)
    .str.strip()
)
df = df[df["review"].str.len().between(cfg["min_chars"], cfg["max_chars"])]
df = df.drop_duplicates(subset="review")
print(f"After cleaning: {len(df)} reviews")

sample = df.groupby("original_rating").sample(n=cfg["per_star"], random_state=cfg["seed"])
sample = sample.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
sample.to_csv(cfg["sample_csv"], index=False)

print(f"Saved {len(sample)} reviews to {cfg['sample_csv']}")
print(sample["original_rating"].value_counts().sort_index())
'''

import pandas as pd
import yaml

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["data"]

df = pd.read_csv(cfg["raw_csv"], usecols=["product_id", "original_rating", "review"])

print(f"Loaded {len(df)} reviews")

df["review"] = (
    df["review"]
    .fillna("")
    .str.replace(r"<[^>]+>", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[df["review"].str.len().between(cfg["min_chars"], cfg["max_chars"])]
df = df.drop_duplicates(subset="review")

print(f"After cleaning: {len(df)} reviews")

sample = (
    df.groupby("original_rating")
    .sample(n=cfg["per_star"], random_state=cfg["seed"])
)

sample = sample.sample(
    frac=1,
    random_state=cfg["seed"]
).reset_index(drop=True)

sample.to_csv(cfg["sample_csv"], index=False)

print(f"Saved {len(sample)} reviews to {cfg['sample_csv']}")
print(sample["original_rating"].value_counts().sort_index())