import json

import pandas as pd

from main import CFG, evaluate, aggregate_aspects, generate_report

paths = CFG["paths"]
df = pd.read_csv(paths["results_csv"]).fillna("")
df["aspects"] = df["aspects"].apply(json.loads)

evaluate(df)
aspect_df = aggregate_aspects(df)
aspect_df.to_csv(paths["aspects_csv"], index=False)
report = generate_report(df, aspect_df)
with open(paths["report_md"], "w", encoding="utf-8") as f:
    f.write(report)
print("Rebuilt metrics, aspects and report from existing results (1 API call).")