import pandas as pd

df = pd.read_parquet("data/traces/rl_train.parquet")
print("rows:", len(df))
print("cols:", list(df.columns))
r = df.iloc[0]
print("data_source:", r["data_source"])
print("ground_truth:", r["reward_model"]["ground_truth"])
print("prompt:", r["prompt"])
