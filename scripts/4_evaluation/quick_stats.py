# This file will be used as template - keeping original for now
import pandas as pd, numpy as np, yaml
from pathlib import Path
CFG = yaml.safe_load(open("configs/config.yaml"))
df = pd.read_csv(CFG["manifest_out"])
print("Total items:", len(df))
print("By source:\n", df.groupby("source").size())
print("By label (top 10):\n", df.groupby("label").size().sort_values(ascending=False).head(10))
