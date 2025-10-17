import pandas as pd, yaml
from pathlib import Path

CFG = yaml.safe_load(open("configs/config.yaml"))

IN_CSV   = Path("artifacts/manifests/msasl_all.csv")          # made by build_msasl_manifest.py
OUT_CSV  = Path("artifacts/manifests/msasl_segments.csv")      # list for downloading/trim
CHOSEN_TXT = Path("artifacts/manifests/selected_msasl_classes.txt")
COUNTS_CSV = Path("artifacts/manifests/msasl_counts.csv")

# ---- knobs (override in config.yaml if you want) ----
TOP_K_CLASSES  = int(CFG.get("msasl_top_k", 20))   # total classes you want
MAX_PER_CLASS  = int(CFG.get("msasl_per_class", 10))  # clips per class
OVER_SAMPLE    = int(CFG.get("msasl_over_sample", 3))  # over-sample factor for fallback
EXPLICIT = [c.strip().lower().replace(" ", "_")
            for c in CFG.get("msasl_selected_classes", [])]  # optional explicit list

# --------- load & clean ----------
df = pd.read_csv(IN_CSV)
df = df[df["duration_sec"] >= 0.5]
df = df[df["yt_url"].notna() & (df["yt_url"].str.len() > 0)]

# --------- compute availability per class ----------
counts = df.groupby("label_text").size().sort_values(ascending=False)
COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
counts.to_csv(COUNTS_CSV, header=["count"])
print(f"Saved availability report → {COUNTS_CSV}")

# --------- pick classes with sufficient candidates ----------
# Only keep labels with at least MAX_PER_CLASS candidates
viable = counts[counts >= MAX_PER_CLASS].index.tolist()

if EXPLICIT:
    # Start with explicit list, but only keep viable ones
    chosen = [c for c in EXPLICIT if c in viable]
    dropped = [c for c in EXPLICIT if c not in viable]
    if dropped:
        print(f"⚠️  Dropped {len(dropped)} labels with <{MAX_PER_CLASS} candidates: {dropped}")
    
    # Backfill to reach TOP_K_CLASSES if needed
    if len(chosen) < TOP_K_CLASSES:
        backfill_pool = [c for c in viable if c not in chosen]
        needed = TOP_K_CLASSES - len(chosen)
        backfill = backfill_pool[:needed]
        chosen.extend(backfill)
        if backfill:
            print(f"✓ Backfilled {len(backfill)} labels to reach {TOP_K_CLASSES}: {backfill}")
else:
    # auto-pick the top classes by count
    chosen = viable[:TOP_K_CLASSES]

chosen = chosen[:TOP_K_CLASSES]  # cap at TOP_K
df = df[df["label_text"].isin(chosen)]

print(f"Selected {len(chosen)} classes with ≥{MAX_PER_CLASS} candidates each")

# --------- rank candidates per class ----------
# prefer diversity & quality: sort by signer_id then by longer duration
df = df.sort_values(["label_text", "signer_id", "duration_sec"],
                    ascending=[True, True, False])

# Over-sample: keep up to OVER_SAMPLE * MAX_PER_CLASS per class, ranked
def top_k_diverse_ranked(g, k):
    g1 = g.drop_duplicates("signer_id", keep="first")
    if len(g1) >= k: 
        result = g1.head(k)
    else:
        rest = g[~g.index.isin(g1.index)]
        result = pd.concat([g1, rest.head(k - len(g1))])
    # Add rank column
    result = result.copy()
    result["rank"] = range(1, len(result) + 1)
    return result

over_sample_count = OVER_SAMPLE * MAX_PER_CLASS
df = df.groupby("label_text", group_keys=False).apply(lambda g: top_k_diverse_ranked(g, over_sample_count))

# save the final class list (for transparency)
CHOSEN_TXT.parent.mkdir(parents=True, exist_ok=True)
CHOSEN_TXT.write_text("\n".join(sorted(df["label_text"].unique())), encoding="utf-8")

# keep what the downloader needs + rank
keep_cols = ["label_text","yt_url","start_time","end_time","signer_id","fps","width","height","rank"]
df[keep_cols].to_csv(OUT_CSV, index=False)

print(f"Wrote {len(df)} candidate rows (up to {over_sample_count} per class) → {OUT_CSV}")
print(f"Saved chosen classes → {CHOSEN_TXT}")
