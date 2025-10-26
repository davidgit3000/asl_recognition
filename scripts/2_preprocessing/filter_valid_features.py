#!/usr/bin/env python3
"""
Filter manifest to only include samples with valid (non-zero) features.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

# Load config
cfg = yaml.safe_load(open("configs/config.yaml"))
manifest_path = Path(cfg["manifest_out"])
features_dir = Path(cfg["artifacts_root"]) / "features"

# Load manifest
df = pd.read_csv(manifest_path)
print(f"Original manifest: {len(df)} samples")

# Check each sample
valid_ids = []
zero_ids = []
missing_ids = []

print("\nScanning features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    feat_path = features_dir / f"{row['id']}.npy"
    
    if not feat_path.exists():
        missing_ids.append(row['id'])
        continue
    
    feat = np.load(feat_path)
    
    # Check if all zeros
    if feat.max() == 0.0 and feat.min() == 0.0:
        zero_ids.append(row['id'])
    else:
        valid_ids.append(row['id'])

# Filter manifest
df_valid = df[df['id'].isin(valid_ids)].copy()

print(f"\nResults:")
print(f"  Valid samples: {len(valid_ids)} ({100*len(valid_ids)/len(df):.1f}%)")
print(f"  Zero samples: {len(zero_ids)} ({100*len(zero_ids)/len(df):.1f}%)")
print(f"  Missing samples: {len(missing_ids)} ({100*len(missing_ids)/len(df):.1f}%)")

# Check class distribution after filtering
print(f"\nClass distribution after filtering:")
print(df_valid['label'].value_counts().head(10))

# FILTER OUT CLASSES WITH TOO FEW SAMPLES (< 100)
MIN_SAMPLES_PER_CLASS = 100
class_counts = df_valid['label'].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
removed_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()

print(f"\n⚠️  Removing {len(removed_classes)} classes with < {MIN_SAMPLES_PER_CLASS} samples:")
for cls in removed_classes:
    print(f"   - {cls}: {class_counts[cls]} samples")

df_valid = df_valid[df_valid['label'].isin(valid_classes)].copy()
print(f"\n✅ After class filtering: {len(df_valid)} samples, {len(valid_classes)} classes")

# Save filtered manifest
output_path = manifest_path.parent / "manifest_filtered.csv"
df_valid.to_csv(output_path, index=False)

print(f"\n✅ Filtered manifest saved to: {output_path}")
print(f"   {len(df_valid)} valid samples")

# Instructions
print(f"\n" + "="*60)
print(f"Next Steps:")
print(f"="*60)
print(f"1. Add this line to configs/config.yaml:")
print(f"   manifest_out_filtered: \"{output_path}\"")
print(f"")
print(f"2. The training script will automatically use the filtered manifest")
print(f"   (dataloader checks for 'manifest_out_filtered' first)")
print(f"")
print(f"3. Run training:")
print(f"   python scripts/3_training/train_baseline.py --epochs 100")
