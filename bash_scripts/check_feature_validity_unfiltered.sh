#!/bin/bash
# Check feature validity BEFORE filtering (using manifest_v1.csv)
# Usage: bash scripts/2_preprocessing/check_feature_validity_unfiltered.sh

echo "============================================================"
echo "Feature Validity Checker (BEFORE FILTERING)"
echo "============================================================"
echo ""

# Activate virtual environment if needed
if [ -d ".venv311" ]; then
    source .venv311/bin/activate
fi

# Run Python script to check features
python3 << 'EOF'
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

# Load config
cfg = yaml.safe_load(open('configs/config.yaml'))
# Use the UNFILTERED manifest (manifest_out, not manifest_out_filtered)
manifest_path = cfg['manifest_out']
features_dir = Path(cfg['artifacts_root']) / 'features'

# Load manifest
df = pd.read_csv(manifest_path)
print(f"Manifest: {manifest_path}")
print(f"Total samples in manifest: {len(df):,}")
print(f"Features directory: {features_dir}")
print("")

# Check each feature file
zero_count = 0
valid_count = 0
missing_count = 0
corrupted_count = 0

zero_samples = []
valid_samples = []
missing_samples = []

print("Scanning features...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
    feat_path = features_dir / f"{row['id']}.npy"
    
    # Check if file exists
    if not feat_path.exists():
        missing_count += 1
        missing_samples.append(row['id'])
        continue
    
    try:
        # Load feature
        feat = np.load(feat_path)
        
        # Check if all zeros
        if feat.max() == 0.0 and feat.min() == 0.0:
            zero_count += 1
            zero_samples.append(row['id'])
        else:
            valid_count += 1
            valid_samples.append(row['id'])
            
    except Exception as e:
        corrupted_count += 1
        print(f"Error loading {row['id']}: {e}")

# Calculate statistics
total_checked = zero_count + valid_count + missing_count + corrupted_count
valid_pct = (valid_count / total_checked * 100) if total_checked > 0 else 0
zero_pct = (zero_count / total_checked * 100) if total_checked > 0 else 0
missing_pct = (missing_count / total_checked * 100) if total_checked > 0 else 0

print("")
print("="*60)
print("RESULTS (BEFORE FILTERING)")
print("="*60)
print(f"✅ Valid features:     {valid_count:,} ({valid_pct:.2f}%)")
print(f"❌ Zero features:      {zero_count:,} ({zero_pct:.2f}%)")
print(f"⚠️  Missing files:      {missing_count:,} ({missing_pct:.2f}%)")
print(f"🔥 Corrupted files:    {corrupted_count:,}")
print(f"📊 Total checked:      {total_checked:,}")
print("")

# Show improvement from filtering
print("="*60)
print("FILTERING IMPACT")
print("="*60)
print(f"Before filtering:  {total_checked:,} samples")
print(f"Valid samples:     {valid_count:,} ({valid_pct:.1f}%)")
print(f"Zero samples:      {zero_count:,} ({zero_pct:.1f}%)")
print(f"Missing samples:   {missing_count:,} ({missing_pct:.1f}%)")
print("")
print(f"💡 Filtering removed {zero_count + missing_count:,} unusable samples")
print(f"💡 Improvement: {valid_pct:.1f}% valid rate")
print("")

# Detailed breakdown by split
if valid_count > 0:
    print("="*60)
    print("BREAKDOWN BY SPLIT (BEFORE FILTERING)")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_valid = len([s for s in valid_samples if s in split_df['id'].values])
        split_zero = len([s for s in zero_samples if s in split_df['id'].values])
        split_missing = len([s for s in missing_samples if s in split_df['id'].values])
        split_total = len(split_df)
        split_valid_pct = (split_valid / split_total * 100) if split_total > 0 else 0
        
        print(f"{split.upper():5s}: {split_valid:,}/{split_total:,} valid ({split_valid_pct:.1f}%), "
              f"{split_zero:,} zero, {split_missing:,} missing")
    print("")

# Detailed breakdown by source
if valid_count > 0:
    print("="*60)
    print("BREAKDOWN BY SOURCE (BEFORE FILTERING)")
    print("="*60)
    
    for source in sorted(df['source'].unique()):
        source_df = df[df['source'] == source]
        source_valid = len([s for s in valid_samples if s in source_df['id'].values])
        source_zero = len([s for s in zero_samples if s in source_df['id'].values])
        source_missing = len([s for s in missing_samples if s in source_df['id'].values])
        source_total = len(source_df)
        source_valid_pct = (source_valid / source_total * 100) if source_total > 0 else 0
        
        print(f"{source:15s}: {source_valid:,}/{source_total:,} valid ({source_valid_pct:.1f}%), "
              f"{source_zero:,} zero, {source_missing:,} missing")
    print("")

# Breakdown by label (show classes with high zero rates)
if zero_count > 0:
    print("="*60)
    print("CLASSES WITH HIGHEST ZERO RATES")
    print("="*60)
    
    label_stats = {}
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_valid = len([s for s in valid_samples if s in label_df['id'].values])
        label_zero = len([s for s in zero_samples if s in label_df['id'].values])
        label_total = len(label_df)
        label_zero_pct = (label_zero / label_total * 100) if label_total > 0 else 0
        label_stats[label] = (label_zero, label_total, label_zero_pct)
    
    # Sort by zero percentage (descending)
    sorted_labels = sorted(label_stats.items(), key=lambda x: x[1][2], reverse=True)
    
    print("Top 10 classes with most zeros:")
    for i, (label, (zeros, total, pct)) in enumerate(sorted_labels[:10], 1):
        print(f"{i:2d}. {label:15s}: {zeros:,}/{total:,} zero ({pct:.1f}%)")
    print("")

# Comparison table
print("="*60)
print("BEFORE vs AFTER FILTERING COMPARISON")
print("="*60)
print(f"{'Metric':<25s} {'Before':>12s} {'After':>12s} {'Change':>12s}")
print("-"*60)
print(f"{'Total samples':<25s} {total_checked:>12,} {valid_count:>12,} {valid_count - total_checked:>12,}")
print(f"{'Valid samples':<25s} {valid_count:>12,} {valid_count:>12,} {0:>12,}")
print(f"{'Zero samples':<25s} {zero_count:>12,} {0:>12,} {-zero_count:>12,}")
print(f"{'Missing samples':<25s} {missing_count:>12,} {0:>12,} {-missing_count:>12,}")
print(f"{'Valid rate':<25s} {valid_pct:>11.1f}% {100.0:>11.1f}% {100.0 - valid_pct:>11.1f}%")
print("")

print("="*60)

EOF

echo ""
echo "✅ Feature validity check (unfiltered) complete!"
echo ""
