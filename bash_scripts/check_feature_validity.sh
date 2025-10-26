#!/bin/bash
# Check feature validity: count valid vs zero features
# Usage: bash scripts/2_preprocessing/check_feature_validity.sh

echo "============================================================"
echo "Feature Validity Checker"
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
manifest_path = cfg.get('manifest_out_filtered', cfg['manifest_out'])
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
print("RESULTS")
print("="*60)
print(f"✅ Valid features:     {valid_count:,} ({valid_pct:.2f}%)")
print(f"❌ Zero features:      {zero_count:,} ({zero_pct:.2f}%)")
print(f"⚠️  Missing files:      {missing_count:,} ({missing_pct:.2f}%)")
print(f"🔥 Corrupted files:    {corrupted_count:,}")
print(f"📊 Total checked:      {total_checked:,}")
print("")

# Detailed breakdown by split
if valid_count > 0:
    print("="*60)
    print("BREAKDOWN BY SPLIT")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_valid = len([s for s in valid_samples if s in split_df['id'].values])
        split_zero = len([s for s in zero_samples if s in split_df['id'].values])
        split_total = len(split_df)
        split_valid_pct = (split_valid / split_total * 100) if split_total > 0 else 0
        
        print(f"{split.upper():5s}: {split_valid:,}/{split_total:,} valid ({split_valid_pct:.1f}%)")
    print("")

# Detailed breakdown by source
if valid_count > 0:
    print("="*60)
    print("BREAKDOWN BY SOURCE")
    print("="*60)
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        source_valid = len([s for s in valid_samples if s in source_df['id'].values])
        source_zero = len([s for s in zero_samples if s in source_df['id'].values])
        source_total = len(source_df)
        source_valid_pct = (source_valid / source_total * 100) if source_total > 0 else 0
        
        print(f"{source:15s}: {source_valid:,}/{source_total:,} valid ({source_valid_pct:.1f}%)")
    print("")

# Feature statistics for valid samples
if valid_count > 0:
    print("="*60)
    print("FEATURE STATISTICS (sample of 100 valid features)")
    print("="*60)
    
    # Sample 100 random valid features
    sample_ids = np.random.choice(valid_samples, min(100, len(valid_samples)), replace=False)
    
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    
    for sample_id in sample_ids:
        feat_path = features_dir / f"{sample_id}.npy"
        feat = np.load(feat_path)
        all_mins.append(feat.min())
        all_maxs.append(feat.max())
        all_means.append(feat.mean())
        all_stds.append(feat.std())
    
    print(f"Min value:    {np.mean(all_mins):.4f} ± {np.std(all_mins):.4f}")
    print(f"Max value:    {np.mean(all_maxs):.4f} ± {np.std(all_maxs):.4f}")
    print(f"Mean value:   {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
    print(f"Std dev:      {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}")
    print("")

# Recommendations
print("="*60)
print("RECOMMENDATIONS")
print("="*60)

if zero_pct > 10:
    print("⚠️  HIGH ZERO RATE DETECTED!")
    print("   - Re-run: python scripts/2_preprocessing/extract_landmarks.py")
    print("   - Then: python scripts/2_preprocessing/preprocess_features.py")
elif zero_pct > 5:
    print("⚠️  MODERATE ZERO RATE")
    print("   - Consider re-running feature extraction for zero samples")
elif zero_pct > 0:
    print("✅ LOW ZERO RATE")
    print("   - Run: python scripts/2_preprocessing/filter_valid_features.py")
    print("   - This will create manifest_filtered.csv excluding zero features")
else:
    print("🎉 PERFECT! All features are valid!")

if missing_count > 0:
    print(f"\n⚠️  {missing_count} MISSING FILES")
    print("   - Re-run: python scripts/2_preprocessing/preprocess_features.py")

print("")
print("="*60)

EOF

echo ""
echo "✅ Feature validity check complete!"
echo ""
