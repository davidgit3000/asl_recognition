import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

# Load config
cfg = yaml.safe_load(open("configs/config.yaml"))
manifest_path = Path(cfg['manifest_out'])

print("="*60)
print("Assigning Train/Val/Test Splits")
print("="*60)

# Load manifest
df = pd.read_csv(manifest_path)
print(f"\nTotal samples: {len(df)}")
print(f"Classes: {df['label'].nunique()}")
print(f"Sources: {df['source'].unique().tolist()}")

# Check if split column exists
if 'split' not in df.columns:
    df['split'] = None

# Strategy: 70% train, 15% val, 15% test
# Use stratified sampling to ensure each class is proportionally represented
print("\nSplit strategy: 70% train, 15% val, 15% test (stratified by label)")

# Handle classes with very few samples (< 3 samples can't be split)
label_counts = df['label'].value_counts()
small_classes = label_counts[label_counts < 3].index.tolist()

if small_classes:
    print(f"\nWarning: {len(small_classes)} classes have < 3 samples:")
    for label in small_classes[:5]:  # Show first 5
        print(f"  {label}: {label_counts[label]} samples")
    print("  These will be assigned to train set only.")
    
    # Assign small classes to train
    df.loc[df['label'].isin(small_classes), 'split'] = 'train'
    
    # Filter out small classes for stratified split
    df_to_split = df[~df['label'].isin(small_classes)].copy()
else:
    df_to_split = df.copy()

# Perform stratified split
try:
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        df_to_split.index,
        test_size=0.3,
        stratify=df_to_split['label'],
        random_state=42
    )
    
    # Second split: val vs test
    temp_df = df_to_split.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_df.index,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )
    
    # Assign splits
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'
    
    print("\n✅ Splits assigned successfully!")
    
except ValueError as e:
    print(f"\n❌ Error during stratified split: {e}")
    print("Falling back to random split without stratification...")
    
    # Fallback: random split without stratification
    train_idx, temp_idx = train_test_split(
        df_to_split.index,
        test_size=0.3,
        random_state=42
    )
    
    temp_df = df_to_split.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_df.index,
        test_size=0.5,
        random_state=42
    )
    
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'
    
    print("✅ Random splits assigned successfully!")

# Print split statistics
print("\n" + "="*60)
print("SPLIT STATISTICS")
print("="*60)

split_counts = df['split'].value_counts()
print(f"\nOverall:")
for split in ['train', 'val', 'test']:
    count = split_counts.get(split, 0)
    pct = 100 * count / len(df)
    print(f"  {split:5s}: {count:5d} samples ({pct:.1f}%)")

# Per-source breakdown
print(f"\nBy source:")
for source in df['source'].unique():
    source_df = df[df['source'] == source]
    print(f"\n  {source}:")
    for split in ['train', 'val', 'test']:
        count = len(source_df[source_df['split'] == split])
        pct = 100 * count / len(source_df) if len(source_df) > 0 else 0
        print(f"    {split:5s}: {count:5d} ({pct:.1f}%)")

# Per-class breakdown (show top 10 classes)
print(f"\nTop 10 classes by sample count:")
top_classes = df['label'].value_counts().head(10).index
for label in top_classes:
    label_df = df[df['label'] == label]
    train_count = len(label_df[label_df['split'] == 'train'])
    val_count = len(label_df[label_df['split'] == 'val'])
    test_count = len(label_df[label_df['split'] == 'test'])
    total = len(label_df)
    print(f"  {label:15s}: train={train_count:4d}, val={val_count:4d}, test={test_count:4d} (total={total:4d})")

# Save updated manifest
df.to_csv(manifest_path, index=False)
print(f"\n✅ Updated manifest saved to: {manifest_path}")

# Create a backup
backup_path = manifest_path.parent / f"{manifest_path.stem}_backup.csv"
df.to_csv(backup_path, index=False)
print(f"✅ Backup saved to: {backup_path}")

print("\n" + "="*60)
print("Split assignment complete!")
print("="*60)
print("\nYou can now use the dataloader with split='train', 'val', or 'test'")
