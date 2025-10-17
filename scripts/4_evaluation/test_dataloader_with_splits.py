import sys
sys.path.insert(0, '.')

from src.dataloader import create_dataloaders
import yaml

# Load config
cfg = yaml.safe_load(open("configs/config.yaml"))

print("="*60)
print("Testing ASL Dataloader with Train/Val/Test Splits")
print("="*60)

# Create dataloaders with splits
print("\nCreating dataloaders...")
train_loader, val_loader, test_loader = create_dataloaders(
    window_size=32,
    stride_train=16,  # More overlap for training
    stride_val=32,    # No overlap for val/test
    batch_size=16,
    num_workers=0,    # Use 0 for testing
    augment_train=True  # Enable augmentation for training
)

# Test each loader
print("\n" + "="*60)
print("TRAIN LOADER")
print("="*60)
for i, (features, labels) in enumerate(train_loader):
    print(f"Batch {i+1}:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label values: {labels[:5].tolist()}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Show label names
    label_names = [train_loader.dataset.get_label_name(l.item()) for l in labels[:5]]
    print(f"  Label names: {label_names}")
    
    if i >= 1:  # Show first 2 batches
        break

print("\n" + "="*60)
print("VAL LOADER")
print("="*60)
for i, (features, labels) in enumerate(val_loader):
    print(f"Batch {i+1}:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label values: {labels[:5].tolist()}")
    
    label_names = [val_loader.dataset.get_label_name(l.item()) for l in labels[:5]]
    print(f"  Label names: {label_names}")
    
    if i >= 0:  # Show first batch
        break

print("\n" + "="*60)
print("TEST LOADER")
print("="*60)
for i, (features, labels) in enumerate(test_loader):
    print(f"Batch {i+1}:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label values: {labels[:5].tolist()}")
    
    label_names = [test_loader.dataset.get_label_name(l.item()) for l in labels[:5]]
    print(f"  Label names: {label_names}")
    
    if i >= 0:  # Show first batch
        break

# Print statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Train:")
print(f"  Samples: {len(train_loader.dataset.df)}")
print(f"  Windows: {len(train_loader.dataset)}")
print(f"  Batches: {len(train_loader)}")
print(f"\nVal:")
print(f"  Samples: {len(val_loader.dataset.df)}")
print(f"  Windows: {len(val_loader.dataset)}")
print(f"  Batches: {len(val_loader)}")
print(f"\nTest:")
print(f"  Samples: {len(test_loader.dataset.df)}")
print(f"  Windows: {len(test_loader.dataset)}")
print(f"  Batches: {len(test_loader)}")

print(f"\nNumber of classes: {train_loader.dataset.num_classes}")
print(f"Class names (first 15): {train_loader.dataset.labels[:15]}")

# Compute class weights for handling imbalanced data
print("\n" + "="*60)
print("CLASS WEIGHTS (for loss function)")
print("="*60)
class_weights = train_loader.dataset.get_class_weights()
print(f"Shape: {class_weights.shape}")
print(f"Range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
print(f"Top 5 weights (most underrepresented):")
top_weights = class_weights.topk(5)
for i, (weight, idx) in enumerate(zip(top_weights.values, top_weights.indices)):
    label = train_loader.dataset.get_label_name(idx.item())
    print(f"  {i+1}. {label}: {weight:.3f}")

print("\n✅ Dataloader test with splits complete!")
print("\nYou can now use these dataloaders for training!")
