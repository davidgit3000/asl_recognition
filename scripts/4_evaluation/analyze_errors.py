#!/usr/bin/env python3
"""Analyze model errors to identify which classes are hardest."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.dataloader import create_dataloaders
from src.models.lstm_model import create_model

# Load model
path = "artifacts/models/lstm_attention_20251020_151032"
model_dir = Path(path)  # Update with your latest model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\nLoading model from: {model_dir}")
print(f"Device: {device}")

model = create_model('lstm_attention', num_classes=26, hidden_dim=512, num_layers=3, dropout=0.25, bidirectional=True)
checkpoint = torch.load(model_dir / 'best.pth', map_location=device)

print(f"\nCheckpoint Info:")
print(f" Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f" Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%" if 'val_acc' in checkpoint else " Val Acc: N/A")
print(f" Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else " Val Loss: N/A")

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"\nModel loaded successfully!")
print(f" Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load data
print("\nLoading test data...")
_, _, test_loader = create_dataloaders(
    window_size=32,
    stride_train=16,
    stride_val=32,
    batch_size=32,
    num_workers=0,
    augment_train=False
)

print(f"\nDataset Info:")
print(f"  Test samples: {len(test_loader.dataset.df)}")
print(f"  Test windows: {len(test_loader.dataset)}")
print(f"  Test batches: {len(test_loader)}")
print(f"  Num classes: {test_loader.dataset.num_classes}")
print(f"  Window size: {test_loader.dataset.window_size}")
print(f"  Stride: {test_loader.dataset.stride}")
print(f"  Augmentation: {test_loader.dataset.augment}")

# Get predictions
all_preds = []
all_labels = []

print("Evaluating model...")
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Overall accuracy
accuracy = (all_preds == all_labels).mean()
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Per-class accuracy
print("\n" + "="*60)
print("Per-Class Performance")
print("="*60)

label_names = test_loader.dataset.labels
report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)

# Sort by F1-score
class_scores = [(name, report[name]['f1-score'], report[name]['support']) 
                for name in label_names if name in report]
class_scores.sort(key=lambda x: x[1])

print("\nWorst 10 classes:")
for name, f1, support in class_scores[:10]:
    print(f"  {name:15s}: F1={f1:.3f}, Support={int(support)}")

print("\nBest 10 classes:")
for name, f1, support in class_scores[-10:]:
    print(f"  {name:15s}: F1={f1:.3f}, Support={int(support)}")

# Confusion matrix for worst classes
print("\n" + "="*60)
print("Analyzing Confusions")
print("="*60)

cm = confusion_matrix(all_labels, all_preds)

# Find most confused pairs
confused_pairs = []
for i in range(len(label_names)):
    for j in range(len(label_names)):
        if i != j and cm[i, j] > 5:  # At least 5 confusions
            confused_pairs.append((label_names[i], label_names[j], cm[i, j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nMost confused class pairs:")
for true_class, pred_class, count in confused_pairs[:15]:
    print(f"  {true_class:10s} → {pred_class:10s}: {count} times")

print(f"\n✅ Analysis complete!")
print(f"\nKey insights:")
print(f"  - Focus on improving worst classes")
print(f"  - Check if confused pairs are visually similar")
print(f"  - Consider collecting more data for low-support classes")
