# ASL Recognition Training Guide

Complete guide to training your ASL recognition model.

## 🚀 Quick Start

### 1. Verify Data is Ready

```bash
# Check dataset statistics
python scripts/4_evaluation/quick_stats.py

# Test dataloader
python scripts/4_evaluation/test_dataloader_with_splits.py
```

Expected output:
- Train: ~18,333 samples
- Val: ~3,928 samples  
- Test: ~3,929 samples
- 45 classes

### 2. Test Model Architecture

```bash
python scripts/3_training/test_model.py
```

Should show:
- ✅ Model test passed!
- ~2.9M parameters

### 3. Start Training

**Basic training (recommended for first run):**
```bash
python scripts/3_training/train_baseline.py
```

**Quick test (fewer epochs):**
```bash
python scripts/3_training/train_baseline.py --epochs 10 --batch-size 16
```

**Full training with attention:**
```bash
python scripts/3_training/train_baseline.py \
    --model-type lstm_attention \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

### 4. Monitor Training

Open TensorBoard in another terminal:
```bash
tensorboard --logdir artifacts/logs
```

Then open http://localhost:6006 to see:
- Training/validation loss curves
- Training/validation accuracy
- Learning rate schedule

## 📊 Model Architecture

### Baseline LSTM
```
Input: [batch, 32, 75, 4]
  ↓
Flatten: [batch, 32, 300]
  ↓
Bidirectional LSTM (2 layers, 256 hidden)
  ↓
Classifier (256 → 45)
  ↓
Output: [batch, 45]
```

**Parameters:** ~2.86M

### LSTM with Attention
```
Input: [batch, 32, 75, 4]
  ↓
Flatten: [batch, 32, 300]
  ↓
Bidirectional LSTM (2 layers, 256 hidden)
  ↓
Temporal Attention
  ↓
Classifier (256 → 45)
  ↓
Output: [batch, 45]
```

**Parameters:** ~2.99M

## 🎯 Training Features

### Data Augmentation (Training Only)
- ✅ **Rotation**: ±15° around z-axis (50% chance)
- ✅ **Scale**: 0.9x to 1.1x (50% chance)
- ✅ **Translation**: ±0.1 units (50% chance)
- ✅ **Temporal shift**: ±5 frames (30% chance)

### Class Balancing
- Uses weighted CrossEntropyLoss
- Weights computed as inverse frequency
- Handles imbalanced dataset (26k letters vs 190 words)

### Optimization
- **Optimizer**: Adam
- **Learning rate**: 0.001 (default)
- **Weight decay**: 1e-5
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: Patience=10 epochs

### Checkpointing
- **best.pth**: Best validation accuracy
- **last.pth**: Last epoch (for resuming)
- **results.json**: Final metrics

## 📈 Expected Results

### Training Progress
```
Epoch 1/50
  Train Loss: 2.5432 | Train Acc: 35.21%
  Val Loss:   2.1234 | Val Acc:   42.15%

Epoch 10/50
  Train Loss: 0.8234 | Train Acc: 75.43%
  Val Loss:   0.9123 | Val Acc:   72.18%

Epoch 30/50
  Train Loss: 0.2145 | Train Acc: 93.21%
  Val Loss:   0.4321 | Val Acc:   88.45%
```

### Final Performance (Expected)
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-92%
- **Test Accuracy**: 85-90%

*Note: Actual results depend on data quality and preprocessing*

## 🔧 Hyperparameter Tuning

### If Overfitting (Train >> Val accuracy)
```bash
# Increase dropout
python scripts/3_training/train_baseline.py --dropout 0.5

# Increase weight decay
python scripts/3_training/train_baseline.py --weight-decay 1e-4

# Reduce model size
python scripts/3_training/train_baseline.py --hidden-dim 128 --num-layers 1
```

### If Underfitting (Both accuracies low)
```bash
# Increase model capacity
python scripts/3_training/train_baseline.py --hidden-dim 512 --num-layers 3

# Train longer
python scripts/3_training/train_baseline.py --epochs 100

# Increase learning rate
python scripts/3_training/train_baseline.py --lr 0.003
```

### If Training is Slow
```bash
# Increase batch size (if memory allows)
python scripts/3_training/train_baseline.py --batch-size 64

# Use more workers
python scripts/3_training/train_baseline.py --num-workers 8

# Reduce window overlap
python scripts/3_training/train_baseline.py --stride-train 32
```

## 📁 Output Structure

After training, you'll have:
```
artifacts/
├── models/
│   └── lstm_20241017_170530/
│       ├── best.pth          # Load this for inference
│       ├── last.pth          # Resume training from here
│       └── results.json      # Final metrics
└── logs/
    └── lstm_20241017_170530/
        └── events.out.tfevents.*  # TensorBoard logs
```

## 🎓 Next Steps After Training

1. **Evaluate on test set** (automatically done at end of training)
2. **Analyze confusion matrix** (to be implemented)
3. **Build inference pipeline** (to be implemented)
4. **Create webcam demo** (to be implemented)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/3_training/train_baseline.py --batch-size 16

# Reduce model size
python scripts/3_training/train_baseline.py --hidden-dim 128
```

### MPS (Apple Silicon) Issues
```bash
# Use CPU if MPS has issues
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Slow Data Loading
```bash
# Reduce workers if CPU-bound
python scripts/3_training/train_baseline.py --num-workers 0
```

### NaN Loss
- Check data preprocessing (look for inf/nan values)
- Reduce learning rate: `--lr 0.0001`
- Check class weights aren't too extreme

## 💡 Tips

1. **Start small**: Train for 10 epochs first to verify everything works
2. **Monitor TensorBoard**: Watch for overfitting/underfitting early
3. **Save your runs**: Keep track of hyperparameters that work well
4. **Use attention**: Usually gives 2-3% better accuracy
5. **Be patient**: Full training takes 2-3 hours

## 📊 Comparing Runs

TensorBoard lets you compare multiple runs:
```bash
tensorboard --logdir artifacts/logs
```

Select multiple runs in the UI to overlay their curves.

Good luck with training! 🚀
