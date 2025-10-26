# Training Scripts

Scripts for training ASL recognition models.

## Available Scripts

### `test_model.py` - Test model architecture
Quick test to verify models work correctly.
```bash
python scripts/3_training/test_model.py
```

### `train_baseline.py` - Train baseline LSTM model
Train a bidirectional LSTM model for ASL recognition.

**Basic usage:**
```bash
python scripts/3_training/train_baseline.py
```

**Custom configuration:**
```bash
python scripts/3_training/train_baseline.py \
    --model-type lstm \
    --hidden-dim 256 \
    --num-layers 2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001
```

**With attention:**
```bash
python scripts/3_training/train_baseline.py --model-type lstm_attention
```

## Model Arguments

- `--model-type`: Model architecture (`lstm` or `lstm_attention`)
- `--hidden-dim`: LSTM hidden dimension (default: 256)
- `--num-layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.3)
- `--bidirectional`: Use bidirectional LSTM (default: True)

## Data Arguments

- `--window-size`: Sequence window size (default: 32)
- `--stride-train`: Stride for training windows (default: 16)
- `--stride-val`: Stride for val/test windows (default: 32)
- `--batch-size`: Batch size (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)

## Training Arguments

- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--patience`: Early stopping patience (default: 10)

## Output

Training creates a timestamped run directory:
```
artifacts/
├── models/
│   └── lstm_20241017_170000/
│       ├── best.pth          # Best model checkpoint
│       ├── last.pth          # Last epoch checkpoint
│       └── results.json      # Final results
└── logs/
    └── lstm_20241017_170000/  # TensorBoard logs
```

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir artifacts/logs
```

Then open http://localhost:6006 in your browser.

## Features

✅ **Bidirectional LSTM** - Captures temporal context in both directions  
✅ **Attention mechanism** - Optional attention for better performance  
✅ **Class weighting** - Handles imbalanced dataset  
✅ **Data augmentation** - Rotation, scale, translation, temporal shift  
✅ **Early stopping** - Prevents overfitting  
✅ **Learning rate scheduling** - ReduceLROnPlateau  
✅ **TensorBoard logging** - Real-time training visualization  
✅ **Checkpoint saving** - Best and last models saved  

## Expected Performance

With default settings on the ASL dataset:
- **Training time**: ~2-3 hours (depending on hardware)
- **Expected accuracy**: 85-95% (varies by dataset quality)
- **Model size**: ~2.9M parameters
