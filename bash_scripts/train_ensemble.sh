#!/bin/bash
# Train ensemble of 5 models with different configurations

# Configuration
EPOCHS=50  # Reduce from 100 to 50 for faster training

echo "Training Ensemble of 5 Models"
echo "=============================="
echo "Epochs per model: $EPOCHS"

# Model 1: LSTM with Attention, large
echo "Training Model 1: LSTM Attention (512 hidden, 3 layers)"
python scripts/3_training/train_baseline.py \
    --model-type lstm_attention \
    --hidden-dim 512 \
    --num-layers 3 \
    --dropout 0.25 \
    --epochs $EPOCHS \
    --lr 0.00063 \
    --batch-size 32 \
    --label-smoothing 0.1

# Model 2: LSTM with Attention, medium
echo "Training Model 2: LSTM Attention (384 hidden, 2 layers)"
python scripts/3_training/train_baseline.py \
    --model-type lstm_attention \
    --hidden-dim 384 \
    --num-layers 2 \
    --dropout 0.3 \
    --epochs $EPOCHS \
    --lr 0.00063 \
    --batch-size 32 \
    --label-smoothing 0.1

# Model 3: Standard LSTM, large
echo "Training Model 3: Standard LSTM (512 hidden, 3 layers)"
python scripts/3_training/train_baseline.py \
    --model-type lstm \
    --hidden-dim 512 \
    --num-layers 3 \
    --dropout 0.2 \
    --epochs $EPOCHS \
    --lr 0.00063 \
    --batch-size 32 \
    --label-smoothing 0.1

# Model 4: LSTM with Attention, deep
echo "Training Model 4: LSTM Attention (256 hidden, 4 layers)"
python scripts/3_training/train_baseline.py \
    --model-type lstm_attention \
    --hidden-dim 256 \
    --num-layers 4 \
    --dropout 0.35 \
    --epochs $EPOCHS \
    --lr 0.00063 \
    --batch-size 32 \
    --label-smoothing 0.1

# Model 5: LSTM with Attention, wide
echo "Training Model 5: LSTM Attention (640 hidden, 2 layers)"
python scripts/3_training/train_baseline.py \
    --model-type lstm_attention \
    --hidden-dim 640 \
    --num-layers 2 \
    --dropout 0.25 \
    --epochs $EPOCHS \
    --lr 0.00063 \
    --batch-size 32 \
    --label-smoothing 0.1

echo "=============================="
echo "All models trained!"
echo "Run ensemble_predict.py to combine predictions"
