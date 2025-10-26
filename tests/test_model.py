#!/usr/bin/env python3
"""Quick test to verify model architecture works."""
import sys
sys.path.insert(0, '.')

import torch
from src.models.lstm_model import create_model

print("="*60)
print("Testing Model Architecture")
print("="*60)

# Create model
model = create_model(
    model_type="lstm",
    num_classes=45,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    bidirectional=True
)

print(f"\nModel: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 4
seq_len = 32
num_landmarks = 75
num_channels = 4

dummy_input = torch.randn(batch_size, seq_len, num_landmarks, num_channels)
print(f"\nInput shape: {dummy_input.shape}")

output = model(dummy_input)
print(f"Output shape: {output.shape}")
print(f"Expected: [{batch_size}, 45]")

assert output.shape == (batch_size, 45), "Output shape mismatch!"

print("\n✅ Model test passed!")

# Test attention model
print("\n" + "="*60)
print("Testing Attention Model")
print("="*60)

model_attn = create_model(
    model_type="lstm_attention",
    num_classes=45,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    bidirectional=True
)

print(f"\nModel: {model_attn.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model_attn.parameters()):,}")

output_attn = model_attn(dummy_input)
print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output_attn.shape}")

assert output_attn.shape == (batch_size, 45), "Output shape mismatch!"

print("\n✅ Attention model test passed!")
