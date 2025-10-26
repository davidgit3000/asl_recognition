# Tests Directory

This directory contains test scripts for validating different components of the ASL recognition pipeline.

## 📁 Test Files

### `test_dataloader_with_splits.py`
**Purpose:** Test the PyTorch dataloader with train/val/test splits

**What it tests:**
- ✅ Dataloader initialization
- ✅ Train/val/test split integrity
- ✅ Batch loading and shapes
- ✅ Label distribution across splits
- ✅ Feature normalization
- ✅ Data augmentation (training only)

**Usage:**
```bash
python tests/test_dataloader_with_splits.py
```

**Expected output:**
- Dataset statistics (samples, windows, classes)
- Batch shape verification
- Sample visualization
- Split distribution analysis

---

### `test_model.py`
**Purpose:** Test model architecture and forward pass

**What it tests:**
- ✅ Model initialization
- ✅ Forward pass with dummy data
- ✅ Output shape verification
- ✅ Parameter count
- ✅ Gradient flow

**Usage:**
```bash
python tests/test_model.py
```

**Expected output:**
- Model architecture summary
- Parameter count
- Forward pass success
- Output shape verification

---

## 🚀 Running All Tests

```bash
# Run all tests sequentially
for test in tests/test_*.py; do
    echo "Running $test..."
    python "$test"
    echo "---"
done
```

---

## 📝 Adding New Tests

When creating new test files:

1. **Naming convention:** `test_<component>.py`
2. **Location:** Place in `tests/` directory
3. **Structure:**
   ```python
   #!/usr/bin/env python3
   """Test description"""
   import sys
   sys.path.insert(0, '.')
   
   # Your test code here
   
   if __name__ == "__main__":
       # Run tests
       print("✅ All tests passed!")
   ```

---

## 🎯 Test Coverage

| Component | Test File | Status |
|-----------|-----------|--------|
| Dataloader | `test_dataloader_with_splits.py` | ✅ |
| Model | `test_model.py` | ✅ |
| Landmark Extraction | - | 📋 TODO |
| Feature Preprocessing | - | 📋 TODO |
| Training Loop | - | 📋 TODO |
| Inference Pipeline | - | 📋 TODO |

---

## 📊 Future Test Ideas

1. **`test_landmark_extraction.py`**
   - Test MediaPipe detection on sample images
   - Verify landmark count and format
   - Test dual-detection fallback

2. **`test_feature_preprocessing.py`**
   - Test normalization
   - Test temporal smoothing
   - Test feature validity

3. **`test_training.py`**
   - Test training loop (1 epoch)
   - Test checkpoint saving/loading
   - Test early stopping

4. **`test_inference.py`**
   - Test real-time inference
   - Test webcam integration
   - Test prediction smoothing

---

## 🔧 Best Practices

- ✅ Keep tests independent (no dependencies between tests)
- ✅ Use small sample data for fast testing
- ✅ Add assertions to verify expected behavior
- ✅ Print clear success/failure messages
- ✅ Clean up temporary files after testing
