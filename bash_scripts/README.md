# Bash Scripts Directory

This directory contains all bash shell scripts (.sh) for the ASL recognition project.

## 📁 Scripts Overview

### 🔽 Data Download

#### `download_kaggle_datasets.sh`
**Purpose:** Download Kaggle ASL datasets using Kaggle API

**Prerequisites:**
- Kaggle API credentials in `~/.kaggle/kaggle.json`
- Kaggle CLI installed (`pip install kaggle`)

**Usage:**
```bash
bash bash_scripts/download_kaggle_datasets.sh
```

**What it downloads:**
- ASL Alphabet dataset → `data/kaggle_asl1/`
- ASL Dataset → `data/kaggle_asl2/`

---

### ✅ Feature Validation

#### `check_feature_validity.sh`
**Purpose:** Check validity of features in the **filtered** dataset

**What it checks:**
- ✅ Valid features (non-zero landmarks)
- ❌ Zero features (failed detection)
- ⚠️ Missing files
- 🔥 Corrupted files

**Usage:**
```bash
bash bash_scripts/check_feature_validity.sh
```

**Output:**
- Overall statistics (valid %, zero %, missing %)
- Breakdown by split (train/val/test)
- Breakdown by source (Kaggle/MS-ASL)
- Feature statistics (min, max, mean, std)
- Recommendations based on results

**Expected result:**
```
✅ Valid features:     68,671 (100.00%)
❌ Zero features:      0 (0.00%)
⚠️  Missing files:      0 (0.00%)
```

---

#### `check_feature_validity_unfiltered.sh`
**Purpose:** Check validity of features in the **unfiltered** dataset (before filtering)

**What it checks:**
- Same as above, but on `manifest_v1.csv` (before filtering)
- Shows impact of filtering process
- Identifies which classes had highest failure rates

**Usage:**
```bash
bash bash_scripts/check_feature_validity_unfiltered.sh
```

**Output:**
- Statistics before filtering
- Classes with highest zero rates (M, N, C, etc.)
- Before vs After comparison table
- Filtering impact analysis

**Expected result:**
```
✅ Valid features:     68,859 (86.07%)
❌ Zero features:      11,144 (13.93%)
⚠️  Missing files:      2 (0.00%)

💡 Filtering removed 11,146 unusable samples
```

---

### 🎓 Model Training

#### `train_ensemble.sh`
**Purpose:** Train ensemble of 5 diverse LSTM models for improved accuracy

**Models trained:**
1. **LSTM Attention (512 hidden, 3 layers)** - Large model
2. **LSTM Attention (384 hidden, 2 layers)** - Medium model
3. **Standard LSTM (512 hidden, 3 layers)** - No attention
4. **LSTM Attention (256 hidden, 4 layers)** - Deep model
5. **LSTM Attention (512 hidden, 2 layers)** - Wide model

**Usage:**
```bash
# Train all 5 models (default: 100 epochs each)
bash bash_scripts/train_ensemble.sh

# Train with custom epochs
bash bash_scripts/train_ensemble.sh 50
```

**Training time:**
- ~6 hours per model
- Total: ~30 hours for all 5 models

**Expected results:**
- Individual models: 85-87% accuracy
- Ensemble (averaged): 88-90% accuracy

**Output:**
- 5 model checkpoints in `artifacts/models/`
- Training logs in `artifacts/logs/`
- Results JSON for each model

---

## 🚀 Quick Reference

| Script | Purpose | Runtime | Output |
|--------|---------|---------|--------|
| `download_kaggle_datasets.sh` | Download Kaggle data | 5-10 min | ~87K images |
| `check_feature_validity.sh` | Validate filtered features | 6-8 sec | Statistics report |
| `check_feature_validity_unfiltered.sh` | Validate unfiltered features | 7-10 sec | Comparison report |
| `train_ensemble.sh` | Train 5 models | ~30 hours | 5 trained models |

---

## 📝 Script Conventions

All bash scripts in this directory follow these conventions:

1. **Shebang:** `#!/bin/bash`
2. **Error handling:** Exit on error with meaningful messages
3. **Virtual environment:** Activate `.venv311` if present
4. **Working directory:** Run from project root
5. **Output:** Clear progress messages and status indicators

---

## 🔧 Making Scripts Executable

If you get permission errors, make scripts executable:

```bash
chmod +x bash_scripts/*.sh
```

Or run with `bash` explicitly:

```bash
bash bash_scripts/script_name.sh
```

---

## 🎯 Typical Workflow

### Initial Setup
```bash
# 1. Download datasets
bash bash_scripts/download_kaggle_datasets.sh

# 2. Process data (use Python scripts)
python scripts/1_data_preparation/combine_kaggle_asl.py
python scripts/1_data_preparation/build_manifest.py
python scripts/2_preprocessing/extract_landmarks.py
python scripts/2_preprocessing/preprocess_features.py

# 3. Check feature validity (before filtering)
bash bash_scripts/check_feature_validity_unfiltered.sh

# 4. Filter invalid features
python scripts/2_preprocessing/filter_valid_features.py

# 5. Check feature validity (after filtering)
bash bash_scripts/check_feature_validity.sh
```

### Training
```bash
# Option A: Train single model
python scripts/3_training/train_baseline.py --model-type lstm_attention

# Option B: Train ensemble
bash bash_scripts/train_ensemble.sh
```

---

## 📊 Expected Outputs

### Feature Validation Scripts
- Console output with statistics
- No files created
- Exit code 0 on success

### Training Scripts
- Model checkpoints: `artifacts/models/*/best.pth`
- Training logs: `artifacts/logs/*/events.out.tfevents.*`
- Results JSON: `artifacts/models/*/results.json`

---

## 🐛 Troubleshooting

### "Permission denied"
```bash
chmod +x bash_scripts/*.sh
```

### "No module named 'pandas'"
```bash
source .venv311/bin/activate
pip install -r requirements.txt
```

### "Kaggle API credentials not found"
```bash
# Setup Kaggle API
mkdir -p ~/.kaggle
# Copy kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### "Command not found: python3"
```bash
# Use python instead
sed -i '' 's/python3/python/g' bash_scripts/*.sh
```

---

## 📈 Future Scripts (Planned)

- `train_cnn.sh` - Train CNN models on raw images
- `train_transformer.sh` - Train Transformer models
- `evaluate_ensemble.sh` - Evaluate ensemble predictions
- `deploy_model.sh` - Export model for deployment
- `benchmark_inference.sh` - Benchmark inference speed

---

## 💡 Tips

1. **Run from project root:** All scripts assume you're in the project root directory
2. **Check logs:** Training scripts output to both console and log files
3. **Monitor progress:** Use `tail -f artifacts/logs/*/events.out.tfevents.*` for live updates
4. **Save time:** Use smaller epochs for testing (e.g., `bash_scripts/train_ensemble.sh 10`)
5. **Parallel training:** Run multiple training scripts in separate terminals (if you have GPU/MPS)

---

## 📄 License

Educational project for CS 4620.
`