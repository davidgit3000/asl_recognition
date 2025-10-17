# ASL Recognition Model

American Sign Language (ASL) recognition system using MediaPipe landmarks and deep learning.

## 📁 Project Structure

```
asl_model/
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── label_map.json           # Label normalization rules
├── data/                        # Raw datasets
│   ├── kaggle_asl_combined/    # Combined Kaggle ASL images
│   ├── microsoft_asl/          # MS-ASL videos and metadata
│   └── personal/               # Personal recordings (optional)
├── artifacts/                   # Generated artifacts
│   ├── landmarks/              # Raw MediaPipe landmarks [T, 543, 4]
│   ├── features/               # Preprocessed features [T, 75, 4]
│   ├── manifests/              # Dataset manifests (CSV)
│   ├── models/                 # Trained model checkpoints
│   └── logs/                   # Training logs
├── src/                        # Core library code
│   ├── data/                   # Data processing modules
│   │   └── dataloader.py      # PyTorch dataloader
│   └── utils/                  # Utility functions
├── scripts/                    # Executable scripts
│   ├── 1_data_preparation/    # Download and organize data
│   ├── 2_preprocessing/       # Extract and preprocess features
│   ├── 3_training/            # Train models
│   └── 4_evaluation/          # Test and visualize
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/davidgit3000/asl_recognition.git
cd asl_recognition
```

### 2. Setup Environment

```bash
# Create virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Data (if not already done)

See `scripts/1_data_preparation/README.md` for detailed steps.

```bash
# Build manifest and assign splits
python scripts/1_data_preparation/build_manifest.py
python scripts/1_data_preparation/assign_splits.py
```

### 4. Extract and Preprocess Features

```bash
# Extract MediaPipe landmarks
python scripts/2_preprocessing/extract_landmarks.py

# Preprocess features for training
python scripts/2_preprocessing/preprocess_features.py
```

### 5. Test Dataloader

```bash
python scripts/4_evaluation/test_dataloader_with_splits.py
```

## 📊 Dataset

- **Kaggle ASL Alphabet**: ~26,000 images (A-Z letters)
- **MS-ASL**: ~190 videos (20 common words)
- **Total**: 26,190 samples, 45 classes
- **Splits**: 70% train, 15% val, 15% test (stratified)

## 🔧 Pipeline Overview

1. **Data Collection** → Download Kaggle + MS-ASL datasets
2. **Manifest Building** → Create unified CSV with metadata
3. **Landmark Extraction** → MediaPipe Holistic (543 landmarks)
4. **Feature Preprocessing** → Normalize, smooth, reduce to 75 landmarks
5. **Train/Val/Test Split** → Stratified 70/15/15 split
6. **Dataloader** → PyTorch DataLoader with windowing
7. **Training** → Train ASL recognition model (to be implemented)
8. **Evaluation** → Test and visualize results

## 📦 Features

- ✅ MediaPipe Holistic landmark extraction
- ✅ Temporal smoothing (Savitzky-Golay filter)
- ✅ Normalization (centered on torso, scaled by shoulder width)
- ✅ Windowed sequences (32 frames by default)
- ✅ Data augmentation (rotation, scale, translation)
- ✅ Class balancing (weighted loss for imbalanced data)
- ✅ Stratified train/val/test splits

## 📝 Documentation

- `KAGGLE_DATASETS_INFO.md` - Kaggle dataset details
- `MSASL_PIPELINE.md` - MS-ASL download pipeline
- `scripts/README.md` - Scripts documentation
- `scripts/*/README.md` - Detailed docs for each stage

## 🎯 Next Steps

1. Implement baseline model (LSTM/Transformer)
2. Train and evaluate
3. Build real-time inference pipeline
4. Create webcam demo

## 📄 License

Educational project for CS 4620.
