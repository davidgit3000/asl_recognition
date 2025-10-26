# ASL Recognition Model

American Sign Language (ASL) recognition system using MediaPipe landmarks and deep learning.

## 📁 Project Structure

```
asl_model/
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── label_map.json           # Label normalization rules
├── data/                        # Raw datasets
│   ├── kaggle_asl1/            # Kaggle ASL Alphabet dataset
│   ├── kaggle_asl2/            # Kaggle ASL Dataset
│   ├── kaggle_asl_combined/    # Combined Kaggle images (A-Z)
│   ├── microsoft_asl/          # MS-ASL videos and metadata
│   └── personal/               # Personal recordings (optional)
├── artifacts/                   # Generated artifacts
│   ├── landmarks/              # Raw MediaPipe landmarks [T, 543, 4]
│   ├── features/               # Preprocessed features [T, 75, 4]
│   ├── manifests/              # Dataset manifests (CSV)
│   ├── models/                 # Trained model checkpoints
│   │   └── lstm_attention_20251020_151032/  # Best model (86.3%)
│   └── logs/                   # TensorBoard training logs
├── src/                        # Core library code
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   └── dataloader.py      # PyTorch dataloader with windowing
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   └── lstm_model.py      # BiLSTM + Attention
│   └── utils/                  # Utility functions
│       └── __init__.py
├── scripts/                    # Executable scripts
│   ├── 1_data_preparation/    # Download and organize data
│   │   ├── combine_kaggle_asl.py
│   │   ├── build_manifest.py
│   │   ├── assign_splits.py
│   │   └── msasl_*.py         # MS-ASL pipeline scripts
│   ├── 2_preprocessing/       # Extract and preprocess features
│   │   ├── extract_landmarks.py
│   │   ├── preprocess_features.py
│   │   └── filter_valid_features.py
│   ├── 3_training/            # Train models
│   │   └── train_baseline.py
│   └── 4_evaluation/          # Evaluate and visualize
│       ├── analyze_errors.py
│       ├── quick_stats.py
│       └── quick_viz.py
├── tests/                      # Test scripts
│   ├── test_dataloader_with_splits.py
│   ├── test_model.py
│   └── README.md
├── bash_scripts/               # Bash shell scripts
│   ├── download_kaggle_datasets.sh
│   ├── check_feature_validity.sh
│   ├── check_feature_validity_unfiltered.sh
│   ├── train_ensemble.sh
│   └── README.md
├── plans/                      # Project planning documents
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── TRAINING_GUIDE.md          # Training documentation
├── KAGGLE_DATASETS_INFO.md    # Kaggle dataset details
├── MSASL_PIPELINE.md          # MS-ASL pipeline guide
└── PROJECT_STRUCTURE.md       # Detailed structure docs
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/davidgit3000/asl_recognition.git
cd asl_recognition
```

### 2. Setup Environment

**Mac/Linux:**
```bash
# Create virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Windows:**
```bash
# Create virtual environment
python -m venv .venv311
.venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Datasets

**Kaggle Datasets (required):**

Option A - Using script (Mac/Linux):
```bash
# Setup Kaggle API credentials first (~/.kaggle/kaggle.json)
bash download_kaggle_datasets.sh
```

Option B - Manual download:
- Download [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) → extract to `data/kaggle_asl1/`
- Download [ASL Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset) → extract to `data/kaggle_asl2/`

**MS-ASL Videos (optional):**
- Download from [MS-ASL Dataset](https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset#overview)
- Then, use automated scripts (see `scripts/1_data_preparation/README.md`)

### 4. Prepare Data

```bash
# Combine Kaggle datasets
python scripts/1_data_preparation/combine_kaggle_asl.py

# Build manifest and assign splits
python scripts/1_data_preparation/build_manifest.py
python scripts/1_data_preparation/assign_splits.py
```

### 5. Extract and Preprocess Features

```bash
# Extract MediaPipe landmarks
python scripts/2_preprocessing/extract_landmarks.py

# Preprocess features for training
python scripts/2_preprocessing/preprocess_features.py
```

### 6. Test Dataloader

```bash
python scripts/4_evaluation/test_dataloader_with_splits.py
```

## 📊 Dataset

### Raw Data
- **Kaggle ASL Alphabet**: ~78,000 images (A-Z letters)
- **Kaggle ASL Dataset**: ~9,000 images (A-Z letters)
- **MS-ASL**: ~190 videos (20 common words)
- **Total Raw**: 87,200 samples

### After Processing & Filtering
- **Valid Features**: 68,671 samples (86.1% extraction success rate)
- **Classes**: 26 (A-Z letters only)
- **Splits**: 70% train (48,074), 15% val (10,278), 15% test (10,319)
- **Removed**: 11,144 zero features + 188 MS-ASL samples (class imbalance)

### Key Statistics
- **Feature extraction success**: 86.1% (dual-detection pipeline)
- **Class balance**: 1,847-3,008 samples per class (well-balanced)
- **Most challenging classes**: M, N (39.8% detection failure due to closed fist)

## 🔧 Pipeline Overview

1. **Data Collection** → Download Kaggle + MS-ASL datasets ✅
2. **Manifest Building** → Create unified CSV with metadata ✅
3. **Landmark Extraction** → Dual-detection (MediaPipe Holistic + Hands fallback) ✅
4. **Feature Preprocessing** → Normalize, smooth, reduce to 75 landmarks ✅
5. **Quality Filtering** → Remove zero features and low-count classes ✅
6. **Train/Val/Test Split** → Stratified 70/15/15 split ✅
7. **Dataloader** → PyTorch DataLoader with windowing ✅
8. **Model Training** → BiLSTM + Attention (86.3% test accuracy) ✅
9. **Evaluation** → Error analysis and confusion matrix 🔄
10. **Inference Pipeline** → Real-time webcam demo 📋

## 📦 Features

### Data Processing
- ✅ **Dual-detection pipeline**: MediaPipe Holistic + Hands fallback (86.1% success)
- ✅ **Temporal smoothing**: Savitzky-Golay filter (window=5, polynomial=2)
- ✅ **Normalization**: Centered on torso, scaled by shoulder width
- ✅ **Quality filtering**: Remove zero features and low-count classes
- ✅ **Windowed sequences**: 32 frames with configurable stride
- ✅ **Data augmentation**: Rotation, scale, translation (training only)
- ✅ **Class balancing**: Weighted loss for imbalanced data
- ✅ **Stratified splits**: 70/15/15 train/val/test

### Model Architecture
- ✅ **BiLSTM + Attention**: 3 layers, 512 hidden units, 17M parameters
- ✅ **Regularization**: Dropout (0.25), weight decay (1e-5), label smoothing (0.1)
- ✅ **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- ✅ **Early stopping**: Patience=10 epochs

### Performance
- ✅ **Test Accuracy**: 86.29% (22.7× better than random)
- ✅ **Validation Accuracy**: 86.79%
- ✅ **Generalization**: Val ≈ Test (no overfitting)
- ✅ **Training Time**: ~6 hours (100 epochs on Apple M4)

## 📝 Documentation

- `KAGGLE_DATASETS_INFO.md` - Kaggle dataset details
- `MSASL_PIPELINE.md` - MS-ASL download pipeline
- `scripts/README.md` - Scripts documentation
- `scripts/*/README.md` - Detailed docs for each stage

## 🎯 Current Status

### ✅ Completed
1. ✅ Data collection and preprocessing (68,671 samples, 26 classes)
2. ✅ Dual-detection landmark extraction (86.1% success rate)
3. ✅ Feature engineering and quality filtering
4. ✅ BiLSTM + Attention model (86.29% test accuracy)
5. ✅ Training pipeline with early stopping and LR scheduling

### 🔄 In Progress
1. 🔄 Error analysis and confusion matrix visualization
2. 🔄 Per-class performance evaluation

### 📋 Next Steps

#### Phase 1: Model Exploration (Week 1-2)
1. **CNN-based models** (for spatial feature extraction)
   - 2D CNN on landmark heatmaps
   - 3D CNN for spatiotemporal features
   - ResNet/EfficientNet backbones

2. **Hybrid models** (combining spatial + temporal)
   - CNN + LSTM (extract spatial features, then temporal modeling)
   - CNN + Transformer (attention over CNN features)
   - Two-stream networks (appearance + motion)

3. **Transformer-based models**
   - Vision Transformer (ViT) for landmark sequences
   - Temporal Transformer with positional encoding
   - BERT-style pre-training on ASL data

4. **Ensemble methods**
   - Train 5 diverse models (different architectures/hyperparameters)
   - Probability averaging or voting
   - Expected: +2-4% accuracy boost

#### Phase 2: Deployment (Week 3-4)
1. **Real-time inference pipeline**
   - Webcam integration with MediaPipe
   - Sliding window prediction (30 FPS)
   - Temporal smoothing for stable predictions

2. **Demo application**
   - GUI with live video feed
   - Top-3 predictions with confidence scores
   - Recording capability for new samples

3. **Model optimization**
   - ONNX export for cross-platform deployment
   - Quantization for faster inference
   - Mobile deployment (TensorFlow Lite)

#### Phase 3: Advanced Features (Optional)
1. **Word-level recognition**
   - Collect more MS-ASL data (500+ samples per word)
   - Sequence-to-sequence models
   - Sentence-level ASL translation

2. **Transfer learning**
   - Fine-tune on personal signing style
   - Few-shot learning for new signs

3. **Multi-modal learning**
   - Combine landmarks + raw video
   - Audio integration (for signed songs)

## 🏆 Model Comparison (Planned)

| Model | Architecture | Params | Expected Acc | Training Time | Notes |
|-------|-------------|--------|--------------|---------------|-------|
| **BiLSTM + Attention** | 3-layer BiLSTM | 17M | 86.3% ✅ | 6h | Current best |
| CNN + LSTM | ResNet18 + 2-layer LSTM | ~15M | 87-89% | 8h | Spatial + temporal |
| 3D CNN | 3D ResNet | ~30M | 85-88% | 10h | End-to-end spatiotemporal |
| Transformer | 6-layer encoder | ~20M | 88-91% | 12h | Pure attention |
| Ensemble (5 models) | Mixed | ~80M | 89-92% | 30h | Best performance |

## 📈 Performance Targets

- ✅ **Baseline (Random)**: 3.8% (1/26 classes)
- ✅ **Current (BiLSTM+Attention)**: 86.3%
- 🎯 **Target (Ensemble)**: 90%+
- 🏆 **SOTA (Published research)**: 92-95%

## 📄 License

Educational project for CS 4620.
