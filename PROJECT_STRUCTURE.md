# Project Structure Overview

## 📁 Directory Organization

```
asl_model/
│
├── 📂 configs/                          Configuration files
│   ├── config.yaml                     Main config (paths, hyperparameters)
│   └── label_map.json                  Label normalization rules
│
├── 📂 data/                            Raw datasets (gitignored)
│   ├── kaggle_asl_combined/           Combined Kaggle ASL images
│   ├── microsoft_asl/                 MS-ASL videos + metadata
│   └── personal/                      Personal recordings (optional)
│
├── 📂 artifacts/                       Generated outputs (gitignored)
│   ├── landmarks/                     Raw MediaPipe landmarks [T, 543, 4]
│   ├── features/                      Preprocessed features [T, 75, 4]
│   ├── manifests/                     Dataset manifests (CSV files)
│   ├── models/                        Trained model checkpoints
│   └── logs/                          Training logs
│
├── 📂 src/                             Core library code (importable)
│   ├── __init__.py
│   ├── data/                          Data processing modules
│   │   ├── __init__.py
│   │   └── dataloader.py             PyTorch Dataset & DataLoader
│   └── utils/                         Utility functions
│       └── __init__.py
│
├── 📂 scripts/                         Executable scripts (organized by stage)
│   ├── README.md                      Scripts documentation
│   │
│   ├── 1_data_preparation/            Step 1: Data collection & organization
│   │   ├── README.md
│   │   ├── combine_kaggle_asl.py     Merge Kaggle datasets
│   │   ├── build_msasl_manifest.py   Build MS-ASL manifest
│   │   ├── msasl_make_list.py        Select MS-ASL classes
│   │   ├── msasl_download_and_trim.py Download MS-ASL videos
│   │   ├── verify_msasl_downloads.py Verify downloads
│   │   ├── build_manifest.py         Build unified manifest
│   │   └── assign_splits.py          Assign train/val/test splits
│   │
│   ├── 2_preprocessing/               Step 2: Feature extraction
│   │   ├── README.md
│   │   ├── extract_landmarks.py      Extract MediaPipe landmarks
│   │   └── preprocess_features.py    Normalize & smooth features
│   │
│   ├── 3_training/                    Step 3: Model training
│   │   └── README.md                 (Placeholder for training scripts)
│   │
│   └── 4_evaluation/                  Step 4: Testing & visualization
│       ├── README.md
│       ├── test_dataloader_with_splits.py Test dataloader with splits
│       ├── quick_stats.py            Dataset statistics
│       └── quick_viz.py              Visualize landmarks
│
├── 📂 plans/                           Project planning documents
│   └── First Progress Report.pdf
│
├── 📄 README.md                        Main project README
├── 📄 PROJECT_STRUCTURE.md             This file
├── 📄 KAGGLE_DATASETS_INFO.md          Kaggle dataset details
├── 📄 MSASL_PIPELINE.md                MS-ASL download pipeline
├── 📄 requirements.txt                 Python dependencies
└── 📄 download_kaggle_datasets.sh      Kaggle download helper
```

## 🔄 Pipeline Flow

```
1. Data Preparation (scripts/1_data_preparation/)
   ↓
2. Preprocessing (scripts/2_preprocessing/)
   ↓
3. Training (scripts/3_training/)
   ↓
4. Evaluation (scripts/4_evaluation/)
```

## 📦 Key Files by Purpose

### Configuration
- `configs/config.yaml` - All paths and hyperparameters
- `configs/label_map.json` - Label normalization rules

### Data Processing
- `src/data/dataloader.py` - PyTorch Dataset/DataLoader implementation
- `scripts/2_preprocessing/extract_landmarks.py` - MediaPipe landmark extraction
- `scripts/2_preprocessing/preprocess_features.py` - Feature normalization & smoothing

### Dataset Management
- `scripts/1_data_preparation/build_manifest.py` - Create master manifest
- `scripts/1_data_preparation/assign_splits.py` - Assign train/val/test splits
- `artifacts/manifests/manifest_v1.csv` - Master dataset manifest

### Testing & Evaluation
- `scripts/4_evaluation/test_dataloader_with_splits.py` - Test dataloader with splits
- `scripts/4_evaluation/quick_stats.py` - Dataset statistics
- `scripts/4_evaluation/quick_viz.py` - Visualize samples

## 🎯 Benefits of This Structure

✅ **Clear separation of concerns** - Each folder has a specific purpose  
✅ **Logical workflow** - Scripts numbered by execution order  
✅ **Easy navigation** - README in each folder explains contents  
✅ **Scalable** - Easy to add new scripts in appropriate folders  
✅ **Clean root** - No clutter, just essential files  
✅ **Importable library** - `src/` can be imported as a Python package  

## 🚀 Quick Commands

```bash
# View dataset stats
python scripts/4_evaluation/quick_stats.py

# Test dataloader
python scripts/4_evaluation/test_dataloader_with_splits.py

# Extract landmarks (if needed)
python scripts/2_preprocessing/extract_landmarks.py

# Preprocess features (if needed)
python scripts/2_preprocessing/preprocess_features.py
```
