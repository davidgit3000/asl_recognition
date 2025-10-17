# Data Preparation Scripts

Scripts for downloading, organizing, and preparing the ASL dataset.

## Workflow Order

1. **Download Kaggle datasets**
   
   **Option A: Using the script (Mac/Linux):**
   ```bash
   # Make sure you have Kaggle API credentials set up (~/.kaggle/kaggle.json)
   bash download_kaggle_datasets.sh
   ```
   
   **Option B: Manual download:**
   - Download from [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
   - Download from [ASL Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
   - Extract to `data/kaggle_asl1/` and `data/kaggle_asl2/`

1a. **Download MS-ASL dataset (optional)**
   - Download from [MS-ASL Dataset](https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset#overview)
   - Then, use the automated download scripts (steps 3-5 below)
   
2. **`combine_kaggle_asl.py`** - Combine multiple Kaggle datasets
   - Merges different Kaggle ASL datasets
   - Deduplicates images
   - Normalizes labels

3. **`build_msasl_manifest.py`** - Build MS-ASL manifest from JSON files
   - Reads MS-ASL metadata
   - Filters viable segments

4. **`msasl_make_list.py`** - Select MS-ASL classes and create download list
   - Chooses top-k classes
   - Over-samples for fallback

5. **`msasl_download_and_trim.py`** - Download and trim MS-ASL videos
   - Downloads from YouTube
   - Trims to exact segments

6. **`verify_msasl_downloads.py`** - Verify download completeness
   - Checks if target number of videos downloaded per class

7. **`build_manifest.py`** - Build unified manifest (Kaggle + MS-ASL)
   - Creates master CSV with all samples
   - Adds metadata (fps, frames, etc.)

8. **`assign_splits.py`** - Assign train/val/test splits
   - Stratified 70/15/15 split
   - Ensures balanced class distribution

## Output
- `artifacts/manifests/manifest_v1.csv` - Master manifest with splits
