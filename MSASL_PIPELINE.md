# MS-ASL Video Download Pipeline

This guide explains how to download ASL videos from the MS-ASL dataset with guaranteed availability.

## Overview

The pipeline ensures you get **up to 10 video examples per word** by:
1. Checking availability before selecting classes
2. Over-sampling candidates (3× by default) to handle download failures
3. Using ranked fallback when videos are private/unavailable
4. Enforcing per-class caps to stop at 10 successful downloads

## Prerequisites

```bash
# Ensure you have ffmpeg installed
brew install ffmpeg

# Install Python dependencies (including yt-dlp)
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml`:

```yaml
# Target number of classes (maximum 20, can be less)
msasl_top_k: 20

# Videos per class (up to 10)
msasl_per_class: 10

# Over-sampling factor for fallback (3× recommended)
msasl_over_sample: 3

# Optional: Explicitly request specific words
# Only words with ≥10 candidates will be kept
# If fewer than msasl_top_k viable, auto-backfill from high-availability words
msasl_selected_classes: 
  - "hello"
  - "thank_you"
  - "please"
  - "yes"
  - "no"
  # ... add more
```

## Pipeline Steps

### Step 1: Build the manifest from MS-ASL JSONs

```bash
python scripts/build_msasl_manifest.py
```

**Output:**
- `artifacts/manifests/msasl_all.csv` — All viable MS-ASL segments after filtering

**What it does:**
- Reads `data/microsoft_asl/ms_asl_json/MSASL_{train,val,test}.json`
- Normalizes labels (e.g., "thank you" → "thank_you")
- Filters out segments with missing URLs, invalid timestamps, or duration <0.5s
- Keeps only labels from your `configs/config.yaml` (excluding single letters)

---

### Step 2: Select classes and generate ranked candidate list

```bash
python scripts/msasl_make_list.py
```

**Output:**
- `artifacts/manifests/msasl_counts.csv` — Availability report (label → count)
- `artifacts/manifests/selected_msasl_classes.txt` — Final chosen classes
- `artifacts/manifests/msasl_segments.csv` — Ranked candidates for download (up to 30 per class)

**What it does:**
- Computes per-class availability from `msasl_all.csv`
- **Only selects labels with ≥ `msasl_per_class` candidates**
- If you provided `msasl_selected_classes`:
  - Keeps only viable ones (drops those with <10 candidates)
  - Backfills from high-availability labels to reach `msasl_top_k`
- Over-samples candidates per class (3× = 30 candidates per class)
- Ranks candidates by signer diversity and duration
- Adds a `rank` column (1 = highest priority)

**Check the output:**
```bash
cat artifacts/manifests/selected_msasl_classes.txt
head artifacts/manifests/msasl_counts.csv
```

---

### Step 3: Download and trim videos with fallback

```bash
python scripts/msasl_download_and_trim.py
```

**Output:**
- `data/microsoft_asl/ms_asl/<label>/*.mp4` — Downloaded video clips
- `artifacts/manifests/download_log.csv` — Detailed log of all attempts

**What it does:**
- Processes candidates in rank order (best first)
- For each class:
  - Downloads YouTube video → `artifacts/tmp/`
  - Trims segment using ffmpeg
  - Saves to `data/microsoft_asl/ms_asl/<label>/`
  - **Stops after 10 successful clips per class**
- Skips duplicates (if file already exists)
- Logs all attempts (success/failure + reason)
- Handles private/removed videos gracefully by trying next-ranked candidate

**Common failure reasons:**
- `DownloadError: Video unavailable` — Private or removed video
- `DownloadError: This video is not available` — Geo-blocked or deleted
- `ffmpeg_failed` — Trim/encoding issue (rare)

---

### Step 4: Verify downloads

```bash
python scripts/verify_msasl_downloads.py
```

**Output:**
- `artifacts/manifests/msasl_verification.csv` — Per-label file counts
- Console summary showing:
  - ✓ Labels with ≥10 clips (sufficient)
  - ⚠️ Labels with <10 clips (insufficient)
  - ❌ Labels with 0 clips (empty)

**What it does:**
- Counts actual `.mp4` files in each label folder
- Flags labels that didn't reach the target

---

## Troubleshooting

### Some labels still have <10 videos after download

**Cause:** Too many private/unavailable videos in the candidate pool.

**Solution:**
1. Check `artifacts/manifests/download_log.csv` to see failure reasons
2. Increase over-sampling: set `msasl_over_sample: 5` in `configs/config.yaml`
3. Re-run Step 2 and Step 3
4. Alternatively, remove that label from `msasl_selected_classes` and let the script auto-backfill

### A label I want was dropped

**Cause:** That label has <10 viable candidates in MS-ASL after filtering.

**Solution:**
1. Check `artifacts/manifests/msasl_counts.csv` to see actual availability
2. Lower `msasl_per_class` (e.g., to 5) if you're okay with fewer examples
3. Or choose a different label

### Download is very slow

**Cause:** YouTube throttling or network issues.

**Solution:**
- The script already uses `yt-dlp` which is robust
- You can pause and resume (script skips existing files)
- Consider running overnight for large datasets

### ffmpeg not found

**Solution:**
```bash
brew install ffmpeg
```

---

## File Structure

```
asl_model/
├── configs/
│   ├── config.yaml              # Main configuration
│   └── label_map.json           # Label normalization rules
├── data/
│   └── microsoft_asl/
│       ├── ms_asl_json/         # MS-ASL metadata JSONs (input)
│       └── ms_asl/              # Downloaded video clips (output)
│           ├── hello/
│           ├── thank_you/
│           └── ...
├── artifacts/
│   ├── manifests/
│   │   ├── msasl_all.csv        # All viable segments
│   │   ├── msasl_counts.csv     # Availability report
│   │   ├── msasl_segments.csv   # Ranked candidates
│   │   ├── selected_msasl_classes.txt
│   │   ├── download_log.csv     # Download attempt log
│   │   └── msasl_verification.csv
│   └── tmp/                     # Temporary YouTube downloads
└── scripts/
    ├── build_msasl_manifest.py
    ├── msasl_make_list.py
    ├── msasl_download_and_trim.py
    └── verify_msasl_downloads.py
```

---

## Quick Start

```bash
# 1. Build manifest
python scripts/build_msasl_manifest.py

# 2. Select classes (with availability checks)
python scripts/msasl_make_list.py

# 3. Download videos (with fallback)
python scripts/msasl_download_and_trim.py

# 4. Verify
python scripts/verify_msasl_downloads.py
```

---

## Advanced: Adjusting the Word List

If you want to change which words to download:

1. Edit `configs/config.yaml`:
   ```yaml
   msasl_selected_classes: ["new_word1", "new_word2", ...]
   ```

2. Re-run from Step 2:
   ```bash
   python scripts/msasl_make_list.py
   python scripts/msasl_download_and_trim.py
   python scripts/verify_msasl_downloads.py
   ```

The pipeline will automatically:
- Check if each word has ≥10 candidates
- Drop words with insufficient data
- Backfill with high-availability words to reach your target count
- Download up to 10 clips per word

---

## Summary

This pipeline guarantees robust video downloads by:
- **Pre-checking availability** before selecting words
- **Over-sampling candidates** (3× by default) to absorb failures
- **Using ranked fallback** to try alternative videos when downloads fail
- **Enforcing per-class caps** to stop at exactly 10 successful clips
- **Logging all attempts** for transparency

You'll end up with up to 10 video examples per selected word, with automatic handling of private/removed videos.
