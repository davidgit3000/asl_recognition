# Kaggle ASL Datasets Structure

## Dataset 1: grassknoted/asl-alphabet (kaggle_asl1)

**Path:** `data/kaggle_asl1/asl_alphabet_train/asl_alphabet_train/`

**Structure:**
```
asl_alphabet_train/
├── A/ (3000 images)
├── B/ (3000 images)
├── C/ (3000 images)
├── D/ (3000 images)
├── E/ (3000 images)
├── F/ (3000 images)
├── G/ (3000 images)
├── H/ (3000 images)
├── I/ (3000 images)
├── J/ (3000 images)
├── K/ (3000 images)
├── L/ (3000 images)
├── M/ (3000 images)
├── N/ (3000 images)
├── O/ (3000 images)
├── P/ (3000 images)
├── Q/ (3000 images)
├── R/ (3000 images)
├── S/ (3000 images)
├── T/ (3000 images)
├── U/ (2969 images)
├── V/ (2748 images)
├── W/ (3000 images)
├── X/ (3000 images)
├── Y/ (3000 images)
├── Z/ (3000 images)
├── del/ (3000 images)
├── nothing/ (3000 images)
└── space/ (3000 images)
```

**Total:** ~86,717 images
**Classes:** A-Z (uppercase), del, nothing, space
**Images per class:** ~3000 (except U: 2969, V: 2748)

---

## Dataset 2: ayuraj/asl-dataset (kaggle_asl2)

**Path:** `data/kaggle_asl2/asl_dataset/`

**Structure:**
```
asl_dataset/
├── a/ (70 images)
├── b/ (70 images)
├── c/ (70 images)
├── d/ (70 images)
├── e/ (70 images)
├── f/ (70 images)
├── g/ (70 images)
├── h/ (70 images)
├── i/ (70 images)
├── j/ (70 images)
├── k/ (70 images)
├── l/ (70 images)
├── m/ (70 images)
├── n/ (70 images)
├── o/ (70 images)
├── p/ (70 images)
├── q/ (70 images)
├── r/ (70 images)
├── s/ (70 images)
├── t/ (65 images)
├── u/ (70 images)
├── v/ (70 images)
├── w/ (70 images)
├── x/ (70 images)
├── y/ (70 images)
├── z/ (70 images)
├── 0/ (70 images)
├── 1/ (70 images)
├── 2/ (70 images)
├── 3/ (70 images)
├── 4/ (70 images)
├── 5/ (70 images)
├── 6/ (70 images)
├── 7/ (70 images)
├── 8/ (70 images)
└── 9/ (70 images)
```

**Total:** ~2,515 images
**Classes:** a-z (lowercase), 0-9
**Images per class:** ~70 (except t: 65)

---

## Combined Dataset Configuration

The `scripts/combine_kaggle_asl.py` script is now configured to:

1. **Source directories:**
   - `data/kaggle_asl1/asl_alphabet_train/asl_alphabet_train/` (grassknoted)
   - `data/kaggle_asl2/asl_dataset/` (ayuraj)

2. **Output directory:**
   - `data/kaggle_asl_combined/`

3. **Processing:**
   - Normalizes lowercase letters (a-z) to uppercase (A-Z)
   - Keeps only A-Z letters (skips digits 0-9 for now)
   - Optionally keeps "nothing" class as negatives (max 200 samples)
   - Skips "space" and "del" classes
   - Deduplicates images using MD5 hash
   - Optional per-class cap (currently disabled)

4. **Expected output:**
   - 26 folders (A-Z) + optional NOTHING folder
   - ~3,070 images per letter (3000 from kaggle_asl1 + 70 from kaggle_asl2)
   - Total: ~80,000+ images for A-Z

---

## Usage

Run the combine script:
```bash
python scripts/combine_kaggle_asl.py
```

This will:
- Create `data/kaggle_asl_combined/` with A-Z folders
- Copy and deduplicate all images
- Print a summary of images per class
