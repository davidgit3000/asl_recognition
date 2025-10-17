import hashlib, shutil, os
from pathlib import Path
from tqdm import tqdm

# INPUTS
SRC_DIRS = [
    Path("data/kaggle_asl1/asl_alphabet_train/asl_alphabet_train"),  # grassknoted dataset
    Path("data/kaggle_asl2/asl_dataset")                              # ayuraj dataset
]
OUT_DIR  = Path("data/kaggle_asl_combined")

# CONFIG — Stage 1 scope: A–Z only
ALLOW_LETTERS = set([chr(c) for c in range(ord('A'), ord('Z')+1)])
USE_NOTHING_AS_NEG = True      # keep a small sample of "NOTHING" as negatives
NEG_MAX_SAMPLES = 200          # limit negatives so they don't dominate
CAP_PER_CLASS = 1000           # limit to 1000 images per class for efficiency

# Map raw folder names to normalized labels
def normalize(name: str):
    n = name.strip().lower()
    # unify common special classes
    if n in {"space", "delete"}:
        return None  # skip for Stage 1
    if n == "nothing":
        return "NOTHING"
    # letters & digits
    if len(n) == 1:
        if n.isalpha():
            return n.upper()
        if n.isdigit():
            return None  # skip digits for Stage 1
    # unknown folder -> skip
    return None

# hash helper to deduplicate identical images
def file_hash(p: Path, chunk=65536):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    seen_hash = set()
    neg_kept = 0

    # create letter folders
    for L in ALLOW_LETTERS:
        (OUT_DIR / L).mkdir(parents=True, exist_ok=True)
    if USE_NOTHING_AS_NEG:
        (OUT_DIR / "NOTHING").mkdir(parents=True, exist_ok=True)

    for src in SRC_DIRS:
        if not src.exists():
            print(f"Skip missing: {src}")
            continue
        for cls_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
            label = normalize(cls_dir.name)
            if label is None:
                continue
            if label == "NOTHING" and not USE_NOTHING_AS_NEG:
                continue
            if label != "NOTHING" and label not in ALLOW_LETTERS:
                continue

            out_c = OUT_DIR / label
            out_c.mkdir(parents=True, exist_ok=True)

            copied_here = counts.get(label, 0)
            for img in tqdm(list(cls_dir.rglob("*")), desc=f"{cls_dir.name} → {label}"):
                if not img.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}: 
                    continue

                if CAP_PER_CLASS and copied_here >= CAP_PER_CLASS:
                    break
                if label == "NOTHING" and neg_kept >= NEG_MAX_SAMPLES:
                    break

                h = file_hash(img)
                if h in seen_hash:
                    continue
                seen_hash.add(h)

                # destination path
                dst = out_c / f"{h}{img.suffix.lower()}"
                try:
                    shutil.copy2(img, dst)
                    copied_here += 1
                    if label == "NOTHING":
                        neg_kept += 1
                except Exception as e:
                    print("copy error:", e)

            counts[label] = copied_here

    # Summary
    total = sum(counts.values())
    print("\nCombined summary:")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")
    print(f"TOTAL images: {total}")

if __name__ == "__main__":
    main()
