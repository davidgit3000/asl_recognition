#!/usr/bin/env python3
"""
Verify MS-ASL downloads by counting actual video files per label.
Reports which labels have empty folders or insufficient clips.
"""
import yaml, csv
from pathlib import Path
from collections import defaultdict

CFG = yaml.safe_load(open("configs/config.yaml"))
CLIPS_DIR = Path(CFG.get("msasl_clips_dir", "./data/microsoft_asl/ms_asl"))
MAX_PER_CLASS = int(CFG.get("msasl_per_class", 10))
REPORT_CSV = Path("artifacts/manifests/msasl_verification.csv")

if not CLIPS_DIR.exists():
    print(f"❌ Clips directory does not exist: {CLIPS_DIR}")
    exit(1)

# Count video files per label
counts = defaultdict(int)
for label_dir in sorted(CLIPS_DIR.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    video_files = list(label_dir.glob("*.mp4"))
    counts[label] = len(video_files)

# Generate report
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["label", "count", "status"])
    w.writeheader()
    
    empty = []
    insufficient = []
    sufficient = []
    
    for label in sorted(counts.keys()):
        count = counts[label]
        if count == 0:
            status = "empty"
            empty.append(label)
        elif count < MAX_PER_CLASS:
            status = "insufficient"
            insufficient.append(label)
        else:
            status = "ok"
            sufficient.append(label)
        
        w.writerow({"label": label, "count": count, "status": status})

# Print summary
print(f"\n{'='*60}")
print(f"MS-ASL Download Verification")
print(f"{'='*60}")
print(f"Clips directory: {CLIPS_DIR}")
print(f"Target per class: {MAX_PER_CLASS}")
print(f"\n✓ Sufficient ({len(sufficient)} labels with ≥{MAX_PER_CLASS} clips):")
for label in sufficient:
    print(f"    {label}: {counts[label]}")

if insufficient:
    print(f"\n⚠️  Insufficient ({len(insufficient)} labels with <{MAX_PER_CLASS} clips):")
    for label in insufficient:
        print(f"    {label}: {counts[label]}/{MAX_PER_CLASS}")

if empty:
    print(f"\n❌ Empty ({len(empty)} labels with 0 clips):")
    for label in empty:
        print(f"    {label}")

print(f"\nTotal labels: {len(counts)}")
print(f"Report saved → {REPORT_CSV}")
print(f"{'='*60}\n")
