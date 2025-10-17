# scripts/msasl_download_and_trim.py
import subprocess, sys, csv, time
from pathlib import Path
import pandas as pd, yaml
from yt_dlp import YoutubeDL
from tqdm import tqdm

CFG = yaml.safe_load(open("configs/config.yaml"))
CSV = Path("artifacts/manifests/msasl_segments.csv")
CLIPS_DIR = Path(CFG.get("msasl_clips_dir","./data/microsoft_asl/ms_asl"))
TMP_DIR   = Path("artifacts/tmp")
LOG_CSV   = Path("artifacts/manifests/download_log.csv")
MAX_PER_CLASS = int(CFG.get("msasl_per_class", 10))
MAX_RETRIES = 3  # Retry network errors

CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
# Sort by label and rank to process in priority order
df = df.sort_values(["label_text", "rank"])

ydl = YoutubeDL({
    "outtmpl": str(TMP_DIR / "%(id)s.%(ext)s"),
    "format": "mp4/bestvideo+bestaudio/best",
    "quiet": True, 
    "noprogress": True
})

def trim_copy(src, dst, start, end):
    cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(src)]
    if end>start: cmd += ["-t", f"{(end-start):.3f}"]
    cmd += ["-c","copy",str(dst)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode==0

def trim_encode(src, dst, start, end):
    cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(src)]
    if end>start: cmd += ["-t", f"{(end-start):.3f}"]
    cmd += ["-c:v","libx264","-c:a","aac","-movflags","+faststart",str(dst)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode==0

# Track successes per class
class_counts = {}
log_rows = []

for _, r in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
    label = str(r["label_text"])
    url   = str(r["yt_url"])
    start = float(r.get("start_time",0.0) or 0.0)
    end   = float(r.get("end_time",0.0) or 0.0)
    rank  = int(r.get("rank", 0))

    # Check if we've already hit the cap for this class
    if class_counts.get(label, 0) >= MAX_PER_CLASS:
        continue

    out_dir = CLIPS_DIR/label
    out_dir.mkdir(parents=True, exist_ok=True)

    status = "failed"
    reason = ""
    dst_path = ""

    # Retry loop for network errors
    for attempt in range(MAX_RETRIES):
        try:
            info = ydl.extract_info(url, download=True)
            vid_id = info.get("id")
            ext = info.get("ext","mp4")
            src = TMP_DIR/f"{vid_id}.{ext}"
            if not src.exists():
                src = TMP_DIR/f"{vid_id}.mp4"  # muxed fallback

            dst = out_dir/f"{vid_id}_{int(start*1000)}_{int(end*1000)}.mp4"
            
            # Skip if already exists
            if dst.exists():
                status = "success"
                reason = "already_exists"
                dst_path = str(dst)
                class_counts[label] = class_counts.get(label, 0) + 1
            else:
                # Try trim with copy codec first, then re-encode
                if trim_copy(src, dst, start, end):
                    status = "success"
                    reason = "trim_copy"
                    dst_path = str(dst)
                    class_counts[label] = class_counts.get(label, 0) + 1
                elif trim_encode(src, dst, start, end):
                    status = "success"
                    reason = "trim_encode"
                    dst_path = str(dst)
                    class_counts[label] = class_counts.get(label, 0) + 1
                else:
                    reason = "ffmpeg_failed"
            break  # Success, exit retry loop
            
        except Exception as e:
            error_msg = str(e)
            reason = f"{type(e).__name__}: {error_msg[:100]}"
            
            # Check if it's a network error worth retrying
            is_network_error = any(x in error_msg.lower() for x in 
                ['nodename', 'servname', 'network', 'connection', 'timeout', 'dns'])
            
            if is_network_error and attempt < MAX_RETRIES - 1:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue  # Retry
            else:
                # Permanent error or max retries reached
                break

    log_rows.append({
        "label_text": label,
        "rank": rank,
        "yt_url": url,
        "start_time": start,
        "end_time": end,
        "status": status,
        "reason": reason,
        "dst_path": dst_path
    })

# Write log
log_cols = ["label_text","rank","yt_url","start_time","end_time","status","reason","dst_path"]
with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=log_cols)
    w.writeheader()
    w.writerows(log_rows)

# Summary
total_success = sum(1 for r in log_rows if r["status"] == "success")
total_fail = len(log_rows) - total_success

print(f"\n{'='*60}")
print(f"Download complete: {total_success} succeeded, {total_fail} failed")
print(f"Log saved → {LOG_CSV}")
print(f"\nPer-class counts:")
for label in sorted(class_counts.keys()):
    count = class_counts[label]
    status_icon = "✓" if count >= MAX_PER_CLASS else "⚠️"
    print(f"  {status_icon} {label}: {count}/{MAX_PER_CLASS}")
print(f"\nClips saved to: {CLIPS_DIR}")
print(f"{'='*60}")
