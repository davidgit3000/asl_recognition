import os, cv2, json, yaml, pandas as pd
from pathlib import Path
from tqdm import tqdm

CFG = yaml.safe_load(open("configs/config.yaml"))
LABEL_MAP = json.load(open("configs/label_map.json")) if Path("configs/label_map.json").exists() else {}

DATA_ROOT = Path(CFG["data_root"])
OUT_CSV  = Path(CFG["manifest_out"]).resolve()
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def norm_label(x:str)->str:
    x = x.strip().lower().replace(" ", "_")
    return LABEL_MAP.get(x, x)

def video_meta(p):
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened(): return dict(fps=None, frames=None, width=None, height=None)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return dict(fps=fps, frames=frames, width=width, height=height)

rows = []

# 2.1 Kaggle ASL Alphabet (images per letter)
kag_dir = Path(CFG["kaggle_asl_dir"])
if kag_dir.exists():
    for label_dir in sorted(kag_dir.glob("*")):
        if not label_dir.is_dir(): continue
        label = norm_label(label_dir.name)
        for img in label_dir.rglob("*"):
            if img.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
            rows.append(dict(
                id=f"kag_{img.stem}",
                source="kaggle",
                path=str(img.resolve()),
                label=label,
                media_type="image",
                fps=0, frames=1, width=None, height=None,
                signer=None, session=None, split=None
            ))

# 2.2 MS-ASL (videos you selected; place them under data/msasl/<label>/*.mp4)
ms_dir = Path(CFG["msasl_clips_dir"])
if ms_dir.exists():
    for label_dir in sorted(ms_dir.glob("*")):
        if not label_dir.is_dir(): continue
        label = norm_label(label_dir.name)
        for vid in label_dir.rglob("*"):
            if vid.suffix.lower() not in {".mp4",".mov",".mkv",".avi"}: continue
            m = video_meta(vid)
            rows.append(dict(
                id=f"msasl_{vid.stem}",
                source="msasl",
                path=str(vid.resolve()),
                label=label,
                media_type="video",
                fps=m["fps"], frames=m["frames"], width=m["width"], height=m["height"],
                signer=None, session=None, split=None
            ))

# 2.3 Personal (videos in personal/S1..S5/<label>/*.mp4)
per_dir = Path(CFG["personal_dir"])
if per_dir.exists():
    for session_dir in sorted(per_dir.glob("S*")):
        session = session_dir.name
        for label_dir in sorted(session_dir.glob("*")):
            if not label_dir.is_dir(): continue
            label = norm_label(label_dir.name)
            for vid in label_dir.rglob("*"):
                if vid.suffix.lower() not in {".mp4",".mov",".mkv",".avi"}: continue
                m = video_meta(vid)
                rows.append(dict(
                    id=f"per_{session}_{vid.stem}",
                    source="personal",
                    path=str(vid.resolve()),
                    label=label,
                    media_type="video",
                    fps=m["fps"], frames=m["frames"], width=m["width"], height=m["height"],
                    signer="you", session=session,
                    # session-wise split rule from your report:
                    split = "train" if session in {"S1","S2","S3"} else ("val" if session=="S4" else ("test" if session=="S5" else None))
                ))

df = pd.DataFrame(rows)
# keep only labels listed in config (helps enforce scope)
allowed = set([l.lower() for l in CFG["labels"]])
df = df[df["label"].isin(allowed)].reset_index(drop=True)

df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows → {OUT_CSV}")
