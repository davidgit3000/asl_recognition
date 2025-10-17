import os, yaml, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2, mediapipe as mp

CFG = yaml.safe_load(open("configs/config.yaml"))
MANIFEST = Path(CFG["manifest_out"])
OUT_DIR = Path(CFG["artifacts_root"]) / "landmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# Order: face(468), pose(33), left(21), right(21) → total 543 points, each (x,y,z,visibility?)
IDX_SIZES = dict(face=468, pose=33, left=21, right=21)

def frame_landmarks(rgb, holo):
    res = holo.process(rgb)
    pts = []
    # face
    if res.face_landmarks and res.face_landmarks.landmark:
        for lm in res.face_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z, 1.0])
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["face"])
    # pose
    if res.pose_landmarks and res.pose_landmarks.landmark:
        for lm in res.pose_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["pose"])
    # left hand
    if res.left_hand_landmarks and res.left_hand_landmarks.landmark:
        for lm in res.left_hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z, 1.0])
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["left"])
    # right hand
    if res.right_hand_landmarks and res.right_hand_landmarks.landmark:
        for lm in res.right_hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z, 1.0])
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["right"])
    return np.array(pts, dtype=np.float32)  # [543,4]

def smooth_ema(arr, alpha=0.4):
    out = np.copy(arr)
    for t in range(1, len(arr)):
        out[t] = alpha*arr[t] + (1-alpha)*out[t-1]
    return out

df = pd.read_csv(MANIFEST)

# OPTIMIZATION 1: Use static_image_mode=True for images, separate model for videos
holo_static = mp_holistic.Holistic(static_image_mode=True, model_complexity=0)   # faster for images
holo_video  = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)  # for videos

# OPTIMIZATION 2: Filter out already processed items upfront
df["out_path"] = df["id"].apply(lambda x: OUT_DIR / f"{x}.npy")
df_todo = df[~df["out_path"].apply(lambda p: p.exists())].copy()
print(f"Processing {len(df_todo)}/{len(df)} items (skipping {len(df) - len(df_todo)} existing)")

processed = 0
for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Extracting landmarks"):
    out_path = row["out_path"]
    
    try:
        if row["media_type"] == "image":
            bgr = cv2.imread(row["path"])
            if bgr is None:
                continue  # skip corrupted images
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pts = frame_landmarks(rgb, holo_static)[None, ...]  # [1, 543, 4]
        else:
            cap = cv2.VideoCapture(row["path"])
            frames = []
            frame_count = 0
            max_frames = 300  # OPTIMIZATION 3: Cap video length to avoid very long videos
            while frame_count < max_frames:
                ok, bgr = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_landmarks(rgb, holo_video))
                frame_count += 1
            cap.release()
            pts = np.stack(frames, axis=0) if frames else np.zeros((0,543,4), np.float32)
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")
        continue

    # normalize by image size using x,y only (z is relative in MediaPipe)
    xy = pts[..., :2]
    pts[..., :2] = np.clip(xy, 0, 1)  # already normalized in [0,1], just ensuring

    # temporal smoothing (simple EMA)
    if len(pts) > 1:
        pts = smooth_ema(pts, alpha=0.4)

    np.save(out_path, pts)  # shape [T, 543, 4]
    processed += 1

print(f"Saved landmarks for {processed} items → {OUT_DIR}")
holo_static.close()
holo_video.close()
