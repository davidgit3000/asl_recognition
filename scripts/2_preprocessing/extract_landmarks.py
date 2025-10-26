import os, yaml, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2, mediapipe as mp

CFG = yaml.safe_load(open("configs/config.yaml"))
MANIFEST = Path(CFG["manifest_out"])
OUT_DIR = Path(CFG["artifacts_root"]) / "landmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
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

def frame_landmarks_hands_only(rgb, hands_detector):
    """
    Extract landmarks using MediaPipe Hands (for hand-only images like Kaggle).
    Returns same format as frame_landmarks: [543, 4] with zeros for face/pose.
    """
    res = hands_detector.process(rgb)
    pts = []
    
    # Face: fill with zeros (not detected)
    pts.extend([[0,0,0,0]]*IDX_SIZES["face"])
    
    # Pose: fill with zeros (not detected)
    pts.extend([[0,0,0,0]]*IDX_SIZES["pose"])
    
    # Hands: extract from MediaPipe Hands
    left_hand_found = False
    right_hand_found = False
    
    if res.multi_hand_landmarks and res.multi_handedness:
        for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            # Determine if left or right hand
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            
            hand_pts = []
            for lm in hand_landmarks.landmark:
                hand_pts.append([lm.x, lm.y, lm.z, 1.0])  # visibility=1.0 for detected hands
            
            if hand_label == "Left" and not left_hand_found:
                # This is the left hand
                left_hand_found = True
                # Store for later (we need to add in correct order)
                left_hand_data = hand_pts
            elif hand_label == "Right" and not right_hand_found:
                # This is the right hand
                right_hand_found = True
                right_hand_data = hand_pts
    
    # Add left hand (or zeros if not found)
    if left_hand_found:
        pts.extend(left_hand_data)
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["left"])
    
    # Add right hand (or zeros if not found)
    if right_hand_found:
        pts.extend(right_hand_data)
    else:
        pts.extend([[0,0,0,0]]*IDX_SIZES["right"])
    
    return np.array(pts, dtype=np.float32)  # [543, 4]

def smooth_ema(arr, alpha=0.4):
    out = np.copy(arr)
    for t in range(1, len(arr)):
        out[t] = alpha*arr[t] + (1-alpha)*out[t-1]
    return out

df = pd.read_csv(MANIFEST)

# OPTIMIZATION 1: Use static_image_mode=True for images, separate model for videos
holo_static = mp_holistic.Holistic(static_image_mode=True, model_complexity=0)   # faster for images
holo_video  = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)  # for videos
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)  # for hand-only images

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
            
            # Try holistic first (for full body)
            pts = frame_landmarks(rgb, holo_static)[None, ...]  # [1, 543, 4]
            
            # If holistic failed (all zeros), try hands-only detection (for Kaggle)
            if pts.max() == 0.0:
                pts = frame_landmarks_hands_only(rgb, hands_static)[None, ...]  # [1, 543, 4]
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
hands_static.close()
