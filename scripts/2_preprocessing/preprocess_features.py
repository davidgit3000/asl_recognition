import os, yaml, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter

CFG = yaml.safe_load(open("configs/config.yaml"))
MANIFEST = Path(CFG["manifest_out"])
LANDMARKS_DIR = Path(CFG["artifacts_root"]) / "landmarks"
FEATURES_DIR = Path(CFG["artifacts_root"]) / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Landmark indices (from extract_landmarks.py)
IDX_FACE = slice(0, 468)
IDX_POSE = slice(468, 468+33)
IDX_LEFT_HAND = slice(468+33, 468+33+21)
IDX_RIGHT_HAND = slice(468+33+21, 468+33+21+21)

# Key pose landmarks for normalization (shoulders, hips) - BEFORE extraction
LEFT_SHOULDER_FULL = 468 + 11   # pose landmark 11 in full array
RIGHT_SHOULDER_FULL = 468 + 12  # pose landmark 12 in full array
LEFT_HIP_FULL = 468 + 23        # pose landmark 23 in full array
RIGHT_HIP_FULL = 468 + 24       # pose landmark 24 in full array

# After extraction (pose only, 33 landmarks starting at index 0)
LEFT_SHOULDER_EXTRACTED = 11    # pose landmark 11 in extracted array
RIGHT_SHOULDER_EXTRACTED = 12   # pose landmark 12 in extracted array
LEFT_HIP_EXTRACTED = 23         # pose landmark 23 in extracted array
RIGHT_HIP_EXTRACTED = 24        # pose landmark 24 in extracted array

def normalize_landmarks_full(pts):
    """
    Normalize landmarks by centering on torso and scaling by shoulder width.
    Works on FULL landmark array (543 points).
    Args:
        pts: [T, 543, 4] array of landmarks (x, y, z, visibility)
    Returns:
        normalized: [T, 543, 4] normalized landmarks
    """
    normalized = np.copy(pts)
    
    for t in range(len(pts)):
        # Get shoulder and hip positions
        left_shoulder = pts[t, LEFT_SHOULDER_FULL, :3]
        right_shoulder = pts[t, RIGHT_SHOULDER_FULL, :3]
        left_hip = pts[t, LEFT_HIP_FULL, :3]
        right_hip = pts[t, RIGHT_HIP_FULL, :3]
        
        # Check if pose landmarks are valid (visibility > 0)
        if pts[t, LEFT_SHOULDER_FULL, 3] > 0 and pts[t, RIGHT_SHOULDER_FULL, 3] > 0:
            # Center point: midpoint of shoulders and hips
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0
            
            # Scale: shoulder width
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            scale = shoulder_width if shoulder_width > 0.01 else 1.0
            
            # Normalize all landmarks
            normalized[t, :, :3] = (pts[t, :, :3] - center) / scale
        # else: keep original (likely all zeros for missing pose)
    
    return normalized

def normalize_landmarks_extracted(pts):
    """
    Normalize extracted landmarks (pose + hands only, 75 points).
    Args:
        pts: [T, 75, 4] array of extracted landmarks (pose 33 + left hand 21 + right hand 21)
    Returns:
        normalized: [T, 75, 4] normalized landmarks
    """
    normalized = np.copy(pts)
    
    for t in range(len(pts)):
        # Get shoulder and hip positions (now at different indices)
        left_shoulder = pts[t, LEFT_SHOULDER_EXTRACTED, :3]
        right_shoulder = pts[t, RIGHT_SHOULDER_EXTRACTED, :3]
        left_hip = pts[t, LEFT_HIP_EXTRACTED, :3]
        right_hip = pts[t, RIGHT_HIP_EXTRACTED, :3]
        
        # Check if pose landmarks are valid (visibility > 0)
        if pts[t, LEFT_SHOULDER_EXTRACTED, 3] > 0 and pts[t, RIGHT_SHOULDER_EXTRACTED, 3] > 0:
            # Center point: midpoint of shoulders and hips
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0
            
            # Scale: shoulder width
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            scale = shoulder_width if shoulder_width > 0.01 else 1.0
            
            # Normalize all landmarks
            normalized[t, :, :3] = (pts[t, :, :3] - center) / scale
        # else: keep original (likely all zeros for missing pose)
    
    return normalized

def smooth_savgol(arr, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to reduce jitter.
    Args:
        arr: [T, N, D] array
        window_length: smoothing window (must be odd)
        polyorder: polynomial order
    Returns:
        smoothed: [T, N, D] array
    """
    if len(arr) < window_length:
        return arr  # too short to smooth
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    smoothed = np.copy(arr)
    T, N, D = arr.shape
    
    # Smooth each dimension independently
    for n in range(N):
        for d in range(D):
            # Only smooth if not all zeros (missing landmarks)
            if np.any(arr[:, n, d] != 0):
                smoothed[:, n, d] = savgol_filter(arr[:, n, d], window_length, polyorder)
    
    return smoothed

def extract_relevant_landmarks(pts):
    """
    Extract only relevant landmarks (pose + hands, skip face for now).
    Face landmarks (468 points) are often noisy and less critical for ASL.
    Args:
        pts: [T, 543, 4] full landmarks
    Returns:
        relevant: [T, 75, 4] (pose 33 + left hand 21 + right hand 21)
    """
    # Extract pose + hands
    pose = pts[:, IDX_POSE, :]      # [T, 33, 4]
    left_hand = pts[:, IDX_LEFT_HAND, :]   # [T, 21, 4]
    right_hand = pts[:, IDX_RIGHT_HAND, :] # [T, 21, 4]
    
    # Concatenate: [T, 75, 4] (33 + 21 + 21)
    relevant = np.concatenate([pose, left_hand, right_hand], axis=1)
    return relevant

def augment_rotation(pts, angle_deg=None):
    """
    Apply random rotation augmentation around z-axis (yaw).
    Args:
        pts: [T, N, 4] landmarks
        angle_deg: rotation angle in degrees (if None, random in [-15, 15])
    Returns:
        rotated: [T, N, 4] rotated landmarks
    """
    if angle_deg is None:
        angle_deg = np.random.uniform(-15, 15)
    
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around z-axis (2D rotation in x-y plane)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    rotated = np.copy(pts)
    rotated[:, :, :3] = pts[:, :, :3] @ R.T
    return rotated

def process_sample(landmark_path, apply_augmentation=False):
    """
    Process a single landmark file.
    Args:
        landmark_path: path to .npy file with raw landmarks
        apply_augmentation: whether to apply rotation augmentation
    Returns:
        features: [T, N, 4] processed features
    """
    # Load raw landmarks
    pts = np.load(landmark_path)  # [T, 543, 4]
    
    if len(pts) == 0:
        return None  # empty video
    
    # 1. Extract relevant landmarks (pose + hands only)
    pts = extract_relevant_landmarks(pts)  # [T, 75, 4]
    
    # 2. Normalize by torso position and scale
    pts = normalize_landmarks_extracted(pts)  # [T, 75, 4]
    
    # 3. Temporal smoothing
    pts = smooth_savgol(pts, window_length=5, polyorder=2)  # [T, 75, 4]
    
    # 4. Optional augmentation
    if apply_augmentation:
        pts = augment_rotation(pts)  # [T, 75, 4]
    
    return pts

# Main processing loop
df = pd.read_csv(MANIFEST)

# Filter out already processed items
df["feature_path"] = df["id"].apply(lambda x: FEATURES_DIR / f"{x}.npy")
df_todo = df[~df["feature_path"].apply(lambda p: p.exists())].copy()
print(f"Processing {len(df_todo)}/{len(df)} items (skipping {len(df) - len(df_todo)} existing)")

processed = 0
for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Preprocessing features"):
    landmark_path = LANDMARKS_DIR / f"{row['id']}.npy"
    feature_path = row["feature_path"]
    
    if not landmark_path.exists():
        print(f"Warning: landmark file not found for {row['id']}")
        continue
    
    try:
        # Process without augmentation for base features
        features = process_sample(landmark_path, apply_augmentation=False)
        
        if features is None or len(features) == 0:
            continue
        
        # Save processed features
        np.save(feature_path, features)
        processed += 1
        
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")
        continue

print(f"\nProcessed {processed} samples → {FEATURES_DIR}")
print(f"Feature shape: [T, 75, 4] (pose 33 + left hand 21 + right hand 21)")
print(f"Normalization: centered on torso, scaled by shoulder width")
print(f"Smoothing: Savitzky-Golay filter (window=5, poly=2)")