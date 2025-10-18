#!/usr/bin/env python3
"""Visualize preprocessed ASL features as a video."""
import cv2
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import sys

# Load config
CFG = yaml.safe_load(open("configs/config.yaml"))
features_dir = Path(CFG["artifacts_root"]) / "features"
output_path = Path(CFG["artifacts_root"]) / "landmarks" / "preview.mp4"

# Load manifest and find a video sample (MS-ASL) for better visualization
df = pd.read_csv(CFG["manifest_out"])

# Try to find an MS-ASL video sample (they have more frames)
msasl_samples = df[df['source'] == 'msasl']
if len(msasl_samples) > 0:
    row = msasl_samples.iloc[0]
    print(f"Using MS-ASL video sample: {row['label']}")
else:
    # Fall back to any sample
    row = df.iloc[0]
    print(f"Using Kaggle image sample: {row['label']} (may be static)")

feature_path = features_dir / f"{row['id']}.npy"

if not feature_path.exists():
    print(f"Error: Feature file not found: {feature_path}")
    print("Run preprocessing first: python scripts/2_preprocessing/preprocess_features.py")
    sys.exit(1)

# Load preprocessed features [T, 75, 4]
pts = np.load(str(feature_path))
print(f"Sample: {row['id']}")
print(f"Label: {row['label']}")
print(f"Features shape: {pts.shape}")
print(f"Frames: {len(pts)}")

# Feature indices (preprocessed features only have pose + hands, no face)
# 33 pose + 21 left hand + 21 right hand = 75 landmarks
pose_idx = slice(0, 33)
lh_idx = slice(33, 54)
rh_idx = slice(54, 75)

# Video settings
H, W = 720, 1280
fps = 15 if len(pts) > 1 else 1  # 1 fps for static images
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

if not out.isOpened():
    print("Error: Could not open video writer")
    sys.exit(1)

# Normalize coordinates to fit canvas
# Features are already normalized around origin, need to shift and scale
xy_coords = pts[:, :, :2]  # [T, 75, 2]
min_x, max_x = xy_coords[:, :, 0].min(), xy_coords[:, :, 0].max()
min_y, max_y = xy_coords[:, :, 1].min(), xy_coords[:, :, 1].max()

print(f"Coordinate range: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")

# Scale to fit in canvas with margin
margin = 100
scale = min((W - 2*margin) / (max_x - min_x + 1e-6), 
            (H - 2*margin) / (max_y - min_y + 1e-6))
center_x = W // 2
center_y = H // 2

print(f"Rendering {len(pts)} frames...")

for t in range(len(pts)):
    canvas = np.zeros((H, W, 3), np.uint8)
    
    # Draw different body parts in different colors
    for idxs, color, name in [
        (pose_idx, (0, 255, 0), "Pose"),      # Green
        (lh_idx, (255, 0, 0), "Left Hand"),   # Blue
        (rh_idx, (0, 0, 255), "Right Hand")   # Red
    ]:
        xy = pts[t, idxs, :2]
        vis = pts[t, idxs, 3] > 0.5  # Visibility threshold
        
        for (x, y), v in zip(xy, vis):
            if v:
                # Transform coordinates to canvas
                canvas_x = int(center_x + (x - (min_x + max_x) / 2) * scale)
                canvas_y = int(center_y + (y - (min_y + max_y) / 2) * scale)
                cv2.circle(canvas, (canvas_x, canvas_y), 4, color, -1)
    
    # Add text overlay
    cv2.putText(canvas, f"Label: {row['label']}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, f"Frame: {t+1}/{len(pts)}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"Source: {row['source']}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Legend
    cv2.putText(canvas, "Pose", (W-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, "Left Hand", (W-200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(canvas, "Right Hand", (W-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    out.write(canvas)

out.release()
print(f"✅ Wrote {output_path}")
print(f"\nTo view: open {output_path}")
