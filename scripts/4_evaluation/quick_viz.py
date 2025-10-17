# scripts/quick_viz.py
import cv2, numpy as np, yaml, pandas as pd
from pathlib import Path
CFG = yaml.safe_load(open("configs/config.yaml"))
land_dir = Path(CFG["artifacts_root"])/"landmarks"

df = pd.read_csv(CFG["manifest_out"])
row = df.iloc[0]
pts = np.load(str(land_dir/f"{row['id']}.npy"))  # [T, 543, 4]

# simple indices for pose (33) + hands (left 21, right 21)
pose_idx = slice(468, 468+33)
lh_idx   = slice(468+33, 468+33+21)
rh_idx   = slice(468+33+21, 468+33+21+21)

H, W = 720, 1280
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("artifacts/landmarks/preview.mp4", fourcc, 15, (W,H))

for t in range(len(pts)):
    canvas = np.zeros((H,W,3), np.uint8)
    for idxs, color in [(pose_idx,(255,255,255)), (lh_idx,(255,255,255)), (rh_idx,(255,255,255))]:
        xy = pts[t, idxs, :2]
        vis = pts[t, idxs, 3] > 0.0
        for (x,y),v in zip(xy, vis):
            if v:
                cv2.circle(canvas, (int(x*W), int(y*H)), 2, color, -1)
    cv2.putText(canvas, f"{row['label']}  t={t+1}/{len(pts)}",(20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    out.write(canvas)
out.release()
print("Wrote artifacts/landmarks/preview.mp4")
