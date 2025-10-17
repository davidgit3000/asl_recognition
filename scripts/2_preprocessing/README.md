# Preprocessing Scripts

Scripts for extracting and preprocessing features from videos/images.

## Workflow Order

1. **`extract_landmarks.py`** - Extract MediaPipe landmarks
   - Processes all videos/images in manifest
   - Extracts 543 landmarks (face, pose, hands)
   - Applies basic smoothing (EMA)
   - Output: `artifacts/landmarks/*.npy` [T, 543, 4]

2. **`preprocess_features.py`** - Preprocess features for training
   - Loads raw landmarks
   - Extracts relevant landmarks (pose + hands only, 75 total)
   - Normalizes by torso position and shoulder width
   - Applies Savitzky-Golay smoothing
   - Output: `artifacts/features/*.npy` [T, 75, 4]

## Output
- `artifacts/landmarks/` - Raw MediaPipe landmarks (543 points)
- `artifacts/features/` - Preprocessed features ready for training (75 points)

## Notes
- Features are normalized and smoothed, ready for dataloader
- Face landmarks (468 points) are excluded to reduce noise and size
