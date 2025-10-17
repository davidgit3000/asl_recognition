import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

class ASLDataset(Dataset):
    """
    PyTorch Dataset for ASL recognition with windowed sequences.
    
    Loads preprocessed features from artifacts/features/ and creates
    fixed-length windows for training.
    """
    
    def __init__(
        self,
        manifest_path: str,
        features_dir: str,
        window_size: int = 32,
        stride: int = 16,
        split: Optional[str] = None,
        source_filter: Optional[List[str]] = None,
        augment: bool = False
    ):
        """
        Args:
            manifest_path: Path to manifest CSV
            features_dir: Directory containing preprocessed .npy files
            window_size: Number of frames per window
            stride: Stride for sliding window (use window_size for non-overlapping)
            split: Filter by split ('train', 'val', 'test', None for all)
            source_filter: Filter by source (e.g., ['kaggle', 'msasl'])
            augment: Apply data augmentation
        """
        self.manifest_path = Path(manifest_path)
        self.features_dir = Path(features_dir)
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        
        # Load manifest
        self.df = pd.read_csv(manifest_path)
        
        # Apply filters
        if split is not None:
            self.df = self.df[self.df['split'] == split]
        if source_filter is not None:
            self.df = self.df[self.df['source'].isin(source_filter)]
        
        # Build label mapping
        self.labels = sorted(self.df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.labels)
        
        # Build windows
        self.windows = self._build_windows()
        
        print(f"ASLDataset initialized:")
        print(f"  Split: {split if split else 'all'}")
        print(f"  Samples: {len(self.df)}")
        print(f"  Windows: {len(self.windows)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Window size: {self.window_size}, Stride: {self.stride}")
    
    def _build_windows(self) -> List[Tuple[str, int, int, int]]:
        """
        Build list of (sample_id, start_frame, end_frame, label_idx) tuples.
        Each tuple represents one training window.
        """
        windows = []
        
        for _, row in self.df.iterrows():
            feature_path = self.features_dir / f"{row['id']}.npy"
            
            if not feature_path.exists():
                continue
            
            # Load to get sequence length
            features = np.load(feature_path)  # [T, 75, 4]
            T = len(features)
            
            label_idx = self.label_to_idx[row['label']]
            
            # Create sliding windows
            if T < self.window_size:
                # If sequence is too short, pad it (will handle in __getitem__)
                windows.append((row['id'], 0, T, label_idx))
            else:
                # Sliding window
                for start in range(0, T - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    windows.append((row['id'], start, end, label_idx))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            features: [window_size, 75, 4] tensor
            label: integer class label
        """
        sample_id, start, end, label_idx = self.windows[idx]
        
        # Load features
        feature_path = self.features_dir / f"{sample_id}.npy"
        features = np.load(feature_path)  # [T, 75, 4]
        
        # Extract window
        window = features[start:end]  # [window_size or less, 75, 4]
        
        # Pad if necessary
        if len(window) < self.window_size:
            pad_length = self.window_size - len(window)
            padding = np.zeros((pad_length, 75, 4), dtype=np.float32)
            window = np.concatenate([window, padding], axis=0)
        
        # Apply augmentation if enabled
        if self.augment:
            window = self._augment(window)
        
        # Convert to tensor
        window = torch.from_numpy(window).float()  # [window_size, 75, 4]
        
        return window, label_idx
    
    def _augment(self, window: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a window.
        Args:
            window: [T, 75, 4] array
        Returns:
            augmented: [T, 75, 4] array
        """
        # Random rotation around z-axis (yaw)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            angle_rad = np.deg2rad(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            window[:, :, :3] = window[:, :, :3] @ R.T
        
        # Random scale
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            window[:, :, :3] *= scale
        
        # Random translation
        if np.random.rand() < 0.5:
            translation = np.random.uniform(-0.1, 0.1, size=3)
            window[:, :, :3] += translation
        
        # Random temporal shift (circular shift)
        if np.random.rand() < 0.3:
            shift = np.random.randint(-5, 5)
            window = np.roll(window, shift, axis=0)
        
        return window
    
    def get_label_name(self, idx: int) -> str:
        """Convert label index to label name."""
        return self.idx_to_label[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.
        Returns inverse frequency weights.
        """
        label_counts = np.zeros(self.num_classes)
        for _, _, _, label_idx in self.windows:
            label_counts[label_idx] += 1
        
        # Inverse frequency
        weights = 1.0 / (label_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        
        return torch.from_numpy(weights).float()


def create_dataloaders(
    config_path: str = "configs/config.yaml",
    window_size: int = 32,
    stride_train: int = 16,
    stride_val: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders from config.
    
    Args:
        config_path: Path to config.yaml
        window_size: Number of frames per window
        stride_train: Stride for training (smaller = more overlap = more data)
        stride_val: Stride for val/test (typically = window_size for no overlap)
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        augment_train: Apply augmentation to training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    manifest_path = cfg['manifest_out']
    features_dir = Path(cfg['artifacts_root']) / 'features'
    
    # Create datasets
    train_dataset = ASLDataset(
        manifest_path=manifest_path,
        features_dir=features_dir,
        window_size=window_size,
        stride=stride_train,
        split='train',
        augment=augment_train
    )
    
    val_dataset = ASLDataset(
        manifest_path=manifest_path,
        features_dir=features_dir,
        window_size=window_size,
        stride=stride_val,
        split='val',
        augment=False
    )
    
    test_dataset = ASLDataset(
        manifest_path=manifest_path,
        features_dir=features_dir,
        window_size=window_size,
        stride=stride_val,
        split='test',
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    # Test the dataloader
    print("Testing ASL Dataloader...")
    
    # Create datasets
    train_loader, val_loader, test_loader = create_dataloaders(
        window_size=32,
        stride_train=16,
        batch_size=8,
        num_workers=0  # Use 0 for debugging
    )
    
    # Test loading a batch
    print("\nLoading first batch from train_loader...")
    for features, labels in train_loader:
        print(f"Features shape: {features.shape}")  # [batch, window_size, 75, 4]
        print(f"Labels shape: {labels.shape}")      # [batch]
        print(f"Labels: {labels}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        break
    
    print("\nDataloader test complete!")