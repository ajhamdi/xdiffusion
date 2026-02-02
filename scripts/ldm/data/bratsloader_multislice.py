import sys
sys.path.append('.')
from typing import Dict, List, Optional, Tuple
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
import pytorch_lightning as pl
import random
import os
import glob
import math
import nibabel as nib
from skimage.transform import resize
import torch.nn.functional as F


class BratsMultiSliceDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 4,
        num_cond_slices: int = 3,
        cond_strategy: str = 'key_slices',
        num_workers: int = 4,
        image_size: int = 256,
        train_val_split: float = 0.85,
        max_train_volumes: int = None,
        max_slices_per_volume: int = None,
        total_view: int = 1,
        train: Optional[Dict] = None,
        validation: Optional[Dict] = None,
        test: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_cond_slices = num_cond_slices
        self.cond_strategy = cond_strategy
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_val_split = train_val_split
        self.max_train_volumes = max_train_volumes
        self.max_slices_per_volume = max_slices_per_volume
        
        # Setup transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ])
        
        # Find all patient volumes
        self.all_paths = self._find_volumes(root_dir)
        
        if len(self.all_paths) == 0:
            raise ValueError(f"No t2f.nii.gz files found in {root_dir}")
        
        print(f"Found {len(self.all_paths)} patient volumes in {root_dir}")
        
        # Split into train/val/test
        n_total = len(self.all_paths)
        n_train = int(n_total * train_val_split)
        n_val = int(n_total * 0.1)
        
        random.seed(42)
        shuffled_paths = self.all_paths.copy()
        random.shuffle(shuffled_paths)
        
        self.train_paths = shuffled_paths[:n_train]
        self.val_paths = shuffled_paths[n_train:n_train + n_val]
        self.test_paths = shuffled_paths[n_train + n_val:]
        
        if max_train_volumes is not None and len(self.train_paths) > max_train_volumes:
            self.train_paths = self.train_paths[:max_train_volumes]
        
        if len(self.test_paths) == 0:
            self.test_paths = self.val_paths
        
        print(f"Train: {len(self.train_paths)}, Val: {len(self.val_paths)}, Test: {len(self.test_paths)}")
    
    def _find_volumes(self, root_dir: str) -> List[str]:
        """Find all T2-FLAIR volumes in the dataset."""
        paths = []
        pattern = os.path.join(root_dir, '**', '*t2f.nii.gz')
        for path in glob.glob(pattern, recursive=True):
            paths.append(path)
        return sorted(paths)
    
    def train_dataloader(self):
        dataset = BratsMultiSliceDataset(
            file_paths=self.train_paths,
            num_cond_slices=self.num_cond_slices,
            cond_strategy=self.cond_strategy,
            image_transforms=self.image_transforms,
            is_training=True,
            max_slices_per_volume=self.max_slices_per_volume
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        dataset = BratsMultiSliceDataset(
            file_paths=self.val_paths,
            num_cond_slices=self.num_cond_slices,
            cond_strategy=self.cond_strategy,
            image_transforms=self.image_transforms,
            is_training=False
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        dataset = BratsMultiSliceDataset(
            file_paths=self.test_paths,
            num_cond_slices=self.num_cond_slices,
            cond_strategy=self.cond_strategy,
            image_transforms=self.image_transforms,
            is_training=False,
            return_all_slices=True
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )


class BratsMultiSliceDataset(Dataset):
    
    MAX_DEPTH = 155
    
    def __init__(
        self,
        file_paths: List[str],
        num_cond_slices: int = 3,
        cond_strategy: str = 'key_slices',
        image_transforms=None,
        is_training: bool = True,
        return_all_slices: bool = False,
        augment: bool = True,
        max_slices_per_volume: int = None
    ):
        self.file_paths = file_paths
        self.num_cond_slices = num_cond_slices
        self.cond_strategy = cond_strategy
        self.image_transforms = image_transforms
        self.is_training = is_training
        self.return_all_slices = return_all_slices
        self.augment = augment and is_training
        self.max_slices_per_volume = max_slices_per_volume
        
        # Precompute all samples
        self.samples = []
        self._prepare_samples()
        
        print(f"BratsMultiSliceDataset: {len(self.samples)} samples")
        print(f"Num cond slices: {num_cond_slices}")
    
    def _prepare_samples(self):
        """Prepare all (volume, conditioning_slices, target_slice) combinations."""
        
        for vol_idx, path in enumerate(self.file_paths):
            patient_id = path.split('/')[-2]
            
            try:
                img = nib.load(path)
                shape = img.shape
                n_slices = min(shape[2], self.MAX_DEPTH)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            
            # Get conditioning slice indices
            cond_indices = self._get_conditioning_indices(n_slices)
            
            if self.return_all_slices:
                # Testing: generate all slices
                for target_idx in range(n_slices):
                    self.samples.append({
                        'vol_path': path,
                        'vol_idx': vol_idx,
                        'patient_id': patient_id,
                        'cond_indices': cond_indices,
                        'target_idx': target_idx,
                        'n_slices': n_slices
                    })
            else:
                # Training/validation
                if self.max_slices_per_volume is not None:
                    n_samples = min(self.max_slices_per_volume, n_slices)
                    center = n_slices // 2
                    half_range = n_samples // 2
                    start_idx = max(0, center - half_range)
                    end_idx = min(n_slices, start_idx + n_samples)
                    target_indices = list(range(start_idx, end_idx))
                else:
                    n_samples_per_vol = n_slices if self.is_training else n_slices // 3
                    target_indices = list(range(0, n_slices, max(1, n_slices // n_samples_per_vol)))
                
                for target_idx in target_indices:
                    self.samples.append({
                        'vol_path': path,
                        'vol_idx': vol_idx,
                        'patient_id': patient_id,
                        'cond_indices': cond_indices,
                        'target_idx': target_idx,
                        'n_slices': n_slices
                    })
    
    def _get_conditioning_indices(self, n_slices: int) -> List[int]:
        """Get indices of conditioning slices based on strategy."""
        
        if self.cond_strategy == 'uniform':
            indices = np.linspace(0, n_slices - 1, self.num_cond_slices + 2)[1:-1]
            return [int(i) for i in indices]
        
        elif self.cond_strategy == 'key_slices':
            if self.num_cond_slices == 1:
                return [n_slices // 2]
            elif self.num_cond_slices == 3:
                return [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
            elif self.num_cond_slices == 5:
                return [n_slices // 6, n_slices // 3, n_slices // 2, 
                        2 * n_slices // 3, 5 * n_slices // 6]
            else:
                return [int(i) for i in np.linspace(0, n_slices - 1, self.num_cond_slices + 2)[1:-1]]
        
        elif self.cond_strategy == 'adaptive':
            return [int(i) for i in np.linspace(0, n_slices - 1, self.num_cond_slices + 2)[1:-1]]
        
        else:
            raise ValueError(f"Unknown cond_strategy: {self.cond_strategy}")
    
    def _load_and_normalize_volume(self, path: str) -> np.ndarray:
        """Load and normalize a NIfTI volume."""
        img = nib.load(path).get_fdata(dtype=np.float32)
        
        # Robust normalization
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        
        # Normalize to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        return img
    
    def _slice_to_tensor(self, slice_data: np.ndarray) -> torch.Tensor:
        """Convert a 2D slice to a transformed tensor."""
        slice_rgb = np.repeat(slice_data[..., np.newaxis], 3, axis=2)
        slice_img = Image.fromarray(np.uint8(slice_rgb * 255.))
        
        if self.image_transforms is not None:
            return self.image_transforms(slice_img)
        else:
            return torch.from_numpy(slice_rgb).float() * 2. - 1.
    
    def _create_optimized_conditioning(
        self, 
        volume: np.ndarray, 
        cond_indices: List[int],
        target_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
     
        n_slices = min(volume.shape[2], self.MAX_DEPTH)
        
        # Normalize depths to [-1, 1]
        cond_depths = [2 * (idx / self.MAX_DEPTH) - 1 for idx in cond_indices]
        target_depth_norm = 2 * (target_idx / self.MAX_DEPTH) - 1
        
        middle_idx = len(cond_indices) // 2
        primary_slice_idx = min(cond_indices[middle_idx], n_slices - 1)
        primary_slice = volume[:, :, primary_slice_idx]
        cond_image = self._slice_to_tensor(primary_slice)
        
        # Compute relative distances from target to each conditioning slice
        rel_distances = [(target_idx - idx) / self.MAX_DEPTH for idx in cond_indices]
        
        # Distance-weighted depth features for interpolation
        inv_distances = [1.0 / (abs(d) + 0.01) for d in rel_distances]  # Add eps to avoid div by 0
        total_inv_dist = sum(inv_distances)
        weights = [w / total_inv_dist for w in inv_distances]
        
        # Weighted depth encoding 
        weighted_depth = sum(w * d for w, d in zip(weights, cond_depths))
        
        # Create depth feature vector
        depth_features = torch.tensor([
            target_depth_norm,  # Target slice position
            weighted_depth,     # Weighted conditioning depth
            *rel_distances      # Relative distances to each conditioning slice
        ], dtype=torch.float32)
        
        return cond_image, depth_features
    
    def _apply_augmentation(
        self, 
        cond_image: torch.Tensor, 
        target_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent augmentation to both conditioning and target images."""
        
        if not self.augment:
            return cond_image, target_image
        
        # Random horizontal flip
        if random.random() > 0.5:
            cond_image = torch.flip(cond_image, dims=[1])
            target_image = torch.flip(target_image, dims=[1])
        
        # Intensity scaling
        scale = 0.9 + 0.2 * random.random()
        cond_image = cond_image * scale
        target_image = target_image * scale
        
        # Clamp
        cond_image = torch.clamp(cond_image, -1, 1)
        target_image = torch.clamp(target_image, -1, 1)
        
        return cond_image, target_image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load volume
        volume = self._load_and_normalize_volume(sample['vol_path'])
        n_slices = min(volume.shape[2], self.MAX_DEPTH)
        
        # Get conditioning indices 
        if self.is_training and random.random() > 0.7:
            cond_indices = [
                min(max(0, idx + random.randint(-2, 2)), n_slices - 1)
                for idx in sample['cond_indices']
            ]
        else:
            cond_indices = sample['cond_indices']
        
        target_idx = sample['target_idx']
        
        # Create optimized conditioning
        cond_image, depth_features = self._create_optimized_conditioning(
            volume, cond_indices, target_idx
        )
        
        # Get target slice
        target_slice = volume[:, :, target_idx]
        target_image = self._slice_to_tensor(target_slice)
        
        # Apply augmentation
        cond_image, target_image = self._apply_augmentation(cond_image, target_image)
        
        # Create T positional encoding
        target_depth_norm = 2 * (target_idx / self.MAX_DEPTH) - 1
        
        T = torch.tensor([
            0.0,  # axis_x (axial scan)
            0.0,  # axis_y
            1.0,  # axis_z (depth axis)
            target_depth_norm  # normalized target depth
        ], dtype=torch.float32)
        
        
        data = {
            'image_target': target_image,
            'image_cond': cond_image,
            'T': T,
            'depth_features': depth_features,  # Extra field for enhanced models
            'filename': f"{sample['patient_id']}_slice_{target_idx:03d}",
            'cond_indices': torch.tensor(cond_indices),
            'target_idx': target_idx,
            'n_slices': n_slices
        }
        
        return data


if __name__ == '__main__':
    # Test the dataloader
    path_nifti = './ASNR-MICCAI-BraTS2023-GLI/'
    
    if os.path.exists(path_nifti):
        print("=" * 60)
        
        dm = BratsMultiSliceDataModule(
            root_dir=path_nifti,
            batch_size=2,
            num_cond_slices=3,
            cond_strategy='key_slices',
            num_workers=0
        )
        
        print("\nTesting train dataloader:")
        for batch in dm.train_dataloader():
            print(f"  image_target shape: {batch['image_target'].shape}")
            print(f"  image_cond shape: {batch['image_cond'].shape}")
            print(f"  T shape: {batch['T'].shape}")
            print(f"  depth_features shape: {batch['depth_features'].shape}")
            print(f"  cond_indices: {batch['cond_indices']}")
            print(f"  target_idx: {batch['target_idx']}")
            print(f"  Sample filenames: {batch['filename']}")
            
        
        print("\n" + "=" * 60)
    else:
        print(f"Test path {path_nifti} not found.")