import sys
sys.path.append('.')
from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config 
from datasets import load_dataset
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import nibabel as nib
from typing import Callable, List, Optional
import glob 
import torch.nn as nn
from skimage.transform import resize
from math import exp
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from ldm.data.convnet import ConvNet

class BratsDatasetModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, use_median=False, axis=None, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        self.use_median = use_median
        self.axis = axis  # 'axial', 'sagittal', 'coronal', or None for random

        dataset_config = {}
        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation
        if test is not None:
            dataset_config = test

        # Handle both dict and OmegaConf config objects
        if 'image_transforms' in dataset_config:
            img_cfg = dataset_config['image_transforms'] if isinstance(dataset_config, dict) else dataset_config.image_transforms
            img_size = img_cfg['size'] if isinstance(img_cfg, dict) else img_cfg.size
            image_transforms = [torchvision.transforms.Resize((img_size, img_size))]
        else:
            image_transforms = [torchvision.transforms.Resize((256, 256))]
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.image_transforms = torchvision.transforms.Compose(image_transforms)

        total_objects = len(next(os.walk(root_dir))[1])

        paths = []
        for s in glob.glob(os.path.join(root_dir, '**/*t2f.nii.gz'), recursive=True):
            paths.append(s)
        
        self.val_paths = paths[math.floor(total_objects / 100. * 90.):] # used last 10% as validation
        self.train_paths = paths[:math.floor(total_objects / 100. * 80.)] # used first 80% as training

    def train_dataloader(self):
        dataset = BratsDataset(root_dir = self.train_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms, use_median=self.use_median, axis=self.axis)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = BratsDataset(root_dir = self.val_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms, use_median=self.use_median, axis=self.axis) 
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(BratsDataset(root_dir=self.test_paths, total_view=self.total_view,
                                          image_transforms=self.image_transforms, use_median=self.use_median, axis=self.axis),
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class BratsDataset(Dataset):
    """
    Dataloader for reading nifti files of 3D brain rotations in range [0,360].
    Image resolution : [240, 240, 155]

    Args:
        -files (List[str]): list of paths to source images
        -transform (Callable): transform to apply to both source and target images
        -preload (bool): load all data when initializing the dataset
        
    Output: 
        -img shape:  (256, 256, 3)
        -target slice index (float) [-1,1]
        -axis of rotation i.e ["001", "010", "100"]
    """
    
    # Mapping from anatomical names to axis codes
    AXIS_MAP = {
        'axial': '001',      # z-axis slices (transverse)
        'sagittal': '100',   # x-axis slices
        'coronal': '010',    # y-axis slices
    }
    
    def __init__(self, root_dir='',
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=1,
        use_median=False,
        axis=None
        ) -> None:


        self.files = root_dir
        self.tform = image_transforms

        if len(self.files) == 0:
            raise ValueError(f'Number of source images must be non-zero')

        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        self.default_trans = default_trans
        self.return_paths = return_paths
        self.total_view = total_view
        self.use_median = use_median
        self.axis = axis  # 'axial', 'sagittal', 'coronal', or None for random

        self.imgs = []
        self.targets = []
        self.rotation_axis = []
        self.depth = [] 
        self.slice = []
        self.filenames = []
        self.input = [] 
            
        for s in self.files:

            id = s.split('/')[-2]

            axes = ['100', '010', '001']
            
            # Use specified axis or random selection
            if self.axis is not None:
                # Use the specified axis for both conditioning and target
                axis_code = self.AXIS_MAP.get(self.axis, '001')  # default to axial
                axis = np.array([axis_code, axis_code])
            else:
                # Random axis selection (original behavior)
                axis = np.random.choice(axes, 2)
            
            self.rotation_axis.append(axis)
            
            img = nib.load(s).get_fdata(dtype=np.float32)
            
            #normalise img 
            sdata = (img-np.min(img))/(np.max(img)-np.min(img) + 1e-5)
            
            # Get dimension sizes for each axis
            dim_sizes = {'100': sdata.shape[0], '010': sdata.shape[1], '001': sdata.shape[2]}
                        
            dat = []
            
            # conditioning img - sample indices based on axis dimension
            cond_axis_size = dim_sizes[axis[0]]
            
            if self.use_median:
                # Use evenly-spaced slices that divide the volume into equal chunks
                # For total_view=1: middle slice (1/2)
                # For total_view=2: slices at 1/3 and 2/3
                # For total_view=3: slices at 1/4, 2/4, 3/4
                # General formula: slice at (i+1)/(total_view+1) for i in range(total_view)
                cond_indices = []
                for i in range(self.total_view):
                    position = (i + 1) / (self.total_view + 1)
                    slice_idx = int(position * cond_axis_size)
                    # Clamp to valid range
                    slice_idx = max(0, min(slice_idx, cond_axis_size - 1))
                    cond_indices.append(slice_idx)
            else:
                # Random sampling (original behavior)
                cond_indices = random.sample(range(cond_axis_size), min(self.total_view, cond_axis_size))
            
            for i in range(self.total_view):
                slice_idx = cond_indices[i] if i < len(cond_indices) else random.randint(0, cond_axis_size - 1)
                if axis[0] == "100":
                    data = sdata[slice_idx,:,:] 
                elif axis[0] == "010":
                    data = sdata[:,slice_idx,:]
                elif axis[0] == "001": 
                    data = sdata[:,:,slice_idx]
                dat.append(data)
        
            # target img - always random selection (not affected by --median)
            target_axis_size = dim_sizes[axis[1]]
            target_idx = random.randint(0, target_axis_size - 1)
            
            if axis[1] == "100":
                target = sdata[target_idx,:,:] 
            elif axis[1] == "010": 
                target = sdata[:,target_idx,:] 
            elif axis[1] == "001":
                target = sdata[:,:,target_idx]  
            
            self.depth.append(2*(target_idx/target_axis_size)-1) 
                
            dat = np.stack(dat)

            # multi-slice aggregation following X-Diffusion paper:
            # x = (1/(K-1)) * sum_{j=1}^{K-1} (x_j * x_{j+1})
            # Element-wise multiplication between consecutive slices, then averaged
            if self.total_view == 1:
                # Single view - just use the slice directly
                img = dat[0]
            else:
                # Compute element-wise product of consecutive slices
                pairwise_products = []
                for i in range(1, self.total_view):
                    # Element-wise multiplication (Hadamard product) between consecutive slices
                    product = dat[i] * dat[i-1]
                    pairwise_products.append(product)
                
                if len(pairwise_products) > 0:
                    # Average over all (K-1) pairwise products
                    img = np.mean(np.stack(pairwise_products, axis=0), axis=0)
                else:
                    img = dat[0]
    
            img = np.repeat(img[..., np.newaxis], 3, axis=2)
            img = Image.fromarray(np.uint8(img * 255.))
            img = self.tform(img)
        
            #normalise target 
            target = (target-np.min(target))/(np.max(target)-np.min(target) + 1e-5)
            target = np.repeat(target[..., np.newaxis], 3, axis=2)
            starget = Image.fromarray(np.uint8(target * 255.))
            starget = self.tform(starget)
            
            self.imgs.append(img)
            self.targets.append(starget)
            self.filenames.append(id)    
      
        print("total scans:",len(self.filenames))
        print("total number of files:",len(self.filenames))
    
         
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
     
        data = {}

        filename = self.filenames[idx]

        if self.return_paths:
            data["path"] = str(filename)

        data["image_target"] = self.targets[idx]
        data["image_cond"] = self.imgs[idx]
        data['filename'] = filename
        
        data["T"] = torch.tensor([int(self.rotation_axis[idx][1][0]), int(self.rotation_axis[idx][1][1]), int(self.rotation_axis[idx][1][2]), self.depth[idx]]) 
        if self.postprocess is not None:
            data = self.postprocess(data)

        return data


class BratsSingleVolumeDataset(Dataset):
    """
    Dataset for loading ALL slices from a single 3D volume sequentially.
    Used for --one_sample mode to generate complete volume inference.
    
    Args:
        nifti_path: Path to the nifti file
        image_transforms: Torchvision transforms to apply
        total_view: Number of conditioning slices for multi-slice conditioning
        use_median: Use evenly-spaced median slices instead of random sampling
        axis: Anatomical axis for slicing ('axial', 'sagittal', 'coronal')
    """
    
    AXIS_MAP = {
        'axial': '001',      # z-axis slices (transverse)
        'sagittal': '100',   # x-axis slices
        'coronal': '010',    # y-axis slices
    }
    
    def __init__(self, nifti_path, image_transforms, total_view=1, use_median=False, axis='axial'):
        self.nifti_path = nifti_path
        self.tform = image_transforms
        self.total_view = total_view
        self.use_median = use_median
        self.axis = axis
        
        # Get axis code
        self.axis_code = self.AXIS_MAP.get(axis, '001')
        
        # Load the 3D volume
        self.volume = nib.load(nifti_path).get_fdata(dtype=np.float32)
        
        # Normalize volume
        self.volume = (self.volume - np.min(self.volume)) / (np.max(self.volume) - np.min(self.volume) + 1e-5)
        
        # Get dimension sizes for each axis
        self.dim_sizes = {'100': self.volume.shape[0], '010': self.volume.shape[1], '001': self.volume.shape[2]}
        
        # Number of slices in the target axis
        self.num_slices = self.dim_sizes[self.axis_code]
        
        # Pre-compute conditioning slices (same for all target slices)
        self.cond_slices = self._compute_conditioning_slices()
        
        # Extract patient ID from path
        self.patient_id = nifti_path.split('/')[-2]
        
        print(f"Loaded volume: {self.patient_id}")
        print(f"Volume shape: {self.volume.shape}")
        print(f"Target axis: {axis} ({self.axis_code})")
        print(f"Number of slices: {self.num_slices}")
        print(f"Conditioning indices: {self.cond_indices}")
    
    def _compute_conditioning_slices(self):
        """Compute the conditioning slices based on median or random selection."""
        cond_axis_size = self.dim_sizes[self.axis_code]
        
        if self.use_median:
            # Use evenly-spaced slices that divide the volume into equal chunks
            cond_indices = []
            for i in range(self.total_view):
                position = (i + 1) / (self.total_view + 1)
                slice_idx = int(position * cond_axis_size)
                slice_idx = max(0, min(slice_idx, cond_axis_size - 1))
                cond_indices.append(slice_idx)
        else:
            # Random sampling
            cond_indices = random.sample(range(cond_axis_size), min(self.total_view, cond_axis_size))
        
        self.cond_indices = cond_indices
        
        # Extract conditioning slices
        cond_data = []
        for slice_idx in cond_indices:
            if self.axis_code == "100":
                data = self.volume[slice_idx, :, :]
            elif self.axis_code == "010":
                data = self.volume[:, slice_idx, :]
            elif self.axis_code == "001":
                data = self.volume[:, :, slice_idx]
            cond_data.append(data)
        
        cond_data = np.stack(cond_data)
        
        # Multi-slice aggregation following X-Diffusion paper
        if self.total_view == 1:
            aggregated = cond_data[0]
        else:
            pairwise_products = []
            for i in range(1, self.total_view):
                product = cond_data[i] * cond_data[i-1]
                pairwise_products.append(product)
            
            if len(pairwise_products) > 0:
                aggregated = np.mean(np.stack(pairwise_products, axis=0), axis=0)
            else:
                aggregated = cond_data[0]
        
        # Convert to 3-channel image
        aggregated = np.repeat(aggregated[..., np.newaxis], 3, axis=2)
        aggregated = Image.fromarray(np.uint8(aggregated * 255.))
        aggregated = self.tform(aggregated)
        
        return aggregated
    
    def __len__(self):
        return self.num_slices
    
    def __getitem__(self, idx):
        """Get a single slice as target with the pre-computed conditioning."""
        # Extract target slice
        if self.axis_code == "100":
            target = self.volume[idx, :, :]
        elif self.axis_code == "010":
            target = self.volume[:, idx, :]
        elif self.axis_code == "001":
            target = self.volume[:, :, idx]
        
        # Normalize and convert to 3-channel
        target = (target - np.min(target)) / (np.max(target) - np.min(target) + 1e-5)
        target = np.repeat(target[..., np.newaxis], 3, axis=2)
        target_img = Image.fromarray(np.uint8(target * 255.))
        target_tensor = self.tform(target_img)
        
        # Compute normalized depth
        depth = 2 * (idx / self.num_slices) - 1
        
        # T tensor: [axis_x, axis_y, axis_z, depth]
        T = torch.tensor([
            int(self.axis_code[0]), 
            int(self.axis_code[1]), 
            int(self.axis_code[2]), 
            depth
        ])
        
        return {
            'image_target': target_tensor,
            'image_cond': self.cond_slices,
            'filename': f"{self.patient_id}_slice_{idx:03d}",
            'T': T,
            'slice_idx': idx
        }


if __name__ == '__main__':
    
    path_nifti = './ASNR-MICCAI-BraTS2023-GLI/'
    d2 = BratsDatasetModuleFromConfig(root_dir = path_nifti, batch_size = 1, total_view = 3, train=True)
    
    for batch in d2.train_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']