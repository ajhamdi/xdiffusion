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

class BratsDatasetModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        # Handle all three possible configs
        if train is not None:
            dataset_config = train
        elif validation is not None:
            dataset_config = validation
        elif test is not None:
            dataset_config = test
        else:
            dataset_config = {}

        # Just use fixed transforms since they're overwritten anyway
        image_transforms = [torchvision.transforms.Resize((256,256))]
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.image_transforms = torchvision.transforms.Compose(image_transforms)

        total_objects = len(next(os.walk(root_dir))[1])

        paths = []
        pattern = os.path.join(root_dir, '**', '*t2f.nii.gz')
        for s in glob.glob(pattern, recursive=True):
            paths.append(s)
        
        print(f"Found {len(paths)} t2f files in {root_dir}")
        
        if len(paths) == 0:
            raise ValueError(f"No t2f.nii.gz files found in {root_dir}")
        
        self.val_paths = paths[math.floor(total_objects / 100. * 90.):]  # used last 10% as validation
        self.train_paths = paths[:math.floor(total_objects / 100. * 80.)]  # used first 80% as training
        self.test_paths = paths  # For testing, use all available data

    def train_dataloader(self):
        dataset = BratsDataset(root_dir=self.train_paths, total_view=self.total_view,
                              image_transforms=self.image_transforms)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = BratsDataset(root_dir=self.val_paths, total_view=self.total_view,
                              image_transforms=self.image_transforms) 
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    # def test_dataloader(self):
    #     dataset = BratsDataset(root_dir=self.test_paths, total_view=self.total_view,
    #                           image_transforms=self.image_transforms)
    #     return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        dataset = BratsDataset(root_dir=self.test_paths, total_view=self.total_view,
                            image_transforms=self.image_transforms)
        # Use regular DataLoader instead of WebLoader
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, 
                     num_workers=self.num_workers, shuffle=False)
class BratsDataset(Dataset):
    """
    Dataloader for reading nifti files of 3D brain volumes and processing all slices.
    Image resolution : [240, 240, 155]
    
    Modified to process ALL slices from a volume instead of random sampling.
    """
    
    def __init__(self, root_dir='',
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=1,
        process_all_slices=True  # New parameter to control slice processing
        ) -> None:

        self.files = root_dir
        # Comment out to process all files, uncomment to test with one file
        self.files = self.files[:1]  # Process just first patient for testing
        
        self.tform = image_transforms

        if len(self.files) == 0:
            raise ValueError(f'Number of source images must be non-zero')

        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        self.default_trans = default_trans
        self.return_paths = return_paths
        self.total_view = total_view
        self.process_all_slices = process_all_slices

        self.imgs = []
        self.targets = []
        self.rotation_axis = []
        self.depth = [] 
        self.slice = []
        self.filenames = []
        self.input = [] 
            
        for s in self.files:
            patient_id = s.split('/')[-2]
            
            # Load image once
            img = nib.load(s).get_fdata(dtype=np.float32)
            
            # Normalize entire volume once
            img_norm = (img - np.min(img))/(np.max(img) - np.min(img) + 1e-5)
            
            # Use only first 155 slices for consistency
            MAX_DIM = 155
            dim0, dim1, dim2 = img.shape  # [240, 240, 155]
            
            if self.process_all_slices:
                # Process ALL slices along the Z-axis (axial view)
                axis_choice = '001'  # Use Z-axis for consistency
                
                for slice_idx in range(min(dim2, MAX_DIM)):  # Process all 155 slices
                    # Use current slice as both input and target for now
                    # You can modify this to use adjacent slices if needed
                    
                    # Get the slice
                    slice_data = img_norm[:,:,slice_idx]
                    
                    # Process as input image
                    img_slice = np.repeat(slice_data[..., np.newaxis], 3, axis=2)
                    img_slice = Image.fromarray(np.uint8(img_slice * 255.))
                    img_slice = self.tform(img_slice)
                    
                    # For target, you could use the same slice or adjacent slice
                    # Here using same slice for simplicity
                    target_slice = slice_data
                    target_slice = np.repeat(target_slice[..., np.newaxis], 3, axis=2)
                    target_slice = Image.fromarray(np.uint8(target_slice * 255.))
                    target_slice = self.tform(target_slice)
                    
                    self.imgs.append(img_slice)
                    self.targets.append(target_slice)
                    self.filenames.append(f"{patient_id}_slice_{slice_idx:03d}")
                    
                    # Store axis and depth information
                    self.rotation_axis.append(['001', '001'])  # Both using Z-axis
                    self.depth.append(2*(slice_idx/MAX_DIM)-1)  # Normalize to [-1, 1]
                    
            else:
                # Original random sampling code (for backward compatibility)
                axes = ['100', '010', '001']
                axis = np.random.choice(axes, 2)
                
                self.rotation_axis.append(axis)
                
                # Sample random indices
                idx = random.sample(range(MAX_DIM), self.total_view + 1)
                
                dat = []
                
                # Conditioning img
                for i in range(self.total_view):
                    if axis[0] == "100":
                        data = img_norm[idx[i],:,:] 
                    elif axis[0] == "010":
                        data = img_norm[:,idx[i],:]
                    elif axis[0] == "001":
                        data = img_norm[:,:,idx[i]]
                    dat.append(data)
                
                # Target img
                if axis[1] == "100":
                    target = img_norm[idx[-1],:,:] 
                elif axis[1] == "010": 
                    target = img_norm[:,idx[-1],:] 
                elif axis[1] == "001":
                    target = img_norm[:,:,idx[-1]]
                
                self.depth.append(2*(idx[-1]/MAX_DIM)-1)
                
                dat = np.stack(dat)
                
                # Multi-slice aggregation 
                if self.total_view > 1:
                    avg_reduction = []
                    for i in range(1, self.total_view):
                        avgdot = torch.einsum("ij,kj->ik", torch.from_numpy(dat[i]).float(), torch.from_numpy(dat[i-1]).float())
                        avg_reduction.append(avgdot)
                    
                    if len(avg_reduction) > 0:
                        input_mdot = torch.stack(avg_reduction)    
                        input_mdot = torch.mean(input_mdot, axis=0)
                        img_data = input_mdot.numpy()
                    else:
                        img_data = dat[0]
                else:
                    img_data = dat[0]
                
                # Process images
                img_data = np.repeat(img_data[..., np.newaxis], 3, axis=2)
                img_data = Image.fromarray(np.uint8(img_data * 255.))
                img_data = self.tform(img_data)
                
                target = np.repeat(target[..., np.newaxis], 3, axis=2)
                starget = Image.fromarray(np.uint8(target * 255.))
                starget = self.tform(starget)
                
                self.imgs.append(img_data)
                self.targets.append(starget)
                self.filenames.append(patient_id)
            
        print(f"Total slices to process: {len(self.filenames)}")
    
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
        
        axis_str = self.rotation_axis[idx][1]  # Get the axis string
        # Convert string to float tensor [1.0, 0.0, 0.0] format
        axis_tensor = [float(axis_str[0]), float(axis_str[1]), float(axis_str[2])]
        data["T"] = torch.tensor(axis_tensor + [self.depth[idx]])
        
        if self.postprocess is not None:
            data = self.postprocess(data)

        return data


if __name__ == '__main__':
    path_nifti = './ASNR-MICCAI-BraTS2023-GLI/'
    d2 = BratsDatasetModuleFromConfig(root_dir=path_nifti, batch_size=1, total_view=3, train=True)
    
    for batch in d2.train_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']