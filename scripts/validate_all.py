"""
Validation Script for 3D Brain Volume Reconstruction
=====================================================

SINGLE-SLICE-TO-FULL-VOLUME reconstruction:
- Input: one conditioning slice (middle slice ~77)
- T_cond positional embedding indicates which slice to generate
- Model generates all 155 slices from that single conditioning slice

Usage:
    python validate_all.py --ckpt /path/to/checkpoint.ckpt --config /path/to/config.yaml --data_path /path/to/ASNR-MICCAI-BraTS2023-GLI/ --output_dir ./validation_results
"""

from contextlib import nullcontext
import numpy as np
import gc
import torch
from einops import rearrange
from ldmv0.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from ldmv0.util import instantiate_from_config
import os 
import nibabel as nib
from ldmv0.data.bratsloader import BratsDatasetModuleFromConfig
from tqdm import tqdm
import argparse
import ldmv0.modules.diffusionmodules.openaimodel as openai_module
from scipy.ndimage import gaussian_filter1d, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage import uniform_filter
import json
from datetime import datetime
import pandas as pd


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred, target, data_range=1.0):
    """Compute Peak Signal-to-Noise Ratio"""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(pred, target, win_size=7):
    """Compute Structural Similarity Index"""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    
    mu_pred = uniform_filter(pred, size=win_size)
    mu_target = uniform_filter(target, size=win_size)
    
    sigma_pred_sq = uniform_filter(pred ** 2, size=win_size) - mu_pred ** 2
    sigma_target_sq = uniform_filter(target ** 2, size=win_size) - mu_target ** 2
    sigma_pred_target = uniform_filter(pred * target, size=win_size) - mu_pred * mu_target
    
    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return np.mean(ssim_map)


def compute_all_metrics(pred, target):

    # Normalize to [0, 1]
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    # Overall metrics
    mse = np.mean((pred_norm - target_norm) ** 2)
    psnr = compute_psnr(pred_norm, target_norm)
    
    # Per-slice SSIM (skip empty slices)
    ssim_values = []
    psnr_values = []
    for i in range(pred.shape[2]):
        if target_norm[:, :, i].max() > 0.01:
            ssim_values.append(compute_ssim(pred_norm[:, :, i], target_norm[:, :, i]))
            psnr_values.append(compute_psnr(pred_norm[:, :, i], target_norm[:, :, i]))
    
    # Correlation
    corr = np.corrcoef(pred_norm.flatten(), target_norm.flatten())[0, 1]
    
    # MAE
    mae = np.mean(np.abs(pred_norm - target_norm))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'psnr': float(psnr),
        'psnr_per_slice_mean': float(np.mean(psnr_values)) if psnr_values else 0,
        'psnr_per_slice_std': float(np.std(psnr_values)) if psnr_values else 0,
        'ssim': float(np.mean(ssim_values)) if ssim_values else 0,
        'ssim_std': float(np.std(ssim_values)) if ssim_values else 0,
        'correlation': float(corr) if not np.isnan(corr) else 0
    }


# =============================================================================
# POST-PROCESSING 
# =============================================================================

def create_brain_mask(volume, threshold=0.05):
    """Create brain mask from the volume itself"""
    mask = volume > threshold
    for i in range(volume.shape[2]):
        if mask[:, :, i].sum() > 100:
            mask[:, :, i] = binary_fill_holes(mask[:, :, i])
            mask[:, :, i] = binary_erosion(mask[:, :, i], iterations=1)
            mask[:, :, i] = binary_dilation(mask[:, :, i], iterations=2)
    return mask.astype(np.float32)


def detect_content_bounds(volume, reference_slice_idx=None, threshold=0.02, min_content_pixels=500):
    """
    Detect first and last slices with meaningful content using the reconstruction itself.
    Filters out inverted/artifact slices that have high background values.
    
    Args:
        volume: Input volume (H, W, D)
        reference_slice_idx: Index of conditioning slice (used to determine normal appearance)
        threshold: Intensity threshold for content detection
        min_content_pixels: Minimum number of pixels above threshold to consider slice as having content
    
    Returns:
        first_content: Index of first slice with content
        last_content: Index of last slice with content
    """
    n_slices = volume.shape[2]
    
    if reference_slice_idx is None:
        reference_slice_idx = n_slices // 2
    
    # Get reference slice statistics 
    ref_slice = volume[:, :, reference_slice_idx]
    ref_corner = np.mean([
        ref_slice[0:25, 0:25].mean(), ref_slice[0:25, -25:].mean(),
        ref_slice[-25:, 0:25].mean(), ref_slice[-25:, -25:].mean()
    ])
    
    valid_slices = []
    
    for i in range(n_slices):
        slice_data = volume[:, :, i]
        slice_mean = slice_data.mean()
        
        # Check corner intensity (background regions)
        corner = np.mean([
            slice_data[0:25, 0:25].mean(), slice_data[0:25, -25:].mean(),
            slice_data[-25:, 0:25].mean(), slice_data[-25:, -25:].mean()
        ])
        
        # Only exclude inverted slices:
        # 1. Very high corner values (bright background) 
        is_inverted_corners = corner > 0.35 and ref_corner < 0.1
        
        # 2. Very high overall mean 
        is_definitely_inverted = slice_mean > 0.5
        
        # 3. Uniform mid-gray artifact
        is_uniform_artifact = np.std(slice_data) < 0.06 and 0.3 < slice_mean < 0.7
        
        if is_inverted_corners or is_definitely_inverted or is_uniform_artifact:
            continue  # Skip this slice, artifact
        
        content_pixels = np.sum(slice_data > threshold)
        
        if content_pixels > min_content_pixels and slice_mean > threshold:
            valid_slices.append(i)
    
    if len(valid_slices) == 0:
        # Fallback: return middle region
        return n_slices // 4, 3 * n_slices // 4
    
    return min(valid_slices), max(valid_slices)


def fix_inverted_slices(volume, reference_slice_idx):
    """
    Fix inverted/artifact slices using the conditioning slice as reference.
        
    Args:
        volume: Input volume (H, W, D)
        reference_slice_idx: Index of the conditioning slice (used as reference for normal appearance)
    
    Returns:
        Corrected volume
    """
    result = volume.copy()
    n_slices = volume.shape[2]
    
    # Use conditioning slice as reference for what "normal" looks like
    ref_slice = volume[:, :, reference_slice_idx]
    ref_corner = np.mean([
        ref_slice[0:25, 0:25].mean(), ref_slice[0:25, -25:].mean(),
        ref_slice[-25:, 0:25].mean(), ref_slice[-25:, -25:].mean()
    ])
    ref_mean = ref_slice.mean()
    
    # Detect content bounds from reconstruction (excluding artifacts)
    first_content, last_content = detect_content_bounds(volume, reference_slice_idx=reference_slice_idx)
    
    zeroed_slices = []
    
    for i in range(n_slices):
        slice_data = volume[:, :, i]
        slice_mean = slice_data.mean()
        
        # Corner intensity (background regions)
        corner = np.mean([
            slice_data[0:25, 0:25].mean(), slice_data[0:25, -25:].mean(),
            slice_data[-25:, 0:25].mean(), slice_data[-25:, -25:].mean()
        ])
        
        # Distance from conditioning slice (normalized 0-1)
        dist_from_ref = abs(i - reference_slice_idx) / n_slices
        
        # Expected intensity based on distance from reference
        # Intensity typically decreases away from center of brain
        expected_mean = ref_mean * max(0.2, 1.0 - dist_from_ref)
        
        # Check if outside expected content bounds
        is_outside_bounds = i < first_content or i > last_content
        
        # Detection criteria
        is_inverted_corners = corner > 0.35 and ref_corner < 0.1
        is_definitely_inverted = slice_mean > 0.5
        is_moderately_bright = slice_mean > 0.3 and slice_mean <= 0.5 and dist_from_ref > 0.25
        is_uniform_artifact = np.std(slice_data) < 0.06 and 0.3 < slice_mean < 0.7
        
        if is_outside_bounds:
            result[:, :, i] = 0
            zeroed_slices.append(i)
        elif is_definitely_inverted or is_inverted_corners:
            # Invert and scale
            inverted = 1.0 - slice_data
            inverted_mean = inverted.mean()
            
            if inverted_mean > 0.5:
                # Too bright after inversion so zero it
                result[:, :, i] = 0
                zeroed_slices.append(i)
            elif inverted_mean > expected_mean * 2 and inverted_mean > 0.15:
                # Scale down to expected intensity
                scale = expected_mean / (inverted_mean + 1e-8)
                scale = np.clip(scale, 0.1, 1.0)
                result[:, :, i] = inverted * scale
            else:
                result[:, :, i] = inverted
        elif is_moderately_bright:
            # Slice is brighter than expected for its position 
            if slice_mean > expected_mean * 2.5:
                # Too bright so zero it
                result[:, :, i] = 0
                zeroed_slices.append(i)
            else:
                # Scale down to expected intensity  
                scale = expected_mean / (slice_mean + 1e-8)
                scale = np.clip(scale, 0.2, 1.0)
                result[:, :, i] = slice_data * scale
        elif is_uniform_artifact:
            result[:, :, i] = 0
            zeroed_slices.append(i)
    
    # Interpolate zeroed slices that are between valid slices 
    result = interpolate_zeroed_slices(result, zeroed_slices, first_content, last_content)
    
    return result


def interpolate_zeroed_slices(volume, zeroed_slices, first_content, last_content):
    """
    Interpolate zeroed slices that fall between valid content slices.
    This helps recover slices that were incorrectly zeroed.
    """
    result = volume.copy()
    n_slices = volume.shape[2]
    
    for i in zeroed_slices:
        # Skip edge slices
        if i <= first_content or i >= last_content:
            continue
        
        # Find nearest non-zero slices before and after
        prev_idx = None
        next_idx = None
        
        for j in range(i - 1, first_content - 1, -1):
            if j not in zeroed_slices and volume[:, :, j].mean() > 0.01:
                prev_idx = j
                break
        
        for j in range(i + 1, last_content + 1):
            if j not in zeroed_slices and volume[:, :, j].mean() > 0.01:
                next_idx = j
                break
        
        if prev_idx is not None and next_idx is not None:
            gap = next_idx - prev_idx
            if gap <= 10:  # Only interpolate small gaps
                weight = (i - prev_idx) / gap
                interpolated = (1 - weight) * result[:, :, prev_idx] + weight * result[:, :, next_idx]
                result[:, :, i] = interpolated
    
    return result


def correct_intensity_profile(volume, reference_slice_idx):
    """
    Correct intensity profile across the volume based on expected brain intensity distribution.
    Brain MRI typically has highest intensity near the center and decreases toward edges.
    """
    result = volume.copy()
    n_slices = volume.shape[2]
    
    ref_slice = volume[:, :, reference_slice_idx]
    ref_mean = ref_slice[ref_slice > 0.02].mean() if (ref_slice > 0.02).sum() > 100 else ref_slice.mean()
    
    for i in range(n_slices):
        slice_data = result[:, :, i]
        mask = slice_data > 0.02
        
        if mask.sum() < 100:
            continue
        
        slice_mean = slice_data[mask].mean()
        
        # Distance from reference
        dist_from_ref = abs(i - reference_slice_idx) / n_slices
        
        # Expected intensity 
        expected_ratio = np.exp(-2.5 * dist_from_ref ** 2)
        expected_mean = ref_mean * max(0.25, expected_ratio)
        
        # Correct if slice is brighter than expected 
        if slice_mean > expected_mean * 1.3 and slice_mean > 0.08:
            scale = expected_mean / (slice_mean + 1e-8)
            scale = np.clip(scale, 0.25, 1.0)
            result[:, :, i] = slice_data * scale
    
    return result


def remove_edge_artifacts(volume, reference_slice_idx=None, margin_slices=2):
    """
    Remove edge slice artifacts based on reconstruction content.
    
    Args:
        volume: Input volume (H, W, D)
        reference_slice_idx: Index of conditioning slice for detecting artifacts
        margin_slices: Number of margin slices to keep beyond detected content
    
    Returns:
        Volume with edge artifacts removed
    """
    result = volume.copy()
    
    first_content, last_content = detect_content_bounds(volume, reference_slice_idx=reference_slice_idx)
    
    # Add small margin for safety
    first_valid = max(0, first_content - margin_slices)
    last_valid = min(volume.shape[2] - 1, last_content + margin_slices)
    
    # Zero out slices outside content region
    for i in range(first_valid):
        result[:, :, i] = 0
    for i in range(last_valid + 1, volume.shape[2]):
        result[:, :, i] = 0
    
    return result


def smooth_intensity_profile(volume, sigma=2.0):
    """
    Smooth intensity variations across slices for consistency.
    
    Args:
        volume: Input volume (H, W, D)
        sigma: Gaussian smoothing sigma for intensity profile
    
    Returns:
        Volume with smoothed intensity profile
    """
    result = volume.copy()
    
    # Compute per-slice mean intensity (only for non-empty regions)
    profile = []
    for i in range(volume.shape[2]):
        slice_data = volume[:, :, i]
        mask = slice_data > 0.05
        if mask.sum() > 100:
            profile.append(slice_data[mask].mean())
        else:
            profile.append(0)
    
    profile = np.array(profile)
    
    # Find valid non-zero region
    valid_mask = profile > 0.01
    if valid_mask.sum() < 3:
        return result
    
    # Smooth the profile
    smoothed_profile = profile.copy()
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        # Interpolate and smooth only valid region
        valid_values = profile[valid_mask]
        smoothed_valid = gaussian_filter1d(valid_values, sigma=sigma)
        
        # Apply correction
        for idx, orig_idx in enumerate(valid_indices):
            if profile[orig_idx] > 0.01:
                scale = smoothed_valid[idx] / (profile[orig_idx] + 1e-8)
                scale = np.clip(scale, 0.5, 2.0)  # Limit correction magnitude
                result[:, :, orig_idx] *= scale
    
    return result


def post_process_volume(recon, cond_slice_idx=None, smooth_sigma=0.8, 
                        intensity_smooth_sigma=2.0, mask_threshold=0.03):
    """
    Complete post-processing pipeline without ground truth.
    
    Args:
        recon: Reconstructed volume (H, W, D)
        cond_slice_idx: Index of conditioning slice (used as reference)
        smooth_sigma: Sigma for z-axis smoothing
        intensity_smooth_sigma: Sigma for intensity profile smoothing
        mask_threshold: Threshold for brain mask creation
    
    Returns:
        Post-processed volume
    """
    if cond_slice_idx is None:
        cond_slice_idx = recon.shape[2] // 2
    
    result = recon.copy().astype(np.float32)
    
    # 1. Fix inverted slices (using conditioning slice as reference)
    result = fix_inverted_slices(result, cond_slice_idx)
    
    # 2. Remove edge artifacts (based on reconstruction content)
    result = remove_edge_artifacts(result, reference_slice_idx=cond_slice_idx)
    
    # 3. Correct intensity profile based on expected brain distribution
    result = correct_intensity_profile(result, cond_slice_idx)
    
    # 4. Create brain mask from reconstruction itself
    mask = create_brain_mask(result, threshold=mask_threshold)
    
    # 5. Enforce black background
    result[mask == 0] = 0
    
    # 6. Smooth intensity profile for consistency (optional)
    if intensity_smooth_sigma > 0:
        result = smooth_intensity_profile(result, sigma=intensity_smooth_sigma)
    
    # 7. Z-smoothing for slice consistency
    if smooth_sigma > 0:
        result = gaussian_filter1d(result, sigma=smooth_sigma, axis=2)
    
    # 8. Final cleanup 
    result = result * mask
    result = np.clip(result, 0, 1)
    
    return result


# =============================================================================
# MODEL
# =============================================================================

def load_model_from_config(config, ckpt, device):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    
    pl_sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = pl_sd["state_dict"]
    del pl_sd
    gc.collect()
    
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    
    model.eval()
    for module in model.children():
        module.to(device)
        torch.cuda.empty_cache()
    
    return model


# Store original timestep_embedding function at module level
_original_timestep_embedding = None
_te_patched = False

@torch.no_grad()
def sample_slice(input_im, T_cond, model, sampler, h, w, ddim_steps, scale, ddim_eta, device):
    """Generate single slice"""
    global _original_timestep_embedding, _te_patched
    
    model = model.to(device)
    sampler.model = model
    
    # Patch timestep_embedding only once
    if not _te_patched:
        _original_timestep_embedding = openai_module.timestep_embedding
        def te_fixed(t, d, max_period=10000, repeat_only=False):
            return _original_timestep_embedding(t, d, max_period, repeat_only).to(device)
        openai_module.timestep_embedding = te_fixed
        _te_patched = True
    
    precision_scope = autocast if torch.cuda.is_available() else nullcontext
    
    with precision_scope("cuda"):
        with model.ema_scope():
            x = input_im
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w').contiguous().float().to(device)
            
            c = model.get_learned_conditioning(x)
            T = T_cond.to(device).contiguous().float()[0][None, None, :].repeat(1, 1, 1)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            
            cond = {
                'c_crossattn': [c],
                'c_concat': [model.encode_first_stage(x).mode().detach()]
            }
            
            if scale != 1.0:
                uc = {
                    'c_concat': [torch.zeros(1, 4, h // 8, w // 8, device=device)],
                    'c_crossattn': [torch.zeros_like(c)]
                }
            else:
                uc = None
            
            samples, _ = sampler.sample(
                S=ddim_steps, conditioning=cond, batch_size=1,
                shape=[4, h // 8, w // 8], verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc, eta=ddim_eta
            )
            
            output = model.decode_first_stage(samples)
            return torch.clamp((output + 1.0) / 2.0, 0.0, 1.0)


# =============================================================================
# VALIDATION
# =============================================================================

def process_volume(dataloader, model, sampler, h, w, ddim_steps, scale, ddim_eta, device):
    """Process all slices for one volume"""
    recon_slices = []
    target_slices = []
    cond_slice_idx = None
    
    for batch_idx, batch in enumerate(dataloader):
        target = batch["image_target"].to(device)
        input_im = batch["image_cond"].to(device)
        T_cond = batch['T'].to(device)
        
        if cond_slice_idx is None and torch.abs(T_cond).sum() < 0.1:
            cond_slice_idx = batch_idx
        
        output = sample_slice(input_im, T_cond, model, sampler, h, w, ddim_steps, scale, ddim_eta, device)
        
        recon_slices.append(output[0].cpu().numpy())
        target_norm = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        target_slices.append(target_norm[0].cpu().numpy())
    
    if cond_slice_idx is None:
        cond_slice_idx = len(recon_slices) // 2
    
    # Stack and format
    recon = np.stack(recon_slices, axis=0)
    target = np.stack(target_slices, axis=0)
    
    if recon.ndim == 4:
        recon = recon[:, 0, :, :] if recon.shape[1] == 3 else recon[:, :, :, 0]
    if target.ndim == 4:
        target = target[:, 0, :, :] if target.shape[1] == 3 else target[:, :, :, 0]
    
    recon = np.transpose(recon, (1, 2, 0)).astype(np.float32)
    target = np.transpose(target, (1, 2, 0)).astype(np.float32)
    
    return recon, target, cond_slice_idx


def validate_all(
    ckpt, config, data_path, output_dir,
    ddim_steps=50, guidance_scale=7.5, ddim_eta=0.0,
    image_size=256, smooth_sigma=0.8, intensity_smooth_sigma=2.0,
    device_idx=0, max_samples=None, save_volumes=False
):
    """Run validation on all samples"""
    
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    config_obj = OmegaConf.load(config)
    model = load_model_from_config(config_obj, ckpt, device)
    sampler = DDIMSampler(model)
    
    # Fix device
    orig_apply = model.apply_model
    def apply_fixed(x, t, c):
        dev = x.device
        if torch.is_tensor(t): t = t.to(dev)
        if isinstance(c, dict):
            c = {k: [v.to(dev) if torch.is_tensor(v) else v for v in vs] if isinstance(vs, list) 
                 else vs.to(dev) if torch.is_tensor(vs) else vs 
                 for k, vs in c.items()}
        return orig_apply(x, t, c)
    model.apply_model = apply_fixed
    
    # Setup dataset
    print("\nSetting up dataset...")
    
    # Validate data path exists and has subdirectories
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(subdirs) == 0:
        raise ValueError(f"Data path has no subdirectories (patient folders): {data_path}")
    
    print(f"Found {len(subdirs)} patient directories in {data_path}")
    
    dataset = BratsDatasetModuleFromConfig(
        root_dir=data_path, batch_size=1, total_view=1,
        test={'validation': False, 'image_transforms': {'size': image_size}},
        num_workers=1
    )
    
    val_paths = dataset.val_paths
    if max_samples:
        val_paths = val_paths[:max_samples]
    
    print(f"\nProcessing {len(val_paths)} validation samples...")
    print("="*70)
    
    all_results = []
    
    for idx, patient_path in enumerate(val_paths):
        patient_name = os.path.basename(patient_path)
        print(f"\n[{idx+1}/{len(val_paths)}] {patient_name}")
        
        try:
            dataset.test_paths = [patient_path]
            dataloader = dataset.test_dataloader()
            
            # Generate reconstruction
            recon_raw, target, cond_idx = process_volume(
                dataloader, model, sampler, image_size, image_size,
                ddim_steps, guidance_scale, ddim_eta, device
            )
        
            # Post-process
            recon_proc = post_process_volume(
                recon_raw, 
                cond_slice_idx=cond_idx, 
                smooth_sigma=smooth_sigma,
                intensity_smooth_sigma=intensity_smooth_sigma
            )
            
            # Metrics
            metrics_proc = compute_all_metrics(recon_proc, target)
            
            result = {
                'patient': patient_name,
                'conditioning_slice': int(cond_idx),
                'processed': metrics_proc
            }
            all_results.append(result)
            
            print(f"  Cond slice: {cond_idx}")
            print(f"  PSNR: {metrics_proc['psnr']:.2f} dB, SSIM: {metrics_proc['ssim']:.4f}")
            
            # Save volumes if requested
            if save_volumes:
                patient_dir = os.path.join(output_dir, patient_name)
                os.makedirs(patient_dir, exist_ok=True)
                
                # Save post-processed reconstruction
                nib.save(nib.Nifti1Image(recon_proc, np.eye(4)), 
                        os.path.join(patient_dir, f'{patient_name}_reconstructed.nii.gz'))
                
                # Save target
                nib.save(nib.Nifti1Image(target, np.eye(4)), 
                        os.path.join(patient_dir, f'{patient_name}_target.nii.gz'))
                
                # Save conditioning slice as a volume (single slice repeated or just the slice)
                cond_slice = target[:, :, cond_idx]
                # Create a volume with just the conditioning slice (for visualization)
                cond_volume = np.zeros_like(target)
                cond_volume[:, :, cond_idx] = cond_slice
                nib.save(nib.Nifti1Image(cond_volume, np.eye(4)), 
                        os.path.join(patient_dir, f'{patient_name}_input_cond.nii.gz'))
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE VALIDATION METRICS")
    print("="*70)
    
    if all_results:
        proc_psnr = [r['processed']['psnr'] for r in all_results if r['processed']['psnr'] != float('inf')]
        proc_ssim = [r['processed']['ssim'] for r in all_results]
        
        print(f"\nNumber of samples: {len(all_results)}")
        
        print(f"\nPROCESSED:")
        print(f"  PSNR: {np.mean(proc_psnr):.2f} ± {np.std(proc_psnr):.2f} dB")
        print(f"  SSIM: {np.mean(proc_ssim):.4f} ± {np.std(proc_ssim):.4f}")
        
        # Save detailed results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'checkpoint': ckpt,
                'ddim_steps': ddim_steps,
                'guidance_scale': guidance_scale,
                'smooth_sigma': smooth_sigma,
                'intensity_smooth_sigma': intensity_smooth_sigma
            },
            'n_samples': len(all_results),
            'aggregate': {
                'processed': {
                    'psnr_mean': float(np.mean(proc_psnr)),
                    'psnr_std': float(np.std(proc_psnr)),
                    'ssim_mean': float(np.mean(proc_ssim)),
                    'ssim_std': float(np.std(proc_ssim))
                },
            },
            'per_patient': all_results
        }
        
        # Save JSON
        json_path = os.path.join(output_dir, 'validation_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Save CSV
        csv_data = []
        for r in all_results:
            csv_data.append({
                'patient': r['patient'],
                'cond_slice': r['conditioning_slice'],
                'proc_psnr': r['processed']['psnr'],
                'proc_ssim': r['processed']['ssim'],
                'proc_mae': r['processed']['mae'],
                'proc_correlation': r['processed']['correlation']
            })
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'validation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate 3D Volume Reconstruction')
    
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config')
    parser.add_argument('--data_path', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--output_dir', type=str, default='./validation_results')
    
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--smooth_sigma', type=float, default=0.8, 
                        help='Z-axis smoothing sigma (0 to disable)')
    parser.add_argument('--intensity_smooth_sigma', type=float, default=2.0,
                        help='Intensity profile smoothing sigma (0 to disable)')
    
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_volumes', action='store_true')
    
    args = parser.parse_args()
    
    validate_all(
        ckpt=args.ckpt,
        config=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        image_size=args.image_size,
        smooth_sigma=args.smooth_sigma,
        intensity_smooth_sigma=args.intensity_smooth_sigma,
        device_idx=args.device_idx,
        max_samples=args.max_samples,
        save_volumes=args.save_volumes
    )