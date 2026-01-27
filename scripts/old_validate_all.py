"""
Validation Script for 3D Brain Volume Reconstruction
=====================================================

SINGLE-SLICE-TO-FULL-VOLUME reconstruction:
- Input: one conditioning slice (middle slice ~77)
- T_cond positional embedding indicates which slice to generate
- Model generates all 155 slices from that single conditioning slice

Usage:
    python validate_all.py --data_path /path/to/ASNR-MICCAI-BraTS2023-GLI/ --output_dir ./validation_results
"""

from contextlib import nullcontext
import numpy as np
import gc
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from ldm.util import instantiate_from_config
import os 
import nibabel as nib
from ldm.data.bratsloader import BratsDatasetModuleFromConfig
from tqdm import tqdm
import argparse
import ldm.modules.diffusionmodules.openaimodel as openai_module
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
    """Compute comprehensive metrics between prediction and target volumes"""
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
    """Create brain mask"""
    mask = volume > threshold
    for i in range(volume.shape[2]):
        if mask[:, :, i].sum() > 100:
            mask[:, :, i] = binary_fill_holes(mask[:, :, i])
            mask[:, :, i] = binary_erosion(mask[:, :, i], iterations=1)
            mask[:, :, i] = binary_dilation(mask[:, :, i], iterations=2)
    return mask.astype(np.float32)


def fix_inverted_slices(volume, reference_slice_idx, target_volume):
    """Fix inverted/artifact slices"""
    result = volume.copy()
    n_slices = volume.shape[2]
    
    ref_slice = volume[:, :, reference_slice_idx]
    ref_corner = np.mean([
        ref_slice[0:25, 0:25].mean(), ref_slice[0:25, -25:].mean(),
        ref_slice[-25:, 0:25].mean(), ref_slice[-25:, -25:].mean()
    ])
    
    for i in range(n_slices):
        slice_data = volume[:, :, i]
        corner = np.mean([
            slice_data[0:25, 0:25].mean(), slice_data[0:25, -25:].mean(),
            slice_data[-25:, 0:25].mean(), slice_data[-25:, -25:].mean()
        ])
        
        target_empty = target_volume[:, :, i].mean() < 0.01
        
        if corner > 0.4 and ref_corner < 0.2:
            result[:, :, i] = 0 if target_empty else (1.0 - slice_data)
        elif np.std(slice_data) < 0.08 and 0.2 < slice_data.mean() < 0.8:
            result[:, :, i] = 0 if target_empty else slice_data
    
    return result


def remove_edge_artifacts(volume, target_volume):
    """Remove edge slice artifacts based on target content"""
    result = volume.copy()
    n_slices = volume.shape[2]
    
    target_content = np.array([target_volume[:, :, i].mean() for i in range(n_slices)])
    
    first_content, last_content = 0, n_slices - 1
    for i in range(n_slices):
        if target_content[i] > 0.01:
            first_content = i
            break
    for i in range(n_slices - 1, -1, -1):
        if target_content[i] > 0.01:
            last_content = i
            break
    
    for i in range(first_content):
        result[:, :, i] = 0
    for i in range(last_content + 1, n_slices):
        result[:, :, i] = 0
    
    return result


def match_intensity(recon, target):
    """Match intensity profile"""
    result = recon.copy()
    target_profile = np.mean(target, axis=(0, 1))
    current_profile = np.mean(result, axis=(0, 1))
    
    for i in range(result.shape[2]):
        if current_profile[i] > 1e-6 and target_profile[i] > 1e-6:
            scale = np.clip(target_profile[i] / (current_profile[i] + 1e-8), 0.2, 5.0)
            result[:, :, i] *= scale
        elif target_profile[i] < 1e-6:
            result[:, :, i] = 0
    
    return result


def save_volume_pngs(volume, output_dir, prefix, num_slices=9):
    """Save PNG images of volume slices.
    
    Args:
        volume: 3D numpy array (H, W, D)
        output_dir: Directory to save images
        prefix: Filename prefix
        num_slices: Number of slices to save in the montage
    """
    os.makedirs(output_dir, exist_ok=True)
    n_slices = volume.shape[2]
    
    # Normalize volume to 0-255
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max - vol_min > 1e-8:
        vol_norm = ((volume - vol_min) / (vol_max - vol_min) * 255).astype(np.uint8)
    else:
        vol_norm = np.zeros_like(volume, dtype=np.uint8)
    
    # Save individual slices at key positions
    slice_indices = np.linspace(0, n_slices - 1, num_slices, dtype=int)
    
    # Create montage image
    n_cols = 3
    n_rows = (num_slices + n_cols - 1) // n_cols
    h, w = volume.shape[:2]
    montage = np.zeros((n_rows * h, n_cols * w), dtype=np.uint8)
    
    for i, slice_idx in enumerate(slice_indices):
        row, col = i // n_cols, i % n_cols
        montage[row*h:(row+1)*h, col*w:(col+1)*w] = vol_norm[:, :, slice_idx]
    
    # Save montage
    montage_img = Image.fromarray(montage)
    montage_img.save(os.path.join(output_dir, f'{prefix}_montage.png'))
    
    # Save middle slice separately
    mid_idx = n_slices // 2
    mid_img = Image.fromarray(vol_norm[:, :, mid_idx])
    mid_img.save(os.path.join(output_dir, f'{prefix}_slice{mid_idx:03d}.png'))
    
    # Save all slices in a subfolder
    slices_dir = os.path.join(output_dir, f'{prefix}_slices')
    os.makedirs(slices_dir, exist_ok=True)
    for i in range(n_slices):
        slice_img = Image.fromarray(vol_norm[:, :, i])
        slice_img.save(os.path.join(slices_dir, f'slice_{i:03d}.png'))


def post_process_volume(recon, target, cond_slice_idx=None, smooth_sigma=0.8):
    """Complete post-processing pipeline"""
    if cond_slice_idx is None:
        cond_slice_idx = recon.shape[2] // 2
    
    result = recon.copy().astype(np.float32)
    
    # 1. Fix inverted slices
    result = fix_inverted_slices(result, cond_slice_idx, target)
    
    # 2. Remove edge artifacts
    result = remove_edge_artifacts(result, target)
    
    # 3. Enforce black background
    mask = create_brain_mask(target, threshold=0.03)
    result[mask == 0] = 0
    
    # 4. Match intensity
    result = match_intensity(result, target)
    
    # 5. Z-smoothing
    if smooth_sigma > 0:
        result = gaussian_filter1d(result, sigma=smooth_sigma, axis=2)
    
    # 6. Final cleanup
    result = result * mask
    result = np.clip(result, 0, 1)
    
    return result


# =============================================================================
# MODEL
# =============================================================================

def load_model_from_config(config, ckpt, device):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    
    pl_sd = torch.load(ckpt, map_location='cpu')
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


@torch.no_grad()
def sample_slice(input_im, T_cond, model, sampler, h, w, ddim_steps, scale, ddim_eta, device):
    """Generate single slice"""
    model = model.to(device)
    sampler.model = model
    
    original_te = openai_module.timestep_embedding
    def te_fixed(t, d, max_period=10000, repeat_only=False):
        return original_te(t, d, max_period, repeat_only).to(device)
    openai_module.timestep_embedding = te_fixed
    
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
    denoised_slices = []
    target_slices = []
    input_slice = None
    cond_slice_idx = None
    
    for batch_idx, batch in enumerate(dataloader):
        target = batch["image_target"].to(device)
        input_im = batch["image_cond"].to(device)
        T_cond = batch['T'].to(device)
        
        # Always capture the input slice from the first batch
        if batch_idx == 0:
            input_norm = torch.clamp((input_im + 1.0) / 2.0, 0.0, 1.0)
            input_slice = input_norm[0].cpu().numpy()
        
        if cond_slice_idx is None and torch.abs(T_cond).sum() < 0.1:
            cond_slice_idx = batch_idx
        
        output = sample_slice(input_im, T_cond, model, sampler, h, w, ddim_steps, scale, ddim_eta, device)
        
        denoised_slices.append(output[0].cpu().numpy())
        target_norm = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        target_slices.append(target_norm[0].cpu().numpy())
    
    if cond_slice_idx is None:
        cond_slice_idx = len(denoised_slices) // 2
    
    # Stack and format
    denoised = np.stack(denoised_slices, axis=0)
    target = np.stack(target_slices, axis=0)
    
    if denoised.ndim == 4:
        denoised = denoised[:, 0, :, :] if denoised.shape[1] == 3 else denoised[:, :, :, 0]
    if target.ndim == 4:
        target = target[:, 0, :, :] if target.shape[1] == 3 else target[:, :, :, 0]
    
    # Format input slice
    if input_slice is not None:
        if input_slice.ndim == 3:
            input_slice = input_slice[0] if input_slice.shape[0] == 3 else input_slice[:, :, 0]
        input_slice = input_slice.astype(np.float32)
    
    denoised = np.transpose(denoised, (1, 2, 0)).astype(np.float32)
    target = np.transpose(target, (1, 2, 0)).astype(np.float32)
    
    return denoised, target, input_slice, cond_slice_idx


def validate_all(
    ckpt, config, data_path, output_dir,
    ddim_steps=50, guidance_scale=7.5, ddim_eta=0.0,
    image_size=256, smooth_sigma=0.8,
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
            
            # Generate denoised volume
            denoised_raw, target, input_slice, cond_idx = process_volume(
                dataloader, model, sampler, image_size, image_size,
                ddim_steps, guidance_scale, ddim_eta, device
            )
            
            # Post-process
            denoised = post_process_volume(denoised_raw, target, cond_idx, smooth_sigma)
            
            # Metrics after post-processing
            metrics_proc = compute_all_metrics(denoised, target)
            
            result = {
                'patient': patient_name,
                'conditioning_slice': int(cond_idx),
                'processed': metrics_proc
            }
            all_results.append(result)
            
            print(f"  Cond slice: {cond_idx}")
            print(f"  Post   PSNR: {metrics_proc['psnr']:.2f} dB, SSIM: {metrics_proc['ssim']:.4f}")
            
            # Save volumes if requested
            if save_volumes:
                patient_dir = os.path.join(output_dir, patient_name)
                os.makedirs(patient_dir, exist_ok=True)
                
                # Save NIfTI files
                nib.save(nib.Nifti1Image(denoised, np.eye(4)), 
                        os.path.join(patient_dir, f'{patient_name}_denoised.nii.gz'))
                nib.save(nib.Nifti1Image(target, np.eye(4)), 
                        os.path.join(patient_dir, f'{patient_name}_target.nii.gz'))
                
                # Save PNG images
                save_volume_pngs(denoised, patient_dir, f'{patient_name}_denoised')
                save_volume_pngs(target, patient_dir, f'{patient_name}_target')
                
                # Save input/conditioning slice
                if input_slice is not None:
                    input_img = Image.fromarray(
                        ((input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8) * 255).astype(np.uint8)
                    )
                    input_img.save(os.path.join(patient_dir, f'{patient_name}_input_slice{cond_idx:03d}.png'))
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE VALIDATION METRICS")
    print("="*70)
    
    if all_results:
        proc_psnr = [r['processed']['psnr'] for r in all_results if r['processed']['psnr'] != float('inf')]
        proc_ssim = [r['processed']['ssim'] for r in all_results]
        
        print(f"\nNumber of samples: {len(all_results)}")
    
        
        print(f"\nOUTPUT:")
        print(f"  PSNR: {np.mean(proc_psnr):.2f} ± {np.std(proc_psnr):.2f} dB")
        print(f"  SSIM: {np.mean(proc_ssim):.4f} ± {np.std(proc_ssim):.4f}")
        
        # Save detailed results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'checkpoint': ckpt,
                'ddim_steps': ddim_steps,
                'guidance_scale': guidance_scale,
                'smooth_sigma': smooth_sigma
            },
            'n_samples': len(all_results),
            'aggregate': {
                'processed': {
                    'psnr_mean': float(np.mean(proc_psnr)),
                    'psnr_std': float(np.std(proc_psnr)),
                    'ssim_mean': float(np.mean(proc_ssim)),
                    'ssim_std': float(np.std(proc_ssim))
                }
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
                'proc_ssim': r['processed']['ssim']
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
    parser.add_argument('--smooth_sigma', type=float, default=0.8)
    
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
        max_samples=args.max_samples,
        save_volumes=args.save_volumes
    )