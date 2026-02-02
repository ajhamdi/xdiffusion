"""
Validation Script for BraTS Multi-Slice Reconstruction
==================================================================
Computes per-slice PSNR/SSIM and aggregates by patient.

Usage:
    python validate_all_multislice.py \
        --ckpt checkpoints/last.ckpt \
        --config configs/sd-brats-multislice.yaml \
        --ddim_steps 200 \
        --guidance_scale 1.0
"""

import argparse
import numpy as np
import torch
from torch import autocast
from contextlib import nullcontext
from tqdm import tqdm
from collections import defaultdict
import json
import os
from scipy.ndimage import gaussian_filter1d

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange


def post_process_volume(pred_volume, z_smooth_sigma=1.2, spatial_smooth_sigma=0.8, 
                               enhance_contrast=True, denoise_strength=0.5):
    """    
    Args:
        pred_volume: Predicted volume [H, W, D]
        z_smooth_sigma: Gaussian smoothing along z-axis
        spatial_smooth_sigma: Spatial smoothing per-slice
        enhance_contrast: Apply adaptive histogram equalization
        denoise_strength: Denoising strength (0-1)
    """
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage import exposure
    
    result = pred_volume.copy().astype(np.float32)
    
    # Denoising with median filter (preserves edges better than Gaussian)
    if denoise_strength > 0:
        # Apply median filter per-slice
        for i in range(result.shape[2]):
            kernel_size = max(3, int(5 * denoise_strength))
            if kernel_size % 2 == 0:
                kernel_size += 1
            result[:, :, i] = median_filter(result[:, :, i], size=kernel_size)
    
    # Spatial smoothing per-slice (mild Gaussian)
    if spatial_smooth_sigma > 0:
        for i in range(result.shape[2]):
            result[:, :, i] = gaussian_filter(result[:, :, i], sigma=spatial_smooth_sigma)
    
    # Z-axis smoothing (inter-slice consistency)
    if z_smooth_sigma > 0:
        result = gaussian_filter1d(result, sigma=z_smooth_sigma, axis=2)
    
    # Adaptive contrast enhancement (CLAHE)
    if enhance_contrast:
        # Apply per-slice CLAHE
        for i in range(result.shape[2]):
            slice_data = result[:, :, i]
            if slice_data.max() > slice_data.min():
                # CLAHE with mild settings
                enhanced = exposure.equalize_adapthist(
                    slice_data, 
                    kernel_size=None,  # Auto
                    clip_limit=0.01,  # Mild enhancement
                    nbins=256
                )
                result[:, :, i] = enhanced
    
    # Normalize to [0, 1]
    result = np.clip(result, 0, 1)
    
    return result


def post_process_volume_light(pred_volume, z_smooth_sigma=0.5):
    """
    Lightweight post-processing.
    """
    result = pred_volume.copy().astype(np.float32)
    
    # Z-axis smoothing only
    if z_smooth_sigma > 0:
        result = gaussian_filter1d(result, sigma=z_smooth_sigma, axis=2)
    
    result = np.clip(result, 0, 1)
    return result


def compute_psnr(pred, target, data_range=1.0):
    """Compute PSNR between pred and target"""
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(data_range ** 2 / mse)


def compute_ssim(pred, target):
    """Simple SSIM computation"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p, mu_t = torch.mean(pred), torch.mean(target)
    sig_p = torch.var(pred)
    sig_t = torch.var(target)
    sig_pt = torch.mean((pred - mu_p) * (target - mu_t))
    return ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))


def load_model(config_path, ckpt_path, device):
    """Load model from checkpoint"""
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    print(f"Loading checkpoint from {ckpt_path}")
    model = instantiate_from_config(config.model)
    
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def validate(model, dataloader, device, ddim_steps=50, guidance_scale=1.0, ddim_eta=0.0):
    """Run validation on dataloader"""
    
    sampler = DDIMSampler(model)
    precision_scope = autocast if torch.cuda.is_available() else nullcontext
    
    # Store slices by patient for 3D volume reconstruction
    patient_slices = defaultdict(lambda: {'pred': [], 'target': [], 'slice_idx': []})
    # Store per-slice metrics
    patient_slice_metrics = defaultdict(lambda: {'psnr': [], 'ssim': []})
    
    print(f"\nRunning validation with {ddim_steps} DDIM steps, guidance scale {guidance_scale}")
    print("=" * 70)
    
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
        try:
            # Get data
            target = batch["image_target"].to(device)
            cond = batch["image_cond"].to(device)
            T = batch["T"].to(device)
            filenames = batch["filename"]
            target_idx = batch["target_idx"]
            
            # Extract patient ID and slice index from filename
            patient_id = filenames[0].split('_slice_')[0] if isinstance(filenames, list) else filenames.split('_slice_')[0]
            slice_idx = target_idx[0].item() if torch.is_tensor(target_idx) else target_idx[0]
            
            # Convert to channels-first if needed
            if target.shape[-1] == 3:
                target = rearrange(target, "b h w c -> b c h w")
                cond = rearrange(cond, "b h w c -> b c h w")
            
            n = target.shape[0]
            h, w = target.shape[2], target.shape[3]
            
            with precision_scope("cuda"):
                with model.ema_scope():
                    # Encode conditioning image with proper scale_factor
                    cond_encoded = model.get_first_stage_encoding(model.encode_first_stage(cond))
                    
                    # Get CLIP embedding
                    clip_emb = model.get_learned_conditioning(cond)
                    
                    # Combine with T
                    T_expanded = T[:, None, :]
                    c_combined = torch.cat([clip_emb, T_expanded], dim=-1)
                    c_proj = model.cc_projection(c_combined)
                    
                    # Create conditioning dict
                    cond_dict = {
                        "c_crossattn": [c_proj],
                        "c_concat": [cond_encoded]
                    }
                    
                    # Unconditional conditioning for guidance
                    if guidance_scale != 1.0:
                        uc = {
                            "c_concat": [torch.zeros_like(cond_encoded)],
                            "c_crossattn": [torch.zeros_like(c_proj)]
                        }
                    else:
                        uc = None
                    
                    # Sample
                    z_shape = [n, 4, h // 8, w // 8]
                    samples, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=cond_dict,
                        batch_size=n,
                        shape=z_shape[1:],
                        verbose=False,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta
                    )
                    
                    # Decode
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                    target_norm = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
                    
                    # Store slices for 3D volume reconstruction
                    for i in range(n):
                        # Take first channel if RGB (grayscale replicated)
                        pred_slice = x_samples[i, 0].cpu().numpy()
                        target_slice = target_norm[i, 0].cpu().numpy()
                        
                        patient_slices[patient_id]['pred'].append(pred_slice)
                        patient_slices[patient_id]['target'].append(target_slice)
                        patient_slices[patient_id]['slice_idx'].append(slice_idx)
                        
                        # Per-slice metrics
                        psnr = compute_psnr(x_samples[i], target_norm[i]).item()
                        ssim = compute_ssim(x_samples[i], target_norm[i]).item()
                        
                        patient_slice_metrics[patient_id]['psnr'].append(psnr)
                        patient_slice_metrics[patient_id]['ssim'].append(ssim)
                    
                    batch_count += 1
        
        except Exception as e:
            print(f"\nError on batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nCompleted validation: {len(patient_slices)} patients, {batch_count} batches")
    
    return patient_slices, patient_slice_metrics


def compute_volume_psnr(pred_volume, target_volume, data_range=1.0):
    """Compute PSNR for entire 3D volume"""
    mse = np.mean((pred_volume - target_volume) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


def aggregate_results(patient_slices, patient_slice_metrics, apply_postprocessing=True):
    """Aggregate per-patient results including 3D volumetric PSNR"""
    
    # Reconstruct 3D volumes and compute volumetric PSNR
    patient_stats = []
    
    for patient_id in patient_slices.keys():
        slices_data = patient_slices[patient_id]
        slice_metrics = patient_slice_metrics[patient_id]
        
        # Sort slices by index
        sorted_indices = np.argsort(slices_data['slice_idx'])
        
        # Stack into 3D volumes [H, W, D]
        pred_volume_raw = np.stack([slices_data['pred'][i] for i in sorted_indices], axis=2)
        target_volume = np.stack([slices_data['target'][i] for i in sorted_indices], axis=2)
        
        # Apply post-processing and compute final PSNR
        if apply_postprocessing:
            pred_volume_final = post_process_volume(
                pred_volume_raw,
                z_smooth_sigma=1.2,
                spatial_smooth_sigma=0.8,
                enhance_contrast=True,
                denoise_strength=0.5
            )
        else:
            pred_volume_final = pred_volume_raw
        
        psnr_3d = compute_volume_psnr(pred_volume_final, target_volume)
        
        patient_stats.append({
            'patient': patient_id,
            'n_slices': len(slices_data['pred']),
            'psnr_3d': float(psnr_3d),
            'psnr_2d_mean': float(np.mean(slice_metrics['psnr'])),
            'psnr_2d_std': float(np.std(slice_metrics['psnr'])),
            'ssim_mean': float(np.mean(slice_metrics['ssim'])),
            'ssim_std': float(np.std(slice_metrics['ssim']))
        })
    
    # Sort by patient name
    patient_stats.sort(key=lambda x: x['patient'])
    
    # Overall stats
    all_psnr_3d = [s['psnr_3d'] for s in patient_stats]
    all_psnr_2d = [psnr for m in patient_slice_metrics.values() for psnr in m['psnr']]
    all_ssim = [ssim for m in patient_slice_metrics.values() for ssim in m['ssim']]
    
    overall = {
        'n_patients': len(patient_slices),
        'n_slices_total': len(all_psnr_2d),
        'psnr_3d_mean': float(np.mean(all_psnr_3d)),
        'psnr_3d_std': float(np.std(all_psnr_3d)),
        'psnr_2d_mean': float(np.mean(all_psnr_2d)),
        'psnr_2d_std': float(np.std(all_psnr_2d)),
        'ssim_mean': float(np.mean(all_ssim)),
        'ssim_std': float(np.std(all_ssim))
    }
    
    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nDataset: {overall['n_patients']} patients, {overall['n_slices_total']} slices")
    print(f"\n3D Volumetric PSNR:  {overall['psnr_3d_mean']:.2f} ± {overall['psnr_3d_std']:.2f} dB")
    print(f"2D Per-Slice PSNR:   {overall['psnr_2d_mean']:.2f} ± {overall['psnr_2d_std']:.2f} dB")
    print("=" * 70)
    
    return overall, patient_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path from config")
    parser.add_argument("--output_dir", type=str, default="./validation_results", help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to validate (for testing)")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.config, args.ckpt, device)
    
    # Override data path if provided
    if args.data_path:
        print(f"Overriding data path to: {args.data_path}")
        config.data.params.root_dir = args.data_path
    
    # Load data
    print("\nLoading validation data...")
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    dataloader = data.val_dataloader()
    
    if args.max_batches:
        print(f"Limiting to {args.max_batches} batches for testing")
        from itertools import islice
        dataloader = islice(dataloader, args.max_batches)
        dataloader = list(dataloader)
    
    # Run validation
    patient_slices, patient_slice_metrics = validate(
        model, dataloader, device,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta
    )
    
    # Aggregate and print results
    overall, patient_stats = aggregate_results(patient_slices, patient_slice_metrics)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'checkpoint': args.ckpt,
        'config': args.config,
        'ddim_steps': args.ddim_steps,
        'guidance_scale': args.guidance_scale,
        'overall': overall,
        'per_patient': patient_stats
    }
    
    output_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()