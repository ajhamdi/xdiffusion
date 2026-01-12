#!/usr/bin/env python3
"""Analyze experiment results from X-Diffusion inference runs."""

import json
import os
from pathlib import Path
from collections import defaultdict

def parse_experiment_name(dirname):
    """Parse experiment directory name to extract parameters.
    
    Format: {total_view}_{axis}_g{guidance_scale}_med_{ddim_steps}
    Example: 1_axial_g75_med_300
    """
    parts = dirname.split('_')
    if len(parts) < 5:
        return None
    
    # Extract guidance scale (g75 -> 7.5, g10 -> 1.0, g100 -> 10.0, etc.)
    gs_str = parts[2].replace('g', '')
    # Convert: 75 -> 7.5, 10 -> 1.0, 100 -> 10.0, 150 -> 15.0, etc.
    if len(gs_str) == 2:
        guidance_scale = float(gs_str) / 10  # 75 -> 7.5
    elif len(gs_str) == 3:
        guidance_scale = float(gs_str) / 10  # 100 -> 10.0, 150 -> 15.0
    else:
        guidance_scale = float(gs_str)
    
    params = {
        'total_view': int(parts[0]),
        'axis': parts[1],
        'guidance_scale': guidance_scale,
        'median': 'med' in dirname,
        'ddim_steps': int(parts[-1])
    }
    return params

def load_metrics(experiment_dir):
    """Load all metrics from an experiment directory."""
    metrics_files = list(Path(experiment_dir).glob('*_metrics.json'))
    
    all_psnr = []
    all_ssim = []
    
    for mf in metrics_files:
        with open(mf, 'r') as f:
            data = json.load(f)
            all_psnr.append(data['psnr']['mean'])
            all_ssim.append(data['ssim']['mean'])
    
    if all_psnr:
        return {
            'psnr_mean': sum(all_psnr) / len(all_psnr),
            'ssim_mean': sum(all_ssim) / len(all_ssim),
            'num_patients': len(all_psnr)
        }
    return None

def main():
    output_dir = Path('/home/hamdiaj/notebooks/X-Diffusion/output')
    
    # Collect all experiment results
    results = []
    
    for exp_dir in sorted(output_dir.iterdir()):
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            params = parse_experiment_name(exp_dir.name)
            if params is None:
                continue  # Skip directories that don't match expected format
            metrics = load_metrics(exp_dir)
            
            if metrics:
                results.append({
                    'name': exp_dir.name,
                    **params,
                    **metrics
                })
    
    # Sort results by PSNR for overall ranking
    results_sorted = sorted(results, key=lambda x: x['psnr_mean'], reverse=True)
    
    print("=" * 100)
    print("EXPERIMENT RESULTS ANALYSIS - X-Diffusion BraTS 2023")
    print("=" * 100)
    
    # Overall ranking table
    print("\n" + "=" * 100)
    print("OVERALL RANKING (Sorted by PSNR)")
    print("=" * 100)
    print(f"{'Rank':<6}{'Experiment':<30}{'View':<6}{'Axis':<10}{'G.Scale':<8}{'Steps':<8}{'PSNR':>10}{'SSIM':>10}{'Patients':>10}")
    print("-" * 100)
    
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:<6}{r['name']:<30}{r['total_view']:<6}{r['axis']:<10}{r['guidance_scale']:<8.1f}{r['ddim_steps']:<8}{r['psnr_mean']:>10.4f}{r['ssim_mean']:>10.4f}{r['num_patients']:>10}")
    
    # Group by parameter variation
    print("\n" + "=" * 100)
    print("ANALYSIS BY PARAMETER VARIATION")
    print("=" * 100)
    
    # Baseline reference
    baseline = [r for r in results if r['name'] == '1_axial_g75_med_300']
    if baseline:
        baseline = baseline[0]
        print(f"\nBaseline: {baseline['name']}")
        print(f"  PSNR: {baseline['psnr_mean']:.4f}, SSIM: {baseline['ssim_mean']:.4f}")
    
    # Axis variation analysis
    print("\n" + "-" * 50)
    print("AXIS VARIATION (view=1, g=7.5, steps=300)")
    print("-" * 50)
    axis_results = [r for r in results if r['total_view'] == '1' and r['guidance_scale'] == '75' and r['ddim_steps'] == '300']
    axis_results = sorted(axis_results, key=lambda x: x['psnr_mean'], reverse=True)
    print(f"{'Axis':<15}{'PSNR':>12}{'SSIM':>12}{'vs Baseline PSNR':>18}{'vs Baseline SSIM':>18}")
    for r in axis_results:
        psnr_diff = r['psnr_mean'] - baseline['psnr_mean'] if baseline else 0
        ssim_diff = r['ssim_mean'] - baseline['ssim_mean'] if baseline else 0
        print(f"{r['axis']:<15}{r['psnr_mean']:>12.4f}{r['ssim_mean']:>12.4f}{psnr_diff:>+18.4f}{ssim_diff:>+18.4f}")
    
    # Guidance scale variation analysis
    print("\n" + "-" * 50)
    print("GUIDANCE SCALE VARIATION (axial, view=1, steps=300)")
    print("-" * 50)
    gs_results = [r for r in results if r['axis'] == 'axial' and r['total_view'] == '1' and r['ddim_steps'] == '300']
    gs_results = sorted(gs_results, key=lambda x: float(x['guidance_scale']))
    print(f"{'G.Scale':<15}{'PSNR':>12}{'SSIM':>12}{'vs Baseline PSNR':>18}{'vs Baseline SSIM':>18}")
    for r in gs_results:
        psnr_diff = r['psnr_mean'] - baseline['psnr_mean'] if baseline else 0
        ssim_diff = r['ssim_mean'] - baseline['ssim_mean'] if baseline else 0
        print(f"{r['guidance_scale']:<15}{r['psnr_mean']:>12.4f}{r['ssim_mean']:>12.4f}{psnr_diff:>+18.4f}{ssim_diff:>+18.4f}")
    
    # DDIM steps variation analysis
    print("\n" + "-" * 50)
    print("DDIM STEPS VARIATION (axial, view=1, g=7.5)")
    print("-" * 50)
    steps_results = [r for r in results if r['axis'] == 'axial' and r['total_view'] == '1' and r['guidance_scale'] == '75']
    steps_results = sorted(steps_results, key=lambda x: int(x['ddim_steps']))
    print(f"{'Steps':<15}{'PSNR':>12}{'SSIM':>12}{'vs Baseline PSNR':>18}{'vs Baseline SSIM':>18}")
    for r in steps_results:
        psnr_diff = r['psnr_mean'] - baseline['psnr_mean'] if baseline else 0
        ssim_diff = r['ssim_mean'] - baseline['ssim_mean'] if baseline else 0
        print(f"{r['ddim_steps']:<15}{r['psnr_mean']:>12.4f}{r['ssim_mean']:>12.4f}{psnr_diff:>+18.4f}{ssim_diff:>+18.4f}")
    
    # Total view variation analysis
    print("\n" + "-" * 50)
    print("TOTAL VIEW VARIATION (axial, g=7.5, steps=300)")
    print("-" * 50)
    view_results = [r for r in results if r['axis'] == 'axial' and r['guidance_scale'] == '75' and r['ddim_steps'] == '300']
    view_results = sorted(view_results, key=lambda x: int(x['total_view']))
    print(f"{'Views':<15}{'PSNR':>12}{'SSIM':>12}{'vs Baseline PSNR':>18}{'vs Baseline SSIM':>18}")
    for r in view_results:
        psnr_diff = r['psnr_mean'] - baseline['psnr_mean'] if baseline else 0
        ssim_diff = r['ssim_mean'] - baseline['ssim_mean'] if baseline else 0
        print(f"{r['total_view']:<15}{r['psnr_mean']:>12.4f}{r['ssim_mean']:>12.4f}{psnr_diff:>+18.4f}{ssim_diff:>+18.4f}")
    
    # Summary and recommendations
    print("\n" + "=" * 100)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 100)
    
    best_overall = results_sorted[0]
    print(f"\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"   Experiment: {best_overall['name']}")
    print(f"   PSNR: {best_overall['psnr_mean']:.4f}, SSIM: {best_overall['ssim_mean']:.4f}")
    
    # Best by SSIM
    results_by_ssim = sorted(results, key=lambda x: x['ssim_mean'], reverse=True)
    best_ssim = results_by_ssim[0]
    print(f"\n📊 BEST SSIM CONFIGURATION:")
    print(f"   Experiment: {best_ssim['name']}")
    print(f"   PSNR: {best_ssim['psnr_mean']:.4f}, SSIM: {best_ssim['ssim_mean']:.4f}")
    
    # Analysis insights
    print("\n" + "-" * 50)
    print("KEY INSIGHTS:")
    print("-" * 50)
    
    # Axis insights
    if axis_results:
        best_axis = max(axis_results, key=lambda x: x['psnr_mean'])
        print(f"• Best axis: {best_axis['axis']} (PSNR: {best_axis['psnr_mean']:.4f})")
    
    # Guidance scale insights
    if gs_results:
        best_gs = max(gs_results, key=lambda x: x['psnr_mean'])
        print(f"• Best guidance scale: {best_gs['guidance_scale']} (PSNR: {best_gs['psnr_mean']:.4f})")
    
    # Steps insights
    if steps_results:
        best_steps = max(steps_results, key=lambda x: x['psnr_mean'])
        print(f"• Best DDIM steps: {best_steps['ddim_steps']} (PSNR: {best_steps['psnr_mean']:.4f})")
    
    # View insights
    if view_results:
        best_view = max(view_results, key=lambda x: x['psnr_mean'])
        print(f"• Best total views: {best_view['total_view']} (PSNR: {best_view['psnr_mean']:.4f})")

if __name__ == '__main__':
    main()

