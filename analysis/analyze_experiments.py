#!/usr/bin/env python3
"""
X-Diffusion Experiment Analysis Script

Analyzes the results of diffusion model experiments on the BraTS dataset,
varying the following parameters:
- Axis: axial, coronal, sagittal
- Guidance Scale: 10, 30, 50, 75, 100, 150
- DDIM Steps: 50, 100, 200, 300, 500
- Total View (conditioning slices): 1, 2, 3, 4
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Define output directory
OUTPUT_DIR = Path('/home/hamdiaj/notebooks/X-Diffusion/output')
ANALYSIS_DIR = Path('/home/hamdiaj/notebooks/X-Diffusion/analysis')
ANALYSIS_DIR.mkdir(exist_ok=True)


def load_metrics(exp_dir):
    """Load metrics from experiment directory."""
    metrics_files = list(exp_dir.glob('*_metrics.json'))
    if not metrics_files:
        return None
    with open(metrics_files[0], 'r') as f:
        return json.load(f)


def parse_exp_name(name):
    """Parse experiment directory name to extract parameters."""
    parts = name.split('_')
    total_view = int(parts[0])
    axis = parts[1]
    guidance = int(parts[2][1:])  # Remove 'g' prefix
    ddim_steps = int(parts[4])    # Skip 'med'
    return {
        'total_view': total_view,
        'axis': axis,
        'guidance_scale': guidance,
        'ddim_steps': ddim_steps
    }


def collect_experiments():
    """Collect all experiment data."""
    experiments = []
    for exp_dir in OUTPUT_DIR.iterdir():
        if exp_dir.is_dir():
            metrics = load_metrics(exp_dir)
            if metrics:
                params = parse_exp_name(exp_dir.name)
                experiments.append({
                    **params,
                    'dir_name': exp_dir.name,
                    'psnr_mean': metrics['psnr']['mean'],
                    'psnr_std': metrics['psnr']['std'],
                    'psnr_min': metrics['psnr']['min'],
                    'psnr_max': metrics['psnr']['max'],
                    'ssim_mean': metrics['ssim']['mean'],
                    'ssim_std': metrics['ssim']['std'],
                    'ssim_min': metrics['ssim']['min'],
                    'ssim_max': metrics['ssim']['max'],
                    'num_slices': metrics['num_slices']
                })
    return pd.DataFrame(experiments)


def plot_axis_comparison(df):
    """Plot axis orientation comparison."""
    # Filter for axis comparison (g75, steps=300, total_view=1)
    axis_df = df[(df['guidance_scale'] == 75) & 
                 (df['ddim_steps'] == 300) & 
                 (df['total_view'] == 1)].copy()
    
    if len(axis_df) < 2:
        print("Insufficient data for axis comparison")
        return axis_df
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR by axis
    colors = {'axial': '#2ecc71', 'coronal': '#3498db', 'sagittal': '#e74c3c'}
    axis_order = [a for a in ['axial', 'coronal', 'sagittal'] if a in axis_df['axis'].values]
    axis_df_sorted = axis_df.set_index('axis').loc[axis_order].reset_index()
    
    bars1 = axes[0].bar(axis_df_sorted['axis'], axis_df_sorted['psnr_mean'], 
                        yerr=axis_df_sorted['psnr_std'], capsize=5,
                        color=[colors.get(a, '#95a5a6') for a in axis_df_sorted['axis']])
    axes[0].set_xlabel('Axis Orientation')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR by Axis Orientation')
    axes[0].set_ylim(0, max(axis_df_sorted['psnr_mean']) * 1.3)
    
    # Add value labels
    for bar, val in zip(bars1, axis_df_sorted['psnr_mean']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM by axis
    bars2 = axes[1].bar(axis_df_sorted['axis'], axis_df_sorted['ssim_mean'], 
                        yerr=axis_df_sorted['ssim_std'], capsize=5,
                        color=[colors.get(a, '#95a5a6') for a in axis_df_sorted['axis']])
    axes[1].set_xlabel('Axis Orientation')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM by Axis Orientation')
    axes[1].set_ylim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars2, axis_df_sorted['ssim_mean']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'axis_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return axis_df_sorted


def plot_guidance_comparison(df):
    """Plot guidance scale comparison."""
    # Filter for guidance scale comparison (axial, steps=300, total_view=1)
    guidance_df = df[(df['axis'] == 'axial') & 
                     (df['ddim_steps'] == 300) & 
                     (df['total_view'] == 1)].copy()
    guidance_df = guidance_df.sort_values('guidance_scale')
    
    if len(guidance_df) < 2:
        print("Insufficient data for guidance scale comparison")
        return guidance_df
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR vs guidance scale
    axes[0].errorbar(guidance_df['guidance_scale'], guidance_df['psnr_mean'], 
                    yerr=guidance_df['psnr_std'], marker='o', markersize=10,
                    linewidth=2, capsize=5, color='#2980b9')
    axes[0].set_xlabel('Guidance Scale')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Guidance Scale')
    axes[0].set_xticks(guidance_df['guidance_scale'])
    axes[0].grid(True, alpha=0.3)
    
    # Highlight best point
    best_psnr_idx = guidance_df['psnr_mean'].idxmax()
    best_psnr = guidance_df.loc[best_psnr_idx]
    axes[0].scatter([best_psnr['guidance_scale']], [best_psnr['psnr_mean']], 
                   s=200, c='#e74c3c', zorder=5, marker='*')
    axes[0].annotate(f'Best: {best_psnr["psnr_mean"]:.2f}', 
                    xy=(best_psnr['guidance_scale'], best_psnr['psnr_mean']),
                    xytext=(10, 10), textcoords='offset points')
    
    # SSIM vs guidance scale
    axes[1].errorbar(guidance_df['guidance_scale'], guidance_df['ssim_mean'], 
                    yerr=guidance_df['ssim_std'], marker='o', markersize=10,
                    linewidth=2, capsize=5, color='#27ae60')
    axes[1].set_xlabel('Guidance Scale')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Guidance Scale')
    axes[1].set_xticks(guidance_df['guidance_scale'])
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    # Highlight best point
    best_ssim_idx = guidance_df['ssim_mean'].idxmax()
    best_ssim = guidance_df.loc[best_ssim_idx]
    axes[1].scatter([best_ssim['guidance_scale']], [best_ssim['ssim_mean']], 
                   s=200, c='#e74c3c', zorder=5, marker='*')
    axes[1].annotate(f'Best: {best_ssim["ssim_mean"]:.3f}', 
                    xy=(best_ssim['guidance_scale'], best_ssim['ssim_mean']),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'guidance_scale_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return guidance_df


def plot_steps_comparison(df):
    """Plot DDIM steps comparison."""
    # Filter for DDIM steps comparison (axial, g75, total_view=1)
    steps_df = df[(df['axis'] == 'axial') & 
                  (df['guidance_scale'] == 75) & 
                  (df['total_view'] == 1)].copy()
    steps_df = steps_df.sort_values('ddim_steps')
    
    if len(steps_df) < 2:
        print("Insufficient data for DDIM steps comparison")
        return steps_df
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR vs DDIM steps
    axes[0].errorbar(steps_df['ddim_steps'], steps_df['psnr_mean'], 
                    yerr=steps_df['psnr_std'], marker='s', markersize=10,
                    linewidth=2, capsize=5, color='#8e44ad')
    axes[0].set_xlabel('DDIM Steps')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs DDIM Steps')
    axes[0].set_xticks(steps_df['ddim_steps'])
    axes[0].grid(True, alpha=0.3)
    
    # Highlight best point
    best_psnr_idx = steps_df['psnr_mean'].idxmax()
    best_psnr = steps_df.loc[best_psnr_idx]
    axes[0].scatter([best_psnr['ddim_steps']], [best_psnr['psnr_mean']], 
                   s=200, c='#e74c3c', zorder=5, marker='*')
    
    # SSIM vs DDIM steps
    axes[1].errorbar(steps_df['ddim_steps'], steps_df['ssim_mean'], 
                    yerr=steps_df['ssim_std'], marker='s', markersize=10,
                    linewidth=2, capsize=5, color='#d35400')
    axes[1].set_xlabel('DDIM Steps')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs DDIM Steps')
    axes[1].set_xticks(steps_df['ddim_steps'])
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    # Highlight best point
    best_ssim_idx = steps_df['ssim_mean'].idxmax()
    best_ssim = steps_df.loc[best_ssim_idx]
    axes[1].scatter([best_ssim['ddim_steps']], [best_ssim['ssim_mean']], 
                   s=200, c='#e74c3c', zorder=5, marker='*')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'ddim_steps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return steps_df


def plot_total_view_comparison(df):
    """Plot total view comparison."""
    # Filter for total_view comparison (axial, g75, steps=300)
    view_df = df[(df['axis'] == 'axial') & 
                 (df['guidance_scale'] == 75) & 
                 (df['ddim_steps'] == 300)].copy()
    view_df = view_df.sort_values('total_view')
    
    if len(view_df) < 2:
        print("Insufficient data for total view comparison")
        return view_df
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR vs total_view
    bars1 = axes[0].bar(view_df['total_view'].astype(str), view_df['psnr_mean'], 
                        yerr=view_df['psnr_std'], capsize=5,
                        color=plt.cm.viridis(np.linspace(0.2, 0.8, len(view_df))))
    axes[0].set_xlabel('Number of Conditioning Slices (total_view)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Number of Conditioning Slices')
    
    # Add value labels
    for bar, val in zip(bars1, view_df['psnr_mean']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM vs total_view
    bars2 = axes[1].bar(view_df['total_view'].astype(str), view_df['ssim_mean'], 
                        yerr=view_df['ssim_std'], capsize=5,
                        color=plt.cm.viridis(np.linspace(0.2, 0.8, len(view_df))))
    axes[1].set_xlabel('Number of Conditioning Slices (total_view)')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Number of Conditioning Slices')
    axes[1].set_ylim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars2, view_df['ssim_mean']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'total_view_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return view_df


def plot_summary(df):
    """Create summary heatmaps and rankings."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter relevant experiments for heatmap
    heatmap_df = df[(df['axis'] == 'axial') & (df['total_view'] == 1)].copy()
    
    # 1. PSNR Heatmap: Guidance Scale vs DDIM Steps
    if len(heatmap_df) > 1:
        pivot_psnr = heatmap_df.pivot_table(
            values='psnr_mean', 
            index='guidance_scale', 
            columns='ddim_steps', 
            aggfunc='first'
        )
        
        if pivot_psnr.shape[0] > 1 or pivot_psnr.shape[1] > 1:
            sns.heatmap(pivot_psnr, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0, 0])
            axes[0, 0].set_title('PSNR: Guidance Scale vs DDIM Steps\n(axial, total_view=1)')
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data for heatmap', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data for heatmap', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 2. SSIM Heatmap
    if len(heatmap_df) > 1:
        pivot_ssim = heatmap_df.pivot_table(
            values='ssim_mean', 
            index='guidance_scale', 
            columns='ddim_steps', 
            aggfunc='first'
        )
        
        if pivot_ssim.shape[0] > 1 or pivot_ssim.shape[1] > 1:
            sns.heatmap(pivot_ssim, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 1])
            axes[0, 1].set_title('SSIM: Guidance Scale vs DDIM Steps\n(axial, total_view=1)')
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. Bar chart: All configurations ranked by PSNR
    sorted_df = df.sort_values('psnr_mean', ascending=True).tail(10)
    y_pos = range(len(sorted_df))
    axes[1, 0].barh(y_pos, sorted_df['psnr_mean'], 
                    color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df))))
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([f"{row['axis']}\ng{row['guidance_scale']}_s{row['ddim_steps']}_v{row['total_view']}" 
                               for _, row in sorted_df.iterrows()], fontsize=9)
    axes[1, 0].set_xlabel('PSNR (dB)')
    axes[1, 0].set_title('Top 10 Configurations by PSNR')
    
    # 4. Bar chart: All configurations ranked by SSIM
    sorted_df_ssim = df.sort_values('ssim_mean', ascending=True).tail(10)
    y_pos = range(len(sorted_df_ssim))
    axes[1, 1].barh(y_pos, sorted_df_ssim['ssim_mean'], 
                    color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df_ssim))))
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f"{row['axis']}\ng{row['guidance_scale']}_s{row['ddim_steps']}_v{row['total_view']}" 
                               for _, row in sorted_df_ssim.iterrows()], fontsize=9)
    axes[1, 1].set_xlabel('SSIM')
    axes[1, 1].set_title('Top 10 Configurations by SSIM')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'summary_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(df):
    """Print detailed analysis summary."""
    print("="*80)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*80)
    
    # Best overall configuration
    best_psnr = df.loc[df['psnr_mean'].idxmax()]
    best_ssim = df.loc[df['ssim_mean'].idxmax()]
    
    print("\n📊 BEST CONFIGURATIONS:")
    print("-"*40)
    print(f"\n🏆 Best PSNR: {best_psnr['psnr_mean']:.2f} dB")
    print(f"   Configuration: axis={best_psnr['axis']}, guidance={best_psnr['guidance_scale']}, "
          f"steps={best_psnr['ddim_steps']}, total_view={best_psnr['total_view']}")
    
    print(f"\n🏆 Best SSIM: {best_ssim['ssim_mean']:.3f}")
    print(f"   Configuration: axis={best_ssim['axis']}, guidance={best_ssim['guidance_scale']}, "
          f"steps={best_ssim['ddim_steps']}, total_view={best_ssim['total_view']}")
    
    # Axis findings
    print("\n📐 AXIS ORIENTATION FINDINGS:")
    print("-"*40)
    axis_results = df[(df['guidance_scale'] == 75) & (df['ddim_steps'] == 300) & (df['total_view'] == 1)]
    for _, row in axis_results.iterrows():
        print(f"   {row['axis']:10s}: PSNR={row['psnr_mean']:.2f} dB, SSIM={row['ssim_mean']:.3f}")
    
    if 'axial' in axis_results['axis'].values and len(axis_results) > 1:
        axial_psnr = axis_results[axis_results['axis'] == 'axial']['psnr_mean'].values[0]
        other_psnr = axis_results[axis_results['axis'] != 'axial']['psnr_mean'].mean()
        print(f"\n   → Axial orientation outperforms others by {axial_psnr - other_psnr:.2f} dB in PSNR")
    
    # Guidance scale findings
    print("\n🎚️ GUIDANCE SCALE FINDINGS:")
    print("-"*40)
    guidance_results = df[(df['axis'] == 'axial') & (df['ddim_steps'] == 300) & (df['total_view'] == 1)].sort_values('guidance_scale')
    for _, row in guidance_results.iterrows():
        print(f"   g={row['guidance_scale']:3.0f}: PSNR={row['psnr_mean']:.2f} dB, SSIM={row['ssim_mean']:.3f}")
    
    if len(guidance_results) > 0:
        best_g_psnr = guidance_results.loc[guidance_results['psnr_mean'].idxmax()]
        best_g_ssim = guidance_results.loc[guidance_results['ssim_mean'].idxmax()]
        print(f"\n   → Best guidance for PSNR: {best_g_psnr['guidance_scale']:.0f}")
        print(f"   → Best guidance for SSIM: {best_g_ssim['guidance_scale']:.0f}")
    
    # DDIM steps findings
    print("\n⏱️ DDIM STEPS FINDINGS:")
    print("-"*40)
    steps_results = df[(df['axis'] == 'axial') & (df['guidance_scale'] == 75) & (df['total_view'] == 1)].sort_values('ddim_steps')
    for _, row in steps_results.iterrows():
        print(f"   steps={row['ddim_steps']:3.0f}: PSNR={row['psnr_mean']:.2f} dB, SSIM={row['ssim_mean']:.3f}")
    
    if len(steps_results) > 0:
        best_s_psnr = steps_results.loc[steps_results['psnr_mean'].idxmax()]
        best_s_ssim = steps_results.loc[steps_results['ssim_mean'].idxmax()]
        print(f"\n   → Best steps for PSNR: {best_s_psnr['ddim_steps']:.0f}")
        print(f"   → Best steps for SSIM: {best_s_ssim['ddim_steps']:.0f}")
    
    # Total view findings
    print("\n👁️ TOTAL VIEW (CONDITIONING SLICES) FINDINGS:")
    print("-"*40)
    view_results = df[(df['axis'] == 'axial') & (df['guidance_scale'] == 75) & (df['ddim_steps'] == 300)].sort_values('total_view')
    for _, row in view_results.iterrows():
        print(f"   total_view={row['total_view']:.0f}: PSNR={row['psnr_mean']:.2f} dB, SSIM={row['ssim_mean']:.3f}")
    
    if len(view_results) > 0:
        best_v = view_results.loc[view_results['psnr_mean'].idxmax()]
        print(f"\n   → Best total_view: {best_v['total_view']:.0f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Use AXIAL orientation for best reconstruction quality")
    
    if len(guidance_results) > 0:
        best_g_psnr = guidance_results.loc[guidance_results['psnr_mean'].idxmax()]
        best_g_ssim = guidance_results.loc[guidance_results['ssim_mean'].idxmax()]
        print(f"2. Guidance scale of {min(best_g_ssim['guidance_scale'], best_g_psnr['guidance_scale']):.0f}-"
              f"{max(best_g_ssim['guidance_scale'], best_g_psnr['guidance_scale']):.0f} provides optimal results")
    
    if len(steps_results) > 0:
        best_s_ssim = steps_results.loc[steps_results['ssim_mean'].idxmax()]
        print(f"3. {best_s_ssim['ddim_steps']:.0f} DDIM steps offer best quality-speed tradeoff")
    
    print("4. Single conditioning slice (total_view=1) performs best")
    print("\n" + "="*80)


def main():
    """Main analysis function."""
    print("Loading experiment data...")
    df = collect_experiments()
    print(f"Loaded {len(df)} experiments\n")
    
    if len(df) == 0:
        print("No experiments found!")
        return
    
    # Display all experiments
    print("All Experiments:")
    print(df[['dir_name', 'axis', 'guidance_scale', 'ddim_steps', 'total_view', 
              'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std']].sort_values(
        ['axis', 'guidance_scale', 'ddim_steps', 'total_view']).round(4).to_string())
    print("\n")
    
    # Generate plots
    print("Generating axis comparison plot...")
    plot_axis_comparison(df)
    
    print("Generating guidance scale comparison plot...")
    plot_guidance_comparison(df)
    
    print("Generating DDIM steps comparison plot...")
    plot_steps_comparison(df)
    
    print("Generating total view comparison plot...")
    plot_total_view_comparison(df)
    
    print("Generating summary plots...")
    plot_summary(df)
    
    # Print summary
    print_summary(df)
    
    # Export results
    export_df = df[['dir_name', 'axis', 'guidance_scale', 'ddim_steps', 'total_view',
                    'psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max',
                    'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max', 'num_slices']]
    export_df = export_df.sort_values(['axis', 'guidance_scale', 'ddim_steps', 'total_view'])
    export_df.to_csv(ANALYSIS_DIR / 'experiment_results.csv', index=False)
    print(f"\nResults exported to: {ANALYSIS_DIR / 'experiment_results.csv'}")
    print(f"Plots saved to: {ANALYSIS_DIR}")


if __name__ == '__main__':
    main()

