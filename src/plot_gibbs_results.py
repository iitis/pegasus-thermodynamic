#!/usr/bin/env python3
"""
Plot Gibbs Sampling Results

Creates a three-panel figure showing:
1. Energy trajectory during burn-in phase
2. Magnetization trajectory during burn-in phase
3. Autocorrelation functions for all collected samples

Usage:
    python plot_gibbs_results.py --results_dir <path_to_results_directory>

Example:
    python plot_gibbs_results.py \
        --results_dir ./gibbs_results/P6_CON_1/BETA_3.000
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

# Set matplotlib style
rcParams['figure.figsize'] = (14, 5)
rcParams['font.size'] = 10


def load_burn_in_energy(results_dir: str) -> pd.DataFrame:
    """Load burn-in energy trajectory from CSV."""
    energy_path = os.path.join(results_dir, 'burn_in_energy.csv')
    if not os.path.exists(energy_path):
        raise FileNotFoundError(f"Burn-in energy file not found: {energy_path}")
    return pd.read_csv(energy_path)


def load_burn_in_magnetization(results_dir: str) -> pd.DataFrame:
    """Load burn-in magnetization trajectory from CSV."""
    mag_path = os.path.join(results_dir, 'burn_in_magnetization.csv')
    if not os.path.exists(mag_path):
        raise FileNotFoundError(f"Burn-in magnetization file not found: {mag_path}")
    return pd.read_csv(mag_path)


def load_sampling_diagnostics(results_dir: str) -> pd.DataFrame:
    """
    Load sampling diagnostics from CSV.
    Expected columns: sample_idx,sweep,energy,magnetization,tau_raw_energy,tau_eff_energy,tau_raw_mag,sampling_interval,beta,sweeps_waited
    """
    diag_path = os.path.join(results_dir, 'sampling_diagnostics.csv')
    if not os.path.exists(diag_path):
        # Fallback to empty if not found (maybe fixed sampling without diagnostics?)
        print(f"Warning: Diagnostics file not found: {diag_path}")
        return pd.DataFrame()
    return pd.read_csv(diag_path)


def plot_results(results_dir: str, output_path: Optional[str] = None) -> None:
    """
    Create a three-panel plot of burn-in energy, burn-in magnetization, and sampling diagnostics.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory from gibbs_sampling_runner
    output_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    # Load data
    print(f"Loading data from: {results_dir}")
    burn_in_df = load_burn_in_energy(results_dir)
    burn_in_mag_df = load_burn_in_magnetization(results_dir)
    diag_df = load_sampling_diagnostics(results_dir)
    
    print(f"Loaded burn-in energy with {len(burn_in_df)} sweeps")
    print(f"Loaded burn-in magnetization with {len(burn_in_mag_df)} sweeps")
    print(f"Loaded sampling diagnostics with {len(diag_df)} samples")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ====================================================================
    # Left panel: Burn-in energy trajectory
    # ====================================================================
    ax1.plot(burn_in_df['sweep'], burn_in_df['energy'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Sweep')
    ax1.set_ylabel('Energy')
    ax1.set_title('Burn-in Energy Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Add some statistics
    if not burn_in_df.empty:
        mean_energy = burn_in_df['energy'].mean()
        ax1.axhline(mean_energy, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {mean_energy:.4f}')
        ax1.legend()

    # ====================================================================
    # Middle panel: Burn-in magnetization trajectory
    # ====================================================================
    ax2.plot(burn_in_mag_df['sweep'], burn_in_mag_df['magnetization'], 'g-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Sweep')
    ax2.set_ylabel('Magnetization')
    ax2.set_title('Burn-in Magnetization Trajectory')
    ax2.grid(True, alpha=0.3)
    
    if not burn_in_mag_df.empty:
        mean_mag = burn_in_mag_df['magnetization'].mean()
        ax2.axhline(mean_mag, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {mean_mag:.4f}')
        ax2.legend()
    
    # ====================================================================
    # Right panel: Sampling Diagnostics (tau_int)
    # ====================================================================
    if not diag_df.empty:
        # Plot Estimated Tau Energy
        ax3.plot(diag_df['sample_idx'], diag_df['tau_eff_energy'], 'o-', markersize=4, color='purple', label=r'Effective $\tau$ (Energy)')
        if 'tau_raw_energy' in diag_df.columns:
             ax3.plot(diag_df['sample_idx'], diag_df['tau_raw_energy'], ':', color='purple', alpha=0.5, label=r'Raw $\tau$ (Energy)')
        
        # Plot Weighted Sampling Interval
        ax3.plot(diag_df['sample_idx'], diag_df['sampling_interval'], 'x--', markersize=4, color='orange', alpha=0.7, label='Sampling Interval')
        
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Sweeps')
        ax3.set_title(r'Autocorrelation $\tau_{int}$ & Sampling Interval')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add text for mean tau
        mean_tau = diag_df['tau_eff_energy'].mean()
        ax3.text(0.05, 0.95, f'Mean $\\tau_{{eff}}$: {mean_tau:.2f}', 
                 transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No Sampling Diagnostics Available', ha='center', va='center')
        ax3.set_title('Sampling Diagnostics')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Gibbs Sampling Results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the results directory from gibbs_sampling_runner")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the figure. If not provided, displays the plot.")
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    
    # Create plot
    plot_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
