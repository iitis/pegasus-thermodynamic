#!/usr/bin/env python3
"""
Validation Script for Gibbs Sampling Results

This script validates pre-generated Gibbs samples by:
1. Loading all samples from a results directory
2. Loading corresponding instance files
3. Recomputing energy and magnetization for each sample
4. Comparing with logged values from diagnostics

Usage:
    python validate_gibbs_samples.py --gibbs_results_dir ./gibbs_results --instances_dir ./data/instances/subpegasus_native
    
Example:
    python validate_gibbs_samples.py \
        --gibbs_results_dir src/gibbs_results \
        --instances_dir data/instances/subpegasus_native
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import dimod
from tqdm import tqdm

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_instance(instance_path: str) -> Tuple[Dict, Dict]:
    """
    Load an Ising model instance from file.
    
    Parameters:
    -----------
    instance_path : str
        Path to instance file (.pkl)
    
    Returns:
    --------
    h : dict
        Linear biases (fields)
    J : dict
        Quadratic biases (couplings)
    """
    with open(instance_path, 'rb') as f:
        data = pickle.load(f)
        
    # Handle different pickle formats
    if isinstance(data, list) and len(data) == 2:
        h, J = data[0], data[1]
    elif isinstance(data, tuple) and len(data) == 2:
        h, J = data
    elif isinstance(data, dict):
        h = data.get('h', {})
        J = data.get('J', {})
    else:
        raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    return h, J


def compute_energy_bqm(spins: dict, h: dict, J: dict) -> float:
    """
    Compute energy using dimod BQM.
    
    Parameters:
    -----------
    spins : dict
        Spin configuration
    h : dict
        Linear biases
    J : dict
        Quadratic biases
    
    Returns:
    --------
    energy : float
    """
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    return float(bqm.energy(spins))


def compute_magnetization(spins: dict) -> float:
    """
    Compute magnetization (mean spin value).
    
    Parameters:
    -----------
    spins : dict
        Spin configuration
    
    Returns:
    --------
    magnetization : float
    """
    return float(np.mean(list(spins.values())))


def sample_array_to_dict(sample: np.ndarray, node_order: np.ndarray) -> dict:
    """
    Convert sample array to dictionary.
    
    Parameters:
    -----------
    sample : np.ndarray
        1D array of spin values
    node_order : np.ndarray
        Order of nodes
    
    Returns:
    --------
    spins : dict
    """
    return {int(node): int(spin) for node, spin in zip(node_order, sample)}


def validate_instance_beta(gibbs_dir: Path, instance_basename: str, beta: float,
                          instances_dir: str, tolerance: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Validate Gibbs samples for a specific instance and beta value.
    
    Parameters:
    -----------
    gibbs_dir : Path
        Path to the Gibbs results directory for this instance/beta
    instance_basename : str
        Instance name (e.g., 'P6_CON_1')
    beta : float
        Inverse temperature
    instances_dir : str
        Directory containing instance files
    tolerance : float
        Tolerance for energy/magnetization comparison
    
    Returns:
    --------
    success : bool
        True if validation passed
    stats : dict
        Statistics about the validation
    """
    stats = {
        'instance': instance_basename,
        'beta': beta,
        'num_samples': 0,
        'energy_matches': 0,
        'mag_matches': 0,
        'max_energy_diff': 0.0,
        'max_mag_diff': 0.0,
        'mean_energy_diff': 0.0,
        'mean_mag_diff': 0.0,
        'errors': []
    }
    
    # Load samples
    samples_path = gibbs_dir / 'gibbs_samples.npz'
    if not samples_path.exists():
        stats['errors'].append(f"Samples file not found: {samples_path}")
        return False, stats
    
    try:
        data = np.load(samples_path, allow_pickle=True)
        samples = data['samples']
        node_order = data['node_order']
        stats['num_samples'] = len(samples)
    except Exception as e:
        stats['errors'].append(f"Error loading samples: {e}")
        return False, stats
    
    # Load diagnostics
    diagnostics_path = gibbs_dir / 'sampling_diagnostics.csv'
    if not diagnostics_path.exists():
        stats['errors'].append(f"Diagnostics file not found: {diagnostics_path}")
        return False, stats
    
    try:
        diagnostics_df = pd.read_csv(diagnostics_path)
    except Exception as e:
        stats['errors'].append(f"Error loading diagnostics: {e}")
        return False, stats
    
    # Check sample count matches
    if len(samples) != len(diagnostics_df):
        stats['errors'].append(
            f"Sample count mismatch: {len(samples)} samples vs {len(diagnostics_df)} diagnostic entries"
        )
        return False, stats
    
    # Load instance
    instance_path = Path(instances_dir) / f"{instance_basename}.pkl"
    if not instance_path.exists():
        stats['errors'].append(f"Instance file not found: {instance_path}")
        return False, stats
    
    try:
        h, J = load_instance(str(instance_path))
    except Exception as e:
        stats['errors'].append(f"Error loading instance: {e}")
        return False, stats
    
    # Validate each sample
    energy_diffs = []
    mag_diffs = []
    
    for i in range(len(samples)):
        sample = samples[i]
        spins_dict = sample_array_to_dict(sample, node_order)
        
        # Compute energy and magnetization
        computed_energy = compute_energy_bqm(spins_dict, h, J)
        computed_mag = compute_magnetization(spins_dict)
        
        # Get logged values
        logged_energy = diagnostics_df.iloc[i]['energy']
        logged_mag = diagnostics_df.iloc[i]['magnetization']
        
        # Compare
        energy_diff = abs(computed_energy - logged_energy)
        mag_diff = abs(computed_mag - logged_mag)
        
        energy_diffs.append(energy_diff)
        mag_diffs.append(mag_diff)
        
        if energy_diff <= tolerance:
            stats['energy_matches'] += 1
        
        if mag_diff <= tolerance:
            stats['mag_matches'] += 1
    
    # Compute statistics
    stats['max_energy_diff'] = np.max(energy_diffs)
    stats['max_mag_diff'] = np.max(mag_diffs)
    stats['mean_energy_diff'] = np.mean(energy_diffs)
    stats['mean_mag_diff'] = np.mean(mag_diffs)
    
    # Check if validation passed
    success = (stats['energy_matches'] == stats['num_samples'] and 
               stats['mag_matches'] == stats['num_samples'])
    
    if not success:
        if stats['energy_matches'] < stats['num_samples']:
            stats['errors'].append(
                f"Energy mismatch in {stats['num_samples'] - stats['energy_matches']} samples "
                f"(max diff: {stats['max_energy_diff']:.2e})"
            )
        if stats['mag_matches'] < stats['num_samples']:
            stats['errors'].append(
                f"Magnetization mismatch in {stats['num_samples'] - stats['mag_matches']} samples "
                f"(max diff: {stats['max_mag_diff']:.2e})"
            )
    
    return success, stats


def find_all_results(gibbs_results_dir: str) -> list:
    """
    Find all instance/beta result directories.
    
    Parameters:
    -----------
    gibbs_results_dir : str
        Root directory containing Gibbs results
    
    Returns:
    --------
    results : list of tuples
        List of (instance_dir_path, instance_basename, beta) tuples
    """
    results = []
    gibbs_path = Path(gibbs_results_dir)
    
    if not gibbs_path.exists():
        print(f"{Colors.FAIL}Error: Gibbs results directory not found: {gibbs_results_dir}{Colors.ENDC}")
        return results
    
    # Iterate through instance directories
    for instance_dir in gibbs_path.iterdir():
        if not instance_dir.is_dir():
            continue
        
        instance_basename = instance_dir.name
        
        # Iterate through beta subdirectories
        for beta_dir in instance_dir.iterdir():
            if not beta_dir.is_dir():
                continue
            
            # Extract beta value from directory name (e.g., "BETA_1.000")
            if beta_dir.name.startswith("BETA_"):
                try:
                    beta = float(beta_dir.name.replace("BETA_", ""))
                    results.append((beta_dir, instance_basename, beta))
                except ValueError:
                    continue
    
    return results


def print_summary(all_stats: list):
    """
    Print summary of validation results.
    
    Parameters:
    -----------
    all_stats : list
        List of stats dictionaries from validation
    """
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.ENDC}")
    print("=" * 80)
    
    total_validated = len(all_stats)
    passed = sum(1 for s in all_stats if not s['errors'])
    failed = total_validated - passed
    total_samples = sum(s['num_samples'] for s in all_stats)
    
    print(f"\nTotal configurations validated: {total_validated}")
    print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")
    print(f"Total samples checked: {total_samples}")
    
    if all_stats:
        all_energy_diffs = [s['max_energy_diff'] for s in all_stats]
        all_mag_diffs = [s['max_mag_diff'] for s in all_stats]
        
        print(f"\nOverall energy difference:")
        print(f"  Max: {np.max(all_energy_diffs):.2e}")
        print(f"  Mean: {np.mean([s['mean_energy_diff'] for s in all_stats]):.2e}")
        
        print(f"\nOverall magnetization difference:")
        print(f"  Max: {np.max(all_mag_diffs):.2e}")
        print(f"  Mean: {np.mean([s['mean_mag_diff'] for s in all_stats]):.2e}")
    
    # Print failures
    if failed > 0:
        print(f"\n{Colors.WARNING}Failed validations:{Colors.ENDC}")
        for s in all_stats:
            if s['errors']:
                print(f"\n  {Colors.FAIL}✗{Colors.ENDC} {s['instance']} (β={s['beta']:.3f})")
                for error in s['errors']:
                    print(f"    - {error}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Gibbs sampling results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--gibbs_results_dir", type=str, required=True,
                        help="Directory containing Gibbs sampling results")
    parser.add_argument("--instances_dir", type=str, required=True,
                        help="Directory containing instance files")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Tolerance for energy/magnetization comparison")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each validation")
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print("GIBBS SAMPLING VALIDATION")
    print("=" * 80)
    print(f"{Colors.ENDC}")
    print(f"Gibbs results directory: {args.gibbs_results_dir}")
    print(f"Instances directory: {args.instances_dir}")
    print(f"Tolerance: {args.tolerance:.2e}")
    print()
    
    # Find all results to validate
    results_to_validate = find_all_results(args.gibbs_results_dir)
    
    if not results_to_validate:
        print(f"{Colors.FAIL}No results found to validate!{Colors.ENDC}")
        sys.exit(1)
    
    print(f"Found {len(results_to_validate)} configurations to validate\n")
    
    # Validate each result
    all_stats = []
    
    for gibbs_dir, instance_basename, beta in tqdm(results_to_validate, desc="Validating"):
        success, stats = validate_instance_beta(
            gibbs_dir, instance_basename, beta, args.instances_dir, args.tolerance
        )
        all_stats.append(stats)
        
        if args.verbose:
            if success:
                print(f"{Colors.OKGREEN}✓{Colors.ENDC} {instance_basename} (β={beta:.3f}): "
                      f"{stats['num_samples']} samples validated")
            else:
                print(f"{Colors.FAIL}✗{Colors.ENDC} {instance_basename} (β={beta:.3f}): "
                      f"Validation failed")
                for error in stats['errors']:
                    print(f"  {error}")
    
    # Print summary
    print_summary(all_stats)
    
    # Exit with appropriate code
    failed_count = sum(1 for s in all_stats if s['errors'])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
