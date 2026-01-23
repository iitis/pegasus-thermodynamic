#!/usr/bin/env python3
"""
Gibbs Sampling Runner Script

This script performs Gibbs sampling with adaptive burn-in and sampling phases.
It monitors energy during burn-in and autocorrelation during sampling.

Usage:
    python gibbs_sampling_runner.py --instance_path <path> --beta <value> [options]

Example:
    python gibbs_sampling_runner.py \
        --instance_path ./data/instances/subpegasus_native/P6_CON_1.pkl \
        --beta 1.0 \
        --num_burn_in_sweeps 1000 \
        --num_sampling_sweeps 100 \
        --num_samples 1000 \
        --output_dir ./gibbs_results
"""

import argparse
import logging
import os
import pickle
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from dimod import BinaryQuadraticModel

# Set up random number generator
rng = np.random.default_rng()

# Configure logging
def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Set up logging with both file and console handlers."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger("gibbs_sampling")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                        datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, f"gibbs_sampling_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(funcName)s | %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def stable_sigmoid(x) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = np.exp(-x)
        return z / (1.0 + z)
    else:
        z = np.exp(x)
        return 1.0 / (1.0 + z)


def get_neighbourhood_from_bqm(bqm: BinaryQuadraticModel) -> dict:
    """Build neighbourhood dictionary from BQM adjacency."""
    neighbourhood = {node: list(bqm.adj[node].keys()) for node in bqm.variables}
    return neighbourhood


def compute_energy_bqm(spins: dict, bqm: BinaryQuadraticModel) -> float:
    """Compute total energy of the spin configuration using BQM."""
    return float(bqm.energy(spins))


def compute_magnetization(spins: dict) -> float:
    """Compute magnetization (mean spin value)."""
    return float(np.mean(list(spins.values())))



def next_pow_two(n: int) -> int:
    """Returns the next power of two greater than or equal to n."""
    i = 1
    while i < n:
        i = i << 1
    return i

def compute_autocorrelation_fft(x: np.ndarray):
    """
    Compute normalized autocorrelation using FFT.

    Returns
    -------
    acf : np.ndarray or None
        Normalized autocorrelation for non-negative lags,
        or None if variance is zero (frozen signal).
    """
    n = len(x)
    var = np.var(x)

    if var == 0.0:
        # signal that system is most likely frozen
        return None  

    n_fft = next_pow_two(2 * n)
    x = x - np.mean(x)

    f_x = np.fft.fft(x, n=n_fft)
    acf = np.fft.ifft(f_x * np.conj(f_x)).real[:n]

    return acf / acf[0]

def tau_cap(beta: float) -> Optional[float]:
    """
    Temperature-dependent cap on integrated autocorrelation time to prevent freezing.
    """
    if beta > 1.5:
        return 2000.0
    elif beta > 0.8:
        return 3000.0
    else:
        return None


def compute_integrated_autocorrelation_time(
    x: np.ndarray,
    max_lag: int | None = None,
):
    """
    Compute integrated autocorrelation time.

    Returns
    -------
    tau : float
        Estimated τ_int, or np.inf if undefined (frozen chain).
    """
    acf = compute_autocorrelation_fft(x)

    if acf is None:
        return np.inf  # <-- frozen regime

    if max_lag is None:
        max_lag = len(acf)

    tau = 0.5
    for k in range(1, max_lag):
        if acf[k] <= 0:
            break
        tau += acf[k]

    return tau


def gibbs_sweep(spins: dict, bqm: BinaryQuadraticModel, neighbourhood: dict, 
                beta: float, nodes: list) -> dict:
    """
    Perform one Gibbs sampling sweep (N single-spin updates).
    
    Parameters:
    -----------
    spins : dict
        Current spin configuration
    bqm : BinaryQuadraticModel
        The Ising model as a BQM
    neighbourhood : dict
        Precomputed neighbour lists
    beta : float
        Inverse temperature
    nodes : list
        List of spin indices
    
    Returns:
    --------
    spins : dict
        Updated spin configuration
    """
    N = len(nodes)
    
    for _ in range(N):
        idx = nodes[rng.integers(0, N)]
        
        # Calculate local field contribution from neighbours
        # Using BQM adjacency for coupling values
        interaction_sum = 0.0
        for j, coupling in bqm.adj[idx].items():
            interaction_sum += coupling * spins[j]
        
        # Get the linear bias (field) from BQM
        h_idx = bqm.get_linear(idx)
        
        # Energy difference for spin flip
        deltaE = 2 * (interaction_sum + h_idx)
        
        # Gibbs update probability
        prob_plus_1 = stable_sigmoid(beta * deltaE)
        
        # Sample new spin
        spins[idx] = rng.choice([-1, 1], p=[1 - prob_plus_1, prob_plus_1])
    
    return spins


def run_burn_in_phase(spins: dict, bqm: BinaryQuadraticModel, neighbourhood: dict,
                      beta: float, nodes: list, num_burn_in_sweeps: Optional[int],
                      output_dir: str, logger: logging.Logger,
                      energy_check_window: int = 100,
                      convergence_threshold: float = 0.01,
                      post_burn_in_sweeps: int = 1000) -> Tuple[dict, pd.DataFrame]:
    """
    Run burn-in phase with energy and magnetization monitoring.
    Also enforces a strict stationarity buffer after convergence.
    """
    logger.info("=" * 60)
    logger.info("BURN-IN PHASE")
    logger.info("=" * 60)
    
    energy_history = []
    magnetization_history = []
    sweep_indices = []
    
    # We will use a deque for online convergence checking to avoid storing everything if runs are long
    # But we also want to save history, so we'll keep the lists but optimize checking.
    
    sweeps_done = 0
    converged = False
    
    if num_burn_in_sweeps is not None:
        # Fixed burn-in
        target_sweeps = num_burn_in_sweeps
        logger.info(f"Running fixed burn-in: {target_sweeps} sweeps")
        
        pbar = tqdm(range(target_sweeps), desc="Burn-in", unit="sweep")
        for i in pbar:
            spins = gibbs_sweep(spins, bqm, neighbourhood, beta, nodes)
            e = compute_energy_bqm(spins, bqm)
            m = compute_magnetization(spins)
            
            energy_history.append(e)
            magnetization_history.append(m)
            sweep_indices.append(sweeps_done)
            sweeps_done += 1
            
            if (i + 1) % 100 == 0:
                pbar.set_description(f"Burn-in | E: {e:.4f} | M: {m:.4f}")
        
        converged = True # Assumed for fixed
        logger.info(f"Fixed burn-in complete.")

    else:
        # Adaptive burn-in
        logger.info(f"Running adaptive burn-in (window={energy_check_window}, thresh={convergence_threshold})")
        logger.info("Monitoring Energy and Magnetization mean/std stability.")
        
        min_sweeps = 2 * energy_check_window
        max_sweeps = 200000 # Safety cap
        
        from collections import deque
        recent_E = deque(maxlen=energy_check_window)
        prev_E = deque(maxlen=energy_check_window)
        recent_M = deque(maxlen=energy_check_window)
        prev_M = deque(maxlen=energy_check_window)
        
        pbar = tqdm(total=max_sweeps, desc="Adaptive Burn-in", unit="sweep")
        
        while not converged and sweeps_done < max_sweeps:
            spins = gibbs_sweep(spins, bqm, neighbourhood, beta, nodes)
            e = compute_energy_bqm(spins, bqm)
            m = compute_magnetization(spins)
            
            energy_history.append(e)
            magnetization_history.append(m)
            sweep_indices.append(sweeps_done)
            
            # Update windows
            recent_E.append(e)
            recent_M.append(m)
            if sweeps_done >= energy_check_window:
                # Keep prev window lagging by check_window
                prev_E.append(energy_history[-energy_check_window - 1])
                prev_M.append(magnetization_history[-energy_check_window - 1])
            
            sweeps_done += 1
            pbar.update(1)
            
            if sweeps_done % 100 == 0:
                pbar.set_description(f"Adap Burn-in | E: {e:.4f} | M: {m:.4f}")
                
            if sweeps_done >= min_sweeps and sweeps_done % 100 == 0:
                # Check convergence
                curr_E_mean, curr_E_std = np.mean(recent_E), np.std(recent_E)
                prev_E_mean, prev_E_std = np.mean(prev_E), np.std(prev_E)
                
                curr_M_mean, curr_M_std = np.mean(recent_M), np.std(recent_M)
                prev_M_mean, prev_M_std = np.mean(prev_M), np.std(prev_M)
                
                # Relative change
                def rel_change(curr, prev):
                    if abs(prev) < 1e-9: return abs(curr - prev)
                    return abs(curr - prev) / abs(prev)
                
                d_E_mean = rel_change(curr_E_mean, prev_E_mean)
                d_E_std = rel_change(curr_E_std, prev_E_std)
                d_M_mean = rel_change(curr_M_mean, prev_M_mean)
                d_M_std = rel_change(curr_M_std, prev_M_std)
                
                if (d_E_mean < convergence_threshold and d_E_std < convergence_threshold and
                    d_M_mean < convergence_threshold and d_M_std < convergence_threshold):
                    converged = True
                    logger.info(f"Converged at sweep {sweeps_done}")
                    logger.info(f"dE_mean: {d_E_mean:.2e}, dE_std: {d_E_std:.2e}")
                    logger.info(f"dM_mean: {d_M_mean:.2e}, dM_std: {d_M_std:.2e}")
        
        pbar.close()
        if not converged:
             logger.warning("Burn-in reached max_sweeps without full convergence.")

    # STATIONARITY BUFFER
    # Strictly throw away these samples, no diagnostics
    if post_burn_in_sweeps > 0:
        logger.info(f"Running buffer: {post_burn_in_sweeps} sweeps")
        # No recording
        pbar_buffer = tqdm(range(post_burn_in_sweeps), desc="Buffer", unit="sweep")
        for i in pbar_buffer:
            spins = gibbs_sweep(spins, bqm, neighbourhood, beta, nodes)
            sweeps_done += 1
            if (i + 1) % 100 == 0:
                m = compute_magnetization(spins)
                pbar_buffer.set_description(f"Buffer | M: {m:.4f}")
            
    # Save Burn-in History
    energy_df = pd.DataFrame({'sweep': sweep_indices, 'energy': energy_history})
    # Also save mag for debugging
    pd.DataFrame({'sweep': sweep_indices, 'magnetization': magnetization_history}).to_csv(
        os.path.join(output_dir, 'burn_in_magnetization.csv'), index=False)
    
    energy_csv_path = os.path.join(output_dir, 'burn_in_energy.csv')
    energy_df.to_csv(energy_csv_path, index=False)
    
    return spins, energy_df



def run_sampling_phase(spins: dict, bqm: BinaryQuadraticModel, neighbourhood: dict,
                       beta: float, nodes: list, num_sampling_sweeps: Optional[int],
                       num_samples: int, output_dir: str, logger: logging.Logger,
                       safety_factor: int = 4) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Run sampling phase with strictly enforced robust energy-based adaptive sampling.
    """
    logger.info("=" * 60)
    logger.info("SAMPLING PHASE (ROBUST | ENERGY-CONTROLLED)")
    logger.info("=" * 60)
    
    from collections import deque
    
    samples = []
    node_list = list(nodes)
    
    # 3. Rolling stationary windows
    # Default window size 500 as robust baseline
    acf_window_size = 500 
    energy_window = deque(maxlen=acf_window_size)
    mag_window = deque(maxlen=acf_window_size)
    
    # State tracking
    sweeps_since_last_sample = 0
    samples_collected = 0
    total_sweeps = 0
    
    # Initial estimates
    current_tau_energy = 10.0
    current_tau_eff = 10.0
    tau_raw_mag = 0.0
    
    # Sampling interval logic
    use_fixed = num_sampling_sweeps is not None
    if use_fixed:
        required_interval = num_sampling_sweeps
        logger.info(f"Using fixed sampling interval: {required_interval}")
    else:
        # Initial conservative interval until window is full
        # We enforce at least one full window before first sample
        required_interval = acf_window_size + 1
        logger.info(f"Using adaptive sampling (Energy-based)")
        logger.info(f"ACF Window Size: {acf_window_size}")

    # Diagnostics file
    diag_csv_path = os.path.join(output_dir, 'sampling_diagnostics.csv')
    with open(diag_csv_path, 'w') as f:
        # 10. Metadata & logging
        f.write("sample_idx,sweep,energy,magnetization,tau_raw_energy,tau_eff_energy,tau_raw_mag,sampling_interval,beta,sweeps_waited\n")

    pbar = tqdm(total=num_samples, desc="Sampling", unit="sample")
    
    # Flag to indicate if we have filled the window at least once
    window_filled = False

    while samples_collected < num_samples:
        # 1. Perform Sweep
        spins = gibbs_sweep(spins, bqm, neighbourhood, beta, nodes)
        total_sweeps += 1
        sweeps_since_last_sample += 1
        
        # 10. Log/Store (Observables)
        e = compute_energy_bqm(spins, bqm)
        m = compute_magnetization(spins)
        
        energy_window.append(e)
        mag_window.append(m)
        
        # Check Window Status
        if not window_filled and len(energy_window) == acf_window_size:
            window_filled = True
            logger.info("ACF Window filled. Starting adaptive estimation.")
            
        # 4. Adaptive Logic (only if not fixed and window is full)
        if not use_fixed and window_filled:
            # We can update tau periodically. 
            # To be robust and responsive, update every sweep or frequently. 
            # Updating every sweep is expensive if window is large, but for 500 it's fine.
            # Let's update every 50 sweeps to balance responsiveness and perf.
            
            if total_sweeps % 50 == 0:
                # 5. Energy-based tau estimation
                e_arr = np.array(energy_window)
                tau_raw_energy = compute_integrated_autocorrelation_time(e_arr)
                frozen = not np.isfinite(tau_raw_energy)
                
                # 6. Temperature-aware tau cap
                cap = tau_cap(beta)
                if frozen:
                    # Frozen regime: τ is undefined, not just "large"
                    current_tau_eff = cap if cap is not None else np.inf
                else:
                    if cap is not None:
                        current_tau_eff = min(tau_raw_energy, cap)
                    else:
                        current_tau_eff = tau_raw_energy

                    
                # 9. Magnetization tau -> diagnostics only
                m_arr = np.array(mag_window)
                tau_raw_mag = compute_integrated_autocorrelation_time(m_arr)
                
                current_tau_energy = tau_raw_energy
                
                # 7. Sampling rule
                # required_interval = int(np.ceil(safety_factor * current_tau_eff))
                # required_interval = max(required_interval, 1) # Safety floor

                # 7. Sampling rule
                if frozen:
                    # Do NOT gate sampling on τ when chain is frozen
                    required_interval = 10
                else:
                    required_interval = int(np.ceil(safety_factor * current_tau_eff))
                    required_interval = max(required_interval, 100)  # at least 100 to be safe
                
                # pbar.set_description(f"Sampling | tau_eff: {current_tau_eff:.1f} | int: {required_interval}")
                if frozen:
                    pbar.set_description("Sampling | frozen phase (τ undefined)")
                else:
                    pbar.set_description(f"Sampling | tau_eff: {current_tau_eff:.1f} | int: {required_interval}")

        elif not use_fixed and not window_filled:
            # Enforce conservative wait
            pbar.set_description(f"Filling buffers: {len(energy_window)}/{acf_window_size}")

        # 7. Sampling Decision Application
        # Force wait if window not full (implicit rule 4: no sampling before window full)
        ready_to_sample = False
        if use_fixed:
            if sweeps_since_last_sample >= required_interval:
                ready_to_sample = True
        else:
            if window_filled and sweeps_since_last_sample >= required_interval:
                ready_to_sample = True
                
        if ready_to_sample:
            # Take Sample
            sample = np.array([spins[node] for node in node_list])
            samples.append(sample)
            
            # Log Diagnostics
            with open(diag_csv_path, 'a') as f:
                f.write(f"{samples_collected},{total_sweeps},{e:.4f},{m:.4f},"
                        f"{current_tau_energy:.4f},{current_tau_eff:.4f},{tau_raw_mag:.4f},"
                        f"{required_interval},{beta},{sweeps_since_last_sample}\n")
            
            samples_collected += 1
            sweeps_since_last_sample = 0
            pbar.update(1)
            
    pbar.close()
    
    return np.array(samples), pd.DataFrame()



def load_instance(instance_path: str, logger: logging.Logger) -> BinaryQuadraticModel:
    """
    Load an Ising model instance from file and create a BQM.
    
    Parameters:
    -----------
    instance_path : str
        Path to instance file (.pkl or .json)
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    bqm : BinaryQuadraticModel
        The loaded Ising model as a BQM
    """
    logger.info(f"Loading instance from: {instance_path}")
    
    if instance_path.endswith('.pkl'):
        with open(instance_path, 'rb') as f:
            data = pickle.load(f)
            # Handle different pickle formats
            if isinstance(data, list) and len(data) == 2:
                # Format: [h, J]
                h, J = data[0], data[1]
            elif isinstance(data, tuple) and len(data) == 2:
                # Format: (h, J)
                h, J = data
            elif isinstance(data, dict):
                # Format: {'h': h, 'J': J}
                h = data.get('h', {})
                J = data.get('J', {})
            else:
                raise ValueError(f"Unexpected pickle format: {type(data)}")
    elif instance_path.endswith('.json'):
        import json
        with open(instance_path, 'r') as f:
            data = json.load(f)
        h = {int(k): v for k, v in data.get('h', {}).items()}
        J = {tuple(map(int, k.strip('()').split(','))): v for k, v in data.get('J', {}).items()}
    else:
        raise ValueError(f"Unsupported file format: {instance_path}")
    
    # Create BQM from h and J
    bqm = BinaryQuadraticModel.from_ising(h, J)
    
    logger.info(f"Instance loaded successfully")
    logger.info(f"  Number of spins: {bqm.num_variables}")
    logger.info(f"  Number of couplings: {bqm.num_interactions}")
    
    # Log field and coupling statistics
    h_values = np.array([bqm.get_linear(v) for v in bqm.variables])
    J_values = np.array([bias for (u, v, bias) in bqm.iter_quadratic()])
    if len(h_values) > 0:
        logger.info(f"  Field range: [{h_values.min():.4f}, {h_values.max():.4f}]")
    if len(J_values) > 0:
        logger.info(f"  Coupling range: [{J_values.min():.4f}, {J_values.max():.4f}]")
    
    return bqm


def main():
    # ==========================================================================
    # PARAMETERS
    # ==========================================================================
    parser = argparse.ArgumentParser(
        description="Gibbs Sampling with Adaptive Burn-in and Sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance file (.pkl or .json)")
    parser.add_argument("--beta", type=float, required=True,
                        help="Inverse temperature for sampling")
    
    # Burn-in parameters
    parser.add_argument("--num_burn_in_sweeps", type=int, default=None,
                        help="Number of burn-in sweeps. None for adaptive burn-in.")
    parser.add_argument("--burn_in_window", type=int, default=100,
                        help="Window size for adaptive burn-in convergence check")
    parser.add_argument("--burn_in_threshold", type=float, default=0.01,
                        help="Convergence threshold for adaptive burn-in")
    parser.add_argument("--post_burn_in_sweeps", type=int, default=1000,
                        help="Number of additional sweeps after burn-in convergence (adaptive mode)")
    
    # Sampling parameters
    parser.add_argument("--num_sampling_sweeps", type=int, default=None,
                        help="Sweeps between samples. None for adaptive (ACF-based).")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to collect")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./gibbs_results",
                        help="Output directory for results")
    parser.add_argument("--output_prefix", type=str, default="gibbs",
                        help="Prefix for output files")
   
    # Other parameters
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # ==========================================================================
    # SETUP
    # ==========================================================================
    
    # Create output directory with instance basename and beta subdirectory
    instance_basename = os.path.splitext(os.path.basename(args.instance_path))[0]
    beta_subdir = f"BETA_{args.beta:.3f}"
    output_dir = os.path.join(args.output_dir, instance_basename, beta_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, args.log_level)
    
    # Set random seed
    global rng
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
    # Log all parameters
    logger.info("=" * 60)
    logger.info("GIBBS SAMPLING RUNNER")
    logger.info("=" * 60)
    logger.info("Parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # ==========================================================================
    # LOAD INSTANCE
    # ==========================================================================
    
    bqm = load_instance(args.instance_path, logger)
    
    # Precompute structures using BQM
    neighbourhood = get_neighbourhood_from_bqm(bqm)
    nodes = list(bqm.variables)
    N = bqm.num_variables
    
    # ==========================================================================
    # INITIALIZE SPINS
    # ==========================================================================
    
    logger.info("Initializing random spin configuration...")
    spins = OrderedDict({v: rng.choice([-1, 1]) for v in bqm.variables})
    
    initial_energy = compute_energy_bqm(spins, bqm)
    initial_mag = compute_magnetization(spins)
    logger.info(f"Initial state:")
    logger.info(f"  Energy: {initial_energy:.4f}")
    logger.info(f"  Magnetization: {initial_mag:.4f}")
    
    # ==========================================================================
    # BURN-IN PHASE
    # ==========================================================================
    
    spins, burn_in_df = run_burn_in_phase(
        spins=spins,
        bqm=bqm,
        neighbourhood=neighbourhood,
        beta=args.beta,
        nodes=nodes,
        num_burn_in_sweeps=args.num_burn_in_sweeps,
        output_dir=output_dir,
        logger=logger,
        energy_check_window=args.burn_in_window,
        convergence_threshold=args.burn_in_threshold,
        post_burn_in_sweeps=args.post_burn_in_sweeps
    )
    
    # ==========================================================================
    # SAMPLING PHASE
    # ==========================================================================
    
    samples, _ = run_sampling_phase(
        spins=spins,
        bqm=bqm,
        neighbourhood=neighbourhood,
        beta=args.beta,
        nodes=nodes,
        num_sampling_sweeps=args.num_sampling_sweeps,
        num_samples=args.num_samples,
        output_dir=output_dir,
        logger=logger
    )
    
    # ==========================================================================
    # SAVE SAMPLES
    # ==========================================================================
    
    logger.info("=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # Save samples to compressed numpy file
    samples_path = os.path.join(output_dir, f'{args.output_prefix}_samples.npz')
    np.savez_compressed(
        samples_path,
        samples=samples,
        beta=args.beta,
        node_order=np.array(nodes),
        h_values=np.array([bqm.get_linear(node) for node in nodes]),
        instance_path=args.instance_path
    )
    logger.info(f"Samples saved to: {samples_path}")
    
    # Save metadata
    metadata = {
        'instance_path': args.instance_path,
        'beta': args.beta,
        'num_burn_in_sweeps': args.num_burn_in_sweeps if args.num_burn_in_sweeps else len(burn_in_df),
        'num_sampling_sweeps': args.num_sampling_sweeps,
        'num_samples': args.num_samples,
        'num_spins': N,
        'seed': args.seed
    }
    
    metadata_path = os.path.join(output_dir, f'{args.output_prefix}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to: {metadata_path}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples collected: {len(samples)}")
    logger.info(f"Sample shape: {samples.shape}")
    
    # Compute final statistics
    sample_energies = []
    for sample in samples:
        s = {node: sample[i] for i, node in enumerate(nodes)}
        sample_energies.append(compute_energy_bqm(s, bqm))
    
    sample_energies = np.array(sample_energies)
    sample_mags = np.mean(samples, axis=1)
    
    logger.info(f"Sample energy: {np.mean(sample_energies):.4f} ± {np.std(sample_energies):.4f}")
    logger.info(f"Sample magnetization: {np.mean(sample_mags):.4f} ± {np.std(sample_mags):.4f}")
    
    logger.info("=" * 60)
    logger.info("Gibbs sampling complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
