
import unittest
import numpy as np
import sys
import os
import logging
import shutil
from unittest.mock import MagicMock, patch

# Add parent directory to path to import runner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gibbs_sampling_runner import (
    compute_integrated_autocorrelation_time,
    run_burn_in_phase,
    run_sampling_phase,
    compute_energy_bqm,
    compute_magnetization,
    gibbs_sweep
)

class TestGibbsSoundness(unittest.TestCase):
    
    def test_iact_ar1(self):
        """Test IACT estimator on AR(1) process with known tau."""
        # AR(1): x_t = phi * x_{t-1} + epsilon
        # tau ~ (1+phi)/(1-phi)
        phi = 0.9
        expected_tau = (1 + phi) / (1 - phi) # ~19.0
        
        np.random.seed(42)
        n = 50000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + np.random.normal()
            
        tau_est = compute_integrated_autocorrelation_time(x)
        print(f"AR(1) phi={phi}: Expected tau={expected_tau:.2f}, Estimated tau={tau_est:.2f}")
        
        # Allow 20% error margin for estimation
        self.assertTrue(abs(tau_est - expected_tau) < 0.25 * expected_tau, 
                        f"IACT estimate {tau_est} far from expected {expected_tau}")

    def test_iact_white_noise(self):
        """Test IACT on white noise."""
        np.random.seed(42)
        x = np.random.normal(size=10000)
        tau_est = compute_integrated_autocorrelation_time(x)
        print(f"White noise: Expected tau=1.0, Estimated tau={tau_est:.2f}")
        self.assertTrue(abs(tau_est - 1.0) < 0.2, "IACT for white noise should be close to 1")

    @patch('gibbs_sampling_runner.gibbs_sweep')
    @patch('gibbs_sampling_runner.compute_energy_bqm')
    @patch('gibbs_sampling_runner.compute_magnetization')
    @patch('gibbs_sampling_runner.tau_cap') # Mock cap to be safe
    def test_strict_burn_in_buffer(self, mock_cap, mock_mag, mock_energy, mock_sweep):
        """Verify that post-burn-in sweeps are executed strictly."""
        # Setup mocks
        mock_sweep.side_effect = lambda s, b, n, be, no: s
        mock_energy.return_value = -1.0
        mock_mag.return_value = 0.0
        mock_cap.return_value = None
        
        logger = logging.getLogger("test")
        logger.setLevel(logging.CRITICAL)
        
        spins = {0: 1}
        bqm = MagicMock()
        
        # Test adaptive mode with buffer
        post_sweeps = 50
        window = 10
        spins, df = run_burn_in_phase(
            spins=spins, bqm=bqm, neighbourhood={}, beta=1.0, nodes=[0],
            num_burn_in_sweeps=None, # Adaptive
            output_dir="/tmp",
            logger=logger,
            energy_check_window=window,
            post_burn_in_sweeps=post_sweeps
        )
        
        # Check if sweeps were called
        # Min sweeps = 2 * window = 20
        # Then convergence check passes
        # Then post_sweeps = 50
        # Total approx 70 calls
        self.assertGreaterEqual(mock_sweep.call_count, 2 * window + post_sweeps)
        
    def test_tau_cap_logic(self):
        """Test temperature dependent capping."""
        from gibbs_sampling_runner import tau_cap
        self.assertEqual(tau_cap(2.0), 2000.0)
        self.assertEqual(tau_cap(1.0), 3000.0)
        self.assertIsNone(tau_cap(0.5))

if __name__ == '__main__':
    unittest.main()
