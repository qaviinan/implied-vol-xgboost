"""
Tests for SVI fitting functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svi import SVICalibrator, ChallengerSurface

class TestSVIFit(unittest.TestCase):
    """Test SVI fitting functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.svi_calibrator = SVICalibrator()
        
        # Create synthetic options data
        np.random.seed(42)
        self.test_data = self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic options data for testing."""
        dates = [datetime.now().date()]
        expiries = [datetime.now().date() + timedelta(days=30)]
        
        data = []
        for date in dates:
            for expiry in expiries:
                dte = (expiry - date).days
                underlying_price = 4000
                
                # Generate strikes around ATM
                strikes = np.arange(3800, 4200, 50)
                
                for strike in strikes:
                    for cp in ['C', 'P']:
                        # Generate synthetic implied volatility with smile
                        moneyness = strike / underlying_price
                        atm_iv = 0.2
                        smile_adjustment = 0.1 * (moneyness - 1) ** 2
                        impl_vol = atm_iv + smile_adjustment + np.random.normal(0, 0.01)
                        impl_vol = max(0.05, min(0.5, impl_vol))
                        
                        data.append({
                            'date': date,
                            'expiry': expiry,
                            'dte': dte,
                            'strike': strike,
                            'call_put': cp,
                            'impl_vol': impl_vol,
                            'underlying_price': underlying_price
                        })
        
        return pd.DataFrame(data)
    
    def test_svi_parameterization(self):
        """Test SVI parameterization function."""
        # Test SVI-JWP function
        k = np.array([-0.1, 0.0, 0.1])
        a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        
        w = self.svi_calibrator.svi_jwp(k, a, b, rho, m, sigma)
        
        # Check that output is positive
        self.assertTrue(np.all(w > 0))
        
        # Check that output has correct shape
        self.assertEqual(len(w), len(k))
    
    def test_svi_fit(self):
        """Test SVI fitting on synthetic data."""
        # Fit SVI to test data
        params = self.svi_calibrator.fit_svi(self.test_data)
        
        # Check that parameters are returned
        self.assertIn('a', params)
        self.assertIn('b', params)
        self.assertIn('rho', params)
        self.assertIn('m', params)
        self.assertIn('sigma', params)
        
        # Check parameter constraints
        self.assertGreater(params['b'], 0)
        self.assertGreater(params['sigma'], 0)
        self.assertLess(abs(params['rho']), 1)
    
    def test_svi_calibrate_surface(self):
        """Test SVI surface calibration."""
        # Calibrate surface
        svi_params = self.svi_calibrator.calibrate_surface(self.test_data)
        
        # Check that parameters are returned for each expiry
        self.assertGreater(len(svi_params), 0)
        
        for expiry, params in svi_params.items():
            self.assertIn('a', params)
            self.assertIn('b', params)
            self.assertIn('rho', params)
            self.assertIn('m', params)
            self.assertIn('sigma', params)
    
    def test_no_arbitrage_check(self):
        """Test no-arbitrage checking."""
        # Create a simple surface function
        def surface_fn(K, T):
            return 0.2  # Constant volatility
        
        strikes = np.array([3800, 3900, 4000, 4100, 4200])
        expiry = 0.25
        
        # Check no-arbitrage
        result = self.svi_calibrator.check_no_arbitrage(surface_fn, strikes, expiry)
        
        # Check that result contains expected keys
        self.assertIn('butterfly_arbitrage', result)
        self.assertIn('calendar_arbitrage', result)
        self.assertIn('no_arbitrage', result)
    
    def test_challenger_surface(self):
        """Test challenger surface (cubic splines)."""
        challenger = ChallengerSurface()
        
        # Fit spline surface
        spline_surfaces = challenger.fit_spline_surface(self.test_data)
        
        # Check that surfaces are returned
        self.assertGreater(len(spline_surfaces), 0)
        
        # Test spline evaluation
        for expiry, spline in spline_surfaces.items():
            # Test evaluation at a point
            k_test = 0.0  # ATM
            w_test = spline(k_test)
            self.assertGreater(w_test, 0)

if __name__ == '__main__':
    unittest.main()
