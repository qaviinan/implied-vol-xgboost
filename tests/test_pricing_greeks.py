"""
Tests for pricing and Greeks functionality.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pricing import (
    bs_price_greeks, implied_volatility, parity_checks,
    bump_and_revalue_greeks, validate_greeks, calculate_portfolio_greeks
)

class TestPricingGreeks(unittest.TestCase):
    """Test pricing and Greeks functionality."""
    
    def setUp(self):
        """Set up test parameters."""
        self.S = 4000.0  # Stock price
        self.K = 4000.0  # Strike price
        self.T = 0.25    # Time to expiry (3 months)
        self.r = 0.05    # Risk-free rate
        self.q = 0.02    # Dividend yield
        self.sigma = 0.2 # Implied volatility
    
    def test_bs_price_greeks_call(self):
        """Test Black-Scholes pricing for call option."""
        result = bs_price_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')
        
        # Check that all required fields are present
        self.assertIn('price', result)
        self.assertIn('delta', result)
        self.assertIn('vega', result)
        self.assertIn('theta', result)
        
        # Check that price is positive
        self.assertGreater(result['price'], 0)
        
        # Check delta bounds for call
        self.assertGreaterEqual(result['delta'], 0)
        self.assertLessEqual(result['delta'], 1)
        
        # Check vega is positive
        self.assertGreater(result['vega'], 0)
    
    def test_bs_price_greeks_put(self):
        """Test Black-Scholes pricing for put option."""
        result = bs_price_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'P')
        
        # Check that all required fields are present
        self.assertIn('price', result)
        self.assertIn('delta', result)
        self.assertIn('vega', result)
        self.assertIn('theta', result)
        
        # Check that price is positive
        self.assertGreater(result['price'], 0)
        
        # Check delta bounds for put
        self.assertGreaterEqual(result['delta'], -1)
        self.assertLessEqual(result['delta'], 0)
        
        # Check vega is positive
        self.assertGreater(result['vega'], 0)
    
    def test_bs_price_greeks_expired(self):
        """Test Black-Scholes pricing for expired option."""
        result = bs_price_greeks(self.S, self.K, 0.0, self.r, self.q, self.sigma, 'C')
        
        # For expired call, price should be max(S-K, 0)
        expected_price = max(0, self.S - self.K)
        self.assertEqual(result['price'], expected_price)
        
        # Delta should be 1 if S > K, 0 otherwise
        expected_delta = 1.0 if self.S > self.K else 0.0
        self.assertEqual(result['delta'], expected_delta)
        
        # Vega should be 0 for expired options
        self.assertEqual(result['vega'], 0.0)
    
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # Calculate option price first
        price = bs_price_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')['price']
        
        # Calculate implied volatility
        iv = implied_volatility(price, self.S, self.K, self.T, self.r, self.q, 'C')
        
        # Check that IV is close to original volatility
        self.assertAlmostEqual(iv, self.sigma, places=2)
    
    def test_parity_checks(self):
        """Test put-call parity checks."""
        # Create test data
        row = pd.Series({
            'strike': self.K,
            'underlying_price': self.S,
            'dte': self.T * 365,
            'r_annualized': self.r,
            'q_div_yield': self.q,
            'call_put': 'C',
            'mid': 100.0  # Placeholder price
        })
        
        # Test parity check (should return True for valid data)
        result = parity_checks(row)
        self.assertIsInstance(result, bool)
    
    def test_bump_and_revalue_greeks(self):
        """Test bump-and-revalue Greeks calculation."""
        result = bump_and_revalue_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')
        
        # Check that all required fields are present
        self.assertIn('delta_num', result)
        self.assertIn('vega_num', result)
        self.assertIn('theta_num', result)
        
        # Check that numerical Greeks are finite
        self.assertTrue(np.isfinite(result['delta_num']))
        self.assertTrue(np.isfinite(result['vega_num']))
        self.assertTrue(np.isfinite(result['theta_num']))
    
    def test_validate_greeks(self):
        """Test Greeks validation."""
        result = validate_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')
        
        # Check that all required fields are present
        self.assertIn('delta_valid', result)
        self.assertIn('vega_valid', result)
        self.assertIn('theta_valid', result)
        self.assertIn('delta_error', result)
        self.assertIn('vega_error', result)
        self.assertIn('theta_error', result)
        
        # Check that validation results are boolean
        self.assertIsInstance(result['delta_valid'], bool)
        self.assertIsInstance(result['vega_valid'], bool)
        self.assertIsInstance(result['theta_valid'], bool)
        
        # Check that errors are non-negative
        self.assertGreaterEqual(result['delta_error'], 0)
        self.assertGreaterEqual(result['vega_error'], 0)
        self.assertGreaterEqual(result['theta_error'], 0)
    
    def test_calculate_portfolio_greeks(self):
        """Test portfolio Greeks calculation."""
        # Create test positions
        positions = pd.DataFrame({
            'quantity': [10, -5],
            'delta': [0.5, -0.3],
            'vega': [20, 15],
            'theta': [-5, -3]
        })
        
        result = calculate_portfolio_greeks(positions)
        
        # Check that all required fields are present
        self.assertIn('portfolio_delta', result)
        self.assertIn('portfolio_vega', result)
        self.assertIn('portfolio_theta', result)
        
        # Check that portfolio Greeks are finite
        self.assertTrue(np.isfinite(result['portfolio_delta']))
        self.assertTrue(np.isfinite(result['portfolio_vega']))
        self.assertTrue(np.isfinite(result['portfolio_theta']))
    
    def test_greeks_consistency(self):
        """Test consistency between analytical and numerical Greeks."""
        # Get analytical Greeks
        analytical = bs_price_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')
        
        # Get numerical Greeks
        numerical = bump_and_revalue_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma, 'C')
        
        # Check that they are reasonably close
        delta_diff = abs(analytical['delta'] - numerical['delta_num'])
        vega_diff = abs(analytical['vega'] - numerical['vega_num'])
        
        # Allow for some numerical error
        self.assertLess(delta_diff, 0.01)
        self.assertLess(vega_diff, 0.1)

if __name__ == '__main__':
    unittest.main()
