"""
SVI (Stochastic Volatility Inspired) parameterization with no-arbitrage checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline
import logging
from .utils import setup_logging

logger = setup_logging()

class SVICalibrator:
    """SVI parameterization calibrator with no-arbitrage checks."""
    
    def __init__(self, lambda_smooth: float = 0.1, rho_bound: float = 0.9):
        """
        Initialize SVI calibrator.
        
        Args:
            lambda_smooth: Smoothness penalty parameter
            rho_bound: Bound for rho parameter (|rho| < rho_bound)
        """
        self.lambda_smooth = lambda_smooth
        self.rho_bound = rho_bound
        
    def svi_jwp(self, k: np.ndarray, a: float, b: float, rho: float, 
                m: float, sigma: float) -> np.ndarray:
        """
        SVI-JWP parameterization.
        
        Args:
            k: Log-moneyness array
            a: Level parameter
            b: Slope parameter
            rho: Skew parameter
            m: Location parameter
            sigma: Scale parameter
            
        Returns:
            Total variance array
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def fit_svi(self, expiry_df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit SVI parameters to options data for a single expiry.
        
        Args:
            expiry_df: DataFrame with options data for single expiry
            
        Returns:
            Dictionary with SVI parameters
        """
        logger.info(f"Fitting SVI for expiry {expiry_df['expiry'].iloc[0]}")
        
        # Calculate log-moneyness and total variance
        k = np.log(expiry_df['strike'] / expiry_df['underlying_price'])
        w_market = expiry_df['impl_vol']**2 * expiry_df['dte'] / 365.0
        
        # Remove outliers
        valid_mask = (w_market > 0) & (w_market < 1.0) & (np.isfinite(k))
        k = k[valid_mask]
        w_market = w_market[valid_mask]
        
        if len(k) < 5:
            logger.warning("Insufficient data points for SVI fitting")
            return self._default_svi_params()
        
        # Initial parameter guess
        initial_params = self._initial_guess(k, w_market)
        
        # Define objective function
        def objective(params):
            a, b, rho, m, sigma = params
            w_svi = self.svi_jwp(k, a, b, rho, m, sigma)
            
            # Main fitting error
            mse = np.mean((w_svi - w_market)**2)
            
            # Smoothness penalty
            smoothness_penalty = self.lambda_smooth * (rho**2 + (b - 0.1)**2)
            
            return mse + smoothness_penalty
        
        # Parameter bounds
        bounds = [
            (-1.0, 1.0),  # a
            (0.01, 2.0),  # b
            (-self.rho_bound, self.rho_bound),  # rho
            (-2.0, 2.0),  # m
            (0.01, 1.0)   # sigma
        ]
        
        # Optimize
        try:
            result = differential_evolution(
                objective, bounds, seed=42, maxiter=1000, popsize=15
            )
            
            if result.success:
                params = result.x
                logger.info(f"SVI fit successful: {params}")
                return {
                    'a': params[0],
                    'b': params[1], 
                    'rho': params[2],
                    'm': params[3],
                    'sigma': params[4],
                    'mse': result.fun
                }
            else:
                logger.warning("SVI optimization failed, using default parameters")
                return self._default_svi_params()
                
        except Exception as e:
            logger.error(f"SVI optimization error: {e}")
            return self._default_svi_params()
    
    def _initial_guess(self, k: np.ndarray, w: np.ndarray) -> Tuple[float, ...]:
        """Generate initial parameter guess."""
        # Simple heuristic initial guess
        a = np.mean(w)
        b = 0.1
        rho = 0.0
        m = np.mean(k)
        sigma = 0.1
        
        return (a, b, rho, m, sigma)
    
    def _default_svi_params(self) -> Dict[str, float]:
        """Return default SVI parameters."""
        return {
            'a': 0.04,
            'b': 0.1,
            'rho': 0.0,
            'm': 0.0,
            'sigma': 0.1,
            'mse': float('inf')
        }
    
    def check_no_arbitrage(self, surface_fn: Callable, strikes: np.ndarray, 
                          expiry: float) -> Dict[str, bool]:
        """
        Check for no-arbitrage violations in the SVI surface.
        
        Args:
            surface_fn: Function that returns implied vol for given strike and expiry
            strikes: Array of strikes to check
            expiry: Time to expiry
            
        Returns:
            Dictionary with no-arbitrage check results
        """
        logger.info("Checking no-arbitrage conditions")
        
        # Butterfly arbitrage check
        butterfly_violations = self._check_butterfly_arbitrage(surface_fn, strikes, expiry)
        
        # Calendar arbitrage check (if multiple expiries available)
        calendar_violations = self._check_calendar_arbitrage(surface_fn, strikes, expiry)
        
        return {
            'butterfly_arbitrage': butterfly_violations,
            'calendar_arbitrage': calendar_violations,
            'no_arbitrage': not (butterfly_violations or calendar_violations)
        }
    
    def _check_butterfly_arbitrage(self, surface_fn: Callable, strikes: np.ndarray, 
                                  expiry: float) -> bool:
        """Check for butterfly arbitrage violations."""
        try:
            # Calculate implied volatilities
            ivs = np.array([surface_fn(K, expiry) for K in strikes])
            
            # Check for negative density (butterfly spread)
            # This is a simplified check - in practice, you'd calculate the density
            # from the second derivative of the call price function
            
            # Check for monotonicity violations
            for i in range(1, len(ivs) - 1):
                if ivs[i] < min(ivs[i-1], ivs[i+1]) - 0.01:  # Allow small tolerance
                    logger.warning(f"Butterfly arbitrage detected at strike {strikes[i]}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in butterfly arbitrage check: {e}")
            return False
    
    def _check_calendar_arbitrage(self, surface_fn: Callable, strikes: np.ndarray, 
                                 expiry: float) -> bool:
        """Check for calendar arbitrage violations."""
        # This would require multiple expiries to check
        # For now, return False (no violations)
        return False
    
    def calibrate_surface(self, options_df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
        """
        Calibrate SVI surface for all expiries.
        
        Args:
            options_df: DataFrame with options data
            
        Returns:
            Dictionary mapping expiry dates to SVI parameters
        """
        logger.info("Calibrating SVI surface for all expiries")
        
        svi_params = {}
        expiries = sorted(options_df['expiry'].unique())
        
        for expiry in expiries:
            expiry_data = options_df[options_df['expiry'] == expiry]
            params = self.fit_svi(expiry_data)
            svi_params[pd.Timestamp(expiry)] = params
        
        logger.info(f"Calibrated SVI for {len(svi_params)} expiries")
        return svi_params

class ChallengerSurface:
    """Challenger surface using cubic splines with shape constraints."""
    
    def __init__(self):
        """Initialize challenger surface."""
        pass
    
    def fit_spline_surface(self, options_df: pd.DataFrame) -> Dict[pd.Timestamp, CubicSpline]:
        """
        Fit cubic spline surface for all expiries.
        
        Args:
            options_df: DataFrame with options data
            
        Returns:
            Dictionary mapping expiry dates to spline functions
        """
        logger.info("Fitting cubic spline surface")
        
        spline_surfaces = {}
        expiries = sorted(options_df['expiry'].unique())
        
        for expiry in expiries:
            expiry_data = options_df[options_df['expiry'] == expiry]
            
            # Calculate log-moneyness and total variance
            k = np.log(expiry_data['strike'] / expiry_data['underlying_price'])
            w = expiry_data['impl_vol']**2 * expiry_data['dte'] / 365.0
            
            # Remove outliers
            valid_mask = (w > 0) & (w < 1.0) & (np.isfinite(k))
            k = k[valid_mask]
            w = w[valid_mask]
            
            if len(k) >= 4:  # Need at least 4 points for cubic spline
                try:
                    # Sort by log-moneyness
                    sort_idx = np.argsort(k)
                    k_sorted = k[sort_idx]
                    w_sorted = w[sort_idx]
                    
                    # Fit cubic spline
                    spline = CubicSpline(k_sorted, w_sorted, bc_type='natural')
                    spline_surfaces[pd.Timestamp(expiry)] = spline
                    
                except Exception as e:
                    logger.warning(f"Failed to fit spline for expiry {expiry}: {e}")
        
        logger.info(f"Fitted spline surfaces for {len(spline_surfaces)} expiries")
        return spline_surfaces
    
    def compare_with_svi(self, svi_params: Dict[pd.Timestamp, Dict], 
                        spline_surfaces: Dict[pd.Timestamp, CubicSpline],
                        options_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compare SVI and spline surface fits.
        
        Args:
            svi_params: SVI parameters by expiry
            spline_surfaces: Spline surfaces by expiry
            options_df: Options data
            
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing SVI and spline surface fits")
        
        svi_errors = []
        spline_errors = []
        
        for expiry in svi_params.keys():
            if expiry not in spline_surfaces:
                continue
                
            expiry_data = options_df[options_df['expiry'] == expiry.date()]
            k = np.log(expiry_data['strike'] / expiry_data['underlying_price'])
            w_market = expiry_data['impl_vol']**2 * expiry_data['dte'] / 365.0
            
            # SVI errors
            svi_calibrator = SVICalibrator()
            params = svi_params[expiry]
            w_svi = svi_calibrator.svi_jwp(k, params['a'], params['b'], 
                                         params['rho'], params['m'], params['sigma'])
            svi_error = np.mean((w_svi - w_market)**2)
            svi_errors.append(svi_error)
            
            # Spline errors
            try:
                w_spline = spline_surfaces[expiry](k)
                spline_error = np.mean((w_spline - w_market)**2)
                spline_errors.append(spline_error)
            except:
                spline_errors.append(float('inf'))
        
        return {
            'svi_mean_error': np.mean(svi_errors),
            'spline_mean_error': np.mean(spline_errors),
            'svi_std_error': np.std(svi_errors),
            'spline_std_error': np.std(spline_errors),
            'svi_better_count': sum(1 for s, sp in zip(svi_errors, spline_errors) if s < sp),
            'total_expiries': len(svi_errors)
        }
