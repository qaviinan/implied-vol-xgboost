"""
Surface interpolation and extrapolation for implied volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Tuple
from scipy.interpolate import interp2d, griddata, RBFInterpolator
from scipy.optimize import minimize_scalar
import logging
from .utils import setup_logging
from .svi import SVICalibrator

logger = setup_logging()

class VolatilitySurface:
    """Implied volatility surface with interpolation and extrapolation."""
    
    def __init__(self, svi_params: Dict[pd.Timestamp, Dict[str, float]]):
        """
        Initialize volatility surface with SVI parameters.
        
        Args:
            svi_params: Dictionary mapping expiry dates to SVI parameters
        """
        self.svi_params = svi_params
        self.svi_calibrator = SVICalibrator()
        
    def make_surface(self) -> Callable[[float, float], float]:
        """
        Create surface function that returns implied vol for given strike and expiry.
        
        Returns:
            Function f(K, T) -> sigma_imp
        """
        def surface_function(K: float, T: float) -> float:
            """
            Get implied volatility for given strike and time to expiry.
            
            Args:
                K: Strike price
                T: Time to expiry (years)
                
            Returns:
                Implied volatility
            """
            # Find closest expiry
            target_date = pd.Timestamp.now() + pd.Timedelta(days=T * 365)
            closest_expiry = min(self.svi_params.keys(), 
                               key=lambda x: abs(x - target_date))
            
            # Get SVI parameters for closest expiry
            params = self.svi_params[closest_expiry]
            
            # Calculate log-moneyness (need underlying price - this is a limitation)
            # For now, assume we have access to current underlying price
            # In practice, this would be passed as a parameter
            S = 4000  # Placeholder - should be current underlying price
            k = np.log(K / S)
            
            # Calculate total variance using SVI
            w = self.svi_calibrator.svi_jwp(k, params['a'], params['b'], 
                                          params['rho'], params['m'], params['sigma'])
            
            # Convert to implied volatility
            if w > 0 and T > 0:
                return np.sqrt(w / T)
            else:
                return 0.2  # Default volatility
            
        return surface_function
    
    def interpolate_surface(self, strikes: np.ndarray, expiries: np.ndarray,
                          method: str = 'linear') -> np.ndarray:
        """
        Interpolate implied volatility surface.
        
        Args:
            strikes: Array of strike prices
            expiries: Array of time to expiry (years)
            method: Interpolation method ('linear', 'cubic', 'rbf')
            
        Returns:
            2D array of implied volatilities
        """
        # Create grid
        K_grid, T_grid = np.meshgrid(strikes, expiries)
        
        # Calculate implied volatilities using SVI
        iv_surface = np.zeros_like(K_grid)
        
        for i, expiry in enumerate(expiries):
            # Find closest SVI parameters
            target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
            closest_expiry = min(self.svi_params.keys(), 
                               key=lambda x: abs(x - target_date))
            params = self.svi_params[closest_expiry]
            
            for j, strike in enumerate(strikes):
                # Calculate log-moneyness
                S = 4000  # Placeholder
                k = np.log(strike / S)
                
                # Calculate total variance
                w = self.svi_calibrator.svi_jwp(k, params['a'], params['b'], 
                                              params['rho'], params['m'], params['sigma'])
                
                # Convert to implied volatility
                if w > 0 and expiry > 0:
                    iv_surface[i, j] = np.sqrt(w / expiry)
                else:
                    iv_surface[i, j] = 0.2
        
        return iv_surface
    
    def get_atm_volatility(self, expiry: float) -> float:
        """
        Get at-the-money implied volatility for given expiry.
        
        Args:
            expiry: Time to expiry (years)
            
        Returns:
            ATM implied volatility
        """
        # Find closest expiry
        target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
        closest_expiry = min(self.svi_params.keys(), 
                           key=lambda x: abs(x - target_date))
        
        params = self.svi_params[closest_expiry]
        
        # ATM corresponds to k = 0 (log-moneyness = 0)
        w_atm = self.svi_calibrator.svi_jwp(0, params['a'], params['b'], 
                                          params['rho'], params['m'], params['sigma'])
        
        if w_atm > 0 and expiry > 0:
            return np.sqrt(w_atm / expiry)
        else:
            return 0.2
    
    def get_volatility_smile(self, expiry: float, strikes: np.ndarray) -> np.ndarray:
        """
        Get volatility smile for given expiry and strikes.
        
        Args:
            expiry: Time to expiry (years)
            strikes: Array of strike prices
            
        Returns:
            Array of implied volatilities
        """
        # Find closest expiry
        target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
        closest_expiry = min(self.svi_params.keys(), 
                           key=lambda x: abs(x - target_date))
        
        params = self.svi_params[closest_expiry]
        S = 4000  # Placeholder
        
        ivs = []
        for strike in strikes:
            k = np.log(strike / S)
            w = self.svi_calibrator.svi_jwp(k, params['a'], params['b'], 
                                          params['rho'], params['m'], params['sigma'])
            
            if w > 0 and expiry > 0:
                ivs.append(np.sqrt(w / expiry))
            else:
                ivs.append(0.2)
        
        return np.array(ivs)
    
    def get_term_structure(self, strike: float, expiries: np.ndarray) -> np.ndarray:
        """
        Get volatility term structure for given strike.
        
        Args:
            strike: Strike price
            expiries: Array of time to expiry (years)
            
        Returns:
            Array of implied volatilities
        """
        S = 4000  # Placeholder
        k = np.log(strike / S)
        
        ivs = []
        for expiry in expiries:
            # Find closest SVI parameters
            target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
            closest_expiry = min(self.svi_params.keys(), 
                               key=lambda x: abs(x - target_date))
            params = self.svi_params[closest_expiry]
            
            w = self.svi_calibrator.svi_jwp(k, params['a'], params['b'], 
                                          params['rho'], params['m'], params['sigma'])
            
            if w > 0 and expiry > 0:
                ivs.append(np.sqrt(w / expiry))
            else:
                ivs.append(0.2)
        
        return np.array(ivs)
    
    def calculate_smile_slope(self, expiry: float, strike: float) -> float:
        """
        Calculate smile slope (first derivative of IV w.r.t. log-moneyness).
        
        Args:
            expiry: Time to expiry (years)
            strike: Strike price
            
        Returns:
            Smile slope
        """
        # Find closest expiry
        target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
        closest_expiry = min(self.svi_params.keys(), 
                           key=lambda x: abs(x - target_date))
        
        params = self.svi_params[closest_expiry]
        S = 4000  # Placeholder
        k = np.log(strike / S)
        
        # Calculate derivative of SVI w.r.t. k
        a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
        
        # Derivative of w(k) w.r.t. k
        dw_dk = b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))
        
        # Convert to IV derivative
        w = self.svi_calibrator.svi_jwp(k, a, b, rho, m, sigma)
        if w > 0 and expiry > 0:
            iv = np.sqrt(w / expiry)
            div_dk = dw_dk / (2 * np.sqrt(w * expiry))
            return div_dk
        else:
            return 0.0
    
    def calculate_smile_curvature(self, expiry: float, strike: float) -> float:
        """
        Calculate smile curvature (second derivative of IV w.r.t. log-moneyness).
        
        Args:
            expiry: Time to expiry (years)
            strike: Strike price
            
        Returns:
            Smile curvature
        """
        # Find closest expiry
        target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry * 365)
        closest_expiry = min(self.svi_params.keys(), 
                           key=lambda x: abs(x - target_date))
        
        params = self.svi_params[closest_expiry]
        S = 4000  # Placeholder
        k = np.log(strike / S)
        
        # Calculate second derivative of SVI w.r.t. k
        a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
        
        # Second derivative of w(k) w.r.t. k
        d2w_dk2 = b * sigma**2 / ((k - m)**2 + sigma**2)**(3/2)
        
        # Convert to IV second derivative
        w = self.svi_calibrator.svi_jwp(k, a, b, rho, m, sigma)
        if w > 0 and expiry > 0:
            iv = np.sqrt(w / expiry)
            # This is a simplified calculation
            return d2w_dk2 / (2 * np.sqrt(w * expiry))
        else:
            return 0.0

class SurfaceBuilder:
    """Builder class for creating volatility surfaces from options data."""
    
    def __init__(self):
        """Initialize surface builder."""
        self.svi_calibrator = SVICalibrator()
    
    def build_surface_from_options(self, options_df: pd.DataFrame) -> VolatilitySurface:
        """
        Build volatility surface from options data.
        
        Args:
            options_df: DataFrame with options data
            
        Returns:
            VolatilitySurface object
        """
        logger.info("Building volatility surface from options data")
        
        # Calibrate SVI parameters for all expiries
        svi_params = self.svi_calibrator.calibrate_surface(options_df)
        
        # Create surface
        surface = VolatilitySurface(svi_params)
        
        logger.info(f"Built surface with {len(svi_params)} expiries")
        return surface
    
    def validate_surface(self, surface: VolatilitySurface, 
                        options_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate surface against market data.
        
        Args:
            surface: VolatilitySurface object
            options_df: Market options data
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating volatility surface")
        
        errors = []
        surface_fn = surface.make_surface()
        
        for _, row in options_df.iterrows():
            # Get market implied volatility
            market_iv = row['impl_vol']
            
            # Get surface implied volatility
            T = row['dte'] / 365.0
            surface_iv = surface_fn(row['strike'], T)
            
            # Calculate error
            error = abs(surface_iv - market_iv)
            errors.append(error)
        
        return {
            'mean_absolute_error': np.mean(errors),
            'root_mean_square_error': np.sqrt(np.mean(np.array(errors)**2)),
            'max_error': np.max(errors),
            'median_error': np.median(errors)
        }
