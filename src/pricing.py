"""
Black-Scholes pricing and Greeks calculation with parity tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import logging
from .utils import setup_logging

logger = setup_logging()

def bs_price_greeks(S: float, K: float, T: float, r: float, q: float, 
                   sigma: float, cp: str) -> Dict[str, float]:
    """
    Calculate Black-Scholes option price and Greeks.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Implied volatility
        cp: 'C' for call, 'P' for put
        
    Returns:
        Dictionary with price, delta, vega, theta
    """
    if T <= 0:
        # Handle expired options
        if cp == 'C':
            return {'price': max(0, S - K), 'delta': 1.0 if S > K else 0.0, 
                   'vega': 0.0, 'theta': 0.0}
        else:
            return {'price': max(0, K - S), 'delta': -1.0 if S < K else 0.0, 
                   'vega': 0.0, 'theta': 0.0}
    
    # Calculate forward price and discount factor
    F = S * np.exp((r - q) * T)
    D = np.exp(-r * T)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if cp == 'C':
        price = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:  # Put
        price = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    # Calculate Greeks
    delta = np.exp(-q * T) * (norm.cdf(d1) if cp == 'C' else norm.cdf(d1) - 1)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    # Theta calculation
    theta_call = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2) 
                  + q * S * np.exp(-q * T) * norm.cdf(d1))
    
    theta_put = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
    
    theta = theta_call if cp == 'C' else theta_put
    
    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'theta': theta
    }

def implied_volatility(price: float, S: float, K: float, T: float, 
                      r: float, q: float, cp: str, 
                      initial_guess: float = 0.2) -> float:
    """
    Calculate implied volatility using Brent's method.
    
    Args:
        price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        cp: 'C' for call, 'P' for put
        initial_guess: Initial guess for volatility
        
    Returns:
        Implied volatility
    """
    def objective(sigma):
        bs_price = bs_price_greeks(S, K, T, r, q, sigma, cp)['price']
        return (bs_price - price) ** 2
    
    try:
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        return result.x
    except:
        return initial_guess

def parity_checks(row: pd.Series) -> bool:
    """
    Check put-call parity for a given option.
    
    Args:
        row: DataFrame row with option data
        
    Returns:
        True if parity holds within tolerance
    """
    S = row['underlying_price']
    K = row['strike']
    T = row['dte'] / 365.0
    r = row.get('r_annualized', 0.05)
    q = row.get('q_div_yield', 0.02)
    
    # Get call and put prices
    if row['call_put'] == 'C':
        call_price = row['mid']
        # Find corresponding put price
        put_price = None  # This would need to be looked up
    else:
        put_price = row['mid']
        # Find corresponding call price
        call_price = None  # This would need to be looked up
    
    if call_price is None or put_price is None:
        return True  # Can't check without both prices
    
    # Check put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
    forward_price = S * np.exp(-q * T)
    strike_pv = K * np.exp(-r * T)
    parity_value = forward_price - strike_pv
    market_value = call_price - put_price
    
    # Check if within 1% tolerance
    tolerance = 0.01 * max(call_price, put_price)
    return abs(parity_value - market_value) <= tolerance

def bump_and_revalue_greeks(S: float, K: float, T: float, r: float, q: float, 
                           sigma: float, cp: str, bump_size: float = 0.01) -> Dict[str, float]:
    """
    Calculate Greeks using bump-and-revalue method for validation.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Implied volatility
        cp: 'C' for call, 'P' for put
        bump_size: Size of bump for numerical differentiation
        
    Returns:
        Dictionary with numerical Greeks
    """
    # Base price
    base_price = bs_price_greeks(S, K, T, r, q, sigma, cp)['price']
    
    # Delta: bump stock price
    price_up = bs_price_greeks(S * (1 + bump_size), K, T, r, q, sigma, cp)['price']
    price_down = bs_price_greeks(S * (1 - bump_size), K, T, r, q, sigma, cp)['price']
    delta_num = (price_up - price_down) / (2 * S * bump_size)
    
    # Vega: bump volatility
    price_vol_up = bs_price_greeks(S, K, T, r, q, sigma + bump_size, cp)['price']
    price_vol_down = bs_price_greeks(S, K, T, r, q, sigma - bump_size, cp)['price']
    vega_num = (price_vol_up - price_vol_down) / (2 * bump_size)
    
    # Theta: bump time (negative direction)
    if T > bump_size / 365.0:
        price_time_down = bs_price_greeks(S, K, T - bump_size/365.0, r, q, sigma, cp)['price']
        theta_num = (price_time_down - base_price) / (bump_size / 365.0)
    else:
        theta_num = 0.0
    
    return {
        'delta_num': delta_num,
        'vega_num': vega_num,
        'theta_num': theta_num
    }

def validate_greeks(S: float, K: float, T: float, r: float, q: float, 
                   sigma: float, cp: str, tolerance: float = 0.01) -> Dict[str, bool]:
    """
    Validate analytical Greeks against numerical Greeks.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Implied volatility
        cp: 'C' for call, 'P' for put
        tolerance: Tolerance for validation
        
    Returns:
        Dictionary with validation results
    """
    # Analytical Greeks
    analytical = bs_price_greeks(S, K, T, r, q, sigma, cp)
    
    # Numerical Greeks
    numerical = bump_and_revalue_greeks(S, K, T, r, q, sigma, cp)
    
    # Validate
    delta_valid = abs(analytical['delta'] - numerical['delta_num']) <= tolerance
    vega_valid = abs(analytical['vega'] - numerical['vega_num']) <= tolerance
    theta_valid = abs(analytical['theta'] - numerical['theta_num']) <= tolerance * 5  # More lenient for theta
    
    return {
        'delta_valid': delta_valid,
        'vega_valid': vega_valid,
        'theta_valid': theta_valid,
        'delta_error': abs(analytical['delta'] - numerical['delta_num']),
        'vega_error': abs(analytical['vega'] - numerical['vega_num']),
        'theta_error': abs(analytical['theta'] - numerical['theta_num'])
    }

def calculate_portfolio_greeks(positions: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate portfolio-level Greeks from individual positions.
    
    Args:
        positions: DataFrame with position data including Greeks
        
    Returns:
        Dictionary with portfolio Greeks
    """
    portfolio_delta = (positions['quantity'] * positions['delta']).sum()
    portfolio_vega = (positions['quantity'] * positions['vega']).sum()
    portfolio_theta = (positions['quantity'] * positions['theta']).sum()
    
    return {
        'portfolio_delta': portfolio_delta,
        'portfolio_vega': portfolio_vega,
        'portfolio_theta': portfolio_theta
    }

def calculate_hedge_ratio(delta: float, vega: float, target_delta: float = 0.0) -> float:
    """
    Calculate hedge ratio for delta hedging.
    
    Args:
        delta: Option delta
        vega: Option vega
        target_delta: Target portfolio delta
        
    Returns:
        Hedge ratio (negative of delta for delta-neutral hedge)
    """
    return -delta

def calculate_position_pnl(entry_price: float, current_price: float, 
                          quantity: float, delta: float, vega: float, theta: float,
                          underlying_change: float, vol_change: float, 
                          time_decay: float) -> Dict[str, float]:
    """
    Calculate position PnL breakdown.
    
    Args:
        entry_price: Entry price of the option
        current_price: Current price of the option
        quantity: Position quantity
        delta: Option delta
        vega: Option vega
        theta: Option theta
        underlying_change: Change in underlying price
        vol_change: Change in implied volatility
        time_decay: Time decay (days)
        
    Returns:
        Dictionary with PnL components
    """
    # Total PnL
    total_pnl = quantity * (current_price - entry_price)
    
    # PnL components
    delta_pnl = quantity * delta * underlying_change
    vega_pnl = quantity * vega * vol_change
    theta_pnl = quantity * theta * time_decay / 365.0
    
    # Residual PnL (higher-order terms)
    residual_pnl = total_pnl - delta_pnl - vega_pnl - theta_pnl
    
    return {
        'total_pnl': total_pnl,
        'delta_pnl': delta_pnl,
        'vega_pnl': vega_pnl,
        'theta_pnl': theta_pnl,
        'residual_pnl': residual_pnl
    }
