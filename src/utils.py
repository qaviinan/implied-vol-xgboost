"""
Utility functions for the VolBoost project.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import warnings

# Set up logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the project."""
    logger = logging.getLogger("volboost")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Configuration management
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_config_path(config_name: str) -> str:
    """Get the full path to a configuration file."""
    return os.path.join(get_project_root(), "config", f"{config_name}.yml")

# Date and time utilities
def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get trading days between start and end dates."""
    # Simple implementation - in practice, you'd use a proper calendar
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    return dates

def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if a date is a trading day."""
    return date.weekday() < 5  # Monday = 0, Friday = 4

def get_next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Get the next trading day."""
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def get_prev_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Get the previous trading day."""
    prev_day = date - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    return prev_day

# Data validation utilities
def validate_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean options data."""
    logger = logging.getLogger("volboost")
    
    initial_count = len(df)
    
    # Remove rows with invalid prices
    df = df[df['bid'] > 0]
    df = df[df['ask'] > df['bid']]
    df = df[df['mid'] > 0.05]
    
    # Remove rows with invalid implied volatility
    df = df[(df['impl_vol'] > 0.01) & (df['impl_vol'] < 5.0)]
    
    # Remove rows with invalid moneyness
    df['moneyness'] = df['strike'] / df['underlying_price']
    df = df[(df['moneyness'] >= 0.5) & (df['moneyness'] <= 1.5)]
    
    # Remove rows with invalid DTE
    df = df[(df['dte'] >= 5) & (df['dte'] <= 365)]
    
    final_count = len(df)
    logger.info(f"Data validation: {initial_count} -> {final_count} rows "
                f"({(initial_count - final_count) / initial_count * 100:.1f}% filtered)")
    
    return df

def validate_underlying_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean underlying data."""
    logger = logging.getLogger("volboost")
    
    initial_count = len(df)
    
    # Remove rows with invalid prices
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    df = df[df['high'] >= df['close']]
    df = df[df['low'] <= df['close']]
    
    # Remove rows with missing data
    df = df.dropna(subset=['close', 'open', 'high', 'low'])
    
    final_count = len(df)
    logger.info(f"Underlying data validation: {initial_count} -> {final_count} rows "
                f"({(initial_count - final_count) / initial_count * 100:.1f}% filtered)")
    
    return df

# Mathematical utilities
def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                default: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling division by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator, out=np.full_like(numerator, default), 
                          where=denominator!=0)
    return result

def clip_extreme_values(x: np.ndarray, lower_percentile: float = 1.0, 
                       upper_percentile: float = 99.0) -> np.ndarray:
    """Clip extreme values to percentiles."""
    lower_bound = np.percentile(x, lower_percentile)
    upper_bound = np.percentile(x, upper_percentile)
    return np.clip(x, lower_bound, upper_bound)

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns over specified periods."""
    return prices.pct_change(periods=periods)

def calculate_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate realized volatility over rolling window."""
    return returns.rolling(window=window).std() * np.sqrt(252)

# Random seed management
def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

# Performance utilities
def time_function(func):
    """Decorator to time function execution."""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = logging.getLogger("volboost")
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Data alignment utilities
def align_market_data(options_df: pd.DataFrame, underlying_df: pd.DataFrame, 
                     rates_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align market data to common timestamps."""
    logger = logging.getLogger("volboost")
    
    # Get common dates
    common_dates = set(options_df['date'].unique()) & set(underlying_df['date'].unique())
    if rates_df is not None:
        common_dates = common_dates & set(rates_df['date'].unique())
    
    common_dates = sorted(common_dates)
    logger.info(f"Aligned data to {len(common_dates)} common dates")
    
    # Filter data to common dates
    options_df = options_df[options_df['date'].isin(common_dates)]
    underlying_df = underlying_df[underlying_df['date'].isin(common_dates)]
    if rates_df is not None:
        rates_df = rates_df[rates_df['date'].isin(common_dates)]
    
    return options_df, underlying_df, rates_df

# Error handling
class VolBoostError(Exception):
    """Base exception for VolBoost project."""
    pass

class DataValidationError(VolBoostError):
    """Exception for data validation errors."""
    pass

class ModelError(VolBoostError):
    """Exception for model-related errors."""
    pass

class TradingError(VolBoostError):
    """Exception for trading-related errors."""
    pass
