"""
VolBoost: Options Volatility Surface ML Strategy

A comprehensive framework for building and backtesting options trading strategies
using SVI volatility surface calibration and machine learning signals.
"""

__version__ = "1.0.0"
__author__ = "VolBoost Team"

# Import main modules
from .ingest import DataIngestion
from .svi import SVICalibrator, ChallengerSurface
from .surface import VolatilitySurface, SurfaceBuilder
from .pricing import (
    bs_price_greeks, implied_volatility, parity_checks,
    bump_and_revalue_greeks, validate_greeks, calculate_portfolio_greeks
)
from .features import FeatureEngineer
from .model import XGBoostModel, PurgedTimeSeriesSplit
from .select import ContractSelector
from .backtest import Backtester
from .risk import RiskManager
from .report import ReportGenerator
from .utils import setup_logging, load_config, get_config_path

__all__ = [
    'DataIngestion',
    'SVICalibrator',
    'ChallengerSurface', 
    'VolatilitySurface',
    'SurfaceBuilder',
    'bs_price_greeks',
    'implied_volatility',
    'parity_checks',
    'bump_and_revalue_greeks',
    'validate_greeks',
    'calculate_portfolio_greeks',
    'FeatureEngineer',
    'XGBoostModel',
    'PurgedTimeSeriesSplit',
    'ContractSelector',
    'Backtester',
    'RiskManager',
    'ReportGenerator',
    'setup_logging',
    'load_config',
    'get_config_path'
]
