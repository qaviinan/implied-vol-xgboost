"""
Feature engineering and label generation for XGBoost model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import logging
from .utils import setup_logging, calculate_returns, calculate_realized_volatility
from .surface import VolatilitySurface
from .pricing import bs_price_greeks

logger = setup_logging()

class FeatureEngineer:
    """Feature engineering for options trading signals."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_config = self.config.get('features', {})
        
    def build_features_labels(self, chain_df: pd.DataFrame, 
                            surface_fn: callable,
                            underlying_df: pd.DataFrame,
                            rates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features and labels for machine learning model.
        
        Args:
            chain_df: Options chain data
            surface_fn: Volatility surface function
            underlying_df: Underlying asset data
            rates_df: Risk-free rates data
            
        Returns:
            DataFrame with features and labels
        """
        logger.info("Building features and labels")
        
        # Merge data
        df = self._merge_market_data(chain_df, underlying_df, rates_df)
        
        # Calculate basic features
        df = self._calculate_basic_features(df)
        
        # Calculate surface features
        df = self._calculate_surface_features(df, surface_fn)
        
        # Calculate realized volatility features
        df = self._calculate_realized_vol_features(df)
        
        # Calculate trading features
        df = self._calculate_trading_features(df)
        
        # Calculate market features
        df = self._calculate_market_features(df)
        
        # Generate labels
        df = self._generate_labels(df)
        
        # Remove rows with missing features
        df = df.dropna()
        
        logger.info(f"Built features and labels: {len(df)} rows")
        return df
    
    def _merge_market_data(self, chain_df: pd.DataFrame, 
                          underlying_df: pd.DataFrame,
                          rates_df: pd.DataFrame) -> pd.DataFrame:
        """Merge options, underlying, and rates data."""
        # Merge underlying data
        df = chain_df.merge(underlying_df, on='date', how='left')
        
        # Merge rates data
        df = df.merge(rates_df, on='date', how='left')
        
        # Forward fill missing rates
        df['r_annualized'] = df['r_annualized'].fillna(method='ffill')
        df['q_div_yield'] = df['q_div_yield'].fillna(method='ffill')
        
        return df
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic option features."""
        # Moneyness
        df['moneyness'] = df['strike'] / df['close']
        df['log_moneyness'] = np.log(df['moneyness'])
        
        # Time to expiry
        df['time_to_expiry'] = df['dte'] / 365.0
        
        # Option type
        df['is_call'] = (df['call_put'] == 'C').astype(int)
        df['is_put'] = (df['call_put'] == 'P').astype(int)
        
        # Spread features
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid']
        df['mid_pct'] = df['mid'] / df['close']
        
        # Volume and open interest
        df['volume_oi_ratio'] = df['volume'] / (df['open_interest'] + 1)
        df['log_volume'] = np.log(df['volume'] + 1)
        df['log_oi'] = np.log(df['open_interest'] + 1)
        
        return df
    
    def _calculate_surface_features(self, df: pd.DataFrame, 
                                  surface_fn: callable) -> pd.DataFrame:
        """Calculate volatility surface features."""
        logger.info("Calculating surface features")
        
        # Get ATM volatility for each date/expiry
        df['atm_iv'] = df.apply(
            lambda row: surface_fn(row['close'], row['time_to_expiry']), axis=1
        )
        
        # Relative implied volatility
        df['rel_iv'] = df['impl_vol'] / df['atm_iv']
        df['iv_skew'] = df['impl_vol'] - df['atm_iv']
        
        # Smile slope and curvature (simplified)
        df['smile_slope'] = df.apply(
            lambda row: self._calculate_smile_slope(row, surface_fn), axis=1
        )
        
        df['smile_curvature'] = df.apply(
            lambda row: self._calculate_smile_curvature(row, surface_fn), axis=1
        )
        
        # Term structure slope
        df['term_slope'] = df.apply(
            lambda row: self._calculate_term_slope(row, surface_fn), axis=1
        )
        
        return df
    
    def _calculate_smile_slope(self, row: pd.Series, surface_fn: callable) -> float:
        """Calculate smile slope at given strike."""
        try:
            # Calculate IV at current strike and slightly higher strike
            iv_current = surface_fn(row['strike'], row['time_to_expiry'])
            iv_higher = surface_fn(row['strike'] * 1.01, row['time_to_expiry'])
            
            # Calculate slope
            slope = (iv_higher - iv_current) / (row['strike'] * 0.01)
            return slope
        except:
            return 0.0
    
    def _calculate_smile_curvature(self, row: pd.Series, surface_fn: callable) -> float:
        """Calculate smile curvature at given strike."""
        try:
            # Calculate IV at three points
            iv_lower = surface_fn(row['strike'] * 0.99, row['time_to_expiry'])
            iv_current = surface_fn(row['strike'], row['time_to_expiry'])
            iv_higher = surface_fn(row['strike'] * 1.01, row['time_to_expiry'])
            
            # Calculate curvature (second derivative approximation)
            curvature = (iv_higher - 2 * iv_current + iv_lower) / (row['strike'] * 0.01)**2
            return curvature
        except:
            return 0.0
    
    def _calculate_term_slope(self, row: pd.Series, surface_fn: callable) -> float:
        """Calculate term structure slope."""
        try:
            # Calculate IV at current expiry and slightly longer expiry
            iv_current = surface_fn(row['strike'], row['time_to_expiry'])
            iv_longer = surface_fn(row['strike'], row['time_to_expiry'] + 0.01)
            
            # Calculate slope
            slope = (iv_longer - iv_current) / 0.01
            return slope
        except:
            return 0.0
    
    def _calculate_realized_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility features."""
        logger.info("Calculating realized volatility features")
        
        # Sort by date for rolling calculations
        df = df.sort_values(['date', 'strike', 'call_put'])
        
        # Calculate underlying returns
        df['underlying_return'] = df.groupby('date')['close'].transform(
            lambda x: x.pct_change()
        )
        
        # Calculate realized volatility for different windows
        vol_windows = self.feature_config.get('realized_vol', {}).get('windows', [5, 10, 20, 30])
        
        for window in vol_windows:
            # Calculate rolling realized volatility
            df[f'realized_vol_{window}d'] = df.groupby('date')['close'].transform(
                lambda x: x.pct_change().rolling(window=window).std() * np.sqrt(252)
            )
            
            # HV-IV gap
            df[f'hv_iv_gap_{window}d'] = df[f'realized_vol_{window}d'] - df['impl_vol']
        
        # Skew z-score
        df['skew_zscore'] = df.groupby('date')['iv_skew'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        return df
    
    def _calculate_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading-related features."""
        logger.info("Calculating trading features")
        
        # Moneyness buckets
        df['moneyness_bucket'] = pd.cut(
            df['moneyness'], 
            bins=[0, 0.9, 0.95, 1.0, 1.05, 1.1, 2.0], 
            labels=['deep_otm', 'otm', 'near_otm', 'atm', 'near_itm', 'itm']
        )
        
        # DTE buckets
        df['dte_bucket'] = pd.cut(
            df['dte'],
            bins=[0, 7, 14, 30, 60, 90, 365],
            labels=['weekly', 'biweekly', 'monthly', 'bimonthly', 'quarterly', 'long_term']
        )
        
        # Liquidity features
        df['liquidity_score'] = (
            df['volume'] / (df['volume'].max() + 1) * 0.5 +
            df['open_interest'] / (df['open_interest'].max() + 1) * 0.5
        )
        
        # Put-call ratio
        df['put_call_ratio'] = df.groupby(['date', 'strike'])['is_put'].transform('sum') / \
                              df.groupby(['date', 'strike'])['is_call'].transform('sum')
        
        return df
    
    def _calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-wide features."""
        logger.info("Calculating market features")
        
        # Underlying returns for different periods
        return_periods = self.feature_config.get('market', {}).get('underlying_returns', [1, 5, 10])
        
        for period in return_periods:
            df[f'underlying_return_{period}d'] = df.groupby('date')['close'].transform(
                lambda x: x.pct_change(periods=period)
            )
        
        # RV ratio (current vs historical)
        df['rv_ratio'] = df['realized_vol_20d'] / df['realized_vol_20d'].rolling(252).mean()
        
        # Volatility regime
        df['vol_regime'] = pd.cut(
            df['realized_vol_20d'],
            bins=3,
            labels=['low_vol', 'medium_vol', 'high_vol']
        )
        
        # Market stress indicators
        df['market_stress'] = (
            (df['realized_vol_20d'] > df['realized_vol_20d'].quantile(0.8)).astype(int) +
            (df['spread_pct'] > df['spread_pct'].quantile(0.8)).astype(int)
        )
        
        return df
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels for machine learning."""
        logger.info("Generating labels")
        
        label_config = self.config.get('labels', {})
        label_type = label_config.get('type', 'delta_hedged_return')
        lookforward_days = label_config.get('lookforward_days', 1)
        
        if label_type == 'delta_hedged_return':
            df = self._generate_delta_hedged_labels(df, lookforward_days)
        elif label_type == 'vol_change':
            df = self._generate_vol_change_labels(df, lookforward_days)
        else:
            raise ValueError(f"Unknown label type: {label_type}")
        
        return df
    
    def _generate_delta_hedged_labels(self, df: pd.DataFrame, 
                                    lookforward_days: int) -> pd.DataFrame:
        """Generate delta-hedged return labels."""
        # Sort by date for forward-looking calculations
        df = df.sort_values(['date', 'strike', 'call_put'])
        
        # Calculate next-day option prices (simplified)
        df['next_option_price'] = df.groupby(['strike', 'call_put'])['mid'].shift(-lookforward_days)
        df['next_underlying_price'] = df.groupby('date')['close'].shift(-lookforward_days)
        
        # Calculate current Greeks
        df['current_delta'] = df.apply(
            lambda row: bs_price_greeks(
                row['close'], row['strike'], row['time_to_expiry'],
                row['r_annualized'], row['q_div_yield'], row['impl_vol'], row['call_put']
            )['delta'], axis=1
        )
        
        # Calculate delta-hedged PnL
        df['option_pnl'] = df['next_option_price'] - df['mid']
        df['hedge_pnl'] = -df['current_delta'] * (df['next_underlying_price'] - df['close'])
        df['delta_hedged_pnl'] = df['option_pnl'] + df['hedge_pnl']
        
        # Scale by option premium
        df['delta_hedged_return'] = df['delta_hedged_pnl'] / (df['mid'] + 1e-8)
        
        # Remove rows where we can't calculate next-day values
        df = df.dropna(subset=['delta_hedged_return'])
        
        return df
    
    def _generate_vol_change_labels(self, df: pd.DataFrame, 
                                  lookforward_days: int) -> pd.DataFrame:
        """Generate implied volatility change labels."""
        # Sort by date for forward-looking calculations
        df = df.sort_values(['date', 'strike', 'call_put'])
        
        # Calculate next-day implied volatility
        df['next_impl_vol'] = df.groupby(['strike', 'call_put'])['impl_vol'].shift(-lookforward_days)
        
        # Calculate volatility change
        df['vol_change'] = df['next_impl_vol'] - df['impl_vol']
        df['vol_change_pct'] = df['vol_change'] / (df['impl_vol'] + 1e-8)
        
        # Remove rows where we can't calculate next-day values
        df = df.dropna(subset=['vol_change'])
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features for model training."""
        # Define feature columns
        feature_cols = [
            # Basic features
            'moneyness', 'log_moneyness', 'time_to_expiry', 'is_call', 'is_put',
            'spread_pct', 'mid_pct', 'volume_oi_ratio', 'log_volume', 'log_oi',
            
            # Surface features
            'atm_iv', 'rel_iv', 'iv_skew', 'smile_slope', 'smile_curvature', 'term_slope',
            
            # Realized vol features
            'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d', 'realized_vol_30d',
            'hv_iv_gap_5d', 'hv_iv_gap_10d', 'hv_iv_gap_20d', 'hv_iv_gap_30d',
            'skew_zscore',
            
            # Trading features
            'liquidity_score', 'put_call_ratio',
            
            # Market features
            'underlying_return_1d', 'underlying_return_5d', 'underlying_return_10d',
            'rv_ratio', 'market_stress'
        ]
        
        # Add categorical features
        categorical_features = ['moneyness_bucket', 'dte_bucket', 'vol_regime']
        
        # One-hot encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())
        
        # Select only available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Add label column
        label_col = 'delta_hedged_return' if 'delta_hedged_return' in df.columns else 'vol_change'
        
        # Select final columns
        final_cols = ['date', 'strike', 'call_put', 'expiry'] + available_features + [label_col]
        final_cols = [col for col in final_cols if col in df.columns]
        
        return df[final_cols]
