"""
Signal ranking and contract selection for trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from .utils import setup_logging

logger = setup_logging()

class ContractSelector:
    """Contract selection based on ML signals and liquidity filters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize contract selector.
        
        Args:
            config: Trading configuration
        """
        self.config = config or {}
        self.selection_config = self.config.get('selection', {})
        self.sizing_config = self.config.get('sizing', {})
        
    def select_contracts(self, scores_df: pd.DataFrame, 
                        cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Select contracts for trading based on signals and filters.
        
        Args:
            scores_df: DataFrame with signal scores and contract data
            cfg: Configuration override
            
        Returns:
            DataFrame with selected contracts
        """
        logger.info("Selecting contracts for trading")
        
        if cfg:
            self.config.update(cfg)
        
        # Apply liquidity filters
        filtered_df = self._apply_liquidity_filters(scores_df)
        
        # Rank by expected PnL
        ranked_df = self._rank_by_signal(filtered_df)
        
        # Select top contracts
        selected_df = self._select_top_contracts(ranked_df)
        
        # Calculate position sizes
        sized_df = self._calculate_position_sizes(selected_df)
        
        logger.info(f"Selected {len(sized_df)} contracts for trading")
        return sized_df
    
    def _apply_liquidity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply liquidity and quality filters."""
        logger.info("Applying liquidity filters")
        
        initial_count = len(df)
        
        # Minimum open interest
        min_oi = self.selection_config.get('min_open_interest', 100)
        df = df[df['open_interest'] >= min_oi]
        
        # Minimum volume
        min_volume = self.selection_config.get('min_volume', 50)
        df = df[df['volume'] >= min_volume]
        
        # Maximum spread percentage
        max_spread_pct = self.selection_config.get('max_spread_pct', 0.3)
        df = df[df['spread_pct'] <= max_spread_pct]
        
        # Minimum expected PnL
        min_expected_pnl = self.sizing_config.get('min_expected_pnl', 0.01)
        df = df[df['expected_pnl'] >= min_expected_pnl]
        
        # Remove options with very short DTE
        min_dte = 5
        df = df[df['dte'] >= min_dte]
        
        # Remove options with very long DTE (optional)
        max_dte = 365
        df = df[df['dte'] <= max_dte]
        
        final_count = len(df)
        logger.info(f"Liquidity filtering: {initial_count} -> {final_count} contracts "
                   f"({(initial_count - final_count) / initial_count * 100:.1f}% filtered)")
        
        return df
    
    def _rank_by_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank contracts by expected PnL signal."""
        logger.info("Ranking contracts by signal")
        
        # Sort by expected PnL (descending)
        df = df.sort_values('expected_pnl', ascending=False)
        
        # Calculate quantile ranks
        top_quantile = self.selection_config.get('top_quantile', 0.2)
        df['signal_rank'] = df['expected_pnl'].rank(pct=True)
        df['in_top_quantile'] = df['signal_rank'] >= (1 - top_quantile)
        
        # Add additional ranking criteria
        df['liquidity_score'] = (
            df['volume'] / df['volume'].max() * 0.4 +
            df['open_interest'] / df['open_interest'].max() * 0.4 +
            (1 - df['spread_pct']) * 0.2
        )
        
        # Combined score
        df['combined_score'] = (
            df['expected_pnl'] * 0.7 +
            df['liquidity_score'] * 0.3
        )
        
        # Re-rank by combined score
        df = df.sort_values('combined_score', ascending=False)
        df['final_rank'] = range(1, len(df) + 1)
        
        return df
    
    def _select_top_contracts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top N contracts for trading."""
        logger.info("Selecting top contracts")
        
        n_contracts = self.selection_config.get('n_contracts', 20)
        
        # Select top contracts
        selected_df = df.head(n_contracts).copy()
        
        # Ensure diversification across expiries
        selected_df = self._ensure_expiry_diversification(selected_df)
        
        # Ensure diversification across strikes
        selected_df = self._ensure_strike_diversification(selected_df)
        
        return selected_df
    
    def _ensure_expiry_diversification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure diversification across expiries."""
        max_concentration = self.config.get('constraints', {}).get('max_concentration', 0.2)
        max_per_expiry = int(len(df) * max_concentration)
        
        if max_per_expiry < 1:
            max_per_expiry = 1
        
        # Group by expiry and select top contracts from each
        diversified_df = []
        for expiry, group in df.groupby('expiry'):
            top_from_expiry = group.head(max_per_expiry)
            diversified_df.append(top_from_expiry)
        
        if diversified_df:
            return pd.concat(diversified_df, ignore_index=True)
        else:
            return df.head(1)  # Fallback to at least one contract
    
    def _ensure_strike_diversification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure diversification across strikes."""
        # Group by moneyness buckets and select from each
        df['moneyness_bucket'] = pd.cut(
            df['moneyness'],
            bins=[0, 0.9, 0.95, 1.0, 1.05, 1.1, 2.0],
            labels=['deep_otm', 'otm', 'near_otm', 'atm', 'near_itm', 'itm']
        )
        
        diversified_df = []
        for bucket, group in df.groupby('moneyness_bucket'):
            if len(group) > 0:
                # Select top contract from each bucket
                top_from_bucket = group.head(1)
                diversified_df.append(top_from_bucket)
        
        if diversified_df:
            return pd.concat(diversified_df, ignore_index=True)
        else:
            return df.head(1)  # Fallback
    
    def _calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on target vega."""
        logger.info("Calculating position sizes")
        
        target_vega = self.sizing_config.get('target_vega', 100000)
        max_contract_weight = self.sizing_config.get('max_contract_weight', 0.1)
        max_book_vega = self.sizing_config.get('max_book_vega', 200000)
        
        # Calculate vega per contract
        df['vega_per_contract'] = df['vega'] * 100  # Assuming 100 shares per contract
        
        # Calculate target quantity based on expected PnL
        total_expected_pnl = df['expected_pnl'].sum()
        if total_expected_pnl > 0:
            df['target_weight'] = df['expected_pnl'] / total_expected_pnl
        else:
            df['target_weight'] = 1.0 / len(df)  # Equal weight
        
        # Apply maximum contract weight constraint
        df['target_weight'] = np.minimum(df['target_weight'], max_contract_weight)
        
        # Normalize weights
        df['target_weight'] = df['target_weight'] / df['target_weight'].sum()
        
        # Calculate target vega allocation
        df['target_vega_allocation'] = df['target_weight'] * target_vega
        
        # Calculate target quantity
        df['target_quantity'] = df['target_vega_allocation'] / df['vega_per_contract']
        
        # Apply constraints
        df['target_quantity'] = np.maximum(df['target_quantity'], 0)  # No short positions
        df['target_quantity'] = np.round(df['target_quantity'])  # Round to whole contracts
        
        # Check total vega constraint
        total_vega = (df['target_quantity'] * df['vega_per_contract']).sum()
        if total_vega > max_book_vega:
            # Scale down proportionally
            scale_factor = max_book_vega / total_vega
            df['target_quantity'] = np.round(df['target_quantity'] * scale_factor)
        
        # Calculate final position values
        df['position_vega'] = df['target_quantity'] * df['vega_per_contract']
        df['position_value'] = df['target_quantity'] * df['mid'] * 100
        df['position_delta'] = df['target_quantity'] * df['delta'] * 100
        
        return df
    
    def validate_selection(self, selected_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate contract selection against constraints."""
        logger.info("Validating contract selection")
        
        constraints = self.config.get('constraints', {})
        
        # Check total vega
        total_vega = selected_df['position_vega'].sum()
        max_vega = constraints.get('max_book_vega', 200000)
        vega_ok = total_vega <= max_vega
        
        # Check concentration
        max_concentration = constraints.get('max_concentration', 0.2)
        max_expiry_weight = selected_df.groupby('expiry')['position_value'].sum().max()
        total_value = selected_df['position_value'].sum()
        concentration_ok = max_expiry_weight / total_value <= max_concentration if total_value > 0 else True
        
        # Check number of contracts
        n_contracts = len(selected_df)
        max_contracts = self.selection_config.get('n_contracts', 20)
        contracts_ok = n_contracts <= max_contracts
        
        return {
            'total_vega': total_vega,
            'max_vega': max_vega,
            'vega_ok': vega_ok,
            'max_expiry_weight': max_expiry_weight,
            'total_value': total_value,
            'concentration_ok': concentration_ok,
            'n_contracts': n_contracts,
            'max_contracts': max_contracts,
            'contracts_ok': contracts_ok,
            'all_constraints_ok': vega_ok and concentration_ok and contracts_ok
        }
    
    def get_selection_summary(self, selected_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of contract selection."""
        if len(selected_df) == 0:
            return {'error': 'No contracts selected'}
        
        return {
            'n_contracts': len(selected_df),
            'total_vega': selected_df['position_vega'].sum(),
            'total_value': selected_df['position_value'].sum(),
            'total_delta': selected_df['position_delta'].sum(),
            'avg_expected_pnl': selected_df['expected_pnl'].mean(),
            'total_expected_pnl': selected_df['expected_pnl'].sum(),
            'expiry_distribution': selected_df['expiry'].value_counts().to_dict(),
            'call_put_distribution': selected_df['call_put'].value_counts().to_dict(),
            'moneyness_distribution': pd.cut(
                selected_df['moneyness'],
                bins=[0, 0.9, 0.95, 1.0, 1.05, 1.1, 2.0],
                labels=['deep_otm', 'otm', 'near_otm', 'atm', 'near_itm', 'itm']
            ).value_counts().to_dict()
        }
