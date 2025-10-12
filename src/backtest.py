"""
Backtesting engine with delta hedging and PnL explain.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from .utils import setup_logging
from .pricing import bs_price_greeks, calculate_position_pnl, calculate_hedge_ratio
from .select import ContractSelector

logger = setup_logging()

class Backtester:
    """Backtesting engine for options trading strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backtester.
        
        Args:
            config: Trading configuration
        """
        self.config = config or {}
        self.trading_config = self.config.get('trading', {})
        self.costs_config = self.config.get('costs', {})
        self.hedging_config = self.trading_config.get('hedging', {})
        self.lifecycle_config = self.trading_config.get('lifecycle', {})
        
        # Initialize contract selector
        self.contract_selector = ContractSelector(config)
        
        # Portfolio state
        self.positions = pd.DataFrame()
        self.cash = 0.0
        self.portfolio_history = []
        self.daily_pnl = []
        
    def run_backtest(self, signals_df: pd.DataFrame, 
                    cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Run backtest on signals data.
        
        Args:
            signals_df: DataFrame with signals and market data
            cfg: Configuration override
            
        Returns:
            DataFrame with daily portfolio PnL and positions
        """
        logger.info("Starting backtest")
        
        if cfg:
            self.config.update(cfg)
        
        # Get unique dates
        dates = sorted(signals_df['date'].unique())
        
        # Initialize portfolio
        self._initialize_portfolio()
        
        # Run backtest day by day
        for i, date in enumerate(dates):
            logger.info(f"Processing date {date} ({i+1}/{len(dates)})")
            
            # Get market data for current date
            daily_data = signals_df[signals_df['date'] == date]
            
            # Update existing positions
            self._update_positions(daily_data, date)
            
            # Exit positions based on lifecycle rules
            self._exit_positions(daily_data, date)
            
            # Select new positions
            new_positions = self._select_new_positions(daily_data, date)
            
            # Enter new positions
            self._enter_positions(new_positions, daily_data, date)
            
            # Delta hedge
            self._delta_hedge(daily_data, date)
            
            # Calculate daily PnL
            daily_pnl = self._calculate_daily_pnl(daily_data, date)
            
            # Record portfolio state
            self._record_portfolio_state(date, daily_pnl)
        
        # Create results DataFrame
        results_df = self._create_results_dataframe()
        
        logger.info("Backtest completed")
        return results_df
    
    def _initialize_portfolio(self):
        """Initialize portfolio state."""
        self.positions = pd.DataFrame()
        self.cash = 0.0
        self.portfolio_history = []
        self.daily_pnl = []
    
    def _update_positions(self, daily_data: pd.DataFrame, date: pd.Timestamp):
        """Update existing positions with current market data."""
        if len(self.positions) == 0:
            return
        
        # Update position prices and Greeks
        for idx, position in self.positions.iterrows():
            # Find current market data for this position
            position_data = daily_data[
                (daily_data['strike'] == position['strike']) &
                (daily_data['call_put'] == position['call_put']) &
                (daily_data['expiry'] == position['expiry'])
            ]
            
            if len(position_data) > 0:
                current_data = position_data.iloc[0]
                
                # Update position with current market data
                self.positions.loc[idx, 'current_price'] = current_data['mid']
                self.positions.loc[idx, 'current_delta'] = current_data.get('delta', position['delta'])
                self.positions.loc[idx, 'current_vega'] = current_data.get('vega', position['vega'])
                self.positions.loc[idx, 'current_theta'] = current_data.get('theta', position['theta'])
                self.positions.loc[idx, 'current_underlying_price'] = current_data['close']
    
    def _exit_positions(self, daily_data: pd.DataFrame, date: pd.Timestamp):
        """Exit positions based on lifecycle rules."""
        if len(self.positions) == 0:
            return
        
        positions_to_exit = []
        
        for idx, position in self.positions.iterrows():
            should_exit = False
            exit_reason = ""
            
            # Check DTE
            if position['dte'] <= self.lifecycle_config.get('exit_dte', 5):
                should_exit = True
                exit_reason = "DTE"
            
            # Check holding period
            holding_days = (date - position['entry_date']).days
            if holding_days >= self.lifecycle_config.get('max_holding_days', 10):
                should_exit = True
                exit_reason = "Max holding period"
            
            # Check signal reversal
            if 'expected_pnl' in position and position['expected_pnl'] < self.lifecycle_config.get('signal_reversal_threshold', -0.5):
                should_exit = True
                exit_reason = "Signal reversal"
            
            # Check illiquidity
            if position.get('spread_pct', 0) > 0.5:  # 50% spread
                should_exit = True
                exit_reason = "Illiquidity"
            
            if should_exit:
                positions_to_exit.append((idx, exit_reason))
        
        # Exit positions
        for idx, exit_reason in positions_to_exit:
            self._exit_position(idx, daily_data, date, exit_reason)
    
    def _exit_position(self, position_idx: int, daily_data: pd.DataFrame, 
                      date: pd.Timestamp, exit_reason: str):
        """Exit a single position."""
        position = self.positions.loc[position_idx]
        
        # Find current market data
        position_data = daily_data[
            (daily_data['strike'] == position['strike']) &
            (daily_data['call_put'] == position['call_put']) &
            (daily_data['expiry'] == position['expiry'])
        ]
        
        if len(position_data) > 0:
            current_data = position_data.iloc[0]
            exit_price = current_data['mid']
            
            # Calculate exit PnL
            exit_pnl = (exit_price - position['entry_price']) * position['quantity'] * 100
            
            # Calculate costs
            exit_costs = self._calculate_trading_costs(
                position['quantity'], exit_price, current_data['spread']
            )
            
            # Net exit PnL
            net_exit_pnl = exit_pnl - exit_costs
            
            # Update cash
            self.cash += net_exit_pnl
            
            # Record exit
            logger.info(f"Exited position: {position['strike']} {position['call_put']} "
                       f"for {exit_reason}, PnL: {net_exit_pnl:.2f}")
        
        # Remove position
        self.positions = self.positions.drop(position_idx)
    
    def _select_new_positions(self, daily_data: pd.DataFrame, 
                            date: pd.Timestamp) -> pd.DataFrame:
        """Select new positions for the day."""
        # Add expected PnL to daily data
        if 'expected_pnl' not in daily_data.columns:
            # Use a simple heuristic if no ML signals available
            daily_data = daily_data.copy()
            daily_data['expected_pnl'] = np.random.normal(0, 0.1, len(daily_data))
        
        # Add Greeks if not present
        if 'delta' not in daily_data.columns:
            daily_data = self._calculate_greeks(daily_data)
        
        # Select contracts
        selected_contracts = self.contract_selector.select_contracts(daily_data)
        
        return selected_contracts
    
    def _enter_positions(self, new_positions: pd.DataFrame, 
                        daily_data: pd.DataFrame, date: pd.Timestamp):
        """Enter new positions."""
        if len(new_positions) == 0:
            return
        
        for _, position in new_positions.iterrows():
            # Calculate entry costs
            entry_costs = self._calculate_trading_costs(
                position['target_quantity'], position['mid'], position['spread']
            )
            
            # Calculate position value
            position_value = position['target_quantity'] * position['mid'] * 100
            
            # Check if we have enough cash
            if self.cash >= position_value + entry_costs:
                # Create position record
                position_record = {
                    'entry_date': date,
                    'strike': position['strike'],
                    'call_put': position['call_put'],
                    'expiry': position['expiry'],
                    'quantity': position['target_quantity'],
                    'entry_price': position['mid'],
                    'current_price': position['mid'],
                    'delta': position.get('delta', 0),
                    'vega': position.get('vega', 0),
                    'theta': position.get('theta', 0),
                    'current_delta': position.get('delta', 0),
                    'current_vega': position.get('vega', 0),
                    'current_theta': position.get('theta', 0),
                    'expected_pnl': position.get('expected_pnl', 0),
                    'current_underlying_price': position['close'],
                    'hedge_quantity': 0.0
                }
                
                # Add to positions
                self.positions = pd.concat([
                    self.positions, 
                    pd.DataFrame([position_record])
                ], ignore_index=True)
                
                # Update cash
                self.cash -= (position_value + entry_costs)
                
                logger.info(f"Entered position: {position['strike']} {position['call_put']} "
                           f"qty: {position['target_quantity']}")
    
    def _delta_hedge(self, daily_data: pd.DataFrame, date: pd.Timestamp):
        """Perform delta hedging."""
        if len(self.positions) == 0:
            return
        
        # Calculate total portfolio delta
        total_delta = (self.positions['quantity'] * self.positions['current_delta'] * 100).sum()
        
        # Check if hedging is needed
        max_delta = self.hedging_config.get('max_delta', 1000)
        hedge_threshold = self.hedging_config.get('hedge_threshold', 0.1)
        
        if abs(total_delta) > max_delta * hedge_threshold:
            # Calculate hedge quantity
            hedge_quantity = -total_delta
            
            # Calculate hedge costs
            underlying_price = daily_data['close'].iloc[0] if len(daily_data) > 0 else 4000
            hedge_costs = self._calculate_hedge_costs(abs(hedge_quantity), underlying_price)
            
            # Update cash
            self.cash -= hedge_costs
            
            # Update hedge quantity in positions
            self.positions['hedge_quantity'] = hedge_quantity / len(self.positions)
            
            logger.info(f"Delta hedge: {hedge_quantity:.0f} shares, cost: {hedge_costs:.2f}")
    
    def _calculate_daily_pnl(self, daily_data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Calculate daily PnL breakdown."""
        if len(self.positions) == 0:
            return {
                'total_pnl': 0.0,
                'option_pnl': 0.0,
                'hedge_pnl': 0.0,
                'theta_pnl': 0.0,
                'costs': 0.0,
                'residual_pnl': 0.0
            }
        
        total_option_pnl = 0.0
        total_hedge_pnl = 0.0
        total_theta_pnl = 0.0
        total_costs = 0.0
        
        underlying_price = daily_data['close'].iloc[0] if len(daily_data) > 0 else 4000
        underlying_change = 0.0  # Simplified - would need previous day's price
        
        for _, position in self.positions.iterrows():
            # Option PnL
            option_pnl = (position['current_price'] - position['entry_price']) * position['quantity'] * 100
            total_option_pnl += option_pnl
            
            # Hedge PnL
            hedge_pnl = position['hedge_quantity'] * underlying_change
            total_hedge_pnl += hedge_pnl
            
            # Theta PnL (simplified)
            theta_pnl = position['current_theta'] * position['quantity'] * 100 / 365.0
            total_theta_pnl += theta_pnl
        
        # Total PnL
        total_pnl = total_option_pnl + total_hedge_pnl + total_theta_pnl - total_costs
        residual_pnl = total_pnl - (total_option_pnl + total_hedge_pnl + total_theta_pnl - total_costs)
        
        return {
            'total_pnl': total_pnl,
            'option_pnl': total_option_pnl,
            'hedge_pnl': total_hedge_pnl,
            'theta_pnl': total_theta_pnl,
            'costs': total_costs,
            'residual_pnl': residual_pnl
        }
    
    def _record_portfolio_state(self, date: pd.Timestamp, daily_pnl: Dict[str, float]):
        """Record portfolio state for the day."""
        portfolio_state = {
            'date': date,
            'cash': self.cash,
            'n_positions': len(self.positions),
            'total_vega': (self.positions['quantity'] * self.positions['current_vega'] * 100).sum() if len(self.positions) > 0 else 0,
            'total_delta': (self.positions['quantity'] * self.positions['current_delta'] * 100).sum() if len(self.positions) > 0 else 0,
            **daily_pnl
        }
        
        self.portfolio_history.append(portfolio_state)
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create results DataFrame from portfolio history."""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate cumulative PnL
        results_df['cumulative_pnl'] = results_df['total_pnl'].cumsum()
        
        # Calculate returns
        results_df['daily_return'] = results_df['total_pnl'] / 1000000  # Assume $1M capital
        
        # Calculate drawdown
        results_df['cumulative_max'] = results_df['cumulative_pnl'].cummax()
        results_df['drawdown'] = results_df['cumulative_pnl'] - results_df['cumulative_max']
        
        return results_df
    
    def _calculate_trading_costs(self, quantity: float, price: float, spread: float) -> float:
        """Calculate trading costs for options."""
        options_config = self.costs_config.get('options', {})
        
        # Half spread cost
        spread_cost = (spread / 2) * quantity * 100 if options_config.get('half_spread', True) else 0
        
        # Commission
        per_contract_fee = options_config.get('per_contract_fee', 0.65)
        commission = per_contract_fee * quantity
        
        # Apply min/max commission
        min_commission = options_config.get('min_commission', 1.0)
        max_commission = options_config.get('max_commission', 50.0)
        commission = max(min_commission, min(commission, max_commission))
        
        return spread_cost + commission
    
    def _calculate_hedge_costs(self, quantity: float, price: float) -> float:
        """Calculate hedging costs for underlying."""
        underlying_config = self.costs_config.get('underlying', {})
        
        # Half spread cost
        spread = price * 0.001  # Assume 0.1% spread
        spread_cost = (spread / 2) * quantity if underlying_config.get('half_spread', True) else 0
        
        # Commission
        commission_per_share = underlying_config.get('commission_per_share', 0.005)
        commission = commission_per_share * quantity
        
        # Apply min/max commission
        min_commission = underlying_config.get('min_commission', 1.0)
        max_commission = underlying_config.get('max_commission', 50.0)
        commission = max(min_commission, min(commission, max_commission))
        
        return spread_cost + commission
    
    def _calculate_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Greeks for options data."""
        greeks_data = []
        
        for _, row in df.iterrows():
            greeks = bs_price_greeks(
                S=row['close'],
                K=row['strike'],
                T=row['dte'] / 365.0,
                r=row.get('r_annualized', 0.05),
                q=row.get('q_div_yield', 0.02),
                sigma=row['impl_vol'],
                cp=row['call_put']
            )
            
            greeks_data.append({
                'delta': greeks['delta'],
                'vega': greeks['vega'],
                'theta': greeks['theta']
            })
        
        greeks_df = pd.DataFrame(greeks_data)
        return pd.concat([df.reset_index(drop=True), greeks_df], axis=1)
    
    def pnl_explain(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explain PnL components for positions.
        
        Args:
            positions_df: DataFrame with position data
            
        Returns:
            DataFrame with PnL explanation
        """
        logger.info("Explaining PnL components")
        
        explanations = []
        
        for _, position in positions_df.iterrows():
            # Calculate PnL components
            underlying_change = position.get('underlying_change', 0)
            vol_change = position.get('vol_change', 0)
            time_decay = position.get('time_decay', 1)
            
            pnl_components = calculate_position_pnl(
                entry_price=position['entry_price'],
                current_price=position['current_price'],
                quantity=position['quantity'],
                delta=position['delta'],
                vega=position['vega'],
                theta=position['theta'],
                underlying_change=underlying_change,
                vol_change=vol_change,
                time_decay=time_decay
            )
            
            explanation = {
                'strike': position['strike'],
                'call_put': position['call_put'],
                'quantity': position['quantity'],
                **pnl_components
            }
            
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)
