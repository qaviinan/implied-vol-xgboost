"""
Risk metrics including VaR/ES and stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from scipy.optimize import minimize
import logging
from .utils import setup_logging

logger = setup_logging()

class RiskManager:
    """Risk management and monitoring for options portfolio."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config or {}
        self.risk_config = self.config.get('risk', {})
        
    def var_es(self, pnl: pd.Series, p: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES).
        
        Args:
            pnl: Series of daily PnL
            p: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            Tuple of (VaR, ES)
        """
        if len(pnl) == 0:
            return 0.0, 0.0
        
        # Calculate VaR (negative of the p-th percentile)
        var = -np.percentile(pnl, (1 - p) * 100)
        
        # Calculate ES (average of losses beyond VaR)
        losses_beyond_var = pnl[pnl <= -var]
        es = -losses_beyond_var.mean() if len(losses_beyond_var) > 0 else var
        
        return var, es
    
    def rolling_var_es(self, pnl: pd.Series, window: int = 252, 
                      p: float = 0.95) -> pd.DataFrame:
        """
        Calculate rolling VaR and ES.
        
        Args:
            pnl: Series of daily PnL
            window: Rolling window size
            p: Confidence level
            
        Returns:
            DataFrame with rolling VaR and ES
        """
        rolling_var = []
        rolling_es = []
        dates = []
        
        for i in range(window, len(pnl)):
            window_pnl = pnl.iloc[i-window:i]
            var, es = self.var_es(window_pnl, p)
            
            rolling_var.append(var)
            rolling_es.append(es)
            dates.append(pnl.index[i])
        
        return pd.DataFrame({
            'date': dates,
            'var': rolling_var,
            'es': rolling_es
        })
    
    def exception_tests(self, pnl: pd.Series, var_series: pd.Series) -> Dict[str, float]:
        """
        Perform exception tests for VaR backtesting.
        
        Args:
            pnl: Series of daily PnL
            var_series: Series of VaR estimates
            
        Returns:
            Dictionary with test results
        """
        # Align data
        common_dates = pnl.index.intersection(var_series.index)
        pnl_aligned = pnl.loc[common_dates]
        var_aligned = var_series.loc[common_dates]
        
        # Count exceptions (losses exceeding VaR)
        exceptions = (pnl_aligned < -var_aligned).sum()
        total_observations = len(pnl_aligned)
        exception_rate = exceptions / total_observations
        
        # Kupiec test (unconditional coverage)
        expected_exceptions = total_observations * (1 - 0.95)  # Assuming 95% VaR
        kupiec_lr = self._kupiec_likelihood_ratio(exceptions, total_observations, 0.05)
        kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_lr, 1)
        
        # Christoffersen test (conditional coverage)
        christoffersen_lr = self._christoffersen_likelihood_ratio(pnl_aligned, var_aligned)
        christoffersen_pvalue = 1 - stats.chi2.cdf(christoffersen_lr, 2)
        
        return {
            'exceptions': exceptions,
            'total_observations': total_observations,
            'exception_rate': exception_rate,
            'expected_exception_rate': 0.05,  # 5% for 95% VaR
            'kupiec_lr': kupiec_lr,
            'kupiec_pvalue': kupiec_pvalue,
            'christoffersen_lr': christoffersen_lr,
            'christoffersen_pvalue': christoffersen_pvalue
        }
    
    def _kupiec_likelihood_ratio(self, exceptions: int, total: int, 
                                expected_rate: float) -> float:
        """Calculate Kupiec likelihood ratio statistic."""
        if exceptions == 0:
            exceptions = 1e-8
        if exceptions == total:
            exceptions = total - 1e-8
        
        # Likelihood under null hypothesis
        l0 = exceptions * np.log(expected_rate) + (total - exceptions) * np.log(1 - expected_rate)
        
        # Likelihood under alternative hypothesis
        actual_rate = exceptions / total
        l1 = exceptions * np.log(actual_rate) + (total - exceptions) * np.log(1 - actual_rate)
        
        # Likelihood ratio
        lr = -2 * (l0 - l1)
        return lr
    
    def _christoffersen_likelihood_ratio(self, pnl: pd.Series, var: pd.Series) -> float:
        """Calculate Christoffersen likelihood ratio statistic."""
        # Create binary series for exceptions
        exceptions = (pnl < -var).astype(int)
        
        # Count transitions
        n00 = ((exceptions[:-1] == 0) & (exceptions[1:] == 0)).sum()
        n01 = ((exceptions[:-1] == 0) & (exceptions[1:] == 1)).sum()
        n10 = ((exceptions[:-1] == 1) & (exceptions[1:] == 0)).sum()
        n11 = ((exceptions[:-1] == 1) & (exceptions[1:] == 1)).sum()
        
        # Calculate transition probabilities
        if n00 + n01 > 0:
            p01 = n01 / (n00 + n01)
        else:
            p01 = 0
        
        if n10 + n11 > 0:
            p11 = n11 / (n10 + n11)
        else:
            p11 = 0
        
        # Unconditional probability
        p = exceptions.mean()
        
        # Likelihood under null hypothesis (independence)
        l0 = n00 * np.log(1 - p) + n01 * np.log(p) + n10 * np.log(1 - p) + n11 * np.log(p)
        
        # Likelihood under alternative hypothesis (Markov chain)
        l1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
        
        # Likelihood ratio
        lr = -2 * (l0 - l1)
        return lr
    
    def stress_report(self, positions: pd.DataFrame, 
                     scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Generate stress test report for different scenarios.
        
        Args:
            positions: DataFrame with current positions
            scenarios: Dictionary of stress scenarios
            
        Returns:
            DataFrame with stress test results
        """
        logger.info("Generating stress test report")
        
        stress_results = []
        
        for scenario_name, scenario_params in scenarios.items():
            # Calculate PnL under stress scenario
            stress_pnl = self._calculate_stress_pnl(positions, scenario_params)
            
            stress_results.append({
                'scenario': scenario_name,
                'total_pnl': stress_pnl,
                'max_loss': min(0, stress_pnl),
                'scenario_params': str(scenario_params)
            })
        
        return pd.DataFrame(stress_results)
    
    def _calculate_stress_pnl(self, positions: pd.DataFrame, 
                            scenario_params: Dict[str, float]) -> float:
        """Calculate PnL under stress scenario."""
        total_pnl = 0.0
        
        for _, position in positions.iterrows():
            # Apply stress scenario to implied volatility
            vol_shock = scenario_params.get('vol_shock', 0.0)
            new_impl_vol = position.get('impl_vol', 0.2) * (1 + vol_shock)
            
            # Apply stress scenario to underlying price
            underlying_shock = scenario_params.get('underlying_shock', 0.0)
            new_underlying_price = position.get('underlying_price', 4000) * (1 + underlying_shock)
            
            # Calculate new option price using Black-Scholes
            from .pricing import bs_price_greeks
            
            new_price = bs_price_greeks(
                S=new_underlying_price,
                K=position['strike'],
                T=position.get('dte', 30) / 365.0,
                r=position.get('r_annualized', 0.05),
                q=position.get('q_div_yield', 0.02),
                sigma=new_impl_vol,
                cp=position['call_put']
            )['price']
            
            # Calculate PnL
            position_pnl = (new_price - position.get('entry_price', position.get('current_price', 0))) * position.get('quantity', 0) * 100
            total_pnl += position_pnl
        
        return total_pnl
    
    def calculate_portfolio_risk_metrics(self, pnl: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics."""
        if len(pnl) == 0:
            return {}
        
        # Basic statistics
        mean_pnl = pnl.mean()
        std_pnl = pnl.std()
        
        # VaR and ES
        var_95, es_95 = self.var_es(pnl, 0.95)
        var_99, es_99 = self.var_es(pnl, 0.99)
        
        # Drawdown metrics
        cumulative_pnl = pnl.cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = pnl[pnl < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = mean_pnl / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = mean_pnl * 252 / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Skewness and kurtosis
        skewness = stats.skew(pnl)
        kurtosis = stats.kurtosis(pnl)
        
        # Tail risk metrics
        tail_ratio = es_95 / var_95 if var_95 > 0 else 0
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio
        }
    
    def monte_carlo_var(self, positions: pd.DataFrame, n_simulations: int = 10000,
                       confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            positions: DataFrame with positions
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for VaR
            
        Returns:
            Dictionary with Monte Carlo VaR results
        """
        logger.info(f"Running Monte Carlo VaR with {n_simulations} simulations")
        
        # Generate random scenarios
        np.random.seed(42)
        
        # Simulate underlying price changes
        underlying_returns = np.random.normal(0, 0.02, n_simulations)  # 2% daily volatility
        
        # Simulate volatility changes
        vol_changes = np.random.normal(0, 0.1, n_simulations)  # 10% vol of vol
        
        # Calculate PnL for each simulation
        simulation_pnls = []
        
        for i in range(n_simulations):
            total_pnl = 0.0
            
            for _, position in positions.iterrows():
                # Apply random shocks
                underlying_shock = underlying_returns[i]
                vol_shock = vol_changes[i]
                
                new_underlying_price = position.get('underlying_price', 4000) * (1 + underlying_shock)
                new_impl_vol = position.get('impl_vol', 0.2) * (1 + vol_shock)
                
                # Calculate new option price
                from .pricing import bs_price_greeks
                
                new_price = bs_price_greeks(
                    S=new_underlying_price,
                    K=position['strike'],
                    T=position.get('dte', 30) / 365.0,
                    r=position.get('r_annualized', 0.05),
                    q=position.get('q_div_yield', 0.02),
                    sigma=new_impl_vol,
                    cp=position['call_put']
                )['price']
                
                # Calculate PnL
                position_pnl = (new_price - position.get('entry_price', position.get('current_price', 0))) * position.get('quantity', 0) * 100
                total_pnl += position_pnl
            
            simulation_pnls.append(total_pnl)
        
        # Calculate VaR and ES
        simulation_pnls = np.array(simulation_pnls)
        var = -np.percentile(simulation_pnls, (1 - confidence_level) * 100)
        es = -simulation_pnls[simulation_pnls <= -var].mean()
        
        return {
            'var': var,
            'es': es,
            'mean_pnl': simulation_pnls.mean(),
            'std_pnl': simulation_pnls.std(),
            'min_pnl': simulation_pnls.min(),
            'max_pnl': simulation_pnls.max()
        }
    
    def check_risk_limits(self, positions: pd.DataFrame, 
                         pnl: pd.Series) -> Dict[str, Any]:
        """Check if portfolio violates risk limits."""
        risk_limits = self.risk_config
        
        violations = []
        
        # Check VaR limits
        var_95, _ = self.var_es(pnl, 0.95)
        max_var_95 = risk_limits.get('max_var_95', 50000)
        if var_95 > max_var_95:
            violations.append(f"VaR 95% exceeds limit: {var_95:.2f} > {max_var_95}")
        
        var_99, _ = self.var_es(pnl, 0.99)
        max_var_99 = risk_limits.get('max_var_99', 100000)
        if var_99 > max_var_99:
            violations.append(f"VaR 99% exceeds limit: {var_99:.2f} > {max_var_99}")
        
        # Check ES limits
        _, es_95 = self.var_es(pnl, 0.95)
        max_es_95 = risk_limits.get('max_es_95', 75000)
        if es_95 > max_es_95:
            violations.append(f"ES 95% exceeds limit: {es_95:.2f} > {max_es_95}")
        
        # Check drawdown limit
        cumulative_pnl = pnl.cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_limit = risk_limits.get('max_drawdown', 0.15)
        if abs(max_drawdown) > max_drawdown_limit * 1000000:  # Assume $1M capital
            violations.append(f"Max drawdown exceeds limit: {abs(max_drawdown):.2f} > {max_drawdown_limit * 1000000}")
        
        return {
            'violations': violations,
            'has_violations': len(violations) > 0,
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'max_drawdown': max_drawdown
        }
