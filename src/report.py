"""
Summary tables and plots for backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import logging
from .utils import setup_logging
from .risk import RiskManager

logger = setup_logging()

class ReportGenerator:
    """Generate reports and visualizations for backtest results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration
        """
        self.config = config or {}
        self.risk_manager = RiskManager(config)
        
    def generate_summary_report(self, backtest_results: pd.DataFrame,
                              positions_history: List[Dict],
                              model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            backtest_results: DataFrame with daily backtest results
            positions_history: List of position snapshots
            model_metrics: Model performance metrics
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary report")
        
        # Basic performance metrics
        performance_metrics = self._calculate_performance_metrics(backtest_results)
        
        # Risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
            backtest_results['total_pnl']
        )
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(backtest_results, positions_history)
        
        # Model metrics
        model_summary = self._summarize_model_metrics(model_metrics)
        
        return {
            'performance': performance_metrics,
            'risk': risk_metrics,
            'trading': trading_metrics,
            'model': model_summary,
            'summary': self._create_executive_summary(performance_metrics, risk_metrics)
        }
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(results) == 0:
            return {}
        
        # Basic returns
        total_return = results['cumulative_pnl'].iloc[-1]
        annualized_return = (1 + total_return / 1000000) ** (252 / len(results)) - 1  # Assume $1M capital
        
        # Volatility
        daily_returns = results['daily_return']
        annualized_vol = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown metrics
        max_drawdown = results['drawdown'].min()
        max_drawdown_pct = max_drawdown / 1000000  # Assume $1M capital
        
        # Win rate
        win_rate = (results['total_pnl'] > 0).mean()
        
        # Average win/loss
        wins = results[results['total_pnl'] > 0]['total_pnl']
        losses = results[results['total_pnl'] < 0]['total_pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
    
    def _calculate_trading_metrics(self, results: pd.DataFrame, 
                                 positions_history: List[Dict]) -> Dict[str, Any]:
        """Calculate trading-specific metrics."""
        if len(results) == 0:
            return {}
        
        # Position metrics
        avg_positions = results['n_positions'].mean()
        max_positions = results['n_positions'].max()
        
        # Vega metrics
        avg_vega = results['total_vega'].mean()
        max_vega = results['total_vega'].max()
        
        # Delta metrics
        avg_delta = results['total_delta'].mean()
        max_delta = results['total_delta'].max()
        
        # Turnover (simplified)
        position_changes = results['n_positions'].diff().abs().sum()
        avg_turnover = position_changes / len(results)
        
        return {
            'avg_positions': avg_positions,
            'max_positions': max_positions,
            'avg_vega': avg_vega,
            'max_vega': max_vega,
            'avg_delta': avg_delta,
            'max_delta': max_delta,
            'avg_turnover': avg_turnover
        }
    
    def _summarize_model_metrics(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize model performance metrics."""
        if not model_metrics:
            return {}
        
        cv_results = model_metrics.get('cv_results', {})
        
        return {
            'mean_r2': cv_results.get('mean_r2', 0),
            'mean_mse': cv_results.get('mean_mse', 0),
            'stability': cv_results.get('stability', 0),
            'n_features': len(cv_results.get('feature_importance', {})),
            'top_features': list(cv_results.get('feature_importance', {}).keys())[:5]
        }
    
    def _create_executive_summary(self, performance: Dict[str, float], 
                                risk: Dict[str, float]) -> Dict[str, str]:
        """Create executive summary."""
        summary = {}
        
        # Performance summary
        if performance.get('sharpe_ratio', 0) > 1.0:
            summary['performance'] = "Strong performance with Sharpe ratio > 1.0"
        elif performance.get('sharpe_ratio', 0) > 0.5:
            summary['performance'] = "Moderate performance with Sharpe ratio > 0.5"
        else:
            summary['performance'] = "Weak performance with Sharpe ratio < 0.5"
        
        # Risk summary
        if risk.get('max_drawdown', 0) > -0.1:
            summary['risk'] = "Low risk with max drawdown < 10%"
        elif risk.get('max_drawdown', 0) > -0.2:
            summary['risk'] = "Moderate risk with max drawdown < 20%"
        else:
            summary['risk'] = "High risk with max drawdown > 20%"
        
        # Overall assessment
        if (performance.get('sharpe_ratio', 0) > 1.0 and 
            risk.get('max_drawdown', 0) > -0.15):
            summary['overall'] = "Strategy shows strong risk-adjusted returns"
        elif (performance.get('sharpe_ratio', 0) > 0.5 and 
              risk.get('max_drawdown', 0) > -0.2):
            summary['overall'] = "Strategy shows acceptable risk-adjusted returns"
        else:
            summary['overall'] = "Strategy needs improvement in risk management"
        
        return summary
    
    def create_performance_plots(self, results: pd.DataFrame, 
                               save_path: Optional[str] = None) -> None:
        """Create performance visualization plots."""
        logger.info("Creating performance plots")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VolBoost Strategy Performance', fontsize=16)
        
        # Cumulative PnL
        axes[0, 0].plot(results['date'], results['cumulative_pnl'])
        axes[0, 0].set_title('Cumulative PnL')
        axes[0, 0].set_ylabel('PnL ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        axes[0, 1].fill_between(results['date'], results['drawdown'], 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(results['date'], results['drawdown'], color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown ($)')
        axes[0, 1].grid(True)
        
        # Daily PnL distribution
        axes[1, 0].hist(results['total_pnl'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Daily PnL Distribution')
        axes[1, 0].set_xlabel('Daily PnL ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Portfolio metrics over time
        axes[1, 1].plot(results['date'], results['total_vega'], label='Vega')
        axes[1, 1].plot(results['date'], results['total_delta'], label='Delta')
        axes[1, 1].set_title('Portfolio Greeks')
        axes[1, 1].set_ylabel('Greeks')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()
    
    def create_risk_plots(self, results: pd.DataFrame, 
                         save_path: Optional[str] = None) -> None:
        """Create risk analysis plots."""
        logger.info("Creating risk plots")
        
        # Calculate rolling VaR and ES
        rolling_risk = self.risk_manager.rolling_var_es(results['total_pnl'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Risk Analysis', fontsize=16)
        
        # Rolling VaR
        axes[0, 0].plot(rolling_risk['date'], rolling_risk['var'], label='VaR 95%')
        axes[0, 0].set_title('Rolling Value at Risk')
        axes[0, 0].set_ylabel('VaR ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Rolling ES
        axes[0, 1].plot(rolling_risk['date'], rolling_risk['es'], label='ES 95%', color='red')
        axes[0, 1].set_title('Rolling Expected Shortfall')
        axes[0, 1].set_ylabel('ES ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # PnL vs VaR
        axes[1, 0].scatter(rolling_risk['var'], results['total_pnl'].iloc[-len(rolling_risk):], alpha=0.6)
        axes[1, 0].plot([0, rolling_risk['var'].max()], [0, -rolling_risk['var'].max()], 'r--', label='VaR Line')
        axes[1, 0].set_title('PnL vs VaR')
        axes[1, 0].set_xlabel('VaR ($)')
        axes[1, 0].set_ylabel('Daily PnL ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Exceptions over time
        exceptions = (results['total_pnl'] < -rolling_risk['var']).astype(int)
        axes[1, 1].plot(results['date'], exceptions.cumsum(), label='Cumulative Exceptions')
        axes[1, 1].set_title('VaR Exceptions')
        axes[1, 1].set_ylabel('Cumulative Exceptions')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk plots saved to {save_path}")
        
        plt.show()
    
    def create_model_plots(self, model_metrics: Dict[str, Any], 
                          save_path: Optional[str] = None) -> None:
        """Create model performance plots."""
        logger.info("Creating model plots")
        
        cv_results = model_metrics.get('cv_results', {})
        if not cv_results:
            logger.warning("No CV results available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance', fontsize=16)
        
        # CV scores
        cv_scores = cv_results.get('cv_scores', [])
        if cv_scores:
            folds = [score['fold'] for score in cv_scores]
            r2_scores = [score['r2'] for score in cv_scores]
            mse_scores = [score['mse'] for score in cv_scores]
            
            axes[0, 0].bar(folds, r2_scores)
            axes[0, 0].set_title('R² Score by Fold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].grid(True)
            
            axes[0, 1].bar(folds, mse_scores)
            axes[0, 1].set_title('MSE by Fold')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].grid(True)
        
        # Feature importance
        feature_importance = cv_results.get('feature_importance', {})
        if feature_importance:
            # Get top 10 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            axes[1, 0].barh(range(len(features)), importances)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].grid(True)
        
        # Model stability
        stability = cv_results.get('stability', 0)
        axes[1, 1].bar(['Stability'], [stability])
        axes[1, 1].set_title('Model Stability (Kendall\'s τ)')
        axes[1, 1].set_ylabel('Stability Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model plots saved to {save_path}")
        
        plt.show()
    
    def export_results_to_csv(self, results: pd.DataFrame, 
                            summary: Dict[str, Any],
                            filepath: str) -> None:
        """Export results to CSV files."""
        logger.info(f"Exporting results to {filepath}")
        
        # Export daily results
        results.to_csv(f"{filepath}_daily_results.csv", index=False)
        
        # Export summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{filepath}_summary.csv", index=False)
        
        logger.info("Results exported successfully")
    
    def generate_html_report(self, results: pd.DataFrame, 
                           summary: Dict[str, Any],
                           filepath: str) -> None:
        """Generate HTML report."""
        logger.info(f"Generating HTML report: {filepath}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VolBoost Strategy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 20px 0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>VolBoost Strategy Report</h1>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <p><strong>Performance:</strong> {summary.get('summary', {}).get('performance', 'N/A')}</p>
                <p><strong>Risk:</strong> {summary.get('summary', {}).get('risk', 'N/A')}</p>
                <p><strong>Overall:</strong> {summary.get('summary', {}).get('overall', 'N/A')}</p>
            </div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add performance metrics
        for key, value in summary.get('performance', {}).items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            html_content += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add risk metrics
        for key, value in summary.get('risk', {}).items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            html_content += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Model Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add model metrics
        for key, value in summary.get('model', {}).items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            html_content += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
        
        html_content += """
            </table>
            
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info("HTML report generated successfully")
