"""
Main entry point for VolBoost strategy demonstration.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    DataIngestion, SVICalibrator, SurfaceBuilder, FeatureEngineer,
    XGBoostModel, ContractSelector, Backtester, RiskManager, ReportGenerator,
    setup_logging, load_config, get_config_path
)

def main():
    """Main demonstration function."""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting VolBoost demonstration")
    
    try:
        # Load configurations
        data_config = load_config(get_config_path("data"))
        model_config = load_config(get_config_path("model"))
        trade_config = load_config(get_config_path("trade"))
        costs_config = load_config(get_config_path("costs"))
        
        # Combine configurations
        config = {
            'data': data_config,
            'model': model_config,
            'trading': trade_config,
            'costs': costs_config
        }
        
        logger.info("Step 1: Data Ingestion")
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Create sample dataset
        options_df, underlying_df, rates_df = data_ingestion.create_sample_dataset()
        logger.info(f"Created sample dataset: {len(options_df)} options, {len(underlying_df)} underlying, {len(rates_df)} rates")
        
        logger.info("Step 2: SVI Surface Calibration")
        # Initialize SVI calibrator
        svi_calibrator = SVICalibrator()
        
        # Calibrate SVI surface
        svi_params = svi_calibrator.calibrate_surface(options_df)
        logger.info(f"Calibrated SVI for {len(svi_params)} expiries")
        
        # Build volatility surface
        surface_builder = SurfaceBuilder()
        surface = surface_builder.build_surface_from_options(options_df)
        
        # Validate surface
        surface_validation = surface_builder.validate_surface(surface, options_df)
        logger.info(f"Surface validation - MAE: {surface_validation['mean_absolute_error']:.4f}")
        
        logger.info("Step 3: Feature Engineering")
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(model_config)
        
        # Build features and labels
        features_df = feature_engineer.build_features_labels(
            options_df, surface.make_surface(), underlying_df, rates_df
        )
        logger.info(f"Built features and labels: {len(features_df)} rows")
        
        # Select features
        features_df = feature_engineer.select_features(features_df)
        logger.info(f"Selected features: {len(features_df.columns)} columns")
        
        logger.info("Step 4: Model Training")
        # Initialize XGBoost model
        xgb_model = XGBoostModel(model_config)
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns if col not in ['date', 'strike', 'call_put', 'expiry', 'delta_hedged_return']]
        X = features_df[['date'] + feature_cols]
        y = features_df['delta_hedged_return']
        
        # Train model
        trained_model = xgb_model.train_model(X, y)
        logger.info("Model training completed")
        
        # Validate model
        model_validation = xgb_model.validate_model(X, y)
        logger.info(f"Model validation - R²: {xgb_model.cv_results['mean_r2']:.4f}")
        
        logger.info("Step 5: Contract Selection")
        # Add expected PnL to features
        features_df['expected_pnl'] = xgb_model.predict_expected_pnl(trained_model, X)
        
        # Add Greeks to features
        from src.pricing import bs_price_greeks
        greeks_data = []
        for _, row in features_df.iterrows():
            greeks = bs_price_greeks(
                S=row.get('close', 4000),
                K=row['strike'],
                T=row.get('dte', 30) / 365.0,
                r=row.get('r_annualized', 0.05),
                q=row.get('q_div_yield', 0.02),
                sigma=row.get('impl_vol', 0.2),
                cp=row['call_put']
            )
            greeks_data.append(greeks)
        
        greeks_df = pd.DataFrame(greeks_data)
        features_df = pd.concat([features_df.reset_index(drop=True), greeks_df], axis=1)
        
        # Initialize contract selector
        contract_selector = ContractSelector(config)
        
        # Select contracts for a specific date
        test_date = features_df['date'].iloc[-1]  # Use last date
        daily_data = features_df[features_df['date'] == test_date]
        
        if len(daily_data) > 0:
            selected_contracts = contract_selector.select_contracts(daily_data)
            logger.info(f"Selected {len(selected_contracts)} contracts for trading")
            
            # Validate selection
            selection_validation = contract_selector.validate_selection(selected_contracts)
            logger.info(f"Selection validation - All constraints OK: {selection_validation['all_constraints_ok']}")
        
        logger.info("Step 6: Backtesting")
        # Initialize backtester
        backtester = Backtester(config)
        
        # Run backtest on subset of data
        backtest_data = features_df.tail(1000)  # Use last 1000 rows for demo
        backtest_results = backtester.run_backtest(backtest_data)
        
        if len(backtest_results) > 0:
            logger.info(f"Backtest completed: {len(backtest_results)} days")
            logger.info(f"Total PnL: ${backtest_results['cumulative_pnl'].iloc[-1]:.2f}")
            logger.info(f"Max Drawdown: ${backtest_results['drawdown'].min():.2f}")
        
        logger.info("Step 7: Risk Analysis")
        # Initialize risk manager
        risk_manager = RiskManager(config)
        
        if len(backtest_results) > 0:
            # Calculate risk metrics
            risk_metrics = risk_manager.calculate_portfolio_risk_metrics(backtest_results['total_pnl'])
            logger.info(f"Risk metrics - VaR 95%: ${risk_metrics['var_95']:.2f}")
            logger.info(f"Risk metrics - ES 95%: ${risk_metrics['es_95']:.2f}")
            logger.info(f"Risk metrics - Sharpe: {risk_metrics['sharpe_ratio']:.4f}")
            
            # Check risk limits
            risk_limits_check = risk_manager.check_risk_limits(selected_contracts if len(selected_contracts) > 0 else pd.DataFrame(), backtest_results['total_pnl'])
            logger.info(f"Risk limits check - Has violations: {risk_limits_check['has_violations']}")
        
        logger.info("Step 8: Report Generation")
        # Initialize report generator
        report_generator = ReportGenerator(config)
        
        if len(backtest_results) > 0:
            # Generate summary report
            summary = report_generator.generate_summary_report(
                backtest_results, [], xgb_model.cv_results
            )
            
            # Create reports directory
            os.makedirs('reports', exist_ok=True)
            
            # Export results
            report_generator.export_results_to_csv(
                backtest_results, summary, 'reports/volboost_results'
            )
            
            # Generate HTML report
            report_generator.generate_html_report(
                backtest_results, summary, 'reports/volboost_report.html'
            )
            
            logger.info("Reports generated in 'reports' directory")
        
        logger.info("VolBoost demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in VolBoost demonstration: {e}")
        raise

if __name__ == "__main__":
    main()
