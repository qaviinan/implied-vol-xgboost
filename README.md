# VolBoost: Options Volatility Surface ML Strategy

A no-arbitrage SVI **options** surface and trained a **gradient-boosted trees (XGBoost)** volatility/return signal to run a delta-hedged, vega-targeted strategy with PnL-explain and VaR/ES backtests.

## Overview

VolBoost is a comprehensive framework for building and backtesting options trading strategies using:

- **SVI (Stochastic Volatility Inspired)** surface calibration with no-arbitrage checks
- **XGBoost** machine learning models for volatility/return prediction
- **Delta-hedged, vega-targeted** trading strategies
- **Comprehensive risk management** with VaR/ES and stress testing
- **PnL attribution** and performance analytics

## Features

### Core Components

1. **Data Ingestion & Cleaning**
   - Options chain data processing
   - Underlying asset data alignment
   - Risk-free rates integration
   - Data quality validation

2. **SVI Surface Calibration**
   - JWP parameterization with constraints
   - No-arbitrage validation (butterfly & calendar)
   - Challenger surface comparison (cubic splines)
   - Surface interpolation and extrapolation

3. **Pricing & Greeks**
   - Black-Scholes pricing with parity checks
   - Analytical Greeks calculation
   - Bump-and-revalue validation
   - Portfolio-level Greeks aggregation

4. **Machine Learning**
   - Feature engineering for options data
   - XGBoost with purged time-series CV
   - Isotonic calibration for score mapping
   - Feature importance stability analysis

5. **Trading Strategy**
   - Contract selection based on ML signals
   - Position sizing with vega targeting
   - Delta hedging with transaction costs
   - Position lifecycle management

6. **Risk Management**
   - VaR/ES calculation and backtesting
   - Exception tests (Kupiec, Christoffersen)
   - Monte Carlo stress testing
   - Risk limit monitoring

7. **Backtesting & Reporting**
   - Historical strategy simulation
   - PnL attribution (vega/theta/costs)
   - Performance analytics and visualization
   - HTML/CSV report generation

## Installation

### Prerequisites

- Python 3.10+
- Conda or Miniconda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VolBoost
   ```

2. **Create conda environment**
   ```bash
   conda env create -f env.yml
   conda activate vol-ml-vega
   ```

3. **Verify installation**
   ```bash
   python -c "import src; print('VolBoost installed successfully!')"
   ```

## Quick Start

### Run the Demo

```bash
python main.py
```

This will:
1. Generate sample options data
2. Calibrate SVI surfaces
3. Train XGBoost model
4. Run backtest simulation
5. Generate performance reports

### Use Individual Components

```python
from src import DataIngestion, SVICalibrator, XGBoostModel, Backtester

# Data ingestion
data_ingestion = DataIngestion()
options_df, underlying_df, rates_df = data_ingestion.create_sample_dataset()

# SVI calibration
svi_calibrator = SVICalibrator()
svi_params = svi_calibrator.calibrate_surface(options_df)

# Model training
model = XGBoostModel()
trained_model = model.train_model(features_df, labels)

# Backtesting
backtester = Backtester()
results = backtester.run_backtest(signals_df)
```

## Configuration

The framework uses YAML configuration files in the `config/` directory:

- `data.yml` - Data sources and filtering rules
- `model.yml` - ML model parameters and features
- `trade.yml` - Trading strategy parameters
- `costs.yml` - Transaction costs and slippage

Example configuration:
```yaml
# config/model.yml
model:
  type: "xgboost"
  xgboost:
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.05
  cv:
    n_splits: 5
    purge_days: 5
    embargo_days: 1
```

## Project Structure

```
VolBoost/
├── README.md
├── env.yml
├── main.py
├── config/
│   ├── data.yml
│   ├── model.yml
│   ├── trade.yml
│   └── costs.yml
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── svi.py
│   ├── surface.py
│   ├── pricing.py
│   ├── features.py
│   ├── model.py
│   ├── select.py
│   ├── backtest.py
│   ├── risk.py
│   ├── report.py
│   └── utils.py
├── notebooks/
│   ├── 00_surface_demo.ipynb
│   ├── 01_signal_oos.ipynb
│   └── 02_backtest_report.ipynb
├── tests/
│   ├── test_svi_fit.py
│   ├── test_pricing_greeks.py
│   ├── test_model_cv.py
│   └── ...
└── reports/
    ├── validation_memo.pdf
    └── summary.md
```

## Key Algorithms

### SVI Parameterization

The SVI-JWP form is used for volatility surface modeling:

```
w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
```

Where:
- `w(k)` = total variance
- `k` = log-moneyness
- `θ = (a,b,ρ,m,σ)` = SVI parameters

### No-Arbitrage Constraints

- **Butterfly arbitrage**: Numerical convexity check on call prices
- **Calendar arbitrage**: Monotonicity of total variance in time
- **Parameter bounds**: `b > 0`, `σ > 0`, `|ρ| < 1`

### Machine Learning Features

- **Surface features**: Smile slope, curvature, ATM IV, term structure
- **Realized volatility**: Multiple windows, HV-IV gap, skew z-scores
- **Trading features**: Moneyness buckets, DTE, spread %, OI/volume
- **Market features**: Underlying returns, RV ratio, volatility regime

### Risk Metrics

- **VaR/ES**: Historical and Monte Carlo methods
- **Exception tests**: Kupiec (unconditional) and Christoffersen (conditional)
- **Stress scenarios**: Vol spikes, vol crush, parallel shifts, skew twists

## Performance Targets

### Surface Quality
- Mean absolute total-variance error ≤ 0.015
- Zero butterfly arbitrage violations
- ≤1% calendar arbitrage violations

### Model Performance
- Purged-CV information ratio ≥ 0.25
- Feature importance stability (Kendall's τ) ≥ 0.4

### Risk Management
- Kupiec LR test p-value ∈ [0.05, 0.95]
- Accounting accuracy within 1e-6

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Key test modules:
- `test_svi_fit.py` - SVI calibration and no-arbitrage checks
- `test_pricing_greeks.py` - Black-Scholes pricing and Greeks
- `test_model_cv.py` - XGBoost model and cross-validation
- `test_backtest_accounting.py` - Backtesting engine
- `test_risk_metrics.py` - Risk calculations

## Notebooks

Interactive demonstrations:

1. **`00_surface_demo.ipynb`** - SVI surface calibration and visualization
2. **`01_signal_oos.ipynb`** - Out-of-sample signal evaluation
3. **`02_backtest_report.ipynb`** - Strategy backtesting and analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended for live trading without proper risk management and regulatory compliance. Past performance does not guarantee future results.

## References

- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Christoffersen, P. (1998). "Evaluating Interval Forecasts"
- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models"
