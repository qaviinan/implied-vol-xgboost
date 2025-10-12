"""
Data ingestion and cleaning module for options chains, underlying, and rates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
from .utils import (
    load_config, get_config_path, validate_options_data, 
    validate_underlying_data, setup_logging, align_market_data
)

logger = setup_logging()

class DataIngestion:
    """Data ingestion and cleaning class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data ingestion with configuration."""
        if config_path is None:
            config_path = get_config_path("data")
        
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        
    def load_options_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load options chain data from file or generate sample data.
        
        Args:
            file_path: Path to options data file. If None, generates sample data.
            
        Returns:
            DataFrame with options chain data
        """
        if file_path is None:
            logger.info("No file path provided, generating sample options data")
            return self._generate_sample_options_data()
        
        try:
            # Try to load from different file formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded options data from {file_path}: {len(df)} rows")
            return self._clean_options_data(df)
            
        except Exception as e:
            logger.error(f"Error loading options data from {file_path}: {e}")
            logger.info("Falling back to sample data generation")
            return self._generate_sample_options_data()
    
    def load_underlying_data(self, symbol: str = "SPY", 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load underlying asset data from Yahoo Finance.
        
        Args:
            symbol: Underlying symbol (default: SPY)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with underlying data
        """
        if start_date is None:
            start_date = self.data_config['start_date']
        if end_date is None:
            end_date = self.data_config['end_date']
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Reset index to get date as column
            df = df.reset_index()
            df['date'] = df['Date'].dt.date
            df = df.drop('Date', axis=1)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            logger.info(f"Loaded underlying data for {symbol}: {len(df)} rows")
            return validate_underlying_data(df)
            
        except Exception as e:
            logger.error(f"Error loading underlying data for {symbol}: {e}")
            return self._generate_sample_underlying_data(start_date, end_date)
    
    def load_rates_data(self, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load risk-free rates data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with rates data
        """
        if start_date is None:
            start_date = self.data_config['start_date']
        if end_date is None:
            end_date = self.data_config['end_date']
        
        try:
            # Use 10-year Treasury yield as proxy for risk-free rate
            ticker = yf.Ticker("^TNX")
            df = ticker.history(start=start_date, end=end_date)
            
            # Reset index and format
            df = df.reset_index()
            df['date'] = df['Date'].dt.date
            df = df.drop('Date', axis=1)
            
            # Use close price as annualized rate
            df['r_annualized'] = df['Close'] / 100.0  # Convert percentage to decimal
            df['q_div_yield'] = 0.02  # Assume 2% dividend yield
            
            df = df[['date', 'r_annualized', 'q_div_yield']]
            
            logger.info(f"Loaded rates data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading rates data: {e}")
            return self._generate_sample_rates_data(start_date, end_date)
    
    def _generate_sample_options_data(self) -> pd.DataFrame:
        """Generate sample options data for testing."""
        logger.info("Generating sample options data")
        
        # Generate date range
        start_date = pd.to_datetime(self.data_config['start_date'])
        end_date = pd.to_datetime(self.data_config['end_date'])
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate sample data
        data = []
        np.random.seed(42)
        
        for date in dates:
            # Generate multiple expiries
            expiries = [date + timedelta(days=d) for d in [7, 14, 30, 60, 90, 180]]
            expiries = [exp for exp in expiries if exp <= end_date]
            
            # Generate underlying price
            underlying_price = 4000 + np.random.normal(0, 50)
            
            for expiry in expiries:
                dte = (expiry - date).days
                if dte < 5:
                    continue
                
                # Generate strikes around money
                atm_strike = underlying_price
                strikes = np.arange(atm_strike * 0.8, atm_strike * 1.2, atm_strike * 0.05)
                
                for strike in strikes:
                    for cp in ['C', 'P']:
                        # Generate implied volatility with smile
                        moneyness = strike / underlying_price
                        atm_iv = 0.2 + np.random.normal(0, 0.05)
                        smile_adjustment = 0.1 * (moneyness - 1) ** 2
                        impl_vol = max(0.05, atm_iv + smile_adjustment + np.random.normal(0, 0.02))
                        
                        # Generate option price using Black-Scholes approximation
                        from .pricing import bs_price_greeks
                        price = bs_price_greeks(
                            S=underlying_price, K=strike, T=dte/365.0, 
                            r=0.05, q=0.02, sigma=impl_vol, cp=cp
                        )['price']
                        
                        # Generate bid-ask spread
                        spread = price * 0.02  # 2% spread
                        bid = price - spread/2
                        ask = price + spread/2
                        
                        data.append({
                            'date': date.date(),
                            'expiry': expiry.date(),
                            'dte': dte,
                            'strike': strike,
                            'call_put': cp,
                            'bid': max(0.01, bid),
                            'ask': ask,
                            'mid': price,
                            'impl_vol': impl_vol,
                            'open_interest': np.random.randint(100, 10000),
                            'volume': np.random.randint(0, 1000),
                            'underlying_price': underlying_price
                        })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated sample options data: {len(df)} rows")
        return validate_options_data(df)
    
    def _generate_sample_underlying_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample underlying data."""
        logger.info("Generating sample underlying data")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)
        
        # Generate price series with random walk
        prices = [4000]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            prices.append(prices[-1] * (1 + change))
        
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            data.append({
                'date': date.date(),
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        df = pd.DataFrame(data)
        return validate_underlying_data(df)
    
    def _generate_sample_rates_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample rates data."""
        logger.info("Generating sample rates data")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)
        
        data = []
        for date in dates:
            data.append({
                'date': date.date(),
                'r_annualized': 0.05 + np.random.normal(0, 0.01),  # 5% ± 1%
                'q_div_yield': 0.02  # 2% dividend yield
            })
        
        df = pd.DataFrame(data)
        return df
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate options data."""
        logger.info("Cleaning options data")
        
        # Ensure required columns exist
        required_cols = ['date', 'expiry', 'strike', 'call_put', 'bid', 'ask', 'mid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['expiry'] = pd.to_datetime(df['expiry']).dt.date
        
        # Calculate DTE
        df['dte'] = (pd.to_datetime(df['expiry']) - pd.to_datetime(df['date'])).dt.days
        
        # Add implied volatility if not present
        if 'impl_vol' not in df.columns:
            logger.warning("Implied volatility not found, calculating from mid prices")
            # This would require Black-Scholes inversion - simplified for now
            df['impl_vol'] = 0.2  # Placeholder
        
        # Add underlying price if not present
        if 'underlying_price' not in df.columns:
            logger.warning("Underlying price not found, using strike as proxy")
            df['underlying_price'] = df['strike']
        
        return validate_options_data(df)
    
    def align_all_data(self, options_df: pd.DataFrame, 
                      underlying_df: pd.DataFrame,
                      rates_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align all market data to common timestamps."""
        logger.info("Aligning all market data")
        
        return align_market_data(options_df, underlying_df, rates_df)
    
    def create_sample_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create a complete sample dataset for testing."""
        logger.info("Creating sample dataset")
        
        options_df = self._generate_sample_options_data()
        underlying_df = self._generate_sample_underlying_data(
            self.data_config['start_date'], 
            self.data_config['end_date']
        )
        rates_df = self._generate_sample_rates_data(
            self.data_config['start_date'],
            self.data_config['end_date']
        )
        
        # Align data
        options_df, underlying_df, rates_df = self.align_all_data(
            options_df, underlying_df, rates_df
        )
        
        logger.info(f"Sample dataset created: {len(options_df)} options, "
                   f"{len(underlying_df)} underlying, {len(rates_df)} rates")
        
        return options_df, underlying_df, rates_df
