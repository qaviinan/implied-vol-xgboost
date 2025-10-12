"""
Tests for model cross-validation functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import XGBoostModel, PurgedTimeSeriesSplit

class TestModelCV(unittest.TestCase):
    """Test model cross-validation functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic time series data
        np.random.seed(42)
        self.n_samples = 1000
        
        # Generate dates
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.n_samples)]
        
        # Generate features
        n_features = 10
        X = np.random.randn(self.n_samples, n_features)
        
        # Generate target with some signal
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(self.n_samples) * 0.1
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X, columns=feature_cols)
        self.X['date'] = dates
        
        self.y = pd.Series(y)
        
        # Model configuration
        self.config = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'cv': {
                'n_splits': 3,
                'purge_days': 2,
                'embargo_days': 1
            }
        }
    
    def test_purged_time_series_split(self):
        """Test purged time series split functionality."""
        splitter = PurgedTimeSeriesSplit(n_splits=3, purge_days=2, embargo_days=1)
        
        splits = list(splitter.split(self.X, self.y))
        
        # Check that we get the expected number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that each split has train and test indices
        for train_idx, test_idx in splits:
            self.assertIsInstance(train_idx, list)
            self.assertIsInstance(test_idx, list)
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            
            # Check that train and test don't overlap
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
    
    def test_xgboost_model_initialization(self):
        """Test XGBoost model initialization."""
        model = XGBoostModel(self.config)
        
        # Check that model is initialized
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)
        self.assertIsNone(model.model)  # Not trained yet
    
    def test_xgboost_model_training(self):
        """Test XGBoost model training."""
        model = XGBoostModel(self.config)
        
        # Train model
        trained_model = model.train_model(self.X, self.y)
        
        # Check that model is trained
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.cv_results)
        
        # Check CV results structure
        cv_results = model.cv_results
        self.assertIn('cv_scores', cv_results)
        self.assertIn('mean_mse', cv_results)
        self.assertIn('mean_r2', cv_results)
        self.assertIn('feature_importance', cv_results)
        self.assertIn('stability', cv_results)
        
        # Check that CV scores are reasonable
        self.assertGreater(cv_results['mean_r2'], -1)  # R² should be > -1
        self.assertLess(cv_results['mean_mse'], 10)    # MSE should be reasonable
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        model = XGBoostModel(self.config)
        model.train_model(self.X, self.y)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check that importance is returned
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check that importance values are non-negative
        for feature, imp in importance.items():
            self.assertGreaterEqual(imp, 0)
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = XGBoostModel(self.config)
        model.train_model(self.X, self.y)
        
        # Create test data
        test_X = self.X.iloc[:100].copy()
        
        # Make predictions
        predictions = model.predict_expected_pnl(model.model, test_X)
        
        # Check that predictions are returned
        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(len(predictions), len(test_X))
        
        # Check that predictions are finite
        self.assertTrue(predictions.isna().sum() == 0)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_model_validation(self):
        """Test model validation."""
        model = XGBoostModel(self.config)
        model.train_model(self.X, self.y)
        
        # Validate model
        validation_results = model.validate_model(self.X, self.y)
        
        # Check that validation results are returned
        self.assertIsInstance(validation_results, dict)
        self.assertIn('sample_size_adequate', validation_results)
        self.assertIn('stability_adequate', validation_results)
        self.assertIn('cv_performance', validation_results)
        
        # Check that sample size is adequate
        self.assertTrue(validation_results['sample_size_adequate'])
    
    def test_calibration(self):
        """Test model calibration."""
        config_with_calibration = self.config.copy()
        config_with_calibration['calibration'] = {'method': 'isotonic'}
        
        model = XGBoostModel(config_with_calibration)
        model.train_model(self.X, self.y)
        
        # Check that calibrator is created
        self.assertIsNotNone(model.calibrator)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = XGBoostModel(self.config)
        model.train_model(self.X, self.y)
        
        # Save model
        model_path = '/tmp/test_model.json'
        model.save_model(model_path)
        
        # Check that file is created
        self.assertTrue(os.path.exists(model_path))
        
        # Create new model and load
        new_model = XGBoostModel(self.config)
        new_model.load_model(model_path)
        
        # Check that model is loaded
        self.assertIsNotNone(new_model.model)
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(model_path.replace('.json', '_calibrator.pkl')):
            os.remove(model_path.replace('.json', '_calibrator.pkl'))

if __name__ == '__main__':
    unittest.main()
