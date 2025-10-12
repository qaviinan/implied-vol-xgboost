"""
XGBoost model with purged CV and calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import kendalltau
import logging
from .utils import setup_logging, set_random_seed

logger = setup_logging()

class PurgedTimeSeriesSplit:
    """Time series cross-validation with purge and embargo."""
    
    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 1):
        """
        Initialize purged time series split.
        
        Args:
            n_splits: Number of splits
            purge_days: Days to purge between train/test
            embargo_days: Days to embargo after test period
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        """
        Generate train/test splits with purge and embargo.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (not used)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Get unique dates
        dates = X['date'].unique()
        dates = pd.to_datetime(dates).sort_values()
        
        # Calculate split points
        n_samples = len(dates)
        split_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test period
            test_start_idx = (i + 1) * split_size
            test_end_idx = min((i + 2) * split_size, n_samples)
            
            test_dates = dates[test_start_idx:test_end_idx]
            
            # Train period (before test)
            train_end_idx = test_start_idx - self.purge_days
            train_dates = dates[:train_end_idx]
            
            # Apply embargo
            if test_end_idx < n_samples:
                embargo_end_idx = min(test_end_idx + self.embargo_days, n_samples)
                embargo_dates = dates[test_end_idx:embargo_end_idx]
            else:
                embargo_dates = pd.DatetimeIndex([])
            
            # Get indices
            train_mask = X['date'].isin(train_dates)
            test_mask = X['date'].isin(test_dates)
            
            train_indices = X[train_mask].index.tolist()
            test_indices = X[test_mask].index.tolist()
            
            yield train_indices, test_indices

class XGBoostModel:
    """XGBoost model with purged CV and calibration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model_config = self.config.get('xgboost', {})
        self.cv_config = self.config.get('cv', {})
        self.calibration_config = self.config.get('calibration', {})
        
        # Set random seed
        set_random_seed(self.model_config.get('random_state', 42))
        
        # Initialize model
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        self.cv_results = None
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train XGBoost model with purged cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model with purged CV")
        
        if config:
            self.config.update(config)
        
        # Feature selection
        if self.config.get('feature_selection', {}).get('enabled', True):
            X = self._select_features(X, y)
        
        # Purged cross-validation
        cv_results = self._cross_validate(X, y)
        self.cv_results = cv_results
        
        # Train final model on full dataset
        self.model = self._train_final_model(X, y)
        
        # Calibrate model
        if self.calibration_config.get('method') == 'isotonic':
            self.calibrator = self._calibrate_model(X, y)
        
        logger.info("Model training completed")
        return self.model
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select top features using mutual information."""
        logger.info("Selecting features using mutual information")
        
        # Get feature columns (exclude date and other non-feature columns)
        feature_cols = [col for col in X.columns if col not in ['date', 'strike', 'call_put', 'expiry']]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X[feature_cols], y, random_state=42)
        
        # Select top features
        top_k = self.config.get('feature_selection', {}).get('top_k', 50)
        top_features_idx = np.argsort(mi_scores)[-top_k:]
        selected_features = [feature_cols[i] for i in top_features_idx]
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Return selected features plus metadata columns
        metadata_cols = ['date', 'strike', 'call_put', 'expiry']
        return_cols = [col for col in metadata_cols if col in X.columns] + selected_features
        
        return X[return_cols]
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform purged cross-validation."""
        logger.info("Performing purged cross-validation")
        
        # Initialize purged CV
        purge_days = self.cv_config.get('purge_days', 5)
        embargo_days = self.cv_config.get('embargo_days', 1)
        n_splits = self.cv_config.get('n_splits', 5)
        
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days
        )
        
        # Get feature columns
        feature_cols = [col for col in X.columns if col not in ['date', 'strike', 'call_put', 'expiry']]
        
        # Cross-validation results
        cv_scores = []
        feature_importances = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx][feature_cols], X.iloc[test_idx][feature_cols]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores.append({
                'fold': fold + 1,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            # Store feature importance
            feature_importances.append(model.feature_importances_)
        
        # Calculate average feature importance
        avg_importance = np.mean(feature_importances, axis=0)
        self.feature_importance = dict(zip(feature_cols, avg_importance))
        
        # Calculate stability
        stability = self._calculate_stability(feature_importances)
        
        return {
            'cv_scores': cv_scores,
            'mean_mse': np.mean([score['mse'] for score in cv_scores]),
            'mean_mae': np.mean([score['mae'] for score in cv_scores]),
            'mean_r2': np.mean([score['r2'] for score in cv_scores]),
            'std_mse': np.std([score['mse'] for score in cv_scores]),
            'feature_importance': self.feature_importance,
            'stability': stability
        }
    
    def _calculate_stability(self, feature_importances: List[np.ndarray]) -> float:
        """Calculate feature importance stability using Kendall's tau."""
        if len(feature_importances) < 2:
            return 0.0
        
        # Calculate pairwise Kendall's tau
        tau_scores = []
        for i in range(len(feature_importances)):
            for j in range(i + 1, len(feature_importances)):
                tau, _ = kendalltau(feature_importances[i], feature_importances[j])
                tau_scores.append(tau)
        
        return np.mean(tau_scores)
    
    def _create_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model with configuration."""
        return xgb.XGBRegressor(
            n_estimators=self.model_config.get('n_estimators', 1000),
            max_depth=self.model_config.get('max_depth', 6),
            learning_rate=self.model_config.get('learning_rate', 0.05),
            subsample=self.model_config.get('subsample', 0.8),
            colsample_bytree=self.model_config.get('colsample_bytree', 0.8),
            reg_alpha=self.model_config.get('reg_alpha', 0.1),
            reg_lambda=self.model_config.get('reg_lambda', 1.0),
            random_state=self.model_config.get('random_state', 42),
            n_jobs=-1
        )
    
    def _train_final_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """Train final model on full dataset."""
        logger.info("Training final model on full dataset")
        
        # Get feature columns
        feature_cols = [col for col in X.columns if col not in ['date', 'strike', 'call_put', 'expiry']]
        
        # Create and train model
        model = self._create_model()
        model.fit(X[feature_cols], y)
        
        return model
    
    def _calibrate_model(self, X: pd.DataFrame, y: pd.Series) -> IsotonicRegression:
        """Calibrate model predictions using isotonic regression."""
        logger.info("Calibrating model predictions")
        
        # Get feature columns
        feature_cols = [col for col in X.columns if col not in ['date', 'strike', 'call_put', 'expiry']]
        
        # Get model predictions
        y_pred = self.model.predict(X[feature_cols])
        
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred, y)
        
        return calibrator
    
    def predict_expected_pnl(self, model: Any, X_new: pd.DataFrame) -> pd.Series:
        """
        Predict expected PnL for new data.
        
        Args:
            model: Trained model
            X_new: New feature matrix
            
        Returns:
            Predicted expected PnL
        """
        # Get feature columns
        feature_cols = [col for col in X_new.columns if col not in ['date', 'strike', 'call_put', 'expiry']]
        
        # Predict
        predictions = model.predict(X_new[feature_cols])
        
        # Apply calibration if available
        if self.calibrator is not None:
            predictions = self.calibrator.predict(predictions)
        
        return pd.Series(predictions, index=X_new.index)
    
    def calibrate_scores(self, y_true: np.ndarray, y_score: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Calibrate model scores to expected PnL.
        
        Args:
            y_true: True labels
            y_score: Model scores
            
        Returns:
            Calibration function
        """
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_score, y_true)
        
        return calibrator.predict
    
    def validate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate model performance and stability."""
        logger.info("Validating model")
        
        validation_config = self.config.get('validation', {})
        
        # Check minimum sample sizes
        min_train = validation_config.get('min_train_samples', 1000)
        min_test = validation_config.get('min_test_samples', 200)
        
        if len(X) < min_train:
            logger.warning(f"Insufficient training samples: {len(X)} < {min_train}")
        
        # Check stability
        stability_threshold = validation_config.get('stability_threshold', 0.4)
        if self.cv_results and self.cv_results['stability'] < stability_threshold:
            logger.warning(f"Low feature importance stability: {self.cv_results['stability']} < {stability_threshold}")
        
        # Check drift (simplified)
        drift_threshold = validation_config.get('drift_threshold', 0.2)
        # This would require historical data to calculate PSI
        
        return {
            'sample_size_adequate': len(X) >= min_train,
            'stability_adequate': self.cv_results['stability'] >= stability_threshold if self.cv_results else False,
            'cv_performance': self.cv_results
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return {}
        
        return self.feature_importance
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        # Save model
        self.model.save_model(filepath)
        
        # Save calibration if available
        if self.calibrator is not None:
            import joblib
            joblib.dump(self.calibrator, filepath.replace('.json', '_calibrator.pkl'))
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        
        # Load calibration if available
        try:
            import joblib
            self.calibrator = joblib.load(filepath.replace('.json', '_calibrator.pkl'))
        except:
            self.calibrator = None
        
        logger.info(f"Model loaded from {filepath}")
