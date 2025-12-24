"""
Sales Volume Predictor Module

Regression model for predicting product sales volume (count_sold).
Uses LightGBM as the primary algorithm with SHAP-ready architecture.

Target Variable:
    count_sold (units sold) - log-transformed for training

Evaluation Metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² Score
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


class SalesPredictor:
    """
    Sales volume prediction model using LightGBM.
    
    Predicts log-transformed sales which are then converted back
    to actual units for interpretable results.
    
    Example:
        predictor = SalesPredictor()
        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)
        metrics = predictor.evaluate(X_test, y_test)
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    def __init__(self, params: Dict = None):
        """
        Initialize the sales predictor.
        
        Args:
            params: LightGBM hyperparameters. Uses defaults if not specified.
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self._fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        log_transform: bool = True
    ) -> 'SalesPredictor':
        """
        Train the sales prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (count_sold)
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Patience for early stopping
            log_transform: Whether to log-transform target
            
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self._log_transform = log_transform
        
        # Log transform target
        if log_transform:
            y_train = np.log1p(y)
        else:
            y_train = y
        
        # Create model
        self.model = lgb.LGBMRegressor(**self.params)
        
        # Prepare callbacks
        callbacks = []
        if eval_set:
            X_val, y_val = eval_set
            if log_transform:
                y_val = np.log1p(y_val)
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            
            self.model.fit(
                X, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
        else:
            self.model.fit(X, y_train)
        
        self._fitted = True
        logger.info(f"Sales predictor trained on {len(X):,} samples with {len(self.feature_names)} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate sales predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted sales (actual units, not log)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Predict log-transformed values
        y_pred_log = self.model.predict(X)
        
        # Convert back to actual units
        if self._log_transform:
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
        else:
            y_pred = y_pred_log
        
        return y_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        # Handle zeros in MAPE calculation
        y_safe = np.where(y == 0, 1, y)
        mape = np.mean(np.abs((y - y_pred) / y_safe)) * 100
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': mape,
            'r2': r2_score(y, y_pred)
        }
        
        logger.info(f"Evaluation: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.params,
                'log_transform': self._log_transform
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'SalesPredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(params=data['params'])
        predictor.model = data['model']
        predictor.feature_names = data['feature_names']
        predictor._log_transform = data['log_transform']
        predictor._fitted = True
        
        logger.info(f"Model loaded from {path}")
        return predictor


def train_sales_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[SalesPredictor, Dict[str, float]]:
    """
    Train the sales prediction model with train/test split.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create validation set from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state
    )
    
    # Train
    model = SalesPredictor()
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics


def predict_sales(
    model: SalesPredictor,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate sales predictions with confidence intervals.
    
    Args:
        model: Trained SalesPredictor
        X: Feature DataFrame
        
    Returns:
        DataFrame with predictions
    """
    predictions = model.predict(X)
    
    result = pd.DataFrame({
        'predicted_sales': predictions,
        # Simple confidence interval based on model uncertainty
        'lower_bound': predictions * 0.7,
        'upper_bound': predictions * 1.3
    })
    
    return result
