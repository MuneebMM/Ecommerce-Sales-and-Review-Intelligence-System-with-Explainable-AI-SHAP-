"""
Review Risk Predictor Module

Binary classification model for predicting negative review probability.
Uses LightGBM with class imbalance handling.

Target Variable:
    is_negative_review (derived: review_rating < 3)

Evaluation Metrics:
    - AUC-ROC
    - Precision, Recall, F1
    - Brier Score
"""

import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss, classification_report, confusion_matrix
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


class ReviewRiskPredictor:
    """
    Review risk prediction model using LightGBM.
    
    Predicts probability of a review being negative (rating < 3).
    Handles class imbalance through scale_pos_weight.
    
    Example:
        predictor = ReviewRiskPredictor()
        predictor.fit(X_train, y_train)
        probabilities = predictor.predict_proba(X_test)
        metrics = predictor.evaluate(X_test, y_test)
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
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
        Initialize the review risk predictor.
        
        Args:
            params: LightGBM hyperparameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = []
        self._fitted = False
        self._threshold = 0.5
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        auto_balance: bool = True
    ) -> 'ReviewRiskPredictor':
        """
        Train the review risk model.
        
        Args:
            X: Feature DataFrame
            y: Binary target Series (0/1)
            eval_set: Optional validation set
            early_stopping_rounds: Patience for early stopping
            auto_balance: Auto-compute class weights for imbalance
            
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        
        # Handle class imbalance
        if auto_balance:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0:
                self.params['scale_pos_weight'] = neg_count / pos_count
                logger.info(f"Class imbalance: {neg_count}:{pos_count}, scale_pos_weight={self.params['scale_pos_weight']:.2f}")
        
        # Create model
        self.model = lgb.LGBMClassifier(**self.params)
        
        # Prepare callbacks
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            self.model.fit(X, y, eval_set=[eval_set], callbacks=callbacks)
        else:
            self.model.fit(X, y)
        
        self._fitted = True
        logger.info(f"Review risk predictor trained on {len(X):,} samples")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of negative review.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities (positive class)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        proba = self.model.predict_proba(X)[:, 1]
        return proba
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict binary class labels.
        
        Args:
            X: Feature DataFrame
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Array of binary predictions
        """
        threshold = threshold or self._threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True binary labels
            
        Returns:
            Dictionary of metrics
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        metrics = {
            'auc_roc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'brier_score': brier_score_loss(y, y_proba)
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        logger.info(f"Evaluation: AUC={metrics['auc_roc']:.3f}, F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}")
        
        return metrics
    
    def set_threshold(self, threshold: float):
        """Set classification threshold."""
        self._threshold = threshold
        logger.info(f"Classification threshold set to {threshold}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
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
                'threshold': self._threshold
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ReviewRiskPredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(params=data['params'])
        predictor.model = data['model']
        predictor.feature_names = data['feature_names']
        predictor._threshold = data['threshold']
        predictor._fitted = True
        
        logger.info(f"Model loaded from {path}")
        return predictor


def train_risk_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[ReviewRiskPredictor, Dict[str, float]]:
    """
    Train the review risk model with train/test split.
    
    Args:
        X: Feature DataFrame
        y: Binary target Series
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train
    )
    
    # Train
    model = ReviewRiskPredictor()
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics


def predict_risk(
    model: ReviewRiskPredictor,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate risk predictions with scores.
    
    Args:
        model: Trained ReviewRiskPredictor
        X: Feature DataFrame
        
    Returns:
        DataFrame with predictions and scores
    """
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    
    result = pd.DataFrame({
        'risk_probability': probabilities,
        'is_high_risk': predictions,
        'risk_level': pd.cut(
            probabilities,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['low', 'medium', 'high', 'critical']
        )
    })
    
    return result
