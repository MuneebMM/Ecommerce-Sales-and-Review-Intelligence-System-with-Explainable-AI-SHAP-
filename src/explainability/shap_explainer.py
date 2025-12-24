"""
SHAP Explainer Module

Core SHAP computation engine for model explainability.
Uses TreeSHAP for efficient explanation of LightGBM models.

Methods:
    - TreeSHAP: For tree-based models (exact, fast)
    - KernelSHAP: For model-agnostic explanations (approximate)
"""

import logging
from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """
    Container for SHAP explanation results.
    
    Attributes:
        shap_values: SHAP values for each feature
        base_value: Expected model output (baseline)
        feature_names: Names of features
        feature_values: Actual feature values for the instance
    """
    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    feature_values: np.ndarray
    
    def get_top_features(self, n: int = 5) -> List[Dict]:
        """Get top N most important features for this prediction."""
        importance = np.abs(self.shap_values)
        indices = np.argsort(importance)[::-1][:n]
        
        return [
            {
                'feature': self.feature_names[i],
                'shap_value': float(self.shap_values[i]),
                'feature_value': float(self.feature_values[i]) if np.isfinite(self.feature_values[i]) else None,
                'direction': 'positive' if self.shap_values[i] > 0 else 'negative'
            }
            for i in indices
        ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'base_value': float(self.base_value),
            'shap_values': {
                name: float(val) 
                for name, val in zip(self.feature_names, self.shap_values)
            },
            'top_drivers': self.get_top_features(5)
        }


class SHAPExplainer:
    """
    Main explainer interface for generating SHAP explanations.
    
    Supports both tree-based models (fast) and general models.
    
    Example:
        explainer = SHAPExplainer(model)
        explanation = explainer.explain(X.iloc[[0]])
        print(explanation.get_top_features())
    """
    
    def __init__(
        self,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        method: str = 'auto'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (SalesPredictor or ReviewRiskPredictor)
            background_data: Background dataset for SHAP baseline
            feature_names: Feature column names
            method: 'tree', 'kernel', or 'auto'
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names or []
        self.method = method
        
        # Extract the underlying LightGBM model
        if hasattr(model, 'model'):
            self._lgb_model = model.model
        else:
            self._lgb_model = model
        
        # Create SHAP explainer
        if method == 'auto' or method == 'tree':
            try:
                self._explainer = shap.TreeExplainer(self._lgb_model)
                self.method = 'tree'
                logger.info("Using TreeSHAP explainer")
            except Exception as e:
                logger.warning(f"TreeSHAP failed, falling back to KernelSHAP: {e}")
                self._explainer = shap.KernelExplainer(
                    self._lgb_model.predict,
                    background_data.sample(min(100, len(background_data))) if background_data is not None else None
                )
                self.method = 'kernel'
        else:
            self._explainer = shap.KernelExplainer(
                self._lgb_model.predict,
                background_data.sample(min(100, len(background_data))) if background_data is not None else None
            )
        
        # Store base value
        self._base_value = None
    
    def explain(self, X: pd.DataFrame) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            X: Single row DataFrame
            
        Returns:
            SHAPExplanation object
        """
        if len(X) > 1:
            logger.warning("Multiple rows provided, using first row only")
            X = X.iloc[[0]]
        
        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)
        
        # Handle binary classification (returns list for each class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Flatten if needed
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()
        
        # Get base value
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0
        
        feature_names = self.feature_names or list(X.columns)
        
        return SHAPExplanation(
            shap_values=shap_values,
            base_value=float(base_value),
            feature_names=feature_names,
            feature_values=X.values.flatten()
        )
    
    def explain_batch(self, X: pd.DataFrame) -> List[SHAPExplanation]:
        """
        Generate SHAP explanations for multiple instances.
        
        Args:
            X: Multi-row DataFrame
            
        Returns:
            List of SHAPExplanation objects
        """
        shap_values = self._explainer.shap_values(X)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get base value
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0
        
        feature_names = self.feature_names or list(X.columns)
        
        explanations = []
        for i in range(len(X)):
            exp = SHAPExplanation(
                shap_values=shap_values[i],
                base_value=float(base_value),
                feature_names=feature_names,
                feature_values=X.iloc[i].values
            )
            explanations.append(exp)
        
        return explanations
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global feature importance from SHAP values.
        
        Args:
            X: Dataset to compute importance over
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self._explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_names = self.feature_names or list(X.columns)
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class TreeSHAPEngine:
    """
    Optimized TreeSHAP engine for production use.
    
    Includes caching and batch optimization.
    """
    
    def __init__(self, model: Any, cache_enabled: bool = True):
        """
        Initialize TreeSHAP engine.
        
        Args:
            model: Trained tree-based model
            cache_enabled: Enable result caching
        """
        self.explainer = SHAPExplainer(model, method='tree')
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, SHAPExplanation] = {}
    
    def explain(self, X: pd.DataFrame, cache_key: str = None) -> SHAPExplanation:
        """
        Generate explanation with optional caching.
        
        Args:
            X: Input features
            cache_key: Optional cache key
            
        Returns:
            SHAPExplanation
        """
        if self.cache_enabled and cache_key and cache_key in self._cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key]
        
        explanation = self.explainer.explain(X)
        
        if self.cache_enabled and cache_key:
            self._cache[cache_key] = explanation
        
        return explanation
    
    def clear_cache(self):
        """Clear explanation cache."""
        self._cache.clear()


def compute_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Convenience function to compute SHAP values.
    
    Args:
        model: Trained model
        X: Features
        
    Returns:
        Array of SHAP values
    """
    explainer = SHAPExplainer(model)
    explanations = explainer.explain_batch(X)
    return np.array([e.shap_values for e in explanations])


def get_feature_importance(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Get global feature importance from SHAP.
    
    Args:
        model: Trained model
        X: Features
        
    Returns:
        Feature importance DataFrame
    """
    explainer = SHAPExplainer(model)
    return explainer.get_feature_importance(X)
