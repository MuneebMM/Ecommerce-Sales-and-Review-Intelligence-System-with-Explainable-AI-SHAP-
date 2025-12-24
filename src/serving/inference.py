"""
Inference Module

Model inference logic with feature retrieval and explanation generation.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.models.sales_predictor import SalesPredictor
from src.models.review_risk_predictor import ReviewRiskPredictor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference orchestrator.
    
    Handles feature preparation, prediction, and explanation generation.
    """
    
    def __init__(
        self,
        sales_model_path: Optional[Path] = None,
        risk_model_path: Optional[Path] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            sales_model_path: Path to saved sales model
            risk_model_path: Path to saved risk model
        """
        self.sales_model: Optional[SalesPredictor] = None
        self.risk_model: Optional[ReviewRiskPredictor] = None
        self.feature_engineer = FeatureEngineer()
        
        if sales_model_path and sales_model_path.exists():
            self.sales_model = SalesPredictor.load(sales_model_path)
        
        if risk_model_path and risk_model_path.exists():
            self.risk_model = ReviewRiskPredictor.load(risk_model_path)
    
    def predict_sales(
        self,
        features: Dict[str, Any],
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate sales prediction with optional explanation.
        
        Args:
            features: Feature dictionary
            include_explanation: Include SHAP explanation
            
        Returns:
            Prediction result dictionary
        """
        if self.sales_model is None:
            raise RuntimeError("Sales model not loaded")
        
        # Prepare features
        X = self._prepare_sales_features(features)
        
        # Predict
        prediction = self.sales_model.predict(X)[0]
        
        result = {
            'predicted_sales': float(prediction),
            'confidence_interval': {
                'lower': float(prediction * 0.7),
                'upper': float(prediction * 1.3)
            }
        }
        
        return result
    
    def predict_risk(
        self,
        features: Dict[str, Any],
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate risk prediction with optional explanation.
        
        Args:
            features: Feature dictionary
            include_explanation: Include SHAP explanation
            
        Returns:
            Prediction result dictionary
        """
        if self.risk_model is None:
            raise RuntimeError("Risk model not loaded")
        
        # Prepare features
        X = self._prepare_risk_features(features)
        
        # Predict
        probability = self.risk_model.predict_proba(X)[0]
        is_high_risk = probability >= 0.5
        
        # Determine risk level
        if probability < 0.2:
            risk_level = 'low'
        elif probability < 0.5:
            risk_level = 'medium'
        elif probability < 0.8:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        result = {
            'risk_probability': float(probability),
            'risk_level': risk_level,
            'is_high_risk': bool(is_high_risk)
        }
        
        return result
    
    def _prepare_sales_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features for sales prediction - direct computation for serving."""
        import numpy as np
        
        # Compute derived features directly
        price = features.get('price', 0)
        stock = features.get('stock', 0)
        discounted_price = features.get('discounted_price', price)
        gold_merchant = features.get('gold_merchant', False)
        is_official = features.get('is_official', False)
        uses_topads = features.get('is_topads', False)
        rating_average = features.get('rating_average', 4.5)
        
        # Compute all possible features
        computed = {
            'price_log': np.log1p(price) if price > 0 else 0,
            'stock_log': np.log1p(stock) if stock > 0 else 0,
            'has_stock': 1 if stock > 0 else 0,
            'low_stock': 1 if stock < 10 else 0,
            'is_preorder': int(features.get('preorder', False)),
            'shop_tier': int(is_official) * 2 + int(gold_merchant),
            'uses_topads': int(uses_topads),
            'discount_pct': max(0, (price - discounted_price) / price) if price > 0 else 0,
            'has_discount': 1 if discounted_price < price else 0,
            'category_encoded': hash(features.get('category', 'Unknown')) % 100,
            'shop_location_encoded': hash(features.get('shop_location', 'Unknown')) % 50,
            'rating_average': rating_average
        }
        
        # Select features that model expects
        required = self.sales_model.feature_names
        X = pd.DataFrame([{f: computed.get(f, 0) for f in required}])
        return X.fillna(0)
    
    def _prepare_risk_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features for risk prediction - direct computation for serving."""
        import numpy as np
        
        # Compute derived features directly
        price = features.get('price', 0)
        discounted_price = features.get('discounted_price', price)
        gold_merchant = features.get('gold_merchant', False)
        is_official = features.get('is_official', False)
        uses_topads = features.get('is_topads', False)
        message_length = features.get('message_length', 0)
        word_count = features.get('word_count', 0)
        has_response = features.get('has_response', False)
        
        # Compute all possible features
        computed = {
            'price_log': np.log1p(price) if price > 0 else 0,
            'shop_tier': int(is_official) * 2 + int(gold_merchant),
            'uses_topads': int(uses_topads),
            'discount_pct': max(0, (price - discounted_price) / price) if price > 0 else 0,
            'has_discount': 1 if discounted_price < price else 0,
            'message_length': message_length,
            'word_count': word_count,
            'has_response': int(has_response),
            'category_encoded': hash(features.get('category', 'Unknown')) % 100,
            'shop_location_encoded': hash(features.get('shop_location', 'Unknown')) % 50,
            'review_hour': 12,
            'review_dayofweek': 3,
            'is_weekend': 0
        }
        
        # Select features that model expects
        required = self.risk_model.feature_names
        X = pd.DataFrame([{f: computed.get(f, 0) for f in required}])
        return X.fillna(0)
    
    @property
    def models_loaded(self) -> Dict[str, bool]:
        """Check which models are loaded."""
        return {
            'sales': self.sales_model is not None,
            'risk': self.risk_model is not None
        }


def predict_with_explanation(
    model: Any,
    X: pd.DataFrame,
    model_type: str = 'risk'
) -> Tuple[float, Optional[Dict]]:
    """
    Generate prediction with SHAP explanation.
    
    Args:
        model: Trained model
        X: Feature DataFrame (single row)
        model_type: 'sales' or 'risk'
        
    Returns:
        Tuple of (prediction, explanation dict)
    """
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        from src.explainability.explanations import format_local_explanation
        
        # Get prediction
        if model_type == 'risk':
            prediction = model.predict_proba(X)[0]
        else:
            prediction = model.predict(X)[0]
        
        # Generate explanation
        explainer = SHAPExplainer(model, feature_names=list(X.columns))
        shap_exp = explainer.explain(X)
        explanation = format_local_explanation(prediction, shap_exp, model_type)
        
        return prediction, explanation.to_dict()
        
    except Exception as e:
        logger.warning(f"Failed to generate explanation: {e}")
        if model_type == 'risk':
            prediction = model.predict_proba(X)[0]
        else:
            prediction = model.predict(X)[0]
        return prediction, None


def batch_predict(
    model: Any,
    X: pd.DataFrame,
    model_type: str = 'risk'
) -> pd.DataFrame:
    """
    Batch prediction for multiple instances.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        model_type: 'sales' or 'risk'
        
    Returns:
        DataFrame with predictions
    """
    if model_type == 'risk':
        probabilities = model.predict_proba(X)
        predictions = model.predict(X)
        
        return pd.DataFrame({
            'risk_probability': probabilities,
            'is_high_risk': predictions
        })
    else:
        predictions = model.predict(X)
        
        return pd.DataFrame({
            'predicted_sales': predictions
        })
