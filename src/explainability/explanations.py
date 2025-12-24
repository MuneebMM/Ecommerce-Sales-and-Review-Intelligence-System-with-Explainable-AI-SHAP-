"""
Explanations Module

Formats and structures SHAP explanation outputs for API responses
and human-readable reports.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Single feature's contribution to prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str  # 'positive' or 'negative'
    contribution_pct: float = 0.0


@dataclass
class Explanation:
    """
    Formatted explanation for a single prediction.
    
    Provides both machine-readable and human-readable formats.
    """
    prediction: float
    base_value: float
    contributions: List[FeatureContribution]
    model_type: str  # 'sales' or 'risk'
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        if self.model_type == 'risk':
            pred_str = f"{self.prediction * 100:.1f}% risk"
        else:
            pred_str = f"{self.prediction:,.0f} units"
        
        top_positive = [c for c in self.contributions if c.direction == 'positive'][:2]
        top_negative = [c for c in self.contributions if c.direction == 'negative'][:2]
        
        lines = [f"Prediction: {pred_str}"]
        
        if top_positive:
            drivers = ", ".join([c.feature_name for c in top_positive])
            lines.append(f"Key drivers ↑: {drivers}")
        
        if top_negative:
            reducers = ", ".join([c.feature_name for c in top_negative])
            lines.append(f"Key reducers ↓: {reducers}")
        
        return " | ".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            'prediction': self.prediction,
            'base_value': self.base_value,
            'model_type': self.model_type,
            'summary': self.get_summary(),
            'contributions': [asdict(c) for c in self.contributions]
        }


class ExplanationFormatter:
    """
    Formats raw SHAP values into structured explanations.
    
    Handles both regression (sales) and classification (risk) outputs.
    """
    
    def __init__(self, model_type: str = 'risk'):
        """
        Initialize formatter.
        
        Args:
            model_type: 'sales' or 'risk'
        """
        self.model_type = model_type
    
    def format(
        self,
        prediction: float,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str],
        feature_values: np.ndarray,
        top_n: int = 10
    ) -> Explanation:
        """
        Format SHAP values into structured explanation.
        
        Args:
            prediction: Model prediction
            shap_values: SHAP values array
            base_value: Expected value
            feature_names: Feature names
            feature_values: Feature values
            top_n: Number of top features to include
            
        Returns:
            Explanation object
        """
        # Calculate contributions
        total_shap = np.sum(np.abs(shap_values))
        
        contributions = []
        for i, (name, shap_val, feat_val) in enumerate(
            zip(feature_names, shap_values, feature_values)
        ):
            contrib = FeatureContribution(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(feat_val) if np.isfinite(feat_val) else 0.0,
                direction='positive' if shap_val > 0 else 'negative',
                contribution_pct=abs(shap_val) / total_shap * 100 if total_shap > 0 else 0.0
            )
            contributions.append(contrib)
        
        # Sort by absolute SHAP value and take top N
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        contributions = contributions[:top_n]
        
        return Explanation(
            prediction=float(prediction),
            base_value=float(base_value),
            contributions=contributions,
            model_type=self.model_type
        )


def format_local_explanation(
    prediction: float,
    shap_explanation,
    model_type: str = 'risk'
) -> Explanation:
    """
    Format a single prediction explanation.
    
    Args:
        prediction: Model prediction
        shap_explanation: SHAPExplanation object
        model_type: 'sales' or 'risk'
        
    Returns:
        Formatted Explanation
    """
    formatter = ExplanationFormatter(model_type)
    
    return formatter.format(
        prediction=prediction,
        shap_values=shap_explanation.shap_values,
        base_value=shap_explanation.base_value,
        feature_names=shap_explanation.feature_names,
        feature_values=shap_explanation.feature_values
    )


def format_global_explanation(
    feature_importance: pd.DataFrame,
    model_type: str = 'risk'
) -> Dict:
    """
    Format global feature importance.
    
    Args:
        feature_importance: DataFrame with feature importance
        model_type: Model type
        
    Returns:
        Dictionary with formatted importance
    """
    return {
        'model_type': model_type,
        'feature_importance': feature_importance.to_dict('records'),
        'top_features': feature_importance.head(10)['feature'].tolist()
    }


def generate_counterfactuals(
    explanation: Explanation,
    n_suggestions: int = 3
) -> List[Dict]:
    """
    Generate simple counterfactual suggestions.
    
    Args:
        explanation: Current explanation
        n_suggestions: Number of suggestions
        
    Returns:
        List of counterfactual suggestions
    """
    suggestions = []
    
    # Find top negative contributors (for risk reduction)
    if explanation.model_type == 'risk':
        positive_contribs = [c for c in explanation.contributions if c.direction == 'positive']
        
        for contrib in positive_contribs[:n_suggestions]:
            suggestion = {
                'feature': contrib.feature_name,
                'current_value': contrib.feature_value,
                'recommendation': f"Consider improving {contrib.feature_name}",
                'impact': f"Could reduce risk by ~{contrib.contribution_pct:.1f}%"
            }
            suggestions.append(suggestion)
    
    return suggestions
