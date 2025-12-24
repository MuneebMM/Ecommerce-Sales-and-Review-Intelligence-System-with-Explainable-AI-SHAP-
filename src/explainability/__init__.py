"""
Explainability Module

SHAP-based explainability engine providing transparent,
actionable explanations for every prediction.

Main Classes:
    SHAPExplainer: Core SHAP computation
    ExplanationFormatter: Format for API/reports
    ExplanationVisualizer: Generate plots
"""

from src.explainability.shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    TreeSHAPEngine,
    compute_shap_values,
    get_feature_importance
)

from src.explainability.explanations import (
    Explanation,
    FeatureContribution,
    ExplanationFormatter,
    format_local_explanation,
    format_global_explanation,
    generate_counterfactuals
)

from src.explainability.visualization import (
    ExplanationVisualizer,
    plot_waterfall,
    plot_summary
)

__all__ = [
    # SHAP Core
    'SHAPExplainer',
    'SHAPExplanation',
    'TreeSHAPEngine',
    'compute_shap_values',
    'get_feature_importance',
    
    # Formatting
    'Explanation',
    'FeatureContribution',
    'ExplanationFormatter',
    'format_local_explanation',
    'format_global_explanation',
    'generate_counterfactuals',
    
    # Visualization
    'ExplanationVisualizer',
    'plot_waterfall',
    'plot_summary',
]
