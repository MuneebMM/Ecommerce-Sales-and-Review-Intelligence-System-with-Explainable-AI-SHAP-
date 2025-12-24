"""
Visualization Module

SHAP visualization components for dashboards and reports.
"""

import logging
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import shap
    SHAP_VIZ_AVAILABLE = True
except ImportError:
    SHAP_VIZ_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """
    Visualization generator for SHAP explanations.
    
    Creates publication-quality plots for analysis and reporting.
    """
    
    def __init__(self, output_dir: str = 'logs/plots'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, visualization disabled")
    
    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str],
        feature_values: np.ndarray,
        title: str = "SHAP Waterfall Plot",
        save_path: Optional[str] = None,
        max_display: int = 10
    ) -> Optional[str]:
        """
        Create waterfall chart showing feature contributions.
        
        Args:
            shap_values: SHAP value array
            base_value: Expected value
            feature_names: Feature names
            feature_values: Feature values
            title: Plot title
            save_path: Path to save plot
            max_display: Max features to show
            
        Returns:
            Path to saved plot or None
        """
        if not MATPLOTLIB_AVAILABLE or not SHAP_VIZ_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_values))[::-1][:max_display]
            
            sorted_shap = shap_values[indices]
            sorted_names = [feature_names[i] for i in indices]
            
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in sorted_shap]
            
            y_pos = np.arange(len(sorted_shap))
            ax.barh(y_pos, sorted_shap, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.invert_yaxis()
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / 'waterfall.png'
            
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved waterfall plot to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            return None
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance (SHAP)",
        save_path: Optional[str] = None,
        max_display: int = 15
    ) -> Optional[str]:
        """
        Create summary plot showing global feature importance.
        
        Args:
            shap_values: 2D array (samples x features)
            feature_names: Feature names
            title: Plot title
            save_path: Path to save plot
            max_display: Max features to show
            
        Returns:
            Path to saved plot or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Compute mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)
            indices = np.argsort(importance)[::-1][:max_display]
            
            sorted_importance = importance[indices]
            sorted_names = [feature_names[i] for i in indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(sorted_importance))
            ax.barh(y_pos, sorted_importance, color='#667eea')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(title)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / 'summary.png'
            
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved summary plot to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            return None


def plot_waterfall(shap_explanation, save_path: str = None) -> Optional[str]:
    """Convenience function for waterfall plot."""
    viz = ExplanationVisualizer()
    return viz.plot_waterfall(
        shap_values=shap_explanation.shap_values,
        base_value=shap_explanation.base_value,
        feature_names=shap_explanation.feature_names,
        feature_values=shap_explanation.feature_values,
        save_path=save_path
    )


def plot_summary(shap_values: np.ndarray, feature_names: List[str], save_path: str = None) -> Optional[str]:
    """Convenience function for summary plot."""
    viz = ExplanationVisualizer()
    return viz.plot_summary(shap_values, feature_names, save_path=save_path)
