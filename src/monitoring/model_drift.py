"""
Model Drift Detection Module

Monitors model performance degradation in production.
Tracks prediction distribution changes and accuracy drops.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ModelDriftResult:
    """Result of model drift detection."""
    timestamp: str
    model_type: str
    prediction_drift: float
    performance_drift: Optional[float]
    is_drifted: bool
    details: Dict
    
    def get_summary(self) -> str:
        status = "DRIFTED" if self.is_drifted else "STABLE"
        return f"Model {self.model_type}: {status} (prediction_drift={self.prediction_drift:.4f})"


class ModelDriftDetector:
    """
    Detects model performance and prediction drift.
    
    Monitors:
    1. Prediction distribution shift (PSI)
    2. Performance metric degradation (if labels available)
    
    Example:
        detector = ModelDriftDetector('risk', baseline_predictions)
        result = detector.detect(current_predictions)
        if result.is_drifted:
            trigger_retraining()
    """
    
    PSI_THRESHOLD = 0.2  # Population Stability Index threshold
    PERFORMANCE_THRESHOLD = 0.1  # 10% performance drop
    
    def __init__(
        self,
        model_type: str,
        baseline_predictions: np.ndarray,
        baseline_labels: Optional[np.ndarray] = None
    ):
        """
        Initialize model drift detector.
        
        Args:
            model_type: 'sales' or 'risk'
            baseline_predictions: Reference prediction distribution
            baseline_labels: Reference labels (for performance tracking)
        """
        self.model_type = model_type
        self.baseline_predictions = baseline_predictions
        self.baseline_labels = baseline_labels
        
        # Compute baseline metrics
        self._baseline_metrics = self._compute_metrics(baseline_predictions, baseline_labels)
    
    def detect(
        self,
        current_predictions: np.ndarray,
        current_labels: Optional[np.ndarray] = None
    ) -> ModelDriftResult:
        """
        Detect drift in model predictions and performance.
        
        Args:
            current_predictions: Current prediction values
            current_labels: Current true labels (if available)
            
        Returns:
            ModelDriftResult with drift metrics
        """
        # Calculate PSI
        psi = self._calculate_psi(self.baseline_predictions, current_predictions)
        
        # Calculate performance drift if labels available
        if current_labels is not None and self.baseline_labels is not None:
            current_metrics = self._compute_metrics(current_predictions, current_labels)
            
            if self.model_type == 'risk':
                baseline_perf = self._baseline_metrics.get('auc', 0.5)
                current_perf = current_metrics.get('auc', 0.5)
            else:
                baseline_perf = self._baseline_metrics.get('r2', 0)
                current_perf = current_metrics.get('r2', 0)
            
            performance_drift = baseline_perf - current_perf
        else:
            performance_drift = None
        
        # Determine if drifted
        is_drifted = psi > self.PSI_THRESHOLD
        if performance_drift is not None:
            is_drifted = is_drifted or performance_drift > self.PERFORMANCE_THRESHOLD
        
        result = ModelDriftResult(
            timestamp=datetime.now().isoformat(),
            model_type=self.model_type,
            prediction_drift=float(psi),
            performance_drift=float(performance_drift) if performance_drift else None,
            is_drifted=is_drifted,
            details={
                'psi': float(psi),
                'psi_threshold': self.PSI_THRESHOLD,
                'baseline_mean': float(np.mean(self.baseline_predictions)),
                'current_mean': float(np.mean(current_predictions))
            }
        )
        
        logger.info(result.get_summary())
        
        return result
    
    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI measures the shift in distribution between baseline and current.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        # Create bins based on baseline
        if len(np.unique(baseline)) <= n_bins:
            # Categorical or low cardinality
            bins = np.unique(np.concatenate([baseline, current]))
        else:
            # Numerical
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(baseline, percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
        
        # Calculate frequencies
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize to proportions
        baseline_prop = baseline_hist / len(baseline)
        current_prop = current_hist / len(current)
        
        # Avoid division by zero
        baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
        current_prop = np.where(current_prop == 0, 0.0001, current_prop)
        
        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
        
        return float(psi)
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        if labels is None:
            return {}
        
        metrics = {}
        
        if self.model_type == 'risk':
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc'] = roc_auc_score(labels, predictions)
            except Exception:
                metrics['auc'] = 0.5
        else:
            from sklearn.metrics import r2_score
            try:
                metrics['r2'] = r2_score(labels, predictions)
            except Exception:
                metrics['r2'] = 0.0
        
        return metrics


class PerformanceMonitor:
    """
    Tracks model performance over time.
    
    Stores metrics history and alerts on degradation.
    """
    
    def __init__(self, model_type: str, alert_threshold: float = 0.1):
        """
        Initialize performance monitor.
        
        Args:
            model_type: 'sales' or 'risk'
            alert_threshold: Performance drop threshold for alerts
        """
        self.model_type = model_type
        self.alert_threshold = alert_threshold
        self._history: List[Dict] = []
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[str] = None
    ):
        """Log metrics to history."""
        self._history.append({
            'timestamp': timestamp or datetime.now().isoformat(),
            **metrics
        })
    
    def get_trend(self, metric: str, n_points: int = 10) -> Dict:
        """Get trend for a specific metric."""
        if len(self._history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent = self._history[-n_points:]
        values = [h.get(metric, 0) for h in recent]
        
        # Simple linear trend
        if len(values) > 1:
            slope = (values[-1] - values[0]) / len(values)
            trend = 'improving' if slope > 0 else 'degrading'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'current': values[-1] if values else 0,
            'change': values[-1] - values[0] if len(values) > 1 else 0
        }
    
    def check_alerts(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check for performance degradation alerts."""
        alerts = []
        
        if len(self._history) == 0:
            return alerts
        
        baseline = self._history[0]
        
        for metric, value in current_metrics.items():
            if metric in baseline:
                change = (baseline[metric] - value) / baseline[metric] if baseline[metric] != 0 else 0
                if change > self.alert_threshold:
                    alerts.append(f"{metric} degraded by {change*100:.1f}%")
        
        return alerts


def check_model_drift(
    model_type: str,
    baseline_preds: np.ndarray,
    current_preds: np.ndarray
) -> ModelDriftResult:
    """Convenience function for model drift detection."""
    detector = ModelDriftDetector(model_type, baseline_preds)
    return detector.detect(current_preds)
