"""
Model Evaluator Module

Comprehensive model evaluation and validation for both models.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Container for evaluation results."""
    model_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        lines = [f"=== {self.model_type.upper()} Model Evaluation ==="]
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class ModelEvaluator:
    """
    Main evaluation orchestrator.
    
    Evaluates both regression (sales) and classification (risk) models.
    """
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "regression"
    ) -> EvaluationReport:
        """
        Evaluate regression model predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model identifier
            
        Returns:
            EvaluationReport
        """
        # Handle zeros for MAPE
        y_safe = np.where(y_true == 0, 1, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_safe)) * 100
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mape,
            'r2': r2_score(y_true, y_pred),
            'median_ae': np.median(np.abs(y_true - y_pred))
        }
        
        report = EvaluationReport(model_type=model_name, metrics=metrics)
        logger.info(report.summary())
        
        return report
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        model_name: str = "classification"
    ) -> EvaluationReport:
        """
        Evaluate classification model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Model identifier
            
        Returns:
            EvaluationReport
        """
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': (y_true == y_pred).mean()
        }
        
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn
        
        report = EvaluationReport(model_type=model_name, metrics=metrics)
        logger.info(report.summary())
        
        return report
    
    def compare_models(
        self,
        reports: List[EvaluationReport],
        primary_metric: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluations.
        
        Args:
            reports: List of evaluation reports
            primary_metric: Metric to sort by
            
        Returns:
            Comparison DataFrame
        """
        rows = []
        for report in reports:
            row = {'model': report.model_type, **report.metrics}
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if primary_metric in df.columns:
            df = df.sort_values(primary_metric, ascending=False)
        
        return df


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Quick regression evaluation."""
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_regression(y_true, y_pred)
    return report.metrics


def evaluate_classification(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Quick classification evaluation."""
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_classification(y_true, y_pred, y_proba)
    return report.metrics
