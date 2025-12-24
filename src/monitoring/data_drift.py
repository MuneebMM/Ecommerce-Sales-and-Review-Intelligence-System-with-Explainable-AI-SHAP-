"""
Data Drift Detection Module

Monitors data distribution changes between training and production data.
Uses statistical tests and distribution comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftLevel(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    drift_score: float
    p_value: float
    drift_level: DriftLevel
    baseline_mean: float
    current_mean: float
    
    @property
    def has_drift(self) -> bool:
        return self.drift_level in [DriftLevel.MEDIUM, DriftLevel.HIGH, DriftLevel.CRITICAL]


@dataclass
class DriftReport:
    """Aggregated drift detection report."""
    timestamp: str
    total_features: int
    drifted_features: int
    results: List[DriftResult]
    overall_drift_level: DriftLevel
    
    def get_summary(self) -> str:
        return (
            f"Drift Report: {self.drifted_features}/{self.total_features} features drifted. "
            f"Overall: {self.overall_drift_level.value}"
        )


class DataDriftDetector:
    """
    Detects distribution drift between reference and current data.
    
    Uses Kolmogorov-Smirnov test for numerical features and
    chi-squared test for categorical features.
    
    Example:
        detector = DataDriftDetector(reference_data)
        report = detector.detect(current_data)
        if report.overall_drift_level == DriftLevel.HIGH:
            trigger_alert()
    """
    
    # Thresholds for drift levels (p-value based)
    THRESHOLDS = {
        DriftLevel.NONE: 0.1,      # p > 0.1
        DriftLevel.LOW: 0.05,      # 0.05 < p <= 0.1
        DriftLevel.MEDIUM: 0.01,   # 0.01 < p <= 0.05
        DriftLevel.HIGH: 0.001,    # 0.001 < p <= 0.01
        DriftLevel.CRITICAL: 0.0   # p <= 0.001
    }
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None
    ):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Training/baseline data
            categorical_features: List of categorical column names
            numerical_features: List of numerical column names
        """
        self.reference = reference_data.copy()
        
        # Auto-detect feature types if not provided
        if numerical_features is None:
            self.numerical_features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        else:
            self.numerical_features = numerical_features
        
        if categorical_features is None:
            self.categorical_features = reference_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        else:
            self.categorical_features = categorical_features
        
        # Pre-compute reference statistics
        self._reference_stats = self._compute_stats(reference_data)
    
    def detect(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current production data
            
        Returns:
            DriftReport with per-feature and overall results
        """
        from datetime import datetime
        
        results = []
        
        # Check numerical features
        for feature in self.numerical_features:
            if feature in current_data.columns and feature in self.reference.columns:
                result = self._detect_numerical_drift(feature, current_data[feature])
                results.append(result)
        
        # Check categorical features
        for feature in self.categorical_features:
            if feature in current_data.columns and feature in self.reference.columns:
                result = self._detect_categorical_drift(feature, current_data[feature])
                results.append(result)
        
        # Determine overall drift level
        if results:
            drift_levels = [r.drift_level.value for r in results]
            drifted = sum(1 for r in results if r.has_drift)
            
            if drifted / len(results) > 0.5:
                overall = DriftLevel.CRITICAL
            elif drifted / len(results) > 0.3:
                overall = DriftLevel.HIGH
            elif drifted / len(results) > 0.1:
                overall = DriftLevel.MEDIUM
            elif drifted > 0:
                overall = DriftLevel.LOW
            else:
                overall = DriftLevel.NONE
        else:
            overall = DriftLevel.NONE
            drifted = 0
        
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            total_features=len(results),
            drifted_features=drifted,
            results=results,
            overall_drift_level=overall
        )
        
        logger.info(report.get_summary())
        
        return report
    
    def _detect_numerical_drift(
        self,
        feature: str,
        current: pd.Series
    ) -> DriftResult:
        """Use KS test for numerical features."""
        reference = self.reference[feature].dropna()
        current = current.dropna()
        
        if len(reference) == 0 or len(current) == 0:
            return DriftResult(
                feature=feature,
                drift_score=0.0,
                p_value=1.0,
                drift_level=DriftLevel.NONE,
                baseline_mean=0.0,
                current_mean=0.0
            )
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(reference, current)
        
        # Determine drift level
        drift_level = self._p_value_to_level(p_value)
        
        return DriftResult(
            feature=feature,
            drift_score=float(statistic),
            p_value=float(p_value),
            drift_level=drift_level,
            baseline_mean=float(reference.mean()),
            current_mean=float(current.mean())
        )
    
    def _detect_categorical_drift(
        self,
        feature: str,
        current: pd.Series
    ) -> DriftResult:
        """Use chi-squared test for categorical features."""
        reference = self.reference[feature].fillna('_missing_')
        current = current.fillna('_missing_')
        
        # Get value counts
        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_freq = [ref_counts.get(c, 0) for c in all_categories]
        cur_freq = [cur_counts.get(c, 0) for c in all_categories]
        
        # Chi-squared test
        try:
            # Use expected frequencies from reference
            chi2, p_value = stats.chisquare(
                f_obs=np.array(cur_freq) * len(current) + 1,  # Add 1 to avoid zeros
                f_exp=np.array(ref_freq) * len(current) + 1
            )
            statistic = float(chi2)
        except Exception:
            statistic = 0.0
            p_value = 1.0
        
        drift_level = self._p_value_to_level(p_value)
        
        return DriftResult(
            feature=feature,
            drift_score=statistic,
            p_value=float(p_value),
            drift_level=drift_level,
            baseline_mean=0.0,  # N/A for categorical
            current_mean=0.0
        )
    
    def _p_value_to_level(self, p_value: float) -> DriftLevel:
        """Convert p-value to drift level."""
        if p_value > 0.1:
            return DriftLevel.NONE
        elif p_value > 0.05:
            return DriftLevel.LOW
        elif p_value > 0.01:
            return DriftLevel.MEDIUM
        elif p_value > 0.001:
            return DriftLevel.HIGH
        else:
            return DriftLevel.CRITICAL
    
    def _compute_stats(self, data: pd.DataFrame) -> Dict:
        """Pre-compute reference statistics."""
        stats_dict = {}
        for col in self.numerical_features:
            if col in data.columns:
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
        return stats_dict


def check_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> DriftReport:
    """Convenience function for drift detection."""
    detector = DataDriftDetector(reference)
    return detector.detect(current)
