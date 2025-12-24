"""
Data Validation Module

Implements data quality checks and validation rules for the Tokopedia dataset.
Based on the data validation checklist specifications.

Key Validations:
    - Schema validation (required columns, data types)
    - Completeness checks (missing value thresholds)
    - Range checks (price > 0, ratings 1-5)
    - List alignment checks (review columns must have matching lengths)
    - NLP quality checks (empty reviews, min length)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity level for validation checks."""
    BLOCKING = "blocking"  # Pipeline must stop
    WARNING = "warning"    # Log alert, continue
    INFO = "info"          # Informational only


@dataclass
class ValidationResult:
    """
    Container for a single validation check result.
    
    Attributes:
        check_name: Name of the validation check
        passed: Whether the check passed
        level: Severity level
        message: Human-readable result message
        details: Additional details (e.g., affected rows)
    """
    check_name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict] = None
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{self.level.value.upper()}] {status}: {self.check_name} - {self.message}"


@dataclass 
class ValidationReport:
    """
    Aggregated validation report for a dataset.
    
    Attributes:
        results: List of individual validation results
        passed: Overall pass/fail status
    """
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all blocking validations passed."""
        return all(
            r.passed for r in self.results 
            if r.level == ValidationLevel.BLOCKING
        )
    
    @property
    def blocking_failures(self) -> List[ValidationResult]:
        """Get list of failed blocking validations."""
        return [r for r in self.results if not r.passed and r.level == ValidationLevel.BLOCKING]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get list of warning-level issues."""
        return [r for r in self.results if not r.passed and r.level == ValidationLevel.WARNING]
    
    def add(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)
        log_level = logging.WARNING if not result.passed else logging.INFO
        logger.log(log_level, str(result))
    
    def summary(self) -> str:
        """Generate a summary string."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        blocking_fails = len(self.blocking_failures)
        warnings = len(self.warnings)
        
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        
        return (
            f"Validation Report: {status}\n"
            f"  Total Checks: {total}\n"
            f"  Passed: {passed}\n"
            f"  Blocking Failures: {blocking_fails}\n"
            f"  Warnings: {warnings}"
        )


# Required columns for the pipeline
REQUIRED_COLUMNS = [
    'product_id',
    'count_sold',
    'price',
    'review_rating',
    'message'
]


class DataValidator:
    """
    Main validation orchestrator for Tokopedia dataset.
    
    Runs all validation checks and aggregates results into a report.
    
    Example:
        validator = DataValidator()
        report = validator.validate(df)
        if not report.passed:
            raise ValueError(f"Validation failed: {report.blocking_failures}")
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise on blocking failures during validate()
        """
        self.strict = strict
    
    def validate(self, df: pd.DataFrame, is_exploded: bool = False) -> ValidationReport:
        """
        Run all validation checks on the dataset.
        
        Args:
            df: DataFrame to validate
            is_exploded: Whether this is review-level (exploded) data
            
        Returns:
            ValidationReport with all results
            
        Raises:
            ValueError: If strict=True and blocking validations fail
        """
        report = ValidationReport()
        
        # Schema checks
        report.add(self._check_required_columns(df))
        report.add(self._check_product_id_type(df))
        
        # Range checks
        report.add(self._check_price_positive(df))
        report.add(self._check_rating_range(df, is_exploded))
        report.add(self._check_discount_logic(df))
        
        # Missing value checks
        report.add(self._check_missing_values(df))
        
        # List alignment (only for non-exploded data)
        if not is_exploded:
            report.add(self._check_list_alignment(df))
        
        # Outlier detection
        report.add(self._check_sales_outliers(df))
        
        # Log summary
        logger.info(report.summary())
        
        # Strict mode: raise on blocking failures
        if self.strict and not report.passed:
            failures = [str(f) for f in report.blocking_failures]
            raise ValueError(f"Blocking validation failures:\n" + "\n".join(failures))
        
        return report
    
    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Check that all required columns exist."""
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        
        if missing:
            return ValidationResult(
                check_name="required_columns",
                passed=False,
                level=ValidationLevel.BLOCKING,
                message=f"Missing required columns: {missing}",
                details={'missing': list(missing)}
            )
        
        return ValidationResult(
            check_name="required_columns",
            passed=True,
            level=ValidationLevel.BLOCKING,
            message="All required columns present"
        )
    
    def _check_product_id_type(self, df: pd.DataFrame) -> ValidationResult:
        """Check that product_id is numeric."""
        if 'product_id' not in df.columns:
            return ValidationResult(
                check_name="product_id_type",
                passed=False,
                level=ValidationLevel.BLOCKING,
                message="product_id column missing"
            )
        
        is_numeric = pd.api.types.is_numeric_dtype(df['product_id'])
        
        return ValidationResult(
            check_name="product_id_type",
            passed=is_numeric,
            level=ValidationLevel.BLOCKING,
            message="product_id is numeric" if is_numeric else "product_id is not numeric"
        )
    
    def _check_price_positive(self, df: pd.DataFrame) -> ValidationResult:
        """Check that price > 0."""
        if 'price' not in df.columns:
            return ValidationResult(
                check_name="price_positive",
                passed=True,
                level=ValidationLevel.WARNING,
                message="price column not present, skipping"
            )
        
        invalid_count = (df['price'] <= 0).sum()
        
        if invalid_count > 0:
            return ValidationResult(
                check_name="price_positive",
                passed=False,
                level=ValidationLevel.BLOCKING,
                message=f"{invalid_count:,} rows have price <= 0",
                details={'invalid_count': invalid_count}
            )
        
        return ValidationResult(
            check_name="price_positive",
            passed=True,
            level=ValidationLevel.BLOCKING,
            message="All prices are positive"
        )
    
    def _check_rating_range(self, df: pd.DataFrame, is_exploded: bool) -> ValidationResult:
        """Check rating values are in valid range [1, 5]."""
        if is_exploded:
            # For exploded data, review_rating is a scalar
            if 'review_rating' not in df.columns:
                return ValidationResult(
                    check_name="rating_range",
                    passed=True,
                    level=ValidationLevel.WARNING,
                    message="review_rating column not present"
                )
            
            ratings = pd.to_numeric(df['review_rating'], errors='coerce')
            invalid = ((ratings < 1) | (ratings > 5)).sum()
            
        else:
            # For product data, rating_average is the aggregate
            if 'rating_average' not in df.columns:
                return ValidationResult(
                    check_name="rating_range",
                    passed=True,
                    level=ValidationLevel.WARNING,
                    message="rating_average column not present"
                )
            
            ratings = df['rating_average'].dropna()
            invalid = ((ratings < 1.0) | (ratings > 5.0)).sum()
        
        if invalid > 0:
            return ValidationResult(
                check_name="rating_range",
                passed=False,
                level=ValidationLevel.BLOCKING,
                message=f"{invalid:,} rows have ratings outside [1, 5] range",
                details={'invalid_count': invalid}
            )
        
        return ValidationResult(
            check_name="rating_range",
            passed=True,
            level=ValidationLevel.BLOCKING,
            message="All ratings in valid range [1, 5]"
        )
    
    def _check_discount_logic(self, df: pd.DataFrame) -> ValidationResult:
        """Check that discounted_price <= price where both exist."""
        if 'discounted_price' not in df.columns or 'price' not in df.columns:
            return ValidationResult(
                check_name="discount_logic",
                passed=True,
                level=ValidationLevel.INFO,
                message="Discount columns not present, skipping"
            )
        
        # Only check where discounted_price is not null
        mask = df['discounted_price'].notna()
        invalid = (df.loc[mask, 'discounted_price'] > df.loc[mask, 'price']).sum()
        
        if invalid > 0:
            return ValidationResult(
                check_name="discount_logic",
                passed=False,
                level=ValidationLevel.WARNING,  # Known data quality issue
                message=f"{invalid:,} rows have discounted_price > price (will be handled by preprocessing)",
                details={'invalid_count': invalid}
            )
        
        return ValidationResult(
            check_name="discount_logic",
            passed=True,
            level=ValidationLevel.BLOCKING,
            message="Discount prices are valid"
        )
    
    def _check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check missing value rates for key columns."""
        key_cols = ['product_id', 'price', 'count_sold']
        key_cols = [c for c in key_cols if c in df.columns]
        
        missing_info = {}
        high_missing = []
        
        for col in key_cols:
            missing_pct = df[col].isna().mean() * 100
            missing_info[col] = f"{missing_pct:.1f}%"
            
            if col == 'product_id' and missing_pct > 0:
                high_missing.append(col)
        
        if high_missing:
            return ValidationResult(
                check_name="missing_values",
                passed=False,
                level=ValidationLevel.BLOCKING,
                message=f"Critical columns have missing values: {high_missing}",
                details=missing_info
            )
        
        return ValidationResult(
            check_name="missing_values",
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Missing value rates: {missing_info}"
        )
    
    def _check_list_alignment(self, df: pd.DataFrame) -> ValidationResult:
        """Check that list columns have matching lengths per row."""
        list_cols = ['review_rating', 'message']
        list_cols = [c for c in list_cols if c in df.columns]
        
        if len(list_cols) < 2:
            return ValidationResult(
                check_name="list_alignment",
                passed=True,
                level=ValidationLevel.INFO,
                message="Not enough list columns to check alignment"
            )
        
        def safe_len(val):
            if isinstance(val, list):
                return len(val)
            return 0
        
        len_col1 = df[list_cols[0]].apply(safe_len)
        len_col2 = df[list_cols[1]].apply(safe_len)
        
        misaligned = (len_col1 != len_col2).sum()
        
        if misaligned > 0:
            return ValidationResult(
                check_name="list_alignment",
                passed=False,
                level=ValidationLevel.WARNING,
                message=f"{misaligned:,} rows have misaligned list lengths (will be handled during explosion)",
                details={'misaligned_count': misaligned}
            )
        
        return ValidationResult(
            check_name="list_alignment",
            passed=True,
            level=ValidationLevel.WARNING,
            message="All list columns properly aligned"
        )
    
    def _check_sales_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Detect and report extreme outliers in count_sold."""
        if 'count_sold' not in df.columns:
            return ValidationResult(
                check_name="sales_outliers",
                passed=True,
                level=ValidationLevel.INFO,
                message="count_sold column not present"
            )
        
        sales = df['count_sold'].dropna()
        
        if len(sales) == 0:
            return ValidationResult(
                check_name="sales_outliers",
                passed=True,
                level=ValidationLevel.INFO,
                message="No sales data to check"
            )
        
        # Compute IQR-based outlier threshold
        q75 = sales.quantile(0.75)
        q25 = sales.quantile(0.25)
        iqr = q75 - q25
        upper_bound = q75 + 3 * iqr
        
        extreme_outliers = (sales > upper_bound).sum()
        max_sales = sales.max()
        
        return ValidationResult(
            check_name="sales_outliers",
            passed=True,  # Outliers are not failures, just info
            level=ValidationLevel.INFO,
            message=f"Found {extreme_outliers:,} extreme outliers (> {upper_bound:,.0f}). Max: {max_sales:,.0f}",
            details={
                'extreme_count': extreme_outliers,
                'threshold': upper_bound,
                'max_value': max_sales
            }
        )


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Quick schema validation check.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if schema is valid, False otherwise
    """
    validator = DataValidator(strict=False)
    report = validator.validate(df)
    return report.passed


def check_completeness(df: pd.DataFrame, required_cols: List[str] = None) -> Dict[str, float]:
    """
    Check completeness (non-null rate) for specified columns.
    
    Args:
        df: DataFrame to check
        required_cols: Columns to check. Defaults to REQUIRED_COLUMNS.
        
    Returns:
        Dictionary mapping column names to completeness percentages
    """
    required_cols = required_cols or REQUIRED_COLUMNS
    
    completeness = {}
    for col in required_cols:
        if col in df.columns:
            completeness[col] = (1 - df[col].isna().mean()) * 100
    
    return completeness
