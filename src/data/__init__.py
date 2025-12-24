"""
Data Layer Module

Handles data ingestion, validation, and preprocessing for the
Review Intelligence System.

Main Classes:
    TokopediaDataLoader: Unified data loading interface
    DataValidator: Validation orchestrator
    DataPreprocessor: Preprocessing pipeline
    
Quick Start:
    from src.data import TokopediaDataLoader, DataValidator, DataPreprocessor
    
    # Load data
    loader = TokopediaDataLoader('tokopedia_products_with_review.csv')
    products = loader.load_products(nrows=1000)
    reviews = loader.load_reviews(nrows=1000)
    
    # Validate
    validator = DataValidator()
    report = validator.validate(products)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.fit_transform(products)
"""

from src.data.ingestion import (
    TokopediaDataLoader,
    load_tokopedia_data,
    load_reviews,
    explode_reviews,
    get_default_loader
)

from src.data.validation import (
    DataValidator,
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    validate_schema,
    check_completeness
)

from src.data.preprocessing import (
    DataPreprocessor,
    TextCleaner,
    clean_product_data,
    clean_review_data,
    handle_missing_values
)

__all__ = [
    # Ingestion
    'TokopediaDataLoader',
    'load_tokopedia_data',
    'load_reviews',
    'explode_reviews',
    'get_default_loader',
    
    # Validation
    'DataValidator',
    'ValidationResult',
    'ValidationReport',
    'ValidationLevel',
    'validate_schema',
    'check_completeness',
    
    # Preprocessing
    'DataPreprocessor',
    'TextCleaner',
    'clean_product_data',
    'clean_review_data',
    'handle_missing_values',
]
