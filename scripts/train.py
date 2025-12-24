#!/usr/bin/env python
"""
Training Script

End-to-end training pipeline for the Review Intelligence System.
Trains both Sales Predictor and Review Risk Predictor models.

Usage:
    python scripts/train.py --data-path tokopedia_products_with_review.csv
    python scripts/train.py --model risk --nrows 1000  # Quick test
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.data.ingestion import load_tokopedia_data, explode_reviews
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.models.sales_predictor import SalesPredictor, train_sales_model
from src.models.review_risk_predictor import ReviewRiskPredictor, train_risk_model
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def train_risk_pipeline(
    data_path: str,
    nrows: int = None,
    save_model: bool = True
):
    """
    Train the Review Risk Predictor.
    
    Args:
        data_path: Path to dataset
        nrows: Number of rows to load (for testing)
        save_model: Whether to save model to disk
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Review Risk Predictor")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading data...")
    df = load_tokopedia_data(data_path, nrows=nrows)
    logger.info(f"Loaded {len(df):,} products")
    
    # Validate
    logger.info("Validating data...")
    validator = DataValidator(strict=False)
    report = validator.validate(df)
    logger.info(report.summary())
    
    # Explode to review level
    logger.info("Exploding to review level...")
    reviews = explode_reviews(df)
    logger.info(f"Created {len(reviews):,} reviews")
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    reviews = preprocessor.fit_transform(reviews)
    
    # Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    reviews = engineer.fit_transform(reviews, target='risk')
    
    # Prepare training data
    feature_cols = [
        'price_log', 'discount_pct', 'has_discount',
        'shop_tier', 'uses_topads',
        'message_length', 'word_count',
        'category_encoded', 'shop_location_encoded'
    ]
    
    available_features = [f for f in feature_cols if f in reviews.columns]
    logger.info(f"Using features: {available_features}")
    
    X = reviews[available_features].fillna(0)
    y = reviews['is_negative_review']
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Train model
    logger.info("Training model...")
    model, metrics = train_risk_model(X, y, test_size=0.2)
    
    logger.info("=" * 40)
    logger.info("RESULTS:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    logger.info("=" * 40)
    
    # Save model
    if save_model:
        model_path = Path("models/review_risk_predictor/model.pkl")
        model.save(model_path)
        
        # Register model
        registry = ModelRegistry("models/")
        registry.register_model(
            model_type="risk",
            version="v1.0",
            artifact_path=str(model_path),
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            features=available_features
        )
        logger.info(f"Model saved and registered: {model_path}")
    
    # Feature importance
    importance = model.get_feature_importance()
    logger.info("Feature Importance:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


def train_sales_pipeline(
    data_path: str,
    nrows: int = None,
    save_model: bool = True
):
    """
    Train the Sales Volume Predictor.
    
    Args:
        data_path: Path to dataset
        nrows: Number of rows to load
        save_model: Whether to save model to disk
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Sales Volume Predictor")
    logger.info("=" * 60)
    
    # Load data (product level)
    logger.info("Loading data...")
    df = load_tokopedia_data(data_path, nrows=nrows, parse_lists=False)
    logger.info(f"Loaded {len(df):,} products")
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor(clean_text=False)
    df = preprocessor.fit_transform(df)
    
    # Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.fit_transform(df, target='sales')
    
    # Prepare training data
    feature_cols = [
        'price_log', 'discount_pct', 'has_discount',
        'stock_log', 'has_stock', 'low_stock',
        'shop_tier', 'uses_topads',
        'category_encoded', 'shop_location_encoded',
        'rating_average'
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    
    # Filter to rows with valid target
    df = df[df['count_sold'].notna() & (df['count_sold'] >= 0)]
    
    X = df[available_features].fillna(0)
    y = df['count_sold']
    
    logger.info(f"Training data shape: X={X.shape}")
    logger.info(f"Target stats: mean={y.mean():.2f}, median={y.median():.2f}")
    
    # Train model
    logger.info("Training model...")
    model, metrics = train_sales_model(X, y, test_size=0.2)
    
    logger.info("=" * 40)
    logger.info("RESULTS:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 40)
    
    # Save model
    if save_model:
        model_path = Path("models/sales_predictor/model.pkl")
        model.save(model_path)
        
        # Register model
        registry = ModelRegistry("models/")
        registry.register_model(
            model_type="sales",
            version="v1.0",
            artifact_path=str(model_path),
            metrics=metrics,
            features=available_features
        )
        logger.info(f"Model saved and registered: {model_path}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Review Intelligence models")
    parser.add_argument(
        "--data-path",
        default="tokopedia_products_with_review.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        "--model",
        choices=["risk", "sales", "all"],
        default="all",
        help="Which model to train"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to load (for testing)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save model to disk"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO", log_file="logs/training.log")
    
    logger.info("=" * 60)
    logger.info("Review Intelligence System - Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"N rows: {args.nrows or 'all'}")
    
    results = {}
    
    if args.model in ["risk", "all"]:
        model, metrics = train_risk_pipeline(
            args.data_path,
            nrows=args.nrows,
            save_model=not args.no_save
        )
        results["risk"] = metrics
    
    if args.model in ["sales", "all"]:
        model, metrics = train_sales_pipeline(
            args.data_path,
            nrows=args.nrows,
            save_model=not args.no_save
        )
        results["sales"] = metrics
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
