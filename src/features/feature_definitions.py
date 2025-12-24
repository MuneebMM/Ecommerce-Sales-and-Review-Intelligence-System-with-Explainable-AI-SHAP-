"""
Feature Definitions Module

Schema definitions for all features used in the system.
Defines feature metadata, types, and groupings for both models.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    TIMESTAMP = "timestamp"


class FeatureSource(Enum):
    """Source of feature data."""
    RAW = "raw"           # Direct from dataset
    DERIVED = "derived"   # Computed feature
    ENCODED = "encoded"   # Encoded categorical


@dataclass
class FeatureSchema:
    """
    Feature metadata and validation rules.
    
    Attributes:
        name: Feature column name
        dtype: Feature data type
        source: Where feature comes from
        nullable: Whether nulls are allowed
        description: Human-readable description
    """
    name: str
    dtype: FeatureType
    source: FeatureSource
    nullable: bool = True
    description: str = ""
    
    def __str__(self):
        return f"{self.name} ({self.dtype.value})"


# =============================================================================
# RAW DATASET COLUMNS
# =============================================================================

RAW_COLUMNS = {
    # Product Identity
    'product_id': FeatureSchema('product_id', FeatureType.NUMERICAL, FeatureSource.RAW, False, 'Unique product identifier'),
    'name': FeatureSchema('name', FeatureType.TEXT, FeatureSource.RAW, False, 'Product name'),
    'category': FeatureSchema('category', FeatureType.CATEGORICAL, FeatureSource.RAW, False, 'Product category'),
    
    # Pricing
    'price': FeatureSchema('price', FeatureType.NUMERICAL, FeatureSource.RAW, False, 'Base price in IDR'),
    'discounted_price': FeatureSchema('discounted_price', FeatureType.NUMERICAL, FeatureSource.RAW, True, 'Discounted price'),
    
    # Inventory
    'stock': FeatureSchema('stock', FeatureType.NUMERICAL, FeatureSource.RAW, False, 'Available stock'),
    'preorder': FeatureSchema('preorder', FeatureType.BINARY, FeatureSource.RAW, False, 'Preorder flag'),
    
    # Shop
    'shop_id': FeatureSchema('shop_id', FeatureType.NUMERICAL, FeatureSource.RAW, False, 'Shop identifier'),
    'shop_location': FeatureSchema('shop_location', FeatureType.CATEGORICAL, FeatureSource.RAW, True, 'Shop location'),
    'gold_merchant': FeatureSchema('gold_merchant', FeatureType.BINARY, FeatureSource.RAW, False, 'Gold merchant flag'),
    'is_official': FeatureSchema('is_official', FeatureType.BINARY, FeatureSource.RAW, False, 'Official store flag'),
    'is_topads': FeatureSchema('is_topads', FeatureType.BINARY, FeatureSource.RAW, False, 'TopAds active'),
    
    # Rating
    'rating_average': FeatureSchema('rating_average', FeatureType.NUMERICAL, FeatureSource.RAW, True, 'Average product rating'),
    
    # Review (post-explosion)
    'review_rating': FeatureSchema('review_rating', FeatureType.NUMERICAL, FeatureSource.RAW, False, 'Individual review rating'),
    'message': FeatureSchema('message', FeatureType.TEXT, FeatureSource.RAW, True, 'Review text'),
    'review_timestamp': FeatureSchema('review_timestamp', FeatureType.TIMESTAMP, FeatureSource.RAW, True, 'Review timestamp'),
    'review_response': FeatureSchema('review_response', FeatureType.TEXT, FeatureSource.RAW, True, 'Seller response'),
}

# =============================================================================
# DERIVED FEATURES
# =============================================================================

DERIVED_FEATURES = {
    # Price features
    'price_log': FeatureSchema('price_log', FeatureType.NUMERICAL, FeatureSource.DERIVED, False, 'Log-transformed price'),
    'price_bucket': FeatureSchema('price_bucket', FeatureType.CATEGORICAL, FeatureSource.DERIVED, False, 'Price range bucket'),
    'discount_pct': FeatureSchema('discount_pct', FeatureType.NUMERICAL, FeatureSource.DERIVED, False, 'Discount percentage'),
    'has_discount': FeatureSchema('has_discount', FeatureType.BINARY, FeatureSource.DERIVED, False, 'Has discount flag'),
    
    # Stock features
    'stock_log': FeatureSchema('stock_log', FeatureType.NUMERICAL, FeatureSource.DERIVED, False, 'Log-transformed stock'),
    'has_stock': FeatureSchema('has_stock', FeatureType.BINARY, FeatureSource.DERIVED, False, 'In stock flag'),
    'low_stock': FeatureSchema('low_stock', FeatureType.BINARY, FeatureSource.DERIVED, False, 'Low stock warning'),
    
    # Shop features
    'shop_tier': FeatureSchema('shop_tier', FeatureType.CATEGORICAL, FeatureSource.DERIVED, False, 'Composite shop quality'),
    'uses_topads': FeatureSchema('uses_topads', FeatureType.BINARY, FeatureSource.DERIVED, False, 'TopAds active'),
    
    # Text features
    'message_length': FeatureSchema('message_length', FeatureType.NUMERICAL, FeatureSource.DERIVED, False, 'Review character count'),
    'word_count': FeatureSchema('word_count', FeatureType.NUMERICAL, FeatureSource.DERIVED, False, 'Review word count'),
    'has_response': FeatureSchema('has_response', FeatureType.BINARY, FeatureSource.DERIVED, False, 'Seller responded'),
    
    # Temporal features
    'review_hour': FeatureSchema('review_hour', FeatureType.NUMERICAL, FeatureSource.DERIVED, True, 'Hour of review'),
    'review_dayofweek': FeatureSchema('review_dayofweek', FeatureType.NUMERICAL, FeatureSource.DERIVED, True, 'Day of week'),
    'is_weekend': FeatureSchema('is_weekend', FeatureType.BINARY, FeatureSource.DERIVED, True, 'Weekend flag'),
    
    # Encoded features
    'category_encoded': FeatureSchema('category_encoded', FeatureType.NUMERICAL, FeatureSource.ENCODED, False, 'Encoded category'),
    'shop_location_encoded': FeatureSchema('shop_location_encoded', FeatureType.NUMERICAL, FeatureSource.ENCODED, True, 'Encoded location'),
}

# =============================================================================
# TARGET VARIABLES
# =============================================================================

TARGETS = {
    'count_sold': FeatureSchema('count_sold', FeatureType.NUMERICAL, FeatureSource.RAW, True, 'Units sold (Sales target)'),
    'count_sold_log': FeatureSchema('count_sold_log', FeatureType.NUMERICAL, FeatureSource.DERIVED, True, 'Log units sold'),
    'is_negative_review': FeatureSchema('is_negative_review', FeatureType.BINARY, FeatureSource.DERIVED, False, 'Negative review (Risk target)'),
}

# =============================================================================
# FEATURE SETS BY MODEL
# =============================================================================

SALES_MODEL_FEATURES: List[str] = [
    # Price
    'price_log', 'price_bucket', 'discount_pct', 'has_discount',
    # Stock
    'stock_log', 'has_stock', 'low_stock', 'is_preorder',
    # Shop
    'shop_tier', 'uses_topads',
    # Categoricals
    'category_encoded', 'shop_location_encoded',
    # Rating (not leakage for sales)
    'rating_average',
]

RISK_MODEL_FEATURES: List[str] = [
    # Price
    'price_log', 'discount_pct', 'has_discount',
    # Shop
    'shop_tier', 'uses_topads',
    # Categoricals
    'category_encoded', 'shop_location_encoded',
    # Text
    'message_length', 'word_count', 'has_response',
    # Temporal
    'review_hour', 'review_dayofweek', 'is_weekend',
]

# Features that must NOT be used (leakage risk)
LEAKAGE_FEATURES: List[str] = [
    'rating_average',      # For risk model - derived from target
    'count_sold',          # For sales model cross-validation splits
    'is_negative_review',  # Target itself
]


def get_feature_schema(name: str) -> Optional[FeatureSchema]:
    """
    Get schema for a feature by name.
    
    Args:
        name: Feature name
        
    Returns:
        FeatureSchema or None if not found
    """
    if name in RAW_COLUMNS:
        return RAW_COLUMNS[name]
    if name in DERIVED_FEATURES:
        return DERIVED_FEATURES[name]
    if name in TARGETS:
        return TARGETS[name]
    return None


def get_features_by_type(dtype: FeatureType) -> List[str]:
    """
    Get all features of a specific type.
    
    Args:
        dtype: Feature type to filter by
        
    Returns:
        List of feature names
    """
    all_features = {**RAW_COLUMNS, **DERIVED_FEATURES}
    return [name for name, schema in all_features.items() if schema.dtype == dtype]


def validate_feature_set(features: List[str], model: str = 'sales') -> Dict[str, bool]:
    """
    Validate that features don't cause data leakage.
    
    Args:
        features: List of feature names
        model: 'sales' or 'risk'
        
    Returns:
        Dict mapping feature names to validity (True = safe)
    """
    leakage_for_model = set(LEAKAGE_FEATURES)
    
    # rating_average is OK for sales model
    if model == 'sales':
        leakage_for_model.discard('rating_average')
    
    return {f: f not in leakage_for_model for f in features}
