"""
Feature Store Module

Central hub for feature management ensuring consistency between 
training and serving while preventing training-serving skew.

Main Classes:
    FeatureEngineer: Feature transformation pipeline
    FeatureStore: Offline feature persistence
    OnlineFeatureStore: Low-latency serving cache
    
Quick Start:
    from src.features import FeatureEngineer, FeatureStore
    
    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.fit_transform(data, target='sales')
    
    # Store for later use
    store = FeatureStore('data/features')
    store.save_features(df, 'training_set')
"""

from src.features.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
    create_product_features,
    create_review_features,
    create_temporal_features
)

from src.features.feature_definitions import (
    FeatureSchema,
    FeatureType,
    FeatureSource,
    SALES_MODEL_FEATURES,
    RISK_MODEL_FEATURES,
    LEAKAGE_FEATURES,
    get_feature_schema,
    get_features_by_type,
    validate_feature_set
)

from src.features.feature_store import (
    FeatureStore,
    OnlineFeatureStore,
    get_training_features,
    get_serving_features,
    materialize_features
)

__all__ = [
    # Engineering
    'FeatureEngineer',
    'FeatureConfig',
    'create_product_features',
    'create_review_features',
    'create_temporal_features',
    
    # Definitions
    'FeatureSchema',
    'FeatureType',
    'FeatureSource',
    'SALES_MODEL_FEATURES',
    'RISK_MODEL_FEATURES',
    'LEAKAGE_FEATURES',
    'get_feature_schema',
    'get_features_by_type',
    'validate_feature_set',
    
    # Store
    'FeatureStore',
    'OnlineFeatureStore',
    'get_training_features',
    'get_serving_features',
    'materialize_features',
]
