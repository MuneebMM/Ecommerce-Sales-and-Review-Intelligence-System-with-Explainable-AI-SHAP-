"""
Feature Engineering Module

Creates features for sales prediction and review risk models.
Features are organized by category and designed to prevent data leakage.

Feature Categories:
    - Product: Price, category, stock features
    - Shop: Merchant tier, location features
    - Review: Rating aggregates, text features
    - Temporal: Time-based patterns
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    include_text_features: bool = True
    include_temporal_features: bool = True
    categorical_encoding: str = 'label'  # 'label' or 'onehot'
    numerical_scaling: bool = False


class FeatureEngineer:
    """
    Main feature transformation pipeline.
    
    Creates features for both Sales Prediction and Review Risk models
    while ensuring no data leakage occurs.
    
    Example:
        engineer = FeatureEngineer()
        X_train = engineer.fit_transform(train_df, target='sales')
        X_test = engineer.transform(test_df)
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature configuration options
        """
        self.config = config or FeatureConfig()
        
        # Encoders learned during fit
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted = False
        
        # Feature lists by model type
        self._sales_features: List[str] = []
        self._risk_features: List[str] = []
    
    def fit(self, df: pd.DataFrame, target: str = 'sales') -> 'FeatureEngineer':
        """
        Learn encoding mappings from training data.
        
        Args:
            df: Training DataFrame
            target: 'sales' or 'risk' to determine feature set
            
        Returns:
            self
        """
        logger.info(f"Fitting feature engineer for {target} model")
        
        # Fit categorical encoders
        categorical_cols = ['category', 'shop_location']
        for col in categorical_cols:
            if col in df.columns:
                self._label_encoders[col] = LabelEncoder()
                # Handle unseen values by adding 'Unknown'
                values = df[col].fillna('Unknown').astype(str)
                self._label_encoders[col].fit(values)
        
        # Fit numerical scalers if enabled
        if self.config.numerical_scaling:
            numerical_cols = ['price', 'stock', 'rating_average']
            for col in numerical_cols:
                if col in df.columns:
                    self._scalers[col] = StandardScaler()
                    valid_values = df[col].dropna().values.reshape(-1, 1)
                    if len(valid_values) > 0:
                        self._scalers[col].fit(valid_values)
        
        self._fitted = True
        logger.info("Feature engineer fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame, target: str = 'sales') -> pd.DataFrame:
        """
        Transform data by creating all features.
        
        Args:
            df: DataFrame to transform
            target: 'sales' or 'risk' to determine feature set
            
        Returns:
            DataFrame with engineered features
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        
        df = df.copy()
        
        # Create features by category
        df = self._create_product_features(df)
        df = self._create_shop_features(df)
        df = self._create_price_features(df)
        
        if self.config.include_temporal_features:
            df = self._create_temporal_features(df)
        
        if self.config.include_text_features and 'message' in df.columns:
            df = self._create_text_features(df)
        
        # Encode categoricals
        df = self._encode_categoricals(df)
        
        # Scale numericals if configured
        if self.config.numerical_scaling:
            df = self._scale_numericals(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target: str = 'sales') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target).transform(df, target)
    
    def get_feature_names(self, target: str = 'sales') -> List[str]:
        """
        Get list of feature names for the specified model.
        
        Args:
            target: 'sales' or 'risk'
            
        Returns:
            List of feature column names
        """
        if target == 'sales':
            return self._get_sales_feature_names()
        else:
            return self._get_risk_feature_names()
    
    def _create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-level features."""
        
        # Stock features
        if 'stock' in df.columns:
            df['stock_log'] = np.log1p(df['stock'])
            df['has_stock'] = (df['stock'] > 0).astype(int)
            df['low_stock'] = (df['stock'] < 10).astype(int)
        
        # Preorder flag
        if 'preorder' in df.columns:
            df['is_preorder'] = df['preorder'].astype(int)
        
        return df
    
    def _create_shop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create shop-level features."""
        
        # Merchant tier (composite feature)
        if 'gold_merchant' in df.columns and 'is_official' in df.columns:
            df['shop_tier'] = (
                df['is_official'].astype(int) * 2 + 
                df['gold_merchant'].astype(int)
            )
        
        # TopAds indicator
        if 'is_topads' in df.columns:
            df['uses_topads'] = df['is_topads'].astype(int)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features."""
        
        if 'price' in df.columns:
            # Log price
            df['price_log'] = np.log1p(df['price'])
            
            # Price buckets
            df['price_bucket'] = pd.cut(
                df['price'],
                bins=[0, 50000, 100000, 500000, 1000000, float('inf')],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(0)
        
        if 'price' in df.columns and 'discounted_price' in df.columns:
            # Proper discount calculation (handle inverted data)
            df['effective_price'] = df[['price', 'discounted_price']].min(axis=1)
            df['original_price'] = df[['price', 'discounted_price']].max(axis=1)
            
            df['discount_amount'] = df['original_price'] - df['effective_price']
            df['discount_pct'] = (df['discount_amount'] / df['original_price']).fillna(0)
            df['discount_pct'] = df['discount_pct'].clip(0, 1)
            df['has_discount'] = (df['discount_pct'] > 0).astype(int)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from review timestamps."""
        
        if 'review_timestamp' in df.columns:
            # Convert timestamp to datetime
            try:
                # Handle both string and numeric timestamps
                if df['review_timestamp'].dtype == 'object':
                    df['review_datetime'] = pd.to_datetime(
                        df['review_timestamp'], 
                        errors='coerce'
                    )
                else:
                    df['review_datetime'] = pd.to_datetime(
                        df['review_timestamp'], 
                        unit='s',
                        errors='coerce'
                    )
                
                # Extract temporal features
                if df['review_datetime'].notna().any():
                    df['review_hour'] = df['review_datetime'].dt.hour.fillna(12)
                    df['review_dayofweek'] = df['review_datetime'].dt.dayofweek.fillna(0)
                    df['review_month'] = df['review_datetime'].dt.month.fillna(6)
                    df['is_weekend'] = df['review_dayofweek'].isin([5, 6]).astype(int)
                    
                    # Time since review (days from most recent)
                    max_date = df['review_datetime'].max()
                    df['days_since_review'] = (max_date - df['review_datetime']).dt.days.fillna(0)
                
            except Exception as e:
                logger.warning(f"Failed to parse timestamps: {e}")
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic text features from review messages."""
        
        def safe_len(x):
            if isinstance(x, str):
                return len(x)
            return 0
        
        def count_exclamations(x):
            if isinstance(x, str):
                return x.count('!')
            return 0
        
        # Message length
        text_col = 'message_clean' if 'message_clean' in df.columns else 'message'
        if text_col in df.columns:
            df['message_length'] = df[text_col].apply(safe_len)
            df['message_length_log'] = np.log1p(df['message_length'])
            
            # Exclamation count (enthusiasm indicator)
            df['exclamation_count'] = df[text_col].apply(count_exclamations)
            
            # Word count
            df['word_count'] = df[text_col].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            
            # Has response from seller
            if 'review_response' in df.columns:
                df['has_response'] = df['review_response'].apply(
                    lambda x: 1 if pd.notna(x) and str(x).strip() not in ['', '[]'] else 0
                )
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        
        for col, encoder in self._label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].fillna('Unknown').astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        return df
    
    def _scale_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply numerical scaling."""
        
        for col, scaler in self._scalers.items():
            if col in df.columns:
                valid_mask = df[col].notna()
                df.loc[valid_mask, f'{col}_scaled'] = scaler.transform(
                    df.loc[valid_mask, col].values.reshape(-1, 1)
                ).flatten()
        
        return df
    
    def _get_sales_feature_names(self) -> List[str]:
        """Get feature names for sales prediction model."""
        return [
            # Price features
            'price_log', 'price_bucket', 'discount_pct', 'has_discount',
            # Stock features
            'stock_log', 'has_stock', 'low_stock', 'is_preorder',
            # Shop features
            'shop_tier', 'uses_topads',
            # Encoded categoricals
            'category_encoded', 'shop_location_encoded',
            # Rating (for sales model, this is OK - not target leakage)
            'rating_average'
        ]
    
    def _get_risk_feature_names(self) -> List[str]:
        """Get feature names for review risk model."""
        return [
            # Price features
            'price_log', 'discount_pct', 'has_discount',
            # Shop features
            'shop_tier', 'uses_topads',
            # Encoded categoricals
            'category_encoded', 'shop_location_encoded',
            # Text features
            'message_length', 'word_count', 'exclamation_count', 'has_response',
            # Temporal
            'review_hour', 'review_dayofweek', 'is_weekend'
        ]


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate product-level features.
    
    Convenience function for quick feature creation.
    
    Args:
        df: Product DataFrame
        
    Returns:
        DataFrame with product features
    """
    engineer = FeatureEngineer()
    return engineer.fit_transform(df, target='sales')


def create_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate review-based features.
    
    Convenience function for review risk model.
    
    Args:
        df: Review DataFrame (exploded)
        
    Returns:
        DataFrame with review features
    """
    engineer = FeatureEngineer()
    return engineer.fit_transform(df, target='risk')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based features.
    
    Args:
        df: DataFrame with review_timestamp column
        
    Returns:
        DataFrame with temporal features
    """
    engineer = FeatureEngineer(FeatureConfig(
        include_text_features=False,
        include_temporal_features=True
    ))
    return engineer.fit_transform(df)
