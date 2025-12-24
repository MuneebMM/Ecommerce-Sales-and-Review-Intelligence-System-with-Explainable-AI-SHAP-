"""
Data Preprocessing Module

Handles data cleaning and transformation operations:
- Missing value imputation
- Price/discount normalization
- Text cleaning for review messages
- Target variable derivation
- Feature type conversions
"""

import re
import logging
from typing import Optional, List, Dict, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Main preprocessing pipeline for Tokopedia data.
    
    Applies transformations in a consistent order to ensure
    reproducibility between training and serving.
    
    Example:
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.fit_transform(df)
    """
    
    def __init__(
        self,
        impute_count_sold: bool = True,
        impute_discounted_price: bool = True,
        create_targets: bool = True,
        clean_text: bool = True
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            impute_count_sold: Fill missing count_sold with 0
            impute_discounted_price: Fill missing discounted_price with price
            create_targets: Create derived target columns
            clean_text: Apply text cleaning to review messages
        """
        self.impute_count_sold = impute_count_sold
        self.impute_discounted_price = impute_discounted_price
        self.create_targets = create_targets
        self.clean_text = clean_text
        
        # Statistics learned during fit
        self._fitted = False
        self._global_rating_mean: float = 4.5
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Learn statistics from training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            self
        """
        # Learn global rating average for imputation
        if 'rating_average' in df.columns:
            self._global_rating_mean = df['rating_average'].mean()
        
        self._fitted = True
        logger.info(f"Preprocessor fitted. Global rating mean: {self._global_rating_mean:.2f}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        # Missing value imputation
        if self.impute_count_sold and 'count_sold' in df.columns:
            df = self._impute_count_sold(df)
        
        if self.impute_discounted_price and 'discounted_price' in df.columns:
            df = self._impute_discounted_price(df)
        
        # Impute rating_average
        if 'rating_average' in df.columns:
            df['rating_average'] = df['rating_average'].fillna(self._global_rating_mean)
        
        # Impute shop_location
        if 'shop_location' in df.columns:
            df['shop_location'] = df['shop_location'].fillna('Unknown')
        
        # Text cleaning
        if self.clean_text and 'message' in df.columns:
            df = self._clean_review_text(df)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Create target variables
        if self.create_targets:
            df = self._create_target_variables(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def _impute_count_sold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing count_sold with 0."""
        missing = df['count_sold'].isna().sum()
        if missing > 0:
            logger.info(f"Imputing {missing:,} missing count_sold values with 0")
            df['count_sold'] = df['count_sold'].fillna(0)
        return df
    
    def _impute_discounted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing discounted_price with price."""
        if 'price' not in df.columns:
            return df
        
        missing = df['discounted_price'].isna().sum()
        if missing > 0:
            logger.info(f"Imputing {missing:,} missing discounted_price values with price")
            df['discounted_price'] = df['discounted_price'].fillna(df['price'])
        return df
    
    def _clean_review_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean review message text.
        
        Handles both list format (product-level) and scalar format (review-level).
        """
        def clean_single_text(text: str) -> str:
            if pd.isna(text) or not isinstance(text, str):
                return ""
            
            # Lowercase
            text = text.lower()
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            return text.strip()
        
        def clean_text_list(texts):
            if isinstance(texts, list):
                return [clean_single_text(t) for t in texts]
            return clean_single_text(texts)
        
        df['message_clean'] = df['message'].apply(clean_text_list)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw columns."""
        
        # Discount ratio
        if 'price' in df.columns and 'discounted_price' in df.columns:
            df['discount_ratio'] = (df['price'] - df['discounted_price']) / df['price']
            df['discount_ratio'] = df['discount_ratio'].clip(0, 1)  # Ensure valid range
            df['has_discount'] = df['discount_ratio'] > 0
        
        # Log-transformed sales (for modeling)
        if 'count_sold' in df.columns:
            df['count_sold_log'] = np.log1p(df['count_sold'])
        
        # Shop quality indicators
        if 'gold_merchant' in df.columns and 'is_official' in df.columns:
            df['shop_tier'] = (
                df['is_official'].astype(int) * 2 + 
                df['gold_merchant'].astype(int)
            )
        
        return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for modeling.
        
        For product-level data: Uses rating_average
        For review-level data: Uses review_rating
        """
        # Check if review-level (scalar review_rating) or product-level (list)
        if 'review_rating' in df.columns:
            # Determine if exploded (scalar) or not (list)
            sample_val = df['review_rating'].iloc[0] if len(df) > 0 else None
            
            if not isinstance(sample_val, list):
                # Review-level: create binary target
                ratings = pd.to_numeric(df['review_rating'], errors='coerce')
                df['is_negative_review'] = (ratings < 3).astype(int)
                df['is_positive_review'] = (ratings >= 4).astype(int)
                logger.info("Created review-level target variables")
        
        return df


class TextCleaner:
    """
    Specialized text cleaner for review messages.
    
    Handles NLP-specific preprocessing:
    - PII masking (emails, phone numbers)
    - Empty/short review filtering
    - Language detection flags
    """
    
    # Regex patterns for PII
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+62|62|0)[\s-]?(?:\d[\s-]?){9,12}\b')
    
    def __init__(self, min_length: int = 3, mask_pii: bool = True):
        """
        Initialize text cleaner.
        
        Args:
            min_length: Minimum review length to keep
            mask_pii: Whether to mask PII in text
        """
        self.min_length = min_length
        self.mask_pii = mask_pii
    
    def clean(self, text: str) -> str:
        """
        Clean a single review text.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Mask PII
        if self.mask_pii:
            text = self.EMAIL_PATTERN.sub('[EMAIL]', text)
            text = self.PHONE_PATTERN.sub('[PHONE]', text)
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def is_valid(self, text: str) -> bool:
        """
        Check if review text meets quality criteria.
        
        Args:
            text: Review text to validate
            
        Returns:
            True if text meets criteria
        """
        if pd.isna(text) or not isinstance(text, str):
            return False
        
        cleaned = text.strip()
        return len(cleaned) >= self.min_length
    
    def filter_valid_reviews(self, df: pd.DataFrame, text_col: str = 'message') -> pd.DataFrame:
        """
        Filter DataFrame to keep only valid reviews.
        
        Args:
            df: DataFrame with review text
            text_col: Column containing review text
            
        Returns:
            Filtered DataFrame
        """
        if text_col not in df.columns:
            return df
        
        original_len = len(df)
        mask = df[text_col].apply(self.is_valid)
        df_filtered = df[mask].copy()
        
        removed = original_len - len(df_filtered)
        if removed > 0:
            logger.info(f"Filtered out {removed:,} invalid reviews (empty or too short)")
        
        return df_filtered


# Convenience functions

def clean_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product-related columns.
    
    Args:
        df: Raw product DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    preprocessor = DataPreprocessor(clean_text=False, create_targets=False)
    return preprocessor.fit_transform(df)


def clean_review_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize review text.
    
    Args:
        df: DataFrame with review data (should be exploded)
        
    Returns:
        DataFrame with cleaned text
    """
    cleaner = TextCleaner(min_length=3, mask_pii=True)
    
    if 'message' in df.columns:
        df = df.copy()
        df['message_clean'] = df['message'].apply(cleaner.clean)
        df = cleaner.filter_valid_reviews(df, 'message_clean')
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Apply imputation strategies to missing values.
    
    Args:
        df: DataFrame with missing values
        strategy: Dict mapping column names to strategies.
                  Strategies: 'zero', 'mean', 'median', 'mode', 'drop'
                  
    Returns:
        DataFrame with imputed values
    """
    default_strategies = {
        'count_sold': 'zero',
        'discounted_price': 'price',  # Fill with price column
        'rating_average': 'mean',
        'shop_location': 'mode'
    }
    
    strategy = strategy or default_strategies
    df = df.copy()
    
    for col, method in strategy.items():
        if col not in df.columns:
            continue
        
        if method == 'zero':
            df[col] = df[col].fillna(0)
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mode':
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
        elif method == 'drop':
            df = df.dropna(subset=[col])
        elif method in df.columns:  # Fill from another column
            df[col] = df[col].fillna(df[method])
    
    return df
