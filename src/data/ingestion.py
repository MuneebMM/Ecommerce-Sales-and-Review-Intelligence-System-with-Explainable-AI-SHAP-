"""
Data Ingestion Module

Provides utilities for loading and reading data from various sources.
Handles the complex nested structure of Tokopedia reviews dataset where
review columns contain stringified Python lists.

Key Features:
    - Safe parsing of stringified lists using ast.literal_eval
    - Product-level loading (raw CSV structure)
    - Review-level loading (exploded one-row-per-review)
    - Batch loading for large files
    - Basic validation during load
"""

import ast
import logging
from pathlib import Path
from typing import Optional, Union, List, Iterator

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Columns that contain stringified lists and need parsing
LIST_COLUMNS = [
    'review_rating',
    'message', 
    'review_time',
    'review_timestamp',
    'review_response',
    'review_like',
    'bad_rating_reason',
    'variant_name'
]

# Core columns that must exist for pipeline to work
REQUIRED_COLUMNS = [
    'product_id',
    'count_sold', 
    'price',
    'review_rating',
    'message'
]


def _safe_parse_list(value: str) -> List:
    """
    Safely parse a stringified Python list.
    
    Args:
        value: String representation of a list, e.g., "['a', 'b']"
        
    Returns:
        Parsed list or empty list if parsing fails
        
    Examples:
        >>> _safe_parse_list("['hello', 'world']")
        ['hello', 'world']
        >>> _safe_parse_list("[1, 2, 3]")
        [1, 2, 3]
        >>> _safe_parse_list("malformed")
        []
    """
    if pd.isna(value) or value == '' or value == '[]':
        return []
    
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
        return [parsed]  # Wrap single values in list
    except (ValueError, SyntaxError, TypeError) as e:
        logger.warning(f"Failed to parse list value: {str(value)[:50]}... Error: {e}")
        return []


def _parse_list_columns(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Parse all list columns in the dataframe from strings to actual Python lists.
    
    Args:
        df: DataFrame with stringified list columns
        columns: List of column names to parse. Defaults to LIST_COLUMNS.
        
    Returns:
        DataFrame with parsed list columns
    """
    columns = columns or LIST_COLUMNS
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            logger.info(f"Parsing list column: {col}")
            df[col] = df[col].apply(_safe_parse_list)
    
    return df


def load_tokopedia_data(
    filepath: Union[str, Path],
    parse_lists: bool = True,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load the Tokopedia products with reviews dataset.
    
    This loads the data at PRODUCT LEVEL (one row per product).
    Review columns will contain lists of reviews.
    
    Args:
        filepath: Path to the CSV file
        parse_lists: Whether to parse stringified lists into Python lists
        nrows: Number of rows to load (for testing/sampling)
        usecols: Specific columns to load
        validate: Whether to run basic validation checks
        
    Returns:
        DataFrame with product-level data
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}")
    
    # Load CSV
    df = pd.read_csv(
        filepath,
        nrows=nrows,
        usecols=usecols,
        low_memory=False
    )
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Validate required columns
    if validate:
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse list columns
    if parse_lists:
        cols_to_parse = [c for c in LIST_COLUMNS if c in df.columns]
        df = _parse_list_columns(df, cols_to_parse)
    
    return df


def load_tokopedia_data_chunked(
    filepath: Union[str, Path],
    chunksize: int = 10000,
    parse_lists: bool = True
) -> Iterator[pd.DataFrame]:
    """
    Load the dataset in chunks for memory-efficient processing.
    
    Args:
        filepath: Path to the CSV file
        chunksize: Number of rows per chunk
        parse_lists: Whether to parse stringified lists
        
    Yields:
        DataFrame chunks
    """
    filepath = Path(filepath)
    
    logger.info(f"Loading dataset in chunks of {chunksize:,} from {filepath}")
    
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, low_memory=False)):
        logger.info(f"Processing chunk {i+1}")
        
        if parse_lists:
            cols_to_parse = [c for c in LIST_COLUMNS if c in chunk.columns]
            chunk = _parse_list_columns(chunk, cols_to_parse)
        
        yield chunk


def explode_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform product-level data to review-level data.
    
    Takes a DataFrame where each row is a product with list columns
    (review_rating, message, etc.) and "explodes" it so each row
    represents a single review.
    
    Args:
        df: Product-level DataFrame with parsed list columns
        
    Returns:
        Review-level DataFrame with one row per review
        
    Example:
        Input (1 row):
            product_id=123, review_rating=[5,3], message=['Good','Bad']
        
        Output (2 rows):
            product_id=123, review_rating=5, message='Good'
            product_id=123, review_rating=3, message='Bad'
    """
    df = df.copy()
    
    # Columns to explode (must be lists)
    explode_cols = [c for c in LIST_COLUMNS if c in df.columns]
    
    if not explode_cols:
        logger.warning("No list columns found to explode")
        return df
    
    # Validate list alignment before explosion
    logger.info("Validating list alignment across columns...")
    
    def get_list_length(val):
        if isinstance(val, list):
            return len(val)
        return 0
    
    # Check if primary review columns have same length
    primary_cols = ['review_rating', 'message']
    primary_cols = [c for c in primary_cols if c in df.columns]
    
    if len(primary_cols) >= 2:
        len_col1 = df[primary_cols[0]].apply(get_list_length)
        len_col2 = df[primary_cols[1]].apply(get_list_length)
        mismatched = (len_col1 != len_col2).sum()
        
        if mismatched > 0:
            logger.warning(f"Found {mismatched:,} rows with misaligned list lengths. These will be handled during explosion.")
    
    # Perform explosion
    logger.info(f"Exploding {len(df):,} product rows into review rows...")
    
    # Explode all list columns together
    # First, ensure all list columns have same length by padding shorter ones
    for col in explode_cols:
        if col not in df.columns:
            continue
        # Ensure column contains lists
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # Find max list length per row for padding
    if 'review_rating' in df.columns:
        df['_list_len'] = df['review_rating'].apply(len)
    elif 'message' in df.columns:
        df['_list_len'] = df['message'].apply(len)
    else:
        df['_list_len'] = 0
    
    # Pad shorter lists with None
    for col in explode_cols:
        if col in df.columns:
            df[col] = df.apply(
                lambda row: row[col] + [None] * (row['_list_len'] - len(row[col])) 
                if len(row[col]) < row['_list_len'] else row[col],
                axis=1
            )
    
    # Drop helper column
    df = df.drop(columns=['_list_len'])
    
    # Explode
    df_exploded = df.explode(explode_cols, ignore_index=True)
    
    # Remove rows where review_rating is None (padding artifacts)
    if 'review_rating' in df_exploded.columns:
        original_len = len(df_exploded)
        df_exploded = df_exploded[df_exploded['review_rating'].notna()]
        removed = original_len - len(df_exploded)
        if removed > 0:
            logger.info(f"Removed {removed:,} padded/empty review rows")
    
    logger.info(f"Explosion complete: {len(df_exploded):,} review rows created")
    
    return df_exploded


def load_reviews(
    filepath: Union[str, Path],
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to load data directly at review level.
    
    Combines load_tokopedia_data() and explode_reviews() into one call.
    
    Args:
        filepath: Path to the CSV file
        nrows: Number of product rows to load before explosion
        
    Returns:
        Review-level DataFrame
    """
    df = load_tokopedia_data(filepath, parse_lists=True, nrows=nrows)
    return explode_reviews(df)


class TokopediaDataLoader:
    """
    Main class for data ingestion operations.
    
    Provides a unified interface for loading Tokopedia data at both
    product level and review level.
    
    Attributes:
        filepath: Path to the dataset
        _product_df: Cached product-level DataFrame
        _review_df: Cached review-level DataFrame
    """
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            filepath: Path to tokopedia_products_with_review.csv
        """
        self.filepath = Path(filepath)
        self._product_df: Optional[pd.DataFrame] = None
        self._review_df: Optional[pd.DataFrame] = None
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")
    
    def load_products(
        self, 
        nrows: Optional[int] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load product-level data (one row per product).
        
        Args:
            nrows: Limit number of rows
            force_reload: Force reload even if cached
            
        Returns:
            Product-level DataFrame
        """
        if self._product_df is None or force_reload or nrows:
            self._product_df = load_tokopedia_data(
                self.filepath, 
                parse_lists=True,
                nrows=nrows
            )
        return self._product_df
    
    def load_reviews(
        self,
        nrows: Optional[int] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load review-level data (one row per review).
        
        Args:
            nrows: Limit number of product rows before explosion
            force_reload: Force reload even if cached
            
        Returns:
            Review-level DataFrame
        """
        if self._review_df is None or force_reload or nrows:
            products = self.load_products(nrows=nrows, force_reload=force_reload)
            self._review_df = explode_reviews(products)
        return self._review_df
    
    def get_stats(self) -> dict:
        """
        Get basic statistics about the loaded data.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'file_path': str(self.filepath),
            'file_size_mb': self.filepath.stat().st_size / (1024 * 1024)
        }
        
        if self._product_df is not None:
            stats['product_count'] = len(self._product_df)
            stats['product_columns'] = list(self._product_df.columns)
        
        if self._review_df is not None:
            stats['review_count'] = len(self._review_df)
        
        return stats
    
    def clear_cache(self):
        """Clear cached DataFrames to free memory."""
        self._product_df = None
        self._review_df = None
        logger.info("Cache cleared")


# Module-level convenience functions
def get_default_loader() -> TokopediaDataLoader:
    """
    Get a data loader configured with the default dataset path.
    
    Returns:
        TokopediaDataLoader instance
    """
    default_path = Path(__file__).parent.parent.parent / 'tokopedia_products_with_review.csv'
    return TokopediaDataLoader(default_path)
