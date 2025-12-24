"""
Feature Store Module

Manages feature persistence and retrieval for training and serving.
Provides a simple file-based implementation suitable for development.
Can be extended to use Feast, Redis, or cloud feature stores.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Simple file-based feature store for the Review Intelligence System.
    
    Features are stored as Parquet files with metadata for versioning.
    
    Example:
        store = FeatureStore('data/features')
        store.save_features(df, 'training_features', version='v1')
        features = store.load_features('training_features')
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize feature store.
        
        Args:
            base_path: Directory to store feature files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.base_path / 'metadata.json'
        self._metadata = self._load_metadata()
    
    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: Optional[str] = None,
        description: str = ""
    ) -> str:
        """
        Save features to the store.
        
        Args:
            df: DataFrame with features
            name: Feature set name (e.g., 'training_features')
            version: Version string. Auto-generated if not provided.
            description: Human-readable description
            
        Returns:
            Version string of saved features
        """
        version = version or datetime.now().strftime('v%Y%m%d_%H%M%S')
        
        # Create versioned directory
        feature_dir = self.base_path / name / version
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        file_path = feature_dir / 'features.parquet'
        df.to_parquet(file_path, index=False)
        
        # Update metadata
        if name not in self._metadata:
            self._metadata[name] = {'versions': {}, 'latest': None}
        
        self._metadata[name]['versions'][version] = {
            'path': str(file_path),
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
            'description': description
        }
        self._metadata[name]['latest'] = version
        
        self._save_metadata()
        
        logger.info(f"Saved features '{name}' version {version}: {len(df):,} rows, {len(df.columns)} columns")
        
        return version
    
    def load_features(
        self,
        name: str,
        version: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load features from the store.
        
        Args:
            name: Feature set name
            version: Version to load. Uses latest if not specified.
            columns: Specific columns to load
            
        Returns:
            DataFrame with features
            
        Raises:
            ValueError: If feature set or version not found
        """
        if name not in self._metadata:
            raise ValueError(f"Feature set '{name}' not found")
        
        version = version or self._metadata[name]['latest']
        
        if version not in self._metadata[name]['versions']:
            raise ValueError(f"Version '{version}' not found for '{name}'")
        
        file_path = self._metadata[name]['versions'][version]['path']
        
        logger.info(f"Loading features '{name}' version {version}")
        
        if columns:
            return pd.read_parquet(file_path, columns=columns)
        return pd.read_parquet(file_path)
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a feature set."""
        if name in self._metadata:
            return self._metadata[name]['latest']
        return None
    
    def list_feature_sets(self) -> List[str]:
        """List all available feature sets."""
        return list(self._metadata.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a feature set."""
        if name in self._metadata:
            return list(self._metadata[name]['versions'].keys())
        return []
    
    def get_feature_info(self, name: str, version: Optional[str] = None) -> Dict:
        """Get metadata about a feature set."""
        if name not in self._metadata:
            return {}
        
        version = version or self._metadata[name]['latest']
        return self._metadata[name]['versions'].get(version, {})
    
    def delete_version(self, name: str, version: str):
        """Delete a specific version of features."""
        if name in self._metadata and version in self._metadata[name]['versions']:
            file_path = Path(self._metadata[name]['versions'][version]['path'])
            if file_path.exists():
                file_path.unlink()
            
            del self._metadata[name]['versions'][version]
            
            # Update latest if needed
            if self._metadata[name]['latest'] == version:
                versions = list(self._metadata[name]['versions'].keys())
                self._metadata[name]['latest'] = versions[-1] if versions else None
            
            self._save_metadata()
            logger.info(f"Deleted features '{name}' version {version}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)


class OnlineFeatureStore:
    """
    In-memory feature store for low-latency serving.
    
    Caches features for quick retrieval during inference.
    In production, this would be backed by Redis or similar.
    """
    
    def __init__(self):
        """Initialize online store."""
        self._cache: Dict[str, Dict] = {}
        self._ttl: Dict[str, datetime] = {}
    
    def set(self, key: str, features: Dict, ttl_seconds: int = 3600):
        """
        Cache features for a product/review.
        
        Args:
            key: Unique identifier (e.g., product_id)
            features: Feature dictionary
            ttl_seconds: Time-to-live in seconds
        """
        self._cache[key] = features
        self._ttl[key] = datetime.now()
    
    def get(self, key: str) -> Optional[Dict]:
        """
        Retrieve cached features.
        
        Args:
            key: Unique identifier
            
        Returns:
            Feature dictionary or None if not found/expired
        """
        if key in self._cache:
            return self._cache[key]
        return None
    
    def delete(self, key: str):
        """Remove features from cache."""
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
    
    def clear(self):
        """Clear all cached features."""
        self._cache.clear()
        self._ttl.clear()
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)


def get_training_features(
    store: FeatureStore,
    feature_set: str,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Retrieve features for model training.
    
    Args:
        store: FeatureStore instance
        feature_set: Name of feature set
        feature_names: List of feature columns to retrieve
        
    Returns:
        DataFrame with requested features
    """
    df = store.load_features(feature_set, columns=feature_names)
    return df


def get_serving_features(
    online_store: OnlineFeatureStore,
    product_id: str
) -> Optional[Dict]:
    """
    Retrieve features for real-time inference.
    
    Args:
        online_store: OnlineFeatureStore instance
        product_id: Product identifier
        
    Returns:
        Feature dictionary or None
    """
    return online_store.get(str(product_id))


def materialize_features(
    df: pd.DataFrame,
    store: FeatureStore,
    name: str,
    description: str = ""
) -> str:
    """
    Compute and store feature values.
    
    Convenience function to save processed features.
    
    Args:
        df: DataFrame with computed features
        store: FeatureStore instance
        name: Feature set name
        description: Description of this version
        
    Returns:
        Version string
    """
    return store.save_features(df, name, description=description)
