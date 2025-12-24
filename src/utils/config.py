"""
Configuration Module

Centralized configuration management using YAML files and environment variables.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the application.
    
    Loads configuration from YAML files with environment variable overrides.
    
    Example:
        config = Config.from_yaml('configs/config.yaml')
        db_url = config.get('database.url')
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return cls({})
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded configuration from {path}")
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Dot-separated key path (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Check environment variable first
        env_key = key.upper().replace('.', '_')
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        
        # Navigate nested config
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: str = 'configs/config.yaml') -> Config:
    """Get or create global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = Config.from_yaml(config_path)
    return _global_config


def get_model_config() -> Dict:
    """Get model-specific configuration."""
    return get_config().get_section('model')


def get_data_config() -> Dict:
    """Get data-specific configuration."""
    return get_config().get_section('data')
