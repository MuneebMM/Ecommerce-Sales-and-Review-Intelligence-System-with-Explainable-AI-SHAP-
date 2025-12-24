"""
Model Registry Module

Single source of truth for production models with versioning and lineage.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Metadata for a model version."""
    model_id: str
    version: str
    model_type: str  # 'sales' or 'risk'
    stage: str
    created_at: str
    metrics: Dict[str, float]
    artifact_path: str
    description: str = ""
    features: List[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Model registry for versioning and stage management.
    
    Example:
        registry = ModelRegistry('models/')
        registry.register_model('sales', 'v1.0', model_path, metrics)
        prod_model = registry.get_production_model('sales')
    """
    
    def __init__(self, base_path: str):
        """
        Initialize registry.
        
        Args:
            base_path: Directory for registry data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.base_path / 'registry.json'
        self._registry = self._load_registry()
    
    def register_model(
        self,
        model_type: str,
        version: str,
        artifact_path: str,
        metrics: Dict[str, float],
        description: str = "",
        features: List[str] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_type: 'sales' or 'risk'
            version: Version string (e.g., 'v1.0')
            artifact_path: Path to saved model file
            metrics: Evaluation metrics
            description: Optional description
            features: List of feature names used
            
        Returns:
            ModelVersion object
        """
        model_id = f"{model_type}_{version}"
        
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            stage=ModelStage.DEVELOPMENT.value,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            artifact_path=str(artifact_path),
            description=description,
            features=features or []
        )
        
        if model_type not in self._registry:
            self._registry[model_type] = {'versions': {}, 'production': None}
        
        self._registry[model_type]['versions'][version] = model_version.to_dict()
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        
        return model_version
    
    def promote_model(self, model_type: str, version: str, stage: ModelStage):
        """
        Move model to a new stage.
        
        Args:
            model_type: Model type
            version: Version to promote
            stage: Target stage
        """
        if model_type not in self._registry:
            raise ValueError(f"Model type '{model_type}' not found")
        
        if version not in self._registry[model_type]['versions']:
            raise ValueError(f"Version '{version}' not found")
        
        self._registry[model_type]['versions'][version]['stage'] = stage.value
        
        # Update production pointer
        if stage == ModelStage.PRODUCTION:
            # Demote current production
            current_prod = self._registry[model_type].get('production')
            if current_prod and current_prod in self._registry[model_type]['versions']:
                self._registry[model_type]['versions'][current_prod]['stage'] = ModelStage.STAGING.value
            
            self._registry[model_type]['production'] = version
        
        self._save_registry()
        logger.info(f"Promoted {model_type}/{version} to {stage.value}")
    
    def get_production_model(self, model_type: str) -> Optional[ModelVersion]:
        """Get the current production model."""
        if model_type not in self._registry:
            return None
        
        prod_version = self._registry[model_type].get('production')
        if not prod_version:
            return None
        
        data = self._registry[model_type]['versions'].get(prod_version)
        if data:
            return ModelVersion(**data)
        return None
    
    def get_model_version(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if model_type not in self._registry:
            return None
        
        data = self._registry[model_type]['versions'].get(version)
        if data:
            return ModelVersion(**data)
        return None
    
    def list_versions(self, model_type: str) -> List[str]:
        """List all versions of a model type."""
        if model_type not in self._registry:
            return []
        return list(self._registry[model_type]['versions'].keys())
    
    def get_model_lineage(self, model_type: str, version: str) -> Dict:
        """Get lineage information for a model."""
        model = self.get_model_version(model_type, version)
        if not model:
            return {}
        
        return {
            'model_id': model.model_id,
            'version': model.version,
            'created_at': model.created_at,
            'features': model.features,
            'metrics': model.metrics
        }
    
    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        if self._registry_file.exists():
            with open(self._registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self._registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2)


def register_model(model_type: str, version: str, path: str, metrics: Dict) -> ModelVersion:
    """Quick registration helper."""
    registry = ModelRegistry('models/')
    return registry.register_model(model_type, version, path, metrics)


def get_production_model(model_type: str) -> Optional[ModelVersion]:
    """Quick production model retrieval."""
    registry = ModelRegistry('models/')
    return registry.get_production_model(model_type)
