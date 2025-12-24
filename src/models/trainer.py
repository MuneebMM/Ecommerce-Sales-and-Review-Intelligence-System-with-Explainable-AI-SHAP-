"""
Model Trainer Module

Orchestrates model training pipeline with cross-validation,
hyperparameter tuning, and experiment tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42
    log_experiments: bool = True


@dataclass
class ExperimentResult:
    """Result of a training experiment."""
    model_type: str
    timestamp: str
    config: Dict
    metrics: Dict
    feature_importance: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type,
            'timestamp': self.timestamp,
            'config': self.config,
            'metrics': self.metrics
        }


class ModelTrainer:
    """
    Main training orchestrator.
    
    Handles training workflow for both Sales and Risk models.
    
    Example:
        trainer = ModelTrainer(config)
        result = trainer.train_sales_model(X, y)
        trainer.save_experiment(result)
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self._experiments: List[ExperimentResult] = []
    
    def train_with_cv(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = None,
        stratified: bool = False
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model with cross-validation.
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Target
            cv_folds: Number of CV folds
            stratified: Use stratified splits (for classification)
            
        Returns:
            Tuple of (best model, CV metrics)
        """
        cv_folds = cv_folds or self.config.cv_folds
        
        if stratified:
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        
        fold_metrics = []
        best_model = None
        best_score = -np.inf
        
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = model_class()
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            fold_metrics.append(metrics)
            
            # Track best model using primary metric
            primary_metric = metrics.get('auc_roc', metrics.get('r2', 0))
            if primary_metric > best_score:
                best_score = primary_metric
                best_model = model
            
            logger.info(f"Fold {fold + 1}/{cv_folds} complete")
        
        # Aggregate metrics
        cv_metrics = {}
        for key in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][key], (int, float)):
                values = [m[key] for m in fold_metrics]
                cv_metrics[f'{key}_mean'] = np.mean(values)
                cv_metrics[f'{key}_std'] = np.std(values)
        
        logger.info(f"CV complete: {cv_metrics}")
        
        return best_model, cv_metrics
    
    def hyperparameter_search(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        n_trials: int = 10
    ) -> Tuple[Dict, float]:
        """
        Simple random search for hyperparameters.
        
        Args:
            model_class: Model class
            X: Features
            y: Target
            param_grid: Dict of parameter lists to sample from
            n_trials: Number of random configurations to try
            
        Returns:
            Tuple of (best params, best score)
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.val_size, random_state=self.config.random_state
        )
        
        best_params = {}
        best_score = -np.inf
        
        logger.info(f"Starting hyperparameter search with {n_trials} trials")
        
        for trial in range(n_trials):
            # Sample random params
            params = {
                key: np.random.choice(values)
                for key, values in param_grid.items()
            }
            
            # Train with these params
            model = model_class(params=params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            score = metrics.get('auc_roc', metrics.get('r2', 0))
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"Trial {trial + 1}: New best score {score:.4f}")
        
        logger.info(f"Best params: {best_params}, Best score: {best_score:.4f}")
        
        return best_params, best_score
    
    def log_experiment(
        self,
        model_type: str,
        config: Dict,
        metrics: Dict,
        feature_importance: pd.DataFrame = None
    ) -> ExperimentResult:
        """
        Log a training experiment.
        
        Args:
            model_type: 'sales' or 'risk'
            config: Training configuration used
            metrics: Evaluation metrics
            feature_importance: Optional feature importance df
            
        Returns:
            ExperimentResult object
        """
        result = ExperimentResult(
            model_type=model_type,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics=metrics,
            feature_importance=feature_importance
        )
        
        self._experiments.append(result)
        logger.info(f"Logged experiment: {model_type} at {result.timestamp}")
        
        return result
    
    def save_experiments(self, path: Path):
        """Save all experiments to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        experiments = [e.to_dict() for e in self._experiments]
        
        with open(path, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
        
        logger.info(f"Saved {len(experiments)} experiments to {path}")
    
    def get_best_experiment(self, model_type: str, metric: str = 'auc_roc') -> Optional[ExperimentResult]:
        """Get best experiment for a model type by metric."""
        relevant = [e for e in self._experiments if e.model_type == model_type]
        
        if not relevant:
            return None
        
        return max(relevant, key=lambda e: e.metrics.get(metric, 0))
