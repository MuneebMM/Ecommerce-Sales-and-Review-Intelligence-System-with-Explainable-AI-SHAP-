"""
Models Module

ML Pipeline for training and serving predictive models.

Main Classes:
    SalesPredictor: Sales volume regression model
    ReviewRiskPredictor: Negative review classification model
    ModelTrainer: Training orchestrator
    ModelRegistry: Model versioning and deployment
"""

from src.models.sales_predictor import (
    SalesPredictor,
    train_sales_model,
    predict_sales
)

from src.models.review_risk_predictor import (
    ReviewRiskPredictor,
    train_risk_model,
    predict_risk
)

from src.models.trainer import (
    ModelTrainer,
    TrainingConfig,
    ExperimentResult
)

from src.models.evaluator import (
    ModelEvaluator,
    EvaluationReport,
    evaluate_regression,
    evaluate_classification
)

from src.models.registry import (
    ModelRegistry,
    ModelVersion,
    ModelStage,
    register_model,
    get_production_model
)

__all__ = [
    # Sales Model
    'SalesPredictor',
    'train_sales_model',
    'predict_sales',
    
    # Risk Model
    'ReviewRiskPredictor', 
    'train_risk_model',
    'predict_risk',
    
    # Training
    'ModelTrainer',
    'TrainingConfig',
    'ExperimentResult',
    
    # Evaluation
    'ModelEvaluator',
    'EvaluationReport',
    'evaluate_regression',
    'evaluate_classification',
    
    # Registry
    'ModelRegistry',
    'ModelVersion',
    'ModelStage',
    'register_model',
    'get_production_model',
]
