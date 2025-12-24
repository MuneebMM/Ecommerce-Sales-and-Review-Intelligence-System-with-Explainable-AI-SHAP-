"""
Serving Module

REST API layer for real-time predictions and explanations.
"""

from src.serving.api import app, create_app
from src.serving.schemas import (
    SalesPredictionRequest,
    SalesPredictionResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from src.serving.inference import (
    InferenceEngine,
    predict_with_explanation,
    batch_predict
)

__all__ = [
    'app',
    'create_app',
    'SalesPredictionRequest',
    'SalesPredictionResponse',
    'RiskPredictionRequest',
    'RiskPredictionResponse',
    'HealthResponse',
    'ModelInfoResponse',
    'InferenceEngine',
    'predict_with_explanation',
    'batch_predict',
]
