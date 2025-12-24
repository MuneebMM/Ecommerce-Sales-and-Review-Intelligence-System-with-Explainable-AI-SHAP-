"""
API Schemas Module

Pydantic models for request/response validation in the FastAPI endpoints.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================

class SalesPredictionRequest(BaseModel):
    """Request for sales volume prediction."""
    product_id: str = Field(..., description="Product identifier")
    price: float = Field(..., gt=0, description="Product price")
    stock: int = Field(default=100, ge=0, description="Available stock")
    category: Optional[str] = Field(default=None, description="Product category")
    shop_location: Optional[str] = Field(default=None, description="Shop location")
    gold_merchant: bool = Field(default=False, description="Gold merchant status")
    is_official: bool = Field(default=False, description="Official store status")
    rating_average: Optional[float] = Field(default=None, ge=1, le=5, description="Average rating")
    include_explanation: bool = Field(default=True, description="Include SHAP explanation")

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "SKU123",
                "price": 150000,
                "stock": 50,
                "category": "Electronics",
                "gold_merchant": True,
                "include_explanation": True
            }
        }


class RiskPredictionRequest(BaseModel):
    """Request for review risk prediction."""
    product_id: str = Field(..., description="Product identifier")
    price: float = Field(..., gt=0, description="Product price")
    message_length: int = Field(default=0, ge=0, description="Review message length")
    word_count: int = Field(default=0, ge=0, description="Review word count")
    category: Optional[str] = Field(default=None, description="Product category")
    gold_merchant: bool = Field(default=False, description="Gold merchant status")
    is_official: bool = Field(default=False, description="Official store status")
    has_response: bool = Field(default=False, description="Seller responded")
    include_explanation: bool = Field(default=True, description="Include SHAP explanation")

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "SKU123",
                "price": 150000,
                "message_length": 50,
                "word_count": 10,
                "include_explanation": True
            }
        }


# =============================================================================
# Response Schemas
# =============================================================================

class FeatureContributionResponse(BaseModel):
    """Single feature contribution in explanation."""
    feature_name: str
    shap_value: float
    direction: str
    contribution_pct: float


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""
    base_value: float
    top_drivers: List[FeatureContributionResponse]
    summary: str


class SalesPredictionResponse(BaseModel):
    """Response for sales prediction."""
    product_id: str
    predicted_sales: float
    confidence_interval: Dict[str, float]
    explanation: Optional[ExplanationResponse] = None
    model_version: str = "v1.0"

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "SKU123",
                "predicted_sales": 245,
                "confidence_interval": {"lower": 200, "upper": 290},
                "model_version": "v1.0"
            }
        }


class RiskPredictionResponse(BaseModel):
    """Response for review risk prediction."""
    product_id: str
    risk_probability: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    is_high_risk: bool
    explanation: Optional[ExplanationResponse] = None
    model_version: str = "v1.0"

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "SKU123",
                "risk_probability": 0.15,
                "risk_level": "low",
                "is_high_risk": False,
                "model_version": "v1.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: Dict[str, bool]
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    version: str
    features: List[str]
    metrics: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
