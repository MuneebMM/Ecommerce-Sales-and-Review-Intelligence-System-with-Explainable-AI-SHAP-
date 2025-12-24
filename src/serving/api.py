"""
API Module

FastAPI-based REST API for the Review Intelligence System.
Provides endpoints for sales prediction and review risk assessment.
"""

import logging
from typing import Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    SalesPredictionRequest,
    SalesPredictionResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from src.serving.inference import InferenceEngine

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Review Intelligence System API",
    description="ML-powered predictions for sales volume and review risk with SHAP explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine (models loaded on startup)
inference_engine: InferenceEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global inference_engine
    
    logger.info("Starting Review Intelligence System API...")
    
    # Initialize engine (models will be loaded if paths exist)
    inference_engine = InferenceEngine(
        sales_model_path=Path("models/sales_predictor/model.pkl"),
        risk_model_path=Path("models/review_risk_predictor/model.pkl")
    )
    
    logger.info(f"Models loaded: {inference_engine.models_loaded}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Review Intelligence System API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=inference_engine.models_loaded if inference_engine else {},
        version="1.0.0"
    )


@app.post("/v1/predict/sales", response_model=SalesPredictionResponse)
async def predict_sales(request: SalesPredictionRequest):
    """
    Predict sales volume for a product.
    
    Returns predicted units sold with confidence interval and SHAP explanation.
    """
    if inference_engine is None or not inference_engine.models_loaded.get('sales'):
        raise HTTPException(status_code=503, detail="Sales model not available")
    
    try:
        # Prepare features from request
        features = {
            'price': request.price,
            'stock': request.stock,
            'category': request.category or 'Unknown',
            'shop_location': request.shop_location or 'Unknown',
            'gold_merchant': request.gold_merchant,
            'is_official': request.is_official,
            'rating_average': request.rating_average or 4.5
        }
        
        # Get prediction
        result = inference_engine.predict_sales(
            features,
            include_explanation=request.include_explanation
        )
        
        return SalesPredictionResponse(
            product_id=request.product_id,
            predicted_sales=result['predicted_sales'],
            confidence_interval=result['confidence_interval'],
            explanation=result.get('explanation'),
            model_version="v1.0"
        )
        
    except Exception as e:
        logger.error(f"Sales prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest):
    """
    Predict negative review risk for a product.
    
    Returns risk probability, level, and SHAP explanation.
    """
    if inference_engine is None or not inference_engine.models_loaded.get('risk'):
        raise HTTPException(status_code=503, detail="Risk model not available")
    
    try:
        # Prepare features from request
        features = {
            'price': request.price,
            'message_length': request.message_length,
            'word_count': request.word_count,
            'category': request.category or 'Unknown',
            'gold_merchant': request.gold_merchant,
            'is_official': request.is_official,
            'has_response': request.has_response
        }
        
        # Get prediction
        result = inference_engine.predict_risk(
            features,
            include_explanation=request.include_explanation
        )
        
        return RiskPredictionResponse(
            product_id=request.product_id,
            risk_probability=result['risk_probability'],
            risk_level=result['risk_level'],
            is_high_risk=result['is_high_risk'],
            explanation=result.get('explanation'),
            model_version="v1.0"
        )
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/model/info/{model_type}", response_model=ModelInfoResponse)
async def get_model_info(model_type: str):
    """Get information about a deployed model."""
    if model_type not in ['sales', 'risk']:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'sales' or 'risk'")
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if model_type == 'sales' and inference_engine.sales_model:
        model = inference_engine.sales_model
    elif model_type == 'risk' and inference_engine.risk_model:
        model = inference_engine.risk_model
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not loaded")
    
    return ModelInfoResponse(
        model_type=model_type,
        version="v1.0",
        features=model.feature_names,
        metrics={}  # Would be populated from model registry
    )


def create_app() -> FastAPI:
    """Factory function to create FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
