# Review Intelligence System

A production-grade MLOps platform for predicting **product sales volume** and **negative review risk** with SHAP-based explainability.

## 🎯 Overview

This system analyzes Tokopedia product and review data to:

1. **Predict Sales Volume** - Forecast units sold using product, shop, and historical features
2. **Predict Negative Review Risk** - Identify products likely to receive negative reviews

Every prediction includes transparent **SHAP-based explanations** for actionable insights.

## 📁 Project Structure

```
├── data/                     # Data storage
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned data
│   └── features/             # Feature store outputs
├── src/                      # Source code
│   ├── data/                 # Data ingestion & validation
│   ├── features/             # Feature engineering & store
│   ├── models/               # ML training & registry
│   ├── explainability/       # SHAP integration
│   ├── serving/              # API layer
│   ├── monitoring/           # Drift detection
│   └── utils/                # Utilities
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── models/                   # Saved model artifacts
├── docker/                   # Docker configuration
└── logs/                     # Application logs
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m src.serving.api
```

## 📊 Dataset

- **File**: `tokopedia_products_with_review.csv`
- **Size**: ~345MB
- **Columns**: 25 (product info, sales, ratings, reviews)

## 🔧 Technology Stack

- **ML Framework**: LightGBM, XGBoost
- **Explainability**: SHAP
- **API**: FastAPI
- **Monitoring**: Prometheus, Evidently
- **Experiment Tracking**: MLflow

## 📖 Documentation

See `configs/` for detailed configuration options.

## 📄 License

MIT License
