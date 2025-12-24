"""
Review Intelligence System - Setup Configuration

Install in development mode: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="review-intelligence-system",
    version="0.1.0",
    description="Production-grade MLOps platform for sales and review risk prediction with SHAP explainability",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "shap>=0.43.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
)
