"""
ML Training Package for Customer Churn Prediction

This package provides modular classes for:
- Feature engineering from customer event logs (features package)
- Model training with multiple algorithms (models package)
- Time-based validation with MLFlow tracking (mlflow_trainer)
- Production-ready feature store pattern (feature_store)
"""

from ml_training.features import BaseFeatureGenerator, FeatureSet1, FeatureSet2
from ml_training.mlflow_trainer import MLFlowTrainer
from ml_training.models import (
    BaseModel,
    CatBoostModel,
    LGBMModel,
    LinearSVMModel,
    RandomForestModel,
    XGBoostModel,
)

__all__ = [
    # Features
    "BaseFeatureGenerator",
    "FeatureSet1",
    "FeatureSet2",
    # Models
    "BaseModel",
    "LGBMModel",
    "RandomForestModel",
    "XGBoostModel",
    "CatBoostModel",
    "LinearSVMModel",
    # Trainer
    "MLFlowTrainer",
]
__version__ = "1.0.0"
