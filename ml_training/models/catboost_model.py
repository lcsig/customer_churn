"""
CatBoost Model Implementation
"""

from catboost import CatBoostClassifier
from models.base_model import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost classifier implementation.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=True):
        """
        Initialize CatBoost model.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: Whether to use balanced class weights (default: True)
        """
        super().__init__(params, random_state, use_class_weights)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the CatBoost model with parameters"""
        default_params = {
            "iterations": 50,
            "learning_rate": 0.01,
            "depth": 3,
            "random_state": self.random_state,
            "loss_function": "Logloss",
            "verbose": False,  # Suppress training output
            "allow_writing_files": False,  # Don't write files during training,
            "auto_class_weights": "Balanced",
        }

        # Use all CPU cores by default (unless user overrides)
        # CatBoost uses `thread_count` for CPU parallelism.
        if "thread_count" not in self.params:
            default_params["thread_count"] = -1

        default_params.update(self.params)
        self.model = CatBoostClassifier(**default_params)

    def fit(self, X, y):
        """
        Train the CatBoost model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Make predictions using CatBoost model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities from CatBoost model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of prediction probabilities
        """
        return self.model.predict_proba(X)
