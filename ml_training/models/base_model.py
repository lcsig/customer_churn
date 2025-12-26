"""
Base Model Abstract Class
All models must inherit from this class
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    All models must implement fit, predict, and predict_proba methods.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=True):
        """
        Initialize the model.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: Whether to use balanced class weights (default: True)
        """
        self.params = params if params is not None else {}
        self.random_state = random_state
        self.use_class_weights = use_class_weights
        self.model = None

        # Add random state to params if not present
        if "random_state" not in self.params:
            self.params["random_state"] = random_state

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on the provided data.

        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Target vector (Series or numpy array)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on the provided data.

        Args:
            X: Feature matrix (DataFrame or numpy array)

        Returns:
            numpy array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Feature matrix (DataFrame or numpy array)

        Returns:
            numpy array of prediction probabilities
        """
        pass

    def get_feature_importance(self):
        """
        Get feature importance from the model.

        Returns:
            numpy array of feature importances, or None if not available
        """
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def get_params(self):
        """
        Get the current model parameters.

        Returns:
            Dictionary of model parameters
        """
        if self.model is not None:
            return self.model.get_params()
        return self.params
