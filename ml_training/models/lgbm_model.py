"""
LightGBM Model Implementation
"""

from lightgbm import LGBMClassifier

from ml_training.models.base_model import BaseModel


class LGBMModel(BaseModel):
    """
    LightGBM classifier implementation.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=True):
        """
        Initialize LightGBM model.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: Whether to use balanced class weights (default: True)
        """
        super().__init__(params, random_state, use_class_weights)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LightGBM model with parameters"""
        default_params = {
            "n_estimators": 50,
            "learning_rate": 0.01,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 10,
            "random_state": self.random_state,
            "verbose": -1,  # Suppress output
        }

        # Use all CPU cores by default (unless user overrides)
        # LightGBM supports `n_jobs` (and aliases like `num_threads`).
        if "n_jobs" not in self.params and "num_threads" not in self.params:
            default_params["n_jobs"] = -1

        # Add class weight if enabled
        if self.use_class_weights and "class_weight" not in self.params:
            default_params["class_weight"] = "balanced"

        default_params.update(self.params)
        self.model = LGBMClassifier(**default_params)

    def fit(self, X, y):
        """
        Train the LightGBM model.

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
        Make predictions using LightGBM model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities from LightGBM model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of prediction probabilities
        """
        return self.model.predict_proba(X)
