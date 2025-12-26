"""
Random Forest Model Implementation
"""

from imblearn.ensemble import BalancedRandomForestClassifier
from models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    """
    Random Forest classifier that can use either standard or balanced implementation.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=True):
        """
        Initialize Random Forest model.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: If True, uses BalancedRandomForestClassifier for handling class imbalance.
                             If False, uses standard RandomForestClassifier. (default: True)
        """
        super().__init__(params, random_state, use_class_weights)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Random Forest model with parameters"""
        default_params = {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": self.random_state,
        }

        # Use all CPU cores by default (unless user overrides)
        if "n_jobs" not in self.params:
            default_params["n_jobs"] = -1

        if self.use_class_weights:
            # Use BalancedRandomForestClassifier for handling class imbalance
            default_params["sampling_strategy"] = "auto"
            default_params["replacement"] = False
            default_params.update(self.params)
            self.model = BalancedRandomForestClassifier(**default_params)
        else:
            # Use standard RandomForestClassifier
            default_params.update(self.params)
            self.model = RandomForestClassifier(**default_params)

    def fit(self, X, y):
        """
        Train the Random Forest model.

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
        Make predictions using Random Forest model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities from Random Forest model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of prediction probabilities
        """
        return self.model.predict_proba(X)
