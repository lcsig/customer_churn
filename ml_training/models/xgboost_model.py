"""
XGBoost Model Implementation
"""

from models.base_model import BaseModel
from xgboost import XGBClassifier


class XGBoostModel(BaseModel):
    """
    XGBoost classifier implementation.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=True):
        """
        Initialize XGBoost model.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: Whether to use balanced class weights (default: True)
        """
        super().__init__(params, random_state, use_class_weights)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model with parameters"""
        default_params = {
            "n_estimators": 50,
            "learning_rate": 0.01,
            "max_depth": 3,
            "min_child_weight": 5,
            "random_state": self.random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        # Use all CPU cores by default (unless user overrides)
        # XGBoost supports `n_jobs` (and older alias `nthread`).
        if "n_jobs" not in self.params and "nthread" not in self.params:
            default_params["n_jobs"] = -1

        # Add scale_pos_weight if enabled (ratio of negative to positive class)
        if self.use_class_weights and "scale_pos_weight" not in self.params:
            default_params["scale_pos_weight"] = 36  # 7 churners vs 255 non-churners

        default_params.update(self.params)
        self.model = XGBClassifier(**default_params)

    def fit(self, X, y):
        """
        Train the XGBoost model.

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
        Make predictions using XGBoost model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities from XGBoost model.

        Args:
            X: Feature matrix

        Returns:
            numpy array of prediction probabilities
        """
        return self.model.predict_proba(X)
