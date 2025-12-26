"""
XGBoost Model Implementation with SMOTE for Class Balancing
"""

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from ml_training.models.base_model import BaseModel


class XGBoostSMOTEModel(BaseModel):
    """
    XGBoost classifier with SMOTE oversampling for class imbalance.
    """

    def __init__(self, params=None, random_state=42, use_class_weights=False, smote_params=None):
        """
        Initialize XGBoost model with SMOTE.

        Args:
            params: Dictionary of model parameters
            random_state: Random state for reproducibility
            use_class_weights: Whether to use balanced class weights (default: False)
            smote_params: Dictionary of SMOTE parameters (optional)
        """
        super().__init__(params, random_state, False)
        self.smote_params = smote_params if smote_params is not None else {}
        self._initialize_model()
        self._initialize_smote()

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
        if "n_jobs" not in self.params and "nthread" not in self.params:
            default_params["n_jobs"] = -1

        default_params.update(self.params)
        self.model = XGBClassifier(**default_params)

    def _initialize_smote(self):
        """Initialize SMOTE with parameters"""
        default_smote_params = {
            "random_state": self.random_state,
            "sampling_strategy": "auto",  # Resample minority class to match majority
        }
        default_smote_params.update(self.smote_params)
        self.smote = SMOTE(**default_smote_params)

    def fit(self, X, y):
        """
        Train the XGBoost model with SMOTE-resampled data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        # Apply SMOTE to balance the classes
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        self.model.fit(X_resampled, y_resampled)
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
