"""
Linear SVM Model Implementation

Notes:
- We use a LinearSVC (fast linear SVM) wrapped in CalibratedClassifierCV so that
  `predict_proba` is available (required by MLFlowTrainer threshold search).
- We also standardize features because SVMs are sensitive to feature scaling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from models.base_model import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class LinearSVMModel(BaseModel):
    """
    Linear SVM classifier with probability outputs via calibration.

    Parameters (via `params` dict):
      - C: float, default 1.0
      - max_iter: int, default 20000
      - tol: float, default 1e-4
      - dual: bool, default True
      - calibrate: bool, default True
      - calibration_method: "sigmoid" | "isotonic", default "sigmoid"
      - calibration_cv: int, default 3
    """

    def __init__(
        self, params: Optional[Dict[str, Any]] = None, random_state=42, use_class_weights=True
    ):
        super().__init__(params, random_state, use_class_weights)
        self._initialize_model()

    def _initialize_model(self):
        default_params: Dict[str, Any] = {
            "C": 1.0,
            "max_iter": 20000,
            "tol": 1e-4,
            "dual": True,
            "random_state": self.random_state,
        }

        # Handle class imbalance
        if self.use_class_weights and "class_weight" not in self.params:
            default_params["class_weight"] = "balanced"

        # Calibration controls (we pop these from params so they don't get passed to LinearSVC)
        calibrate = bool(self.params.pop("calibrate", True))
        calibration_method = self.params.pop("calibration_method", "sigmoid")
        calibration_cv = self.params.pop("calibration_cv", 3)

        default_params.update(self.params)

        # Scale features then train LinearSVC
        base_estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(**default_params)),
            ]
        )

        if calibrate:
            # sklearn changed arg name from base_estimator -> estimator
            try:
                self.model = CalibratedClassifierCV(
                    estimator=base_estimator, method=calibration_method, cv=calibration_cv
                )
            except TypeError:
                self.model = CalibratedClassifierCV(
                    base_estimator=base_estimator, method=calibration_method, cv=calibration_cv
                )
        else:
            # No calibration => no predict_proba. We'll provide a sigmoid transform of decision_function
            # for compatibility with the trainer.
            self.model = base_estimator

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        # Fallback for non-calibrated LinearSVC pipeline:
        # Convert decision_function to a pseudo-probability via sigmoid.
        decision = self.model.decision_function(X)
        decision = np.asarray(decision).ravel()
        proba_pos = 1.0 / (1.0 + np.exp(-decision))
        proba_neg = 1.0 - proba_pos
        return np.vstack([proba_neg, proba_pos]).T

    def get_feature_importance(self):
        """
        For a linear model, expose |coef_| as a feature-importance-like signal.
        Returns None if coefficients are not available.
        """
        try:
            # CalibratedClassifierCV stores per-fold estimators in calibrated_classifiers_ after fit
            if (
                hasattr(self.model, "calibrated_classifiers_")
                and self.model.calibrated_classifiers_
            ):
                cc0 = self.model.calibrated_classifiers_[0]
                est = getattr(cc0, "estimator", getattr(cc0, "base_estimator", None))
            else:
                est = self.model

            # Our estimator is a Pipeline(scaler, svc)
            if hasattr(est, "named_steps") and "svc" in est.named_steps:
                svc = est.named_steps["svc"]
            else:
                svc = est

            if hasattr(svc, "coef_"):
                return np.abs(np.asarray(svc.coef_)).ravel()
        except Exception:
            return None
        return None
