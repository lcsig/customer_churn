"""
Features Package
Contains abstract base class and concrete feature set implementations
"""

from ml_training.features.base_features import BaseFeatureGenerator
from ml_training.features.feature_set1 import FeatureSet1
from ml_training.features.feature_set2 import FeatureSet2

__all__ = [
    "BaseFeatureGenerator",
    "FeatureSet1",  # Batch processing (for historical data)
    "FeatureSet2",  # Enhanced feature set with churn prediction features
]
