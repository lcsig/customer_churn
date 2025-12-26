"""
Base Feature Generator Abstract Class
All feature generators must inherit from this class
"""

from abc import ABC, abstractmethod


class BaseFeatureGenerator(ABC):
    """
    Abstract base class for feature generators.
    All feature generators must implement load_data, generate_features, and generate_labels.
    """

    def __init__(self, data_path):
        """
        Initialize the feature generator.

        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.df = None

    @abstractmethod
    def load_data(self):
        """
        Load data from the data_path.
        Must set self.df and return self.

        Returns:
            self
        """
        pass

    @abstractmethod
    def generate_features(
        self, cutoff_date=None, observation_window_days=30, active_users_only=True
    ):
        """
        Generate features for users using a sliding observation window.

        Args:
            cutoff_date: datetime object. The end of the observation window. If None, use max date in data.
            observation_window_days: Number of days to look back from cutoff_date for feature generation.
            active_users_only: If True, only include users who haven't cancelled yet by cutoff_date

        Returns:
            DataFrame with features per userId
        """
        pass

    @abstractmethod
    def generate_labels(self, cutoff_date, label_window_days=30):
        """
        Generate churn labels based on events after cutoff_date.

        Args:
            cutoff_date: datetime object. The date to split features from labels.
            label_window_days: Number of days after cutoff to look for churn events.

        Returns:
            DataFrame with userId and churn label (1 = churned, 0 = not churned)
        """
        pass
