"""
MLFlow Trainer Class
Handles training, testing, and MLFlow logging with time-based validation
"""

import random
from contextlib import nullcontext
from datetime import timedelta

import mlflow
import mlflow.sklearn
import numpy as np
from features.base_features import BaseFeatureGenerator
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid


class MLFlowTrainer:
    """
    Handles model training with time-based validation and MLFlow experiment tracking.
    Implements a holdout test strategy with rolling time windows.
    """

    def __init__(
        self,
        feature_generator,
        model_class,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name=None,
        observation_window_days=28,
        label_window_days=14,
        use_class_weights=True,
        enable_mlflow=True,
        test_size=0.2,
        random_state=42,
    ):
        """
        Initialize the MLFlow Trainer.

        Args:
            feature_generator: Instance of BaseFeatureGenerator (e.g., FeatureSet1)
            model_class: BaseModel class (not instance) to be instantiated during training
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of the MLFlow experiment. If None, auto-generated as
                           "churn_prediction_{model_name}_{features_name}"
            observation_window_days: Number of days to look back from cutoff_date for features (default: 30)
            label_window_days: Number of days after cutoff_date to look for churn labels (default: 7)
            use_class_weights: Whether to use balanced class weights for imbalanced data (default: True)
            enable_mlflow: Whether to enable MLFlow logging (default: True). Set to False for dry runs.
            test_size: Proportion of users for holdout test (default 0.2)
            random_state: Random state for reproducibility
        """
        if not isinstance(feature_generator, BaseFeatureGenerator):
            raise TypeError("feature_generator must be an instance of BaseFeatureGenerator")

        if not issubclass(model_class, BaseModel):
            raise TypeError("model_class must be a subclass of BaseModel")

        self.feature_generator = feature_generator
        self.model_class = model_class
        self.enable_mlflow = enable_mlflow

        # Auto-generate experiment name if not provided
        if experiment_name is None:
            model_name = model_class.__name__
            features_name = feature_generator.__class__.__name__
            experiment_name = f"churn_task_{model_name}_{features_name}"

        self.experiment_name = experiment_name
        self.observation_window_days = observation_window_days
        self.label_window_days = label_window_days
        self.use_class_weights = use_class_weights
        self.test_size = test_size
        self.random_state = random_state

        # Set up MLFlow only if enabled
        if self.enable_mlflow:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

        # User split
        self.train_users = None
        self.test_users = None

        print("MLFlow Trainer initialized")
        print(f"MLFlow logging: {'ENABLED' if enable_mlflow else 'DISABLED (dry run)'}")
        if self.enable_mlflow:
            print(f"Tracking URI: {tracking_uri}")
            print(f"Experiment: {experiment_name}")
        print(f"Observation window: {observation_window_days} days")
        print(f"Label window: {label_window_days} days")
        print(f"Use class weights: {use_class_weights}")
        print(f"Test size: {test_size}")

    def split_users(self, user_ids):
        """
        Randomly split users into train and test groups.
        This split is permanent for all time windows.

        Args:
            user_ids: List or array of user IDs

        Returns:
            tuple: (train_users, test_users)
        """
        user_ids = sorted(set(user_ids))  # Remove duplicates and sort for determinism

        # Shuffle with fixed random state
        random.seed(self.random_state)
        random.shuffle(user_ids)

        # Split
        split_idx = int(len(user_ids) * (1 - self.test_size))
        self.train_users = set(user_ids[:split_idx])
        self.test_users = set(user_ids[split_idx:])

        print("\nUser Split:")
        print(f"  Training users: {len(self.train_users)} ({(1-self.test_size)*100:.0f}%)")
        print(f"  Holdout users: {len(self.test_users)} ({self.test_size*100:.0f}%)")

        return self.train_users, self.test_users

    def time_based_validation(self, params, cutoff_dates):
        """
        Perform time-based validation across multiple cutoff dates.
        Uses a single optimal threshold across all weeks for consistency.

        Args:
            params: Dictionary of model parameters
            cutoff_dates: List of datetime objects for cutoff dates (e.g., 4 weeks)

        Returns:
            dict: Metrics including F1 scores for each week and average
        """
        model_name = self.model_class.__name__
        print(f"\nTime-Based Validation for {model_name} with params: {params}")
        print("=" * 80)

        # First pass: collect predictions from all weeks
        week_data = []

        for week_idx, cutoff_date in enumerate(cutoff_dates, 1):
            print(f"\nWeek {week_idx} - Cutoff Date: {cutoff_date.date()}")
            print("-" * 80)

            # Generate features for active users up to cutoff date
            features = self.feature_generator.generate_features(
                cutoff_date=cutoff_date,
                observation_window_days=self.observation_window_days,
                active_users_only=True,
            )

            # Generate labels based on events after cutoff date
            labels = self.feature_generator.generate_labels(
                cutoff_date=cutoff_date, label_window_days=self.label_window_days
            )

            # Merge features and labels
            data = features.merge(labels, on="userId", how="inner")

            # Split by pre-defined user groups
            train_data = data[data["userId"].isin(self.train_users)]
            test_data = data[data["userId"].isin(self.test_users)]

            if len(train_data) == 0 or len(test_data) == 0:
                print(f"  Skipping week {week_idx}: insufficient data")
                continue

            # Check if there are any churned users
            if train_data["churn"].sum() == 0:
                print(f"  Skipping week {week_idx}: no churned users in training set")
                continue

            # Prepare features and labels
            X_train = train_data.drop(["userId", "churn"], axis=1)
            y_train = train_data["churn"]
            X_test = test_data.drop(["userId", "churn"], axis=1)
            y_test = test_data["churn"]

            # Handle any missing values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)

            print(
                f"  Train: {len(X_train)} users, {y_train.sum()} churned ({y_train.mean()*100:.1f}%)"
            )
            print(
                f"  Test:  {len(X_test)} users, {y_test.sum()} churned ({y_test.mean()*100:.1f}%)"
            )

            # Train model - instantiate the model class with params
            model = self.model_class(
                params=params,
                random_state=self.random_state,
                use_class_weights=self.use_class_weights,
            )
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Store predictions and actuals for later threshold optimization
            week_data.append(
                {
                    "week_idx": week_idx,
                    "y_test": y_test,
                    "y_pred_proba": y_pred_proba,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "train_churn_rate": y_train.mean(),
                    "test_churn_rate": y_test.mean(),
                }
            )

            # Diagnostic output
            print(
                f"  Prediction probabilities - Min: {y_pred_proba.min():.4f}, "
                f"Max: {y_pred_proba.max():.4f}, Mean: {y_pred_proba.mean():.4f}"
            )

            if y_test.sum() > 0:
                churned_probas = y_pred_proba[y_test.values == 1]
                print(
                    f"  Actual churned users probabilities: {[f'{p:.3f}' for p in churned_probas]}"
                )

        # Find optimal threshold across ALL weeks
        print("\n" + "=" * 80)
        print("Finding optimal threshold across all weeks...")
        print("=" * 80)

        best_avg_f1 = -1
        best_threshold = 0.5

        for threshold in [
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
        ]:
            f1_scores = []
            for week in week_data:
                y_pred_temp = (week["y_pred_proba"] >= threshold).astype(int)
                f1_temp = f1_score(week["y_test"], y_pred_temp, zero_division=0)
                f1_scores.append(f1_temp)

            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            print(f"  Threshold {threshold:.2f}: Average F1 = {avg_f1:.4f}")

            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_threshold = threshold

        print(f"\n>>> Optimal threshold: {best_threshold:.2f} (Average F1 = {best_avg_f1:.4f})")

        # Second pass: Apply optimal threshold to all weeks and calculate metrics
        print("\n" + "=" * 80)
        print(f"Applying threshold {best_threshold:.2f} to all weeks:")
        print("=" * 80)

        week_scores = []
        week_metrics = []

        for week in week_data:
            week_idx = week["week_idx"]
            y_test = week["y_test"]
            y_pred_proba = week["y_pred_proba"]

            # Apply the optimal threshold
            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # Calculate metrics
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            week_scores.append(f1)
            week_metrics.append(
                {
                    "week": week_idx,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "threshold": best_threshold,
                    "train_size": week["train_size"],
                    "test_size": week["test_size"],
                    "train_churn_rate": week["train_churn_rate"],
                    "test_churn_rate": week["test_churn_rate"],
                }
            )

            print(
                f"\nWeek {week_idx}: F1={f1:.4f}, Precision={precision:.4f}, "
                f"Recall={recall:.4f}, Predicted={y_pred.sum()}, Actual={y_test.sum()}"
            )

        # Calculate average metrics across all weeks
        if week_metrics:
            avg_f1 = np.mean([m["f1"] for m in week_metrics])
            avg_precision = np.mean([m["precision"] for m in week_metrics])
            avg_recall = np.mean([m["recall"] for m in week_metrics])
            avg_accuracy = np.mean([m["accuracy"] for m in week_metrics])
        else:
            avg_f1 = avg_precision = avg_recall = avg_accuracy = 0

        print("\n" + "=" * 80)
        print(f"Summary - {len(week_scores)} weeks with threshold {best_threshold:.2f}:")
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print("=" * 80)

        return {
            "avg_f1": avg_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_accuracy": avg_accuracy,
            "optimal_threshold": best_threshold,
            "week_scores": week_scores,
            "week_metrics": week_metrics,
            "num_weeks": len(week_scores),
        }

    def grid_search_with_time_validation(self, param_grid, cutoff_dates):
        """
        Perform grid search with time-based validation.
        Each parameter combination is evaluated across multiple time windows.

        Args:
            param_grid: Dictionary of parameter grids
            cutoff_dates: List of datetime objects for time validation

        Returns:
            dict: Best parameters and their metrics
        """
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        model_name = self.model_class.__name__
        print(f"\n{'='*80}")
        print(f"GRID SEARCH: {model_name.upper()}")
        print(f"{'='*80}")
        print(f"Total parameter combinations: {len(param_combinations)}")
        print(f"Time windows: {len(cutoff_dates)}")
        print(f"Total training runs: {len(param_combinations) * len(cutoff_dates)}")

        best_params = None
        best_avg_f1 = -1
        best_avg_precision = 0
        best_avg_recall = 0
        best_avg_accuracy = 0
        best_optimal_threshold = 0.5
        all_results = []

        # Iterate through all parameter combinations
        for param_idx, params in enumerate(param_combinations, 1):
            print(f"\n{'='*80}")
            print(f"Parameter Combination {param_idx}/{len(param_combinations)}")
            print(f"{'='*80}")

            # Conditionally use MLFlow context manager or dummy context
            if self.enable_mlflow:
                run_context = mlflow.start_run(run_name=f"{model_name}_params_{param_idx}")
            else:
                # Use a dummy context manager that does nothing
                run_context = nullcontext()

            with run_context:
                # Log parameters (only if MLFlow is enabled)
                if self.enable_mlflow:
                    mlflow.log_param("model_class", model_name)
                    mlflow.log_param("observation_window_days", self.observation_window_days)
                    mlflow.log_param("label_window_days", self.label_window_days)
                    mlflow.log_param("use_class_weights", self.use_class_weights)
                    mlflow.log_param("test_size", self.test_size)
                    mlflow.log_param("random_state", self.random_state)
                    mlflow.log_param("experiment_name", self.experiment_name)
                    mlflow.log_param("train_users", len(self.train_users))
                    mlflow.log_param("test_users", len(self.test_users))
                    mlflow.log_param("cutoff_dates", cutoff_dates)
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)

                # Perform time-based validation
                results = self.time_based_validation(
                    params=params,
                    cutoff_dates=cutoff_dates,
                )

                avg_f1 = results["avg_f1"]
                avg_precision = results["avg_precision"]
                avg_recall = results["avg_recall"]
                avg_accuracy = results["avg_accuracy"]
                optimal_threshold = results["optimal_threshold"]

                # Log metrics (only if MLFlow is enabled)
                if self.enable_mlflow:
                    mlflow.log_param("optimal_threshold", optimal_threshold)
                    mlflow.log_metric("avg_f1_score", avg_f1)
                    mlflow.log_metric("avg_precision", avg_precision)
                    mlflow.log_metric("avg_recall", avg_recall)
                    mlflow.log_metric("avg_accuracy", avg_accuracy)
                    mlflow.log_metric("num_weeks_evaluated", results["num_weeks"])

                    # Log weekly metrics
                    for week_metric in results["week_metrics"]:
                        week = week_metric["week"]
                        mlflow.log_metric(f"week_{week}_f1", week_metric["f1"])
                        mlflow.log_metric(f"week_{week}_precision", week_metric["precision"])
                        mlflow.log_metric(f"week_{week}_recall", week_metric["recall"])
                        mlflow.log_metric(f"week_{week}_accuracy", week_metric["accuracy"])

                # Store results
                all_results.append({"params": params, "avg_f1": avg_f1, "results": results})

                # Update best parameters
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_avg_precision = avg_precision
                    best_avg_recall = avg_recall
                    best_avg_accuracy = avg_accuracy
                    best_optimal_threshold = optimal_threshold
                    best_params = params
                    print(
                        f"\n>>> NEW BEST: F1={best_avg_f1:.4f}, "
                        f"Precision={best_avg_precision:.4f}, Recall={best_avg_recall:.4f}, "
                        f"Threshold={best_optimal_threshold:.2f}"
                    )

        print("\n" + "=" * 80)
        print("GRID SEARCH COMPLETE")
        print("=" * 80)
        print("Best Average Metrics:")
        print(f"  F1 Score: {best_avg_f1:.4f}")
        print(f"  Precision: {best_avg_precision:.4f}")
        print(f"  Recall: {best_avg_recall:.4f}")
        print(f"  Accuracy: {best_avg_accuracy:.4f}")
        print(f"  Optimal Threshold: {best_optimal_threshold:.2f}")
        print(f"Best Parameters: {best_params}")

        return {
            "best_params": best_params,
            "best_avg_f1": best_avg_f1,
            "best_avg_precision": best_avg_precision,
            "best_avg_recall": best_avg_recall,
            "best_avg_accuracy": best_avg_accuracy,
            "best_optimal_threshold": best_optimal_threshold,
            "all_results": all_results,
        }

    def train_final_model(self, params, final_cutoff_date):
        """
        Train a final model on all data up to final_cutoff_date.

        Args:
            params: Model parameters
            final_cutoff_date: Final cutoff date for training

        Returns:
            Trained model instance
        """
        model_name = self.model_class.__name__
        print(f"\n{'='*80}")
        print("TRAINING FINAL MODEL")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Parameters: {params}")
        print(f"Cutoff Date: {final_cutoff_date.date()}")

        # Generate features and labels
        features = self.feature_generator.generate_features(
            cutoff_date=final_cutoff_date,
            observation_window_days=self.observation_window_days,
            active_users_only=True,
        )

        labels = self.feature_generator.generate_labels(
            cutoff_date=final_cutoff_date, label_window_days=self.label_window_days
        )

        # Merge
        data = features.merge(labels, on="userId", how="inner")

        # Prepare data
        X = data.drop(["userId", "churn"], axis=1)
        y = data["churn"]
        X = X.fillna(0)

        print(f"Training on {len(X)} users, {y.sum()} churned ({y.mean()*100:.1f}%)")

        # Train model - instantiate the model class with params
        model = self.model_class(
            params=params,
            random_state=self.random_state,
            use_class_weights=self.use_class_weights,
        )
        model.fit(X, y)

        print("Training complete!")

        return model

    def generate_cutoff_dates(self, start_date, windows_count=3, interval_days=7):
        """
        Generate a list of cutoff dates for time-based validation.

        Args:
            start_date: Starting date (datetime object)
            windows_count: Number of windows to generate
            interval_days: Days between each cutoff

        Returns:
            List of datetime objects
        """
        cutoff_dates = []
        for i in range(windows_count):
            cutoff_date = start_date + timedelta(days=i * interval_days)
            cutoff_dates.append(cutoff_date)

        return cutoff_dates
