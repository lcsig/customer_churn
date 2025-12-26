"""
Example Usage of the ML Training Pipeline

This script demonstrates how to use the ABC-based ML training pipeline:
1. FeatureSet1 (inherits from BaseFeatureGenerator) - Generate features from customer data
2. Model classes (inherit from BaseModel) - LGBM, RandomForest, XGBoost ... etc
3. MLFlowTrainer - Train with time-based validation and MLFlow logging
"""

from datetime import datetime

from features import FeatureSet1
from mlflow_tracking_uri import get_mlflow_tracking_uri
from mlflow_trainer import MLFlowTrainer
from models import CatBoostModel, LGBMModel, RandomForestModel, XGBoostModel

ENABLE_MLFLOW = True


def main():
    """Main execution function"""

    data_path = "data_clean_process/customer_churn_cleaned.json"
    feature_gen = FeatureSet1(data_path)
    feature_gen.load_data()

    print(f"Data loaded: {len(feature_gen.df)} events")
    print(f"Unique users: {feature_gen.df['userId'].nunique()}")
    print(f"Date range: {feature_gen.df['ts_dt'].min()} to {feature_gen.df['ts_dt'].max()}")

    # Choose which model to use (LGBMModel, RandomForestModel, or XGBoostModel)
    model_class = CatBoostModel
    print(f"Selected model: {model_class.__name__}")

    tracking_uri = get_mlflow_tracking_uri()
    print(f"Tracking URI: {tracking_uri}")

    trainer = MLFlowTrainer(
        feature_generator=feature_gen,
        model_class=model_class,
        tracking_uri=tracking_uri,
        experiment_name=None,
        observation_window_days=28,
        label_window_days=14,
        enable_mlflow=ENABLE_MLFLOW,
        test_size=0.3,
        random_state=42,
    )

    all_users = feature_gen.df["userId"].unique()
    train_users, test_users = trainer.split_users(all_users)

    # Example: 4 windows starting from October 29, 2018 (first week of November) - keep seasonality in mind
    # We need to keep seasonality in mind because we are using a rolling window approach.
    start_date = datetime(2018, 10, 29)
    cutoff_dates = trainer.generate_cutoff_dates(
        start_date=start_date, windows_count=4, interval_days=7
    )

    print("Cutoff dates for time-based validation:")
    for i, date in enumerate(cutoff_dates, 1):
        print(f"  Window {i}: {date.date()}")

    # Define parameter grid based on the selected model
    # Using simpler parameters suitable for small sample sizes (11-14 churned users)
    if model_class == LGBMModel:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05],
            "min_child_samples": [1, 2, 3, 5],
        }
    elif model_class == RandomForestModel:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 3, 5],
        }
    elif model_class == XGBoostModel:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05],
            "min_child_weight": [0.5, 1, 2],
        }
    elif model_class == CatBoostModel:
        param_grid = {
            "iterations": [50, 100, 200],
            "depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05],
            "min_data_in_leaf": [1, 2, 3, 5],
        }
    else:
        param_grid = {}

    print(f"Parameter grid: {param_grid}")

    results = trainer.grid_search_with_time_validation(
        param_grid=param_grid, cutoff_dates=cutoff_dates
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Best Parameters: {results['best_params']}")
    print("\nBest Average Metrics:")
    print(f"  F1 Score:  {results['best_avg_f1']:.4f}")
    print(f"  Precision: {results['best_avg_precision']:.4f}")
    print(f"  Recall:    {results['best_avg_recall']:.4f}")
    print(f"  Accuracy:  {results['best_avg_accuracy']:.4f}")
    print(f"  Optimal Threshold: {results['best_optimal_threshold']:.2f}")

    final_model = trainer.train_final_model(
        params=results["best_params"],
        final_cutoff_date=cutoff_dates[-1],  # Use last cutoff date
    )

    print(f"\nFinal {model_class.__name__} trained successfully!")

    # =========================================================================
    # OPTIONAL: Feature Importance
    # =========================================================================
    print("\n" + "=" * 80)
    print("Feature Importance (Top 20)")
    print("=" * 80)

    feature_importance = final_model.get_feature_importance()
    if feature_importance is not None:
        # Get feature names
        sample_features = feature_gen.generate_features(
            cutoff_date=cutoff_dates[-1],
            observation_window_days=30,
            active_users_only=True,
        )
        feature_names = [col for col in sample_features.columns if col != "userId"]

        # Create importance dataframe
        import pandas as pd

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        print(importance_df.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    if ENABLE_MLFLOW:
        print(f"Check MLFlow UI at: {tracking_uri}")
    else:
        print("MLFlow logging was disabled (dry run mode)")


if __name__ == "__main__":
    # Run the main example
    main()
