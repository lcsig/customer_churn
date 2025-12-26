"""
ML Training Pipeline with Manual Parameters and Visualizations

This script demonstrates training with pre-defined best parameters and includes:
1. Manual parameter specification (no grid search)
2. PCA visualization of feature space
3. Feature importance visualization
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features import FeatureSet1
from mlflow_tracking_uri import get_mlflow_tracking_uri
from mlflow_trainer import MLFlowTrainer
from models import CatBoostModel, LGBMModel, LinearSVMModel, RandomForestModel, XGBoostModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca(X, y, title="PCA Visualization", save_path=None):
    """Generate PCA visualization of features"""
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Print explained variance
    explained_var = pca.explained_variance_ratio_
    print(
        f"  PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}, Total: {explained_var.sum():.2%}"
    )

    # Create plot
    plt.figure(figsize=(12, 8))
    classes = np.unique(y)
    colors = ["blue", "red"]
    labels = ["Not Churned", "Churned"]

    for i, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[i],
            label=labels[i],
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidths=0.5,
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return pca


def plot_feature_importance(
    feature_importance, feature_names, top_n=20, model_name=None, save_path=None
):
    """Plot feature importance as a horizontal bar chart"""
    # Create importance dataframe
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    # Get top N features
    top_features = importance_df.head(top_n)

    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_features["importance"], color=colors)

    # Customize plot
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance", fontsize=12, fontweight="bold")
    plt.ylabel("Feature", fontsize=12, fontweight="bold")

    # Add model name to title if provided
    title = f"Top {top_n} Feature Importance"
    if model_name:
        title = f"{model_name} - {title}"
    plt.title(title, fontsize=16, fontweight="bold")

    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return importance_df


def main():
    """Main execution function"""
    # Resolve output paths relative to this script (not the current working directory)
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = "data_clean_process/customer_churn_cleaned.json"
    feature_gen = FeatureSet1(data_path)
    feature_gen.load_data()

    print(f"Dataset: {len(feature_gen.df)} events, {feature_gen.df['userId'].nunique()} users")

    # Define model and parameters
    model_class = RandomForestModel

    if model_class == LGBMModel:
        best_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_samples": 3,
        }
    elif model_class == RandomForestModel:
        best_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
        }
    elif model_class == XGBoostModel:
        best_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 1,
        }
    elif model_class == CatBoostModel:
        best_params = {
            "iterations": 200,
            "depth": 5,
            "learning_rate": 0.05,
            "min_data_in_leaf": 3,
        }
    elif model_class == LinearSVMModel:
        best_params = {
            "C": 1.0,
            "calibration_method": "sigmoid",
            "calibration_cv": 3,
            "max_iter": 20000,
        }
    else:
        best_params = {}

    print(f"Model: {model_class.__name__}, Params: {best_params}")

    # Setup trainer
    tracking_uri = get_mlflow_tracking_uri()

    trainer = MLFlowTrainer(
        feature_generator=feature_gen,
        model_class=model_class,
        tracking_uri=tracking_uri,
        experiment_name=None,
        observation_window_days=28,
        label_window_days=14,
        enable_mlflow=False,
        test_size=0.3,
        random_state=42,
    )

    all_users = feature_gen.df["userId"].unique()
    train_users, test_users = trainer.split_users(all_users)

    start_date = datetime(2018, 10, 29)
    cutoff_dates = trainer.generate_cutoff_dates(
        start_date=start_date, windows_count=4, interval_days=7
    )
    final_cutoff_date = cutoff_dates[0]

    # Train model
    print(f"\nTraining with cutoff date: {final_cutoff_date.date()}")
    final_model = trainer.train_final_model(
        params=best_params,
        final_cutoff_date=final_cutoff_date,
    )

    # Generate features and labels
    all_features = feature_gen.generate_features(
        cutoff_date=final_cutoff_date,
        observation_window_days=trainer.observation_window_days,
        active_users_only=True,
    )
    labels = feature_gen.generate_labels(
        cutoff_date=final_cutoff_date,
        label_window_days=trainer.label_window_days,
    )
    data = all_features.merge(labels, on="userId", how="inner")
    feature_cols = [col for col in all_features.columns if col != "userId"]

    # PCA Visualization
    print("\n[PCA Visualization]")
    X_all = data[feature_cols].values
    y_all = data["churn"].values
    plot_pca(
        X_all,
        y_all,
        save_path=images_dir / "pca_visualization.png",
    )

    # Feature Importance
    print("\n[Feature Importance]")
    feature_importance = final_model.get_feature_importance()
    if feature_importance is not None:
        model_name = model_class.__name__.replace("Model", "")
        importance_df = plot_feature_importance(
            feature_importance,
            feature_cols,
            top_n=20,
            model_name=model_name,
            save_path=images_dir / f"feature_importance_{model_name}.png",
        )
        print("\nTop 20 Features:")
        print(importance_df.head(20).to_string(index=False))
    else:
        print("Feature importance not available.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nModel: {model_class.__name__}")
    print(f"Parameters: {best_params}")
    print("\nGenerated files:")
    print(f"  - {images_dir / 'pca_visualization.png'}")
    model_name = model_class.__name__.replace("Model", "")
    print(f"  - {images_dir / f'feature_importance_{model_name}.png'}")
    print("=" * 60)


if __name__ == "__main__":
    # Run the main example
    main()
