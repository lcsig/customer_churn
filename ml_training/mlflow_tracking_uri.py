import os


def get_mlflow_tracking_uri(default: str = "http://127.0.0.1:5000") -> str:
    """
    Build MLflow tracking URI from environment variables.

    Logic is intentionally kept identical/simplistic to `ml_training/main.py`:
    - Read MLFLOW_TRACKING_URI, MLFLOW_USERNAME, MLFLOW_PASSWORD
    - If username+password exist and "://" is in the URI, inject creds:
        https://user:pass@host/path # pragma: allowlist secret
      Otherwise, return the URI as-is.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default)
    username = os.getenv("MLFLOW_USERNAME")
    password = os.getenv("MLFLOW_PASSWORD")

    if username and password and "://" in tracking_uri:
        scheme, rest = tracking_uri.split("://", 1)
        tracking_uri = f"{scheme}://{username}:{password}@{rest}"

    return tracking_uri
