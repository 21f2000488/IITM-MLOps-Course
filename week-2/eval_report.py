#!/usr/bin/env python3
"""Evaluation script: load model and data, print accuracy, confusion matrix, and classification report.

This script will attempt to fetch the latest registered model from MLflow Model Registry
(registered name: "iris-decision-tree") and use it for evaluation. If MLflow cannot be reached
or no registered model is found, it falls back to searching the local `artifacts/` directory for a
`.joblib` file.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn import metrics

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "http://136.114.83.43:8100"
REGISTERED_MODEL_NAME = "iris-decision-tree"


def _download_latest_registered_model(name: str, dst: Path) -> Path:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    # get latest version (highest version number)
    try:
        versions = client.get_latest_versions(name)
    except Exception as e:
        print(f"Error fetching versions for registered model '{name}' from MLflow: {e}")
        return None
    if not versions:
        return None
    # prefer versions in stage 'Production' if any
    prod = [v for v in versions if v.current_stage == 'Production']
    chosen = prod[0] if prod else versions[-1]

    # download the model artifact for the chosen version
    try:
        dst.mkdir(parents=True, exist_ok=True)
        local_path = client.download_artifacts(chosen.run_id, "model", dst=str(dst))
        print(f"Downloaded MLflow registered model: name={name}, version={chosen.version}, run_id={chosen.run_id}")
        # mlflow.sklearn.save_model writes a directory; we'll point to that
        return Path(local_path)
    except Exception as e:
        print(f"Error downloading artifacts for model '{name}' version {chosen.version} (run_id={chosen.run_id}): {e}")
        return None


def find_model(artifacts_dir: Path):
    # First try MLflow model registry
    ml_model = _download_latest_registered_model(REGISTERED_MODEL_NAME, artifacts_dir / "mlflow")
    if ml_model is not None:
        return ml_model

    # Fallback: local joblib in artifacts_dir
    if not artifacts_dir.exists():
        return None
    models = sorted(artifacts_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0] if models else None


def _load_model_from_registry(name: str):
    """Try to load the model directly from the MLflow Model Registry using the models:/ URI.
    Returns (model, description) on success or (None, None) on failure.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(name)
    except Exception as e:
        print(f"Error fetching versions for registered model '{name}' from MLflow: {e}")
        return None, None
    if not versions:
        return None, None
    prod = [v for v in versions if v.current_stage == 'Production']
    chosen = prod[0] if prod else versions[-1]

    model_uri = f"models:/{name}/{chosen.version}"
    try:
        print(f"Attempting to load model from registry URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from registry: name={name}, version={chosen.version}, run_id={chosen.run_id}")
        return model, f"registry:{model_uri} (version={chosen.version}, run_id={chosen.run_id})"
    except Exception as e:
        print(f"Error loading model from registry URI {model_uri}: {e}")
        return None, None


def main(data_path: Path = Path("data/iris.csv"), artifacts_dir: Path = Path("artifacts")):
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    # Preferred: try loading directly from MLflow Model Registry
    model, model_source = _load_model_from_registry(REGISTERED_MODEL_NAME)
    if model is None:
        # Fall back to existing file-based behavior
        model_path = find_model(Path(artifacts_dir))
        if model_path is None:
            print(f"ERROR: no model found in MLflow registry or {artifacts_dir}", file=sys.stderr)
            sys.exit(3)

        print(f"Loading model from: {model_path}")
        if model_path.is_dir():
            try:
                model = mlflow.sklearn.load_model(str(model_path))
                model_source = f"local_mlflow_artifact:{model_path}"
                print(f"Loaded model from MLflow artifact path: {model_path}")
            except Exception as e:
                print(f"Error loading model from artifact path {model_path}: {e}")
                sys.exit(6)
        else:
            try:
                model = joblib.load(model_path)
                model_source = f"local_joblib:{model_path}"
                print(f"Loaded local joblib model: {model_path}")
            except Exception as e:
                print(f"Error loading local joblib model {model_path}: {e}")
                sys.exit(7)
    else:
        print(f"Using model from MLflow registry: {model_source}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not set(feature_cols).issubset(df.columns) or "species" not in df.columns:
        print("ERROR: data missing required columns", file=sys.stderr)
        sys.exit(4)

    X = df[feature_cols]
    y = df["species"]

    if not hasattr(model, "predict"):
        print("ERROR: loaded object has no predict method", file=sys.stderr)
        sys.exit(5)

    preds = model.predict(X)

    acc = metrics.accuracy_score(y, preds)
    labels = sorted(y.unique())
    cm = metrics.confusion_matrix(y, preds, labels=labels)
    cr = metrics.classification_report(y, preds)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    # pretty print confusion matrix with labels
    header = "\t" + "\t".join(labels)
    print(header)
    for lab, row in zip(labels, cm):
        print(lab + "\t" + "\t".join(str(x) for x in row))

    print("\nClassification Report:")
    print(cr)


if __name__ == "__main__":
    # default locations (CI will run this from repo root)
    main(data_path=Path("week-2") / "data" / "iris.csv", artifacts_dir=Path("week-2") / "artifacts")
