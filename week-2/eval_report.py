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

MLFLOW_TRACKING_URI = "http://34.72.133.126:8100"
REGISTERED_MODEL_NAME = "iris-decision-tree"


def _download_latest_registered_model(name: str, dst: Path) -> Path:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    # get latest version (highest version number)
    try:
        versions = client.get_latest_versions(name)
    except Exception:
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
        # mlflow.sklearn.save_model writes a directory; we'll point to that
        return Path(local_path)
    except Exception:
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


def main(data_path: Path = Path("data/iris.csv"), artifacts_dir: Path = Path("artifacts")):
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    model_path = find_model(Path(artifacts_dir))
    if model_path is None:
        print(f"ERROR: no model found in MLflow registry or {artifacts_dir}", file=sys.stderr)
        sys.exit(3)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not set(feature_cols).issubset(df.columns) or "species" not in df.columns:
        print("ERROR: data missing required columns", file=sys.stderr)
        sys.exit(4)

    X = df[feature_cols]
    y = df["species"]

    print(f"Loading model from: {model_path}")
    # joblib.load works for joblib files; MLflow models are typically directories and can be loaded via mlflow.sklearn.load_model
    if model_path.is_dir():
        model = mlflow.sklearn.load_model(str(model_path))
    else:
        model = joblib.load(model_path)

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
