#!/usr/bin/env python3
"""
Train an Iris Decision Tree model.

- Expects input CSV at `data/iris.csv` by default.
- Trains a DecisionTreeClassifier and prints accuracy to stdout.
- Logs params, metrics and the trained model to MLflow (no local model file saved).
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import mlflow
import mlflow.sklearn

# MLflow configuration
MLFLOW_TRACKING_URI = "http://34.29.222.152:8100"
MLFLOW_EXPERIMENT_NAME = "iris-experiment"
REGISTERED_MODEL_NAME = "iris-decision-tree"


def main(data_path: str = "data/iris.csv", output_dir: str = "artifacts") -> None:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")

    df = pd.read_csv(data_path)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not set(feature_cols).issubset(df.columns) or "species" not in df.columns:
        raise ValueError(
            "Expected columns missing. CSV must contain the feature columns: "
            "sepal_length, sepal_width, petal_length, petal_width and a 'species' column."
        )

    train, test = train_test_split(df, test_size=0.4, stratify=df["species"], random_state=42)
    X_train = train[feature_cols]
    y_train = train["species"]
    X_test = test[feature_cols]
    y_test = test["species"]

    # Model hyperparameters (expose some common DT params for logging)
    clf_kwargs = {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "random_state": 1,
        "ccp_alpha": 0.0,
    }

    model = DecisionTreeClassifier(**clf_kwargs)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    print(f"The accuracy of the Decision Tree is {acc:.3f}")

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            **{k: v for k, v in clf_kwargs.items() if v is not None},
        })

        # Log metrics
        mlflow.log_metric("accuracy", float(acc))

        # Log the model and register it with the registry
        # This will both save the model artifact under the run and attempt to register it
        try:
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
            print(f"Logged and registered model to MLflow under experiment '{MLFLOW_EXPERIMENT_NAME}'")
        except Exception as e:
            # If registration fails (e.g., registry not enabled), still log the model artifact
            print(f"Warning: model registration failed: {e}. Falling back to logging without registration.")
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"MLflow run id: {run.info.run_id}")

    # NOTE: by design we no longer save a local joblib model file; models are persisted in MLflow.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Decision Tree model")
    parser.add_argument("--data", default="data/iris.csv", help="Path to iris csv (default: data/iris.csv)")
    parser.add_argument("--output", default="artifacts", help="Output directory for model artifact (ignored; MLflow used)")
    args = parser.parse_args()
    main(args.data, args.output)
