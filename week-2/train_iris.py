#!/usr/bin/env python3
"""
Train an Iris Decision Tree model.

- Expects input CSV at `data/iris.csv` by default.
- Trains a DecisionTreeClassifier and prints accuracy to stdout.
- Logs params, metrics and the trained model to MLflow (no local model file saved).
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

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

    # Model hyperparameters (updated)
    clf_kwargs = {
        "criterion": "entropy",
        "splitter": "best",
        "max_depth": 5,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "ccp_alpha": 0.01,
    }

    # Perform a small grid search (CV) to tune a few hyperparameters
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "max_features": [None, "sqrt"],
    }

    base_clf = DecisionTreeClassifier(random_state=clf_kwargs.get("random_state", None))
    gs = GridSearchCV(base_clf, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    best_params = gs.best_params_
    best_cv_score = gs.best_score_
    print(f"GridSearchCV best params: {best_params}, best_cv_accuracy={best_cv_score:.4f}")

    # infer signature using training inputs and predictions from the best model
    try:
        preds_train = best_model.predict(X_train)
        signature = infer_signature(X_train, preds_train)
        input_example = X_train.head(1)
    except Exception as e:
        signature = None
        input_example = None
        print(f"Warning: could not infer model signature: {e}")

    preds = best_model.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    print(f"The accuracy of the Decision Tree is {acc:.3f}")

    # Use best_model going forward
    model = best_model

    # build a simple schema description and log it
    schema = {
        "features": {col: str(X_train[col].dtype) for col in feature_cols},
        "target": str(y_train.dtype),
    }
    print(f"Inferred schema: {json.dumps(schema)}")

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log dataset sizes and initial classifier kwargs
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))
        mlflow.log_param("initial_clf_kwargs", json.dumps(clf_kwargs))

        # Log grid-search details
        mlflow.log_param("grid_search_param_grid", json.dumps(param_grid))

        # Log best params from GridSearchCV (if available)
        try:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_cv_accuracy", float(best_cv_score))
        except Exception:
            # best_params may not exist if GS failed; continue
            print("Warning: best_params or best_cv_score not available to log to MLflow")

        # Log evaluation metric on the test set
        mlflow.log_metric("accuracy", float(acc))

        # set tags for features/target
        mlflow.set_tag("features", ",".join(feature_cols))
        mlflow.set_tag("target", "species")
        mlflow.log_param("schema", json.dumps(schema))

        # Log the model artifact (with signature and input example if available)
        if signature is not None and input_example is not None:
            mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=input_example)
            print(f"Logged model with inferred signature to MLflow (run_id={run.info.run_id})")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"Logged model without signature to MLflow (run_id={run.info.run_id})")

        # Explicitly register the just-logged model URI and capture the model version
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_version = None
        try:
            mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            registered_version = mv.version
            print(f"Registered model to MLflow: name={REGISTERED_MODEL_NAME}, version={registered_version}, stage={mv.current_stage}")
            # also log the registered version as a run tag/param
            mlflow.set_tag("registered_model_version", str(registered_version))
        except Exception as e:
            # If registration fails (e.g., registry not enabled), still inform
            print(f"Warning: model registration failed: {e}. Model is available under run artifacts at {model_uri}.")

        print(f"MLflow run id: {run.info.run_id}")

    # NOTE: by design we no longer save a local joblib model file; models are persisted in MLflow.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Decision Tree model")
    parser.add_argument("--data", default="data/iris.csv", help="Path to iris csv (default: data/iris.csv)")
    parser.add_argument("--output", default="artifacts", help="Output directory for model artifact (ignored; MLflow used)")
    args = parser.parse_args()
    main(args.data, args.output)
