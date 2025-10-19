#!/usr/bin/env python3
"""Evaluation script: load model and data, print accuracy, confusion matrix, and classification report.

This script finds a joblib model in the `artifacts/` directory (first match), loads
`data/iris.csv`, runs predictions and prints human-readable metrics to stdout.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn import metrics


def find_model(artifacts_dir: Path):
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
        print(f"ERROR: no model (.joblib) found in {artifacts_dir}", file=sys.stderr)
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
