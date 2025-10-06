#!/usr/bin/env python3
"""
Train an Iris Decision Tree model.

- Expects input CSV at `data/iris.csv` by default.
- Trains a DecisionTreeClassifier and prints accuracy to stdout.
- Saves the trained model artifact into the `artifacts/` directory (timestamped filename).
"""

import os
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib


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

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = metrics.accuracy_score(preds, y_test)
    print(f"The accuracy of the Decision Tree is {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(output_dir) / f"model_{timestamp}.joblib"
    joblib.dump(model, out_path)
    print(f"Saved model artifact to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Decision Tree model")
    parser.add_argument("--data", default="data/iris.csv", help="Path to iris csv (default: data/iris.csv)")
    parser.add_argument("--output", default="artifacts", help="Output directory for model artifact")
    args = parser.parse_args()
    main(args.data, args.output)
