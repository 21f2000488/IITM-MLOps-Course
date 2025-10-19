import pandas as pd
from pathlib import Path
import pytest


def _data_path():
    # week-2/tests/... -> parents[1] == week-2
    base = Path(__file__).resolve().parents[1]
    return base / "data" / "iris.csv"


def test_iris_csv_exists_and_columns():
    data_path = _data_path()
    if not data_path.exists():
        pytest.skip(f"Data file not found at {data_path} - run `dvc pull` in week-2 or run CI to fetch data")

    df = pd.read_csv(data_path)
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_cols.issubset(set(df.columns)), f"Missing expected columns. Found: {df.columns.tolist()}"


def test_no_missing_values():
    data_path = _data_path()
    if not data_path.exists():
        pytest.skip("Data missing; skipped data validation")
    df = pd.read_csv(data_path)
    # Ensure there are no NA values in feature or target columns
    assert df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].isnull().sum().sum() == 0


def test_class_balance():
    data_path = _data_path()
    if not data_path.exists():
        pytest.skip("Data missing; skipped data validation")
    df = pd.read_csv(data_path)
    counts = df["species"].value_counts()
    # Expect at least 2 samples per class (iris dataset typically has 50 each)
    assert (counts >= 2).all(), f"Unexpectedly small class counts: {counts.to_dict()}"
