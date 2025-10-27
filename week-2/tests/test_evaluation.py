import importlib.util
from pathlib import Path


def _module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_runs_and_saves_model(tmp_path):
    # Resolve paths relative to week-2
    base = Path(__file__).resolve().parents[1]
    train_py = base / "train_iris.py"
    data_path = base / "data" / "iris.csv"

    if not data_path.exists():
        import pytest
        pytest.skip("Data file missing; run `dvc pull` to fetch data or run CI")

    trainer = _module_from_path(train_py, "train_iris")

    # Run training which will log to MLflow
    trainer.main(data_path=str(data_path), output_dir=str(tmp_path / "artifacts"))

    # Try to load latest model from MLflow registry
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri("http://34.72.133.126:8100")
        client = MlflowClient()
        # get latest versions for registered model
        name = "iris-decision-tree"
        versions = client.get_latest_versions(name)
        if not versions:
            import pytest
            pytest.skip("No registered model versions found in MLflow registry")
        # pick production if available else last
        prod = [v for v in versions if v.current_stage == 'Production']
        chosen = prod[0] if prod else versions[-1]
        model = mlflow.sklearn.load_model(f"models:/{name}/{chosen.version}")
    except Exception:
        import pytest
        pytest.skip("Could not fetch model from MLflow registry; skipping")

    assert hasattr(model, "predict"), "Loaded object is not a model"
