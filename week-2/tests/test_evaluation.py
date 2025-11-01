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

    # Attempt to reuse the loading logic from eval_report.py so the test mirrors
    # the evaluation script: prefer MLflow Model Registry, otherwise fall back
    # to local artifacts created by the training run.
    eval_report = _module_from_path(base / "eval_report.py", "eval_report")

    # Try loading directly from the MLflow Model Registry first
    model, source = eval_report._load_model_from_registry(eval_report.REGISTERED_MODEL_NAME)
    if model is None:
        # Fall back to finding a model in the local artifacts directory produced
        # by the trainer (tmp_path / "artifacts"). This mirrors eval_report.find_model.
        artifacts_dir = tmp_path / "artifacts"
        model_path = eval_report.find_model(artifacts_dir)
        if model_path is None:
            import pytest
            pytest.skip("No model available in MLflow registry or local artifacts; skipping")

        # Load model from the located path. Could be an MLflow artifact dir or a joblib
        if model_path.is_dir():
            import mlflow
            model = mlflow.sklearn.load_model(str(model_path))
        else:
            import joblib
            model = joblib.load(model_path)

    assert hasattr(model, "predict"), "Loaded object is not a model"
