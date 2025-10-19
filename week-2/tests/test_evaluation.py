import importlib.util
from pathlib import Path
import joblib


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

    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()

    trainer.main(data_path=str(data_path), output_dir=str(out_dir))

    # Check there is at least one model file
    files = list(out_dir.glob("model_*.joblib"))
    assert len(files) >= 1, "No model files were saved by training"

    # Load model and do a quick predict sanity check
    model = joblib.load(files[0])
    # the iris model should have a predict method
    assert hasattr(model, "predict"), "Loaded object is not a model"
