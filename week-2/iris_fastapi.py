import os
from typing import List, Optional, Dict, Any

import mlflow
from mlflow import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://136.114.83.43:8100")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "iris-decision-tree")
MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR")  # optional local fallback


class PredictRequest(BaseModel):
    # Accept either a list of features or named feature values
    features: Optional[List[float]] = None
    sepal_length: Optional[float] = None
    sepal_width: Optional[float] = None
    petal_length: Optional[float] = None
    petal_width: Optional[float] = None


app = FastAPI(title="Iris model API")


def _load_latest_model() -> (Optional[Any], Optional[str]):
    """Attempt to load the latest registered model from MLflow Model Registry.
    Returns (model, source_description) or (None, None) on failure.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
        if not versions:
            return None, None
        prod = [v for v in versions if v.current_stage == 'Production']
        chosen = prod[0] if prod else versions[-1]
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{chosen.version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, f"registry:{model_uri} (version={chosen.version})"
    except Exception as e:
        print(f"Warning: failed loading from MLflow registry: {e}")
        # last-resort: try to load from local artifact dir if provided
        if MODEL_ARTIFACT_DIR:
            try:
                # If a path to MLflow artifact directory is provided, try loading
                p = os.path.abspath(MODEL_ARTIFACT_DIR)
                if os.path.isdir(p):
                    model = mlflow.sklearn.load_model(p)
                    return model, f"local_artifact:{p}"
            except Exception as e2:
                print(f"Warning: failed loading from local artifact dir: {e2}")
        return None, None


@app.on_event("startup")
def startup_event():
    model, source = _load_latest_model()
    app.state.model = model
    app.state.model_source = source


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model", None)),
        "model_source": getattr(app.state, "model_source", None),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build feature vector
    feature_order = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if req.features:
        if len(req.features) != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features in 'features' list")
        df = pd.DataFrame([req.features], columns=feature_order)
    else:
        try:
            vals = [
                req.sepal_length,
                req.sepal_width,
                req.petal_length,
                req.petal_width,
            ]
            if any(v is None for v in vals):
                raise ValueError("Missing one or more feature values")
            df = pd.DataFrame([vals], columns=feature_order)
        except Exception:
            raise HTTPException(status_code=400, detail="Provide features list or all named feature values")

    try:
        preds = model.predict(df)
        result = {"predicted_label": preds[0]}
        # include probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0].tolist()
            result["predicted_proba"] = probs
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("iris_fastapi:app", host="0.0.0.0", port=int(os.getenv("PORT", 8200)), reload=False)
