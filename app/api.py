from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from src.prediction import load_artifacts, predict_new_data
from src.config import load_config
from src.pipeline import MODEL_REGISTRY
from app.schemas import (
    SinglePredictRequest,
    SinglePredictResponse,
    BatchPredictResponse,
    BatchPredictItem,
    ErrorResponse,
)

app = FastAPI(
    title="Twitter Fake Account Detection API",
    description="Multi-model API for detecting fake Twitter accounts",
    version="0.1.0",
)


def _build_single_df(req: SinglePredictRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "followers_count": req.followers_count,
        "friends_count": req.friends_count,
        "post_count": req.post_count,
        "location": req.location,
        "lang": req.lang,
        "description": req.description,
        "created_at": req.created_at,
    }])


def _load_model(model_name: str):
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    model_path = models_dir / f"{model_name}_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    lang_mapping_path = models_dir / "lang_mapping.pkl"

    for p in [model_path, scaler_path, lang_mapping_path]:
        if not p.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {p.name}. Train the model first.",
            )

    return load_artifacts(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        lang_mapping_path=str(lang_mapping_path),
    )


@app.get("/")
def root():
    return {
        "message": "Twitter Fake Account Detection API",
        "endpoints": {
            "GET /": "This message",
            "POST /predict": "Single prediction",
            "POST /predict/batch": "Batch prediction from file",
        },
    }


@app.post(
    "/predict",
    response_model=SinglePredictResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def predict_single(
    request: SinglePredictRequest,
    model_name: str = Query("slp", description="Model to use for prediction"),
):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}",
        )

    model, scaler, lang_mapping = _load_model(model_name)
    df_input = _build_single_df(request)
    result_df = predict_new_data(df_input, model, scaler, lang_mapping)
    row = result_df.iloc[0]

    return SinglePredictResponse(
        prediction=row["prediction"],
        fake_probability=float(row["fake_probability"]),
        risk_level=row["risk_level"],
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def predict_batch(
    file: UploadFile = File(..., description="CSV or XLSX file with required columns"),
    model_name: str = Query("slp", description="Model to use for prediction"),
):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}",
        )

    if not file.filename:
        raise HTTPException(status_code=422, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix == ".csv":
        df_input = pd.read_csv(file.file)
    elif suffix == ".xlsx":
        df_input = pd.read_excel(file.file)
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file format. Upload CSV or XLSX.",
        )

    required_cols = [
        "followers_count", "friends_count", "post_count",
        "location", "lang", "description", "created_at",
    ]
    missing = [c for c in required_cols if c not in df_input.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}",
        )

    model, scaler, lang_mapping = _load_model(model_name)
    result_df = predict_new_data(df_input, model, scaler, lang_mapping)

    results = [
        BatchPredictItem(
            prediction=row["prediction"],
            fake_probability=float(row["fake_probability"]),
            risk_level=row["risk_level"],
        )
        for _, row in result_df.iterrows()
    ]

    return BatchPredictResponse(
        total=len(results),
        fake_count=int((result_df["prediction"] == "Fake").sum()),
        high_risk_count=int((result_df["risk_level"] == "High").sum()),
        results=results,
    )
