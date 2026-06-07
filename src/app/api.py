from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io

from src.pipeline.churn_inference import model_inference, business_inference, retention_inference

app = FastAPI(
    title="Customer Churn Intelligence API",
    description=(
        "An end-to-end customer retention intelligence API. "
        "Predicts churn risk by LTV segment, explains model decisions and contribution via SHAP, "
        "assigns behavioral personas, and recommends targeted retention strategies."
    ),
    version="0.1.0"
)


class TestDataRequest(BaseModel):
    n_samples: Optional[int] = None  # None = full test set
    output: str = "full"

@app.get("/health")
def health():
    """Health check for monitoring and load balancer."""
    return {"status": "ok"}


@app.post("/predict/upload")
async def predict_from_upload(file: UploadFile = File(...)):
    """Accept CSV upload and return predictions."""
    try:    
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result = model_inference(data=df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/test")
def predict_from_test(request: TestDataRequest):
    """Use internal test dataset. n_samples=None uses full set, otherwise random sample."""
    try:
        result = model_inference(data=None, n_samples=request.n_samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business/upload")
async def business_from_upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = business_inference(data=df)
    return result

@app.post("/business/test")
def business_from_test(request: TestDataRequest):
    result = business_inference(data=None, n_samples=request.n_samples)
    return result

@app.post("/retention/upload")
async def retention_from_upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = retention_inference(data=df)
    return result

@app.post("/retention/test")
def retention_from_test(request: TestDataRequest):
    result = retention_inference(data=None, n_samples=request.n_samples)
    return result
    
    