from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import logging
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Метрики Prometheus
PREDICTION_HISTOGRAM = Histogram(
    'model_prediction_value',
    'Histogram of model prediction values',
    buckets=[0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 15e6, 20e6, float('inf')]
)

REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total model predictions'
)

ERROR_COUNTER = Counter(
    'http_errors_total',
    'Total HTTP errors by type',
    ['error_type']
)

app = FastAPI(title="Car Price Prediction Service", version="2.0")

# Загрузка модели
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    logger.info(f"Model expects {model.n_features_in_} features: {getattr(model, 'feature_names_in_', 'No feature names')}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class CarFeatures(BaseModel):
    year: int
    mileage: int
    engine_volume: float
    horsepower: int
    brand: str
    model: str
    transmission: str
    fuel_type: str

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

@app.middleware("http")
async def monitor_requests(request, call_next):
    response = await call_next(request)
    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    if 400 <= response.status_code < 500:
        ERROR_COUNTER.labels(error_type='4xx').inc()
    elif 500 <= response.status_code < 600:
        ERROR_COUNTER.labels(error_type='5xx').inc()
    
    return response

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Received features: {features}")
        
        # Создаем DataFrame с правильными именами фич которые ожидает модель
        features_df = pd.DataFrame([[
            features.year,           # feature1
            features.mileage,        # feature2  
            features.engine_volume,  # feature3
            features.horsepower      # feature4
        ]], columns=['feature1', 'feature2', 'feature3', 'feature4'])
        
        logger.info(f"Prepared features for model: {features_df.values.tolist()}")
        
        # Получаем предсказание
        prediction = model.predict(features_df)[0]
        
        # Логируем метрики
        PREDICTION_HISTOGRAM.observe(prediction)
        PREDICTION_COUNTER.inc()
        
        logger.info(f"Prediction made: {prediction}")
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version="2.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ERROR_COUNTER.labels(error_type='prediction_error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "version": "2.0"}

@app.get("/")
async def root():
    return {
        "message": "Car Price Prediction Service v2.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "metrics": "/metrics",
            "predict": "/predict"
        }
    }
