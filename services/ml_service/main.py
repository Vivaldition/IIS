from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY, Gauge
import time
import psutil
import os
import sys


sys.path.append('/app/models')
from api_handler import FastAPIHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

CPU_USAGE = Gauge('process_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('process_memory_bytes', 'Memory usage in bytes')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['endpoint'])
PROCESS_UPTIME = Gauge('process_uptime_seconds', 'Process uptime in seconds')
startup_time = time.time()

app = FastAPI(title="Car Price Prediction Service", version="2.0")


try:
    model_handler = FastAPIHandler(model_path='/app/car_price_model.pkl')
    logger.info("Model handler initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model handler: {e}")
    model_handler = None

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

def update_system_metrics():
    """Обновление системных метрик"""
    try:
        CPU_USAGE.set(psutil.cpu_percent())
        process = psutil.Process(os.getpid())
        MEMORY_USAGE.set(process.memory_info().rss)
        PROCESS_UPTIME.set(time.time() - startup_time)
    except Exception as e:
        logger.warning(f"Error updating system metrics: {e}")

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()
    update_system_metrics()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
    
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
    if model_handler is None or model_handler.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Received features: {features}")
        
        
        features_dict = features.dict()
        
        
        prediction = model_handler.predict(features_dict)[0]
        
        
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
    update_system_metrics()
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

@app.get("/health")
async def health():
    update_system_metrics()
    return {
        "status": "healthy", 
        "model_loaded": model_handler is not None and model_handler.model is not None, 
        "version": "2.0",
        "uptime": time.time() - startup_time
    }

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
