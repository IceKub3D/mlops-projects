from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import os
import pickle
import numpy as np
from prometheus_client import Counter, Histogram
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# -----------------------------
# Prometheus Metrics
# -----------------------------
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint", "method", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["endpoint", "method"]
)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Graduate Placement Prediction API")

# -----------------------------
# Load Model on Startup
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Schemas
# -----------------------------
class PlacementFeatures(BaseModel):
    age: int
    cgpa: float
    gpa: float
    test_score: float
    internships: int
    projects: int
    soft_skills: int

# -----------------------------
# Security — API Token Check
# -----------------------------
API_TOKEN = os.getenv("API_TOKEN", "test-token-123")

def verify_token(authorization: str = Header(...)):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")
    return True

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    REQUEST_COUNT.labels("/", "GET", 200).inc()
    return {"message": "Graduate Placement Prediction API is running!"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(features: PlacementFeatures, valid=Depends(verify_token)):

    import time
    start_time = time.time()

    try:
        data = np.array([[features.age, features.cgpa, features.gpa,
                          features.test_score, features.internships,
                          features.projects, features.soft_skills]])

        prediction = model.predict(data)[0]
        status = 200

        return {"placement_prediction": int(prediction)}

    except Exception as e:
        status = 500
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_COUNT.labels("/predict", "POST", status).inc()
        REQUEST_LATENCY.labels("/predict", "POST").observe(time.time() - start_time)

# -----------------------------
# Prometheus Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
