# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import numpy as np
import xgboost as xgb

# Initialize FastAPI app
app = FastAPI(
    title="Graduate Placement Prediction API",
    description="Predicts whether a graduate student will be placed based on their profile.",
    version="1.0.0"
)

# Load trained model
# Get absolute path to this file’s directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

model = joblib.load(MODEL_PATH)

# Define input schema
class PlacementInput(BaseModel):
    gpa: float
    test_score: float
    # Add all other numeric/categorical features your model expects in order

@app.get("/")
def home():
    return {"message": "Graduate Placement Prediction API is running!"}

@app.post("/predict")
def predict(data: PlacementInput):
    # Convert input to numpy array
    features = np.array([[data.gpa, data.test_score]])
    # Make prediction
    prediction = model.predict(features)
    # If XGBoost returns a probability or integer
    result = int(prediction[0])
    return {"placement_prediction": result}
