from fastapi import FastAPI
import joblib
import os

app = FastAPI()

# Load environment variables from ConfigMap / Secret
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/xgb_model.pkl")
API_VERSION = os.getenv("API_VERSION", "v1")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")

# Utility function to load model safely
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at path: {path}")
    return joblib.load(path)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    model = load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"API version: {API_VERSION} | Log level: {LOG_LEVEL}")

@app.get("/")
def home():
    return {"message": "Graduate Placement Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: dict):
    """
    Example prediction endpoint.
    You should adjust 'features' to match your actual input structure.
    """
    try:
        # Convert request dict to model input
        input_data = list(features.values())
        prediction = model.predict([input_data])
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
