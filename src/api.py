from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

from model import LogisticRegressionScratch

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(title="Heart Disease Risk Prediction API")

# -----------------------------
# Enable CORS (IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow frontend from anywhere
    allow_credentials=True,
    allow_methods=["*"],        # allow POST, OPTIONS
    allow_headers=["*"],
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_DIR = r"D:\Heart Disease prediction\models"

model = LogisticRegressionScratch(learning_rate=0.01)
model.weights = np.load(os.path.join(MODEL_DIR, "weights.npy"))
model.bias = np.load(os.path.join(MODEL_DIR, "bias.npy"))

# -----------------------------
# Input schema
# -----------------------------
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# -----------------------------
# Risk interpretation
# -----------------------------
def interpret_risk(prob):
    if prob < 0.4:
        return "LOW"
    elif prob < 0.7:
        return "MODERATE"
    else:
        return "HIGH"

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_risk(data: PatientData):
    X = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    probability = float(model.predict_proba(X)[0])
    risk_level = interpret_risk(probability)

    return {
        "risk_level": risk_level,
        "risk_probability": round(probability, 3)
    }
