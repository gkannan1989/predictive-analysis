from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ChurnIQ – Churn Prediction API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = joblib.load("app/churn_model.pkl")

# Request body schema
class CustomerData(BaseModel):
    age: int
    subscription_months: int
    login_freq: int

# API endpoint
@app.post("/predict_churn")
def predict_churn(data: CustomerData):
    features = np.array([[data.age, data.subscription_months, data.login_freq]])
    probability = float(model.predict_proba(features)[0][1])
    return {"churn_probability": round(probability, 3)}
