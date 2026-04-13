from fastapi import FastAPI
import joblib
import numpy as np

# Load model once at startup
model = joblib.load('model.joblib')

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(features: list):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}