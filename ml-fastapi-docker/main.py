from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Credit Risk API",
    description="API for credit risk prediction"
)

# Request model
class PredictRequest(BaseModel):
    features: list[float]

# Load model
try:
    model = joblib.load("model.joblib")
    logger.info("Model loaded successfully")
    logger.info(f"Expected features: {model.n_features_in_}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


# @app.get("/")
# def root():
#     return {
#         "message": "ML API is running",
#         "status": "active"
#     }


# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "expected_features": model.n_features_in_ if model else None
#     }


# @app.get("/model-info")
# def model_info():
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     return {
#         "model_type": type(model).__name__,
#         "expected_features": model.n_features_in_
#     }


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        features = request.features

        logger.info(f"Received features: {features}")
        logger.info(f"Number of received features: {len(features)}")

        expected_features = model.n_features_in_
        received_features = len(features)

        if received_features != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Wrong number of features. Model expects {expected_features}, but received {received_features}."
            )

        input_array = np.array(features).reshape(1, -1)

        prediction = model.predict(input_array)

        result = {
            "prediction": int(prediction[0]),
            "features_used": received_features
        }

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array)

            result["probability"] = {
                "good_credit_risk": float(probability[0][0]),
                "bad_credit_risk": float(probability[0][1])
            }

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


