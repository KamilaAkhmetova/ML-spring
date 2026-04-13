# ML Model Deployment with FastAPI and Docker

## Project Description
This project demonstrates a complete MLOps workflow: training a machine learning model and deploying it as a REST API using **FastAPI** and **Docker**.

The model predicts credit risk (good/bad credit) using the **German Credit Dataset** from UCI Machine Learning Repository.

## Technologies Used
- **Python 3.10** - Programming language
- **FastAPI** - Web framework for building APIs
- **Uvicorn** - ASGI server
- **Scikit-learn** - Machine learning library
- **Docker** - Containerization platform
- **Joblib** - Model serialization


## Run Steps

Without Docker
1. Install dependencies
   ```bash
   pip install -r requirements.txt
2. Train the model
    python train.py
3. Run the API 
    uvicorn main:app --reload --port 8080
4. Open in browser
    http://localhost:8080/ - check API status
    http://localhost:8080/docs - Swagger documentation

With Docker
1. Build Docker image
    docker build -t ml-api .
2. Run container
    docker run -p 8080:8000 ml-api
3. Test in browser
    http://localhost:8080/
    http://localhost:8080/docs