import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow experiment
mlflow.set_experiment("German_Credit_Risk_3_Features")