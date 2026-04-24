import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load dataset
statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

# Preprocess target
y['class'] = y['class'].map({1: 0, 2: 1})

# Preprocess specific columns
X['Attribute19'] = X['Attribute19'].map({'A191': 0, 'A192': 1})
X['Attribute20'] = X['Attribute20'].map({'A201': 1, 'A202': 0})

# One-hot encode categorical columns
categorical_cols = ['Attribute10', 'Attribute14', 'Attribute15', 'Attribute1',
                    'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7',
                    'Attribute9', 'Attribute12', 'Attribute17']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train.values.ravel())

# metrics
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("German_Credit_Risk")

with mlflow.start_run(run_name="LR_Baseline"):
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("C", 1.0)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    
    # Save and log model
    joblib.dump(pipeline, "model.joblib")
    mlflow.log_artifact("model.joblib")
    
    # Register model
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="GermanCreditRiskModel"
    )

# Save model
# joblib.dump(pipeline, 'model.joblib')

# print(f"Training accuracy: {pipeline.score(X_train, y_train):.4f}")