import streamlit as st
import requests
import os

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("German Credit Risk Prediction")
st.markdown(
    "This app predicts whether a customer is a **good** (0) or **bad** (1) credit risk."
)

st.sidebar.header("Customer Information")

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

col1, col2, col3 = st.columns(3)

with col1:
    duration = st.number_input(
        "Credit Duration (months)",
        min_value=1,
        max_value=72,
        value=24
    )

with col2:
    amount = st.number_input(
        "Credit Amount (DM)",
        min_value=1,
        max_value=50000,
        value=5000
    )

with col3:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=35
    )


if st.button("Predict Credit Risk", type="primary"):
    with st.spinner("Calling ML API..."):

        # These must match the features used during training
        sample_features = [
            float(duration),
            float(amount),
            float(age)
        ]

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": sample_features}
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]

                if prediction == 0:
                    st.success("Low Risk - Customer is likely to be a good credit risk.")
                else:
                    st.error("High Risk - Customer is likely to be a bad credit risk.")

                st.info(f"Prediction value: {prediction}")

                if "probability" in result:
                    st.subheader("Prediction probability")

                    good_prob = result["probability"]["good_credit_risk"]
                    bad_prob = result["probability"]["bad_credit_risk"]

                    st.write(f"Good credit risk probability: {good_prob:.2%}")
                    st.write(f"Bad credit risk probability: {bad_prob:.2%}")

                st.subheader("Features sent to the model")
                st.write({
                    "duration_months": duration,
                    "credit_amount_DM": amount,
                    "age_years": age
                })

            else:
                st.error(f"API error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to ML API. Make sure the FastAPI server is running.")

        except Exception as e:
            st.error(f"Error: {str(e)}")


with st.expander("About the Model"):
    st.markdown("""
    - **Model**: Logistic Regression
    - **Dataset**: German Credit Risk Dataset
    - **Target**: 0 = Good credit risk, 1 = Bad credit risk
    - **Features used in this app**:
        - Credit duration, months
        - Credit amount, DM
        - Age, years
    """)


# 1st version
# import streamlit as st
# import requests
# import json
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# st.title("German Credit Risk Prediction")
# st.markdown("This app predicts whether a customer is a **good** (0) or **bad** (1) credit risk.")

# st.sidebar.header("Customer Information")

# col1, col2 = st.columns(2)

# with col1:
#     duration = st.number_input("Credit Duration (months)", min_value=0, max_value=72, value=24)
#     amount = st.number_input("Credit Amount (DM)", min_value=0, max_value=50000, value=5000)
#     age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)

# with col2:
#     purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"])
#     housing = st.selectbox("Housing", ["A151", "A152", "A153"])
#     job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])


# if st.button("Predict Credit Risk", type="primary"):
#     with st.spinner("Calling ML API..."):
        
#         sample_features = [duration, amount, age]
        
#         try:
#             response = requests.post("http://localhost:8000/predict", json={"features": sample_features})
#             # response = requests.post("http://fastapi-app:8000/predict", json={"features": sample_features})
            
#             if response.status_code == 200:
#                 prediction = response.json()["prediction"]
                
#                 if prediction == 0:
#                     st.success("✅ **Low Risk** - Customer is likely to be a good credit risk!")
#                     st.balloons()
#                 else:
#                     st.error("⚠️ **High Risk** - Customer is likely to be a bad credit risk!")
                
#                 st.info(f"Prediction value: {prediction}")
#             else:
#                 st.error(f"API error: {response.status_code}")
                
#         except requests.exceptions.ConnectionError:
#             st.error("Cannot connect to ML API. Make sure the FastAPI server is running.")
#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# with st.expander("ℹ️ About the Model"):
#     st.markdown("""
#     - **Model**: Logistic Regression
#     - **Dataset**: German Credit Risk (UCI)
#     - **Target**: 0 = Good credit risk, 1 = Bad credit risk
#     - **Features**: Various customer attributes (duration, credit amount, age, etc.)
#     """)