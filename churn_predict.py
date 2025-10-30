import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# -------------------------------
# Custom CSS for styling
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f9fbfc;
    }
    .main-title {
        font-size: 38px;
        color: #003366;
        text-align: center;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #4f4f4f;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-top: 25px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model and Scaler
# -------------------------------
model = joblib.load('customer_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------------
# Header
# -------------------------------
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331966.png", width=120)
st.markdown('<p class="main-title">📉 Customer Churn Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict whether a customer is likely to leave based on their information.</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Layout: Sidebar + Main
# -------------------------------
st.sidebar.header("🧾 Input Customer Details")

# Sidebar Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 10)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges (for reference)", min_value=0.0, max_value=10000.0, value=3000.0)

# -------------------------------
# Create input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# -------------------------------
# Manual Encoding (match training)
# -------------------------------
encoding_map = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

for col in encoding_map.keys():
    mapping = {label: idx for idx, label in enumerate(encoding_map[col])}
    input_data[col] = input_data[col].map(mapping)

# -------------------------------
# Prepare for Prediction
# -------------------------------
X_input = input_data.drop('TotalCharges', axis=1)
X_scaled = scaler.transform(X_input)

# -------------------------------
# Prediction Section
# -------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=250)
with col2:
    if st.button("🔍 Predict Churn", use_container_width=True):
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class='result-card'>
                <h3 style='color:#b30000;'>⚠️ Customer is <b>likely to churn</b></h3>
                <p>Churn Probability: <b>{proba:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/620/620851.png", width=150)
        else:
            st.markdown(f"""
            <div class='result-card'>
                <h3 style='color:#008000;'>✅ Customer is <b>not likely to churn</b></h3>
                <p>Churn Probability: <b>{proba:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=150)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<div class='footer'>Built with ❤️ using Streamlit, Scikit-learn, and Python.</div>", unsafe_allow_html=True)
