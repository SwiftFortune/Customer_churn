import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Customer Churn EDA Dashboard", layout="wide")

st.title("📉 Customer Churn Data Analysis Dashboard")
st.markdown("""
This interactive dashboard helps explore patterns and insights from the **Telco Customer Churn** dataset.
Use the sidebar to upload your file or rely on the sample dataset.
""")

# -------------------------------
# Sidebar: Upload CSV
# -------------------------------
st.sidebar.header("🔽 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using default Telco Customer Churn dataset.")
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -------------------------------
# Data Cleaning
# -------------------------------
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Handle missing and inconsistent TotalCharges
if 'TotalCharges' in data.columns:
    data['TotalCharges'] = data['TotalCharges'].replace(" ", np.nan)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
    data = data.dropna(subset=['TotalCharges'])

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.subheader("🔍 Analysis Options")
show_raw = st.sidebar.checkbox("Show Raw Data", False)

if show_raw:
    st.subheader("📄 Raw Data")
    st.dataframe(data.head())

# -------------------------------
# Tabs for sections
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", 
    "📊 Categorical Analysis", 
    "⏳ Tenure Insights", 
    "🌐 Internet & Contract", 
    "💳 Payment & Charges", 
    "📉 Correlations"
])

# -------------------------------
# Overview Tab
# -------------------------------
with tab1:
    st.subheader("📈 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", data.shape[0])
    col2.metric("Features", data.shape[1])
    churn_rate = data['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

    st.markdown("### Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.countplot(x='Churn', data=data, palette='Set2', ax=ax)
    st.pyplot(fig)

    st.markdown("### Dataset Info")
    st.write(data.describe())

# -------------------------------
# Categorical Analysis
# -------------------------------
with tab2:
    st.subheader("📊 Churn by Categorical Features")
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    selected_col = st.selectbox("Select a categorical column", categorical_cols)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(x=selected_col, hue='Churn', data=data, palette='pastel', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------
# Tenure Insights
# -------------------------------
with tab3:
    st.subheader("⏳ Churn by Tenure Group")

    data['tenure_group'] = pd.cut(data['tenure'], 
                                  bins=[0, 12, 24, 48, 72], 
                                  labels=['0–12 months', '13–24 months', '25–48 months', '49–72 months'])
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(x='tenure_group', hue='Churn', data=data, palette='viridis', ax=ax)
    st.pyplot(fig)

# -------------------------------
# Internet & Contract Tab
# -------------------------------
with tab4:
    st.subheader("🌐 Internet Service and Contract Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.countplot(x='InternetService', hue='Churn', data=data, palette='Set3', ax=ax1)
        ax1.set_title("Churn by Internet Service")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.countplot(x='Contract', hue='Churn', data=data, palette='Spectral', ax=ax2)
        ax2.set_title("Churn by Contract Type")
        st.pyplot(fig2)

# -------------------------------
# Payment & Charges Tab
# -------------------------------
with tab5:
    st.subheader("💳 Payment Method & Charges")

    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.countplot(x='PaymentMethod', hue='Churn', data=data, palette='husl', ax=ax3)
        ax3.set_title("Churn by Payment Method")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(6,4))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=data, palette='coolwarm', ax=ax4)
        ax4.set_title("Monthly Charges vs Churn")
        st.pyplot(fig4)

    st.markdown("### Total Charges vs Churn")
    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.boxplot(x='Churn', y='TotalCharges', data=data, palette='coolwarm', ax=ax5)
    st.pyplot(fig5)

# -------------------------------
# Correlation Tab
# -------------------------------
with tab6:
    st.subheader("📉 Correlation Heatmap (Numerical Features)")

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(data[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Customer Churn Analysis Dashboard")
