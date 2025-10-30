# 📞 Customer Churn Prediction – Telco Dataset

## 🧠 Project Overview
This project aims to predict **customer churn** for a telecom company — identifying customers who are likely to discontinue the service.  
By analyzing customer demographics, account information, and service usage patterns, this model helps improve **retention strategies** and reduce revenue loss.

---

## 📂 Dataset Information
**Dataset Name:** WA_Fn-UseC_-Telco-Customer-Churn.csv  
**Source:** [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

### Key Columns:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents  
- **Account Info:** tenure, Contract, PaperlessBilling, PaymentMethod  
- **Service Usage:** InternetService, OnlineSecurity, StreamingTV, etc.  
- **Target Variable:** `Churn` (Yes/No)

---

## 🔍 Exploratory Data Analysis (EDA)
The dataset was analyzed to uncover key trends and correlations using **Matplotlib** and **Seaborn**.

### 🧩 Key Insights
- Senior citizens and customers with month-to-month contracts are more likely to churn.
- Customers without online security or tech support show higher churn rates.
- Churn is higher among customers paying via electronic checks.
- Long-tenure customers are more loyal and less likely to churn.

### 📊 Visualizations
The notebook includes:
- **Churn Distribution**
- **Gender vs Churn**
- **Contract Type vs Churn**
- **Internet Service vs Churn**
- **Monthly Charges Distribution**
- **Tenure vs Churn**
- **Correlation Heatmap**

> A variety of colorful and informative charts were added to make the EDA visually appealing.

---

## ⚙️ Data Preprocessing
Steps performed before modeling:
1. **Dropped unnecessary columns** (`customerID`)
2. **Handled missing values** in `TotalCharges`
3. **Encoded categorical features**
4. **Scaled numerical features**
5. **Split data** into training and testing sets

---

## 🤖 Machine Learning Models
Several models were trained and compared for churn prediction:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting  
- XGBoost  

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

---

## 🚀 Streamlit App
A **Streamlit web app** was developed for interactive churn prediction.  
Users can input customer details and instantly see whether the customer is likely to churn.

**App Features:**
- Interactive UI for input
- Real-time churn prediction
- Model accuracy display
- Data visualization and insights section

---

## 🧩 Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**
- **Streamlit**

---

