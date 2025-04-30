# 10Academy-Kifiya-Week-8-9

# **🚀 Building a Fraud Detection System: An In-Depth Look at Transaction Security**  

## **🚀 Introduction**  
Fraudulent financial transactions pose a major challenge for both e-commerce platforms and banking institutions. At **Adey Innovations Inc.**, a **robust fraud detection system** is being developed to analyze transaction patterns, identify anomalies, and build **machine learning models** capable of detecting fraud in real-time. With the rise of online transactions, fraudsters continuously find new ways to exploit vulnerabilities. Here is a sample **modular, scalable, and industry-standard fraud detection system** using **machine learning, deep learning, and interactive dashboards**.  

This project is divided into multiple phases, from **data preprocessing to model training and dashboard visualization**—while ensuring a streamlined and production-ready pipeline, providing an overview of the data preparation steps, key insights from exploratory data analysis (EDA), and the foundation built for machine learning models.  
---

## **The Challenge**  
I aimed to build an **end-to-end fraud detection system** that:  

✔ **Preprocesses transaction data** for effective model training.  
✔ **Builds multiple ML & deep learning models** for fraud detection.  
✔ **Tracks experiments** using MLflow for model comparison.  
✔ **Serves predictions** via an API for real-time fraud detection.  
✔ **Visualizes insights** using Flask and Dash.  

**Tech Stack:**  
- **Data Processing:** Pandas, NumPy, DBT  
- **ML Models:** Logistic Regression, Decision Trees, Random Forest, Gradient Boosting  
- **Deep Learning Models:** MLP, CNN, RNN, LSTM (PyTorch)  
- **Experiment Tracking:** MLflow  
- **Model Deployment:** Flask, Docker  
- **Dashboard:** Flask + Dash  

---

## **📊 Understanding the Data**  
Working with two key datasets:  

1️⃣ **E-commerce Fraud Data (`Fraud_Data.csv`)**  
- Contains transaction details, user attributes, device/browser information, and geolocation data.  
- Goal: Identify fraud cases using user behavior, purchase patterns, and IP-based geolocation.  

2️⃣ **Credit Card Fraud Data (`creditcard.csv`)**  
- Contains anonymized transaction features (`V1–V28` from PCA) and timestamps.  
- Goal: Detect fraudulent transactions based on these latent features.  

Additionally, `IpAddress_to_Country.csv` is used to **map IP addresses to countries**, helping us **analyze fraud patterns across locations**.  

---

## **Step 1: Data Analysis & Preprocessing**

Fraud detection starts with **clean and structured data**. I scraped and collected transaction data, then performed **feature engineering** to extract useful insights.  

### **🛠️ Key Data Preprocessing Steps**  

#### **1️⃣ Handling Missing Data**  
- Checked for missing values and **imputed or dropped** them as needed.  
- Ensured **data completeness before moving to feature engineering**.  

#### **2️⃣ Data Cleaning & Type Corrections**  
- Converted timestamps (`signup_time`, `purchase_time`) into **datetime format**.  
- Mapped `ip_address` values to corresponding countries.  
- Standardized categorical values (`browser`, `source`) for consistency.  

#### **3️⃣ Feature Engineering for Fraud Analysis**  
New features were extracted to enhance fraud detection:  
✅ **Transaction Velocity:** Number of transactions per user/device within a time window.  
✅ **Time-Based Features:** Hour of the day, day of the week (fraud patterns often follow time trends).  
✅ **Geolocation Analysis:** Linking transactions to country-level fraud trends.  

#### **4️⃣ Encoding & Scaling**  
- **Encoded categorical variables** (e.g., `source`, `browser`).  
- **Normalized purchase values** to ensure fair weightage in models.  

---

### **📈 Exploratory Data Analysis (EDA) & Key Insights**  

#### **📌 Correlation Analysis**  
**feature correlations** were examined to remove redundant variables and identify **key fraud indicators**.  
🔹 **Purchase value and transaction velocity** showed strong fraud-related trends.  
🔹 Credit card PCA features (`V1–V28`) showed **hidden patterns useful for fraud detection**.  

#### **📌 Fraud Trends Over Time**  
- **E-commerce fraud cases** spiked during certain hours (potential bot activity).  
- **Credit card fraud cases** showed **high concentration in specific transaction windows**.  

#### **📌 Device & Browser-Based Fraud**  
- Certain devices showed **disproportionate fraud cases**, indicating possible misuse.  
- Less popular browsers had **higher fraud rates**, suggesting attackers use obscure setups.  

#### **📌 Feature Importance Analysis**  
Using a **Random Forest model**, key fraud indicators were identified:  
✅ **For E-commerce:** `purchase_value`, `transaction_velocity`, `signup_time` patterns.  
✅ **For Credit Cards:** Specific **PCA-transformed features** heavily influenced fraud classification.  

---
#### **📌 Sample Preprocessing Code**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv("data/fraud_data.csv")

# Encode categorical variables
encoder = LabelEncoder()
df["device"] = encoder.fit_transform(df["device"])
df["browser"] = encoder.fit_transform(df["browser"])

# Scale numerical features
scaler = StandardScaler()
df[["amount", "transaction_time"]] = scaler.fit_transform(df[["amount", "transaction_time"]])

# Save preprocessed data
df.to_csv("data/processed_fraud_data.csv", index=False)
```
---
---

## **Step 2: Model Training (ML & Deep Learning)**  
To detect fraud, I trained multiple models and **experimented with different architectures**.  

### **🔹 ML Models Used**  
✅ **Logistic Regression** – Simple but effective for binary classification.  
✅ **Random Forest** – Robust against overfitting.  
✅ **Gradient Boosting** – Strong performance on structured data.  

### **🔹 Deep Learning Models (PyTorch)**  
✅ **MLP** – Fully connected neural network.  
✅ **CNN** – Feature extraction from tabular data.  
✅ **RNN & LSTM** – Sequence-based fraud detection.  

### **📌 Sample ML Model Code (Random Forest)**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load processed data
df = pd.read_csv("data/processed_fraud_data.csv")

# Define features and target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, "models/random_forest.pkl")
```
---

## **Step 3: Experiment Tracking with MLflow**  
To ensure reproducibility, I integrated **MLflow** for experiment tracking. MLflow helped in:  

✔ Comparing different models.  
✔ Logging hyperparameters and metrics.  
✔ Versioning trained models.  

### **📌 MLflow Integration**
```python
import mlflow
mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest")
```

---

## **Step 4: Real-Time Fraud Detection API**  
I developed a **Flask API** to serve model predictions in real time.  

### **📌 Sample API Endpoint**
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/random_forest.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"fraud": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
```

---

## **Step 5: Interactive Fraud Dashboard with Flask & Dash**  
To **visualize fraud trends**, I created an interactive dashboard using **Dash**.  

### **🔹 Dashboard Insights**
✔ **Total transactions, fraud cases & fraud percentages.**  
✔ **Line chart** showing fraud cases over time.  
✔ **Bar chart** comparing fraud cases across devices.  
✔ **Geographical fraud analysis.**  

### **📌 Sample Dashboard Code**
```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = pd.read_csv("data/fraud_data.csv")
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df_trends = df[df["is_fraud"] == 1].groupby(df["transaction_date"].dt.date).size().reset_index(name="Fraud Cases")

app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    dcc.Graph(figure=px.line(df_trends, x="transaction_date", y="Fraud Cases", title="Fraud Cases Over Time")),
])

if __name__ == "__main__":
    app.run_server(debug=True)
```

---

## **Final Thoughts & Key Takeaways**  
Building a **fraud detection system** requires **modularization, scalability, and industry best practices**. Here’s what I learned:  

✔ **Preprocessing is key** → Feature engineering significantly improves performance.  
✔ **Model diversity helps** → Combining ML and DL provides better fraud detection accuracy.  
✔ **Experiment tracking saves time** → MLflow ensures reproducibility.  
✔ **APIs enable real-time inference** → Flask serves predictions instantly.  
✔ **Dashboards provide insights** → Visualizing fraud trends helps decision-makers.  

### **🚀 Next Steps**  
🔹 **Optimize models further** using feature selection & hyperparameter tuning.  
🔹 **Deploy the API with Docker** for production readiness.  
🔹 **Improve dashboard UX** for better visualization.  

This project was an exciting challenge, and I’m excited to **scale it further**!
---
