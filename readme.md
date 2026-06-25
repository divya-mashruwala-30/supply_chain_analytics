# 📦 Supply Chain Analytics

An end-to-end Machine Learning project that predicts **Late Delivery Risk** for supply chain orders and performs **Customer Segmentation** using unsupervised learning. The project also includes a **Streamlit web application** for interactive predictions and customer segmentation.

---

## 🚀 Project Overview

Efficient supply chain management is crucial for improving customer satisfaction and reducing operational costs. This project addresses two important business problems:

1. **Late Delivery Risk Prediction**
   - Predict whether an order is likely to be delivered late.
   - Helps logistics teams identify high-risk shipments and take proactive actions.

2. **Customer Segmentation**
   - Group customers based on purchasing behaviour.
   - Enables targeted marketing, loyalty programs, and personalized business strategies.

---

## 📂 Project Structure

```text
Supply_Chain_Analytics/
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_Late_Delivery_Prediction.ipynb
│   └── 03_Customer_Segmentation.ipynb
│
├── app/
│   ├── streamlit_app.py
│   ├── classifier.pkl
│   ├── customer_segmentation_kmeans.pkl
│   ├── customer_segmentation_scaler.pkl
│   └── categories.pkl
│
├── requirements.txt
├── README.md
└── dataset/
```

---

# 📊 Dataset

**Dataset:** DataCo Smart Supply Chain Dataset

The dataset contains information related to:

- Customer Information
- Product Details
- Order Information
- Shipping Details
- Sales
- Discounts
- Delivery Status

### Dataset Size

- **Orders:** 180,519
- **Customers:** 20,652

---

# 🎯 Business Problems

## 1️⃣ Late Delivery Risk Prediction

Objective:

Predict whether an order is likely to be delivered late.

**Target Variable**

- Late_delivery_risk
  - 1 → Late Delivery
  - 0 → On-Time Delivery

### Feature Engineering

Some engineered features include:

- Discount_to_Sales
- Customer_Total_Orders
- Order_Month
- Order_DayOfWeek
- Is_Weekend

### Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest ✅
- XGBoost
- CatBoost

### Final Model

**Random Forest Classifier**

| Metric | Score |
|---------|-------|
| Accuracy | 81.36% |
| Precision | 88.84% |
| Recall | 75.50% |
| F1 Score | 81.62% |
| ROC-AUC | 0.91 |

---

# 👥 Customer Segmentation

Customer-level aggregation was performed to create behavioural features.

### Features Used

- Customer_Total_Orders
- Order_Item_Total
- Order_Item_Quantity
- Order_Item_Discount

### Clustering Algorithms Evaluated

- K-Means ✅
- Hierarchical Clustering
- DBSCAN

### Selecting the Number of Clusters

The optimal number of clusters was determined using:

- Elbow Method
- Silhouette Score
- Hierarchical Clustering (Dendrogram)
- DBSCAN Comparison

Final choice:

**K = 4**

### Customer Segments

| Segment | Characteristics |
|----------|----------------|
| ⭐ Premium Customers | Highest spending and purchase frequency |
| 🛒 Frequent Customers | Regular purchases with high spending |
| 👥 Regular Customers | Moderate purchasing behaviour |
| 🌱 Occasional Customers | Low spending and infrequent purchases |

---

# 🌐 Streamlit Application

The project includes a Streamlit application with three sections.

### 🏠 Home

- Project Overview
- Dataset Information
- ML Workflow

### 🚚 Late Delivery Prediction

Predicts whether an order has a high or low risk of late delivery.

### 👥 Customer Segmentation

Predicts the customer segment based on purchasing behaviour.

---

# 🛠 Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Supply-Chain-Analytics.git
```

Move into the project directory

```bash
cd Supply-Chain-Analytics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

# 📈 Project Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ├──────────────┐
      ▼              ▼
Classification   Clustering
(Random Forest)  (K-Means)
      │              │
      ▼              ▼
Late Delivery   Customer Segments
Prediction
      │
      ▼
Business Insights
```

---

# 💼 Business Impact

### Late Delivery Prediction

- Identify high-risk shipments
- Improve logistics planning
- Reduce delivery delays
- Increase customer satisfaction

### Customer Segmentation

- Personalized marketing campaigns
- Customer retention
- Loyalty programs
- Better business decisions

---

# 📌 Future Improvements

- Deploy the application on Streamlit Community Cloud.
- Add interactive dashboards with Plotly.
- Integrate real-time order prediction using APIs.
- Add explainability using SHAP values.
- Automate model retraining with new data.

---

# 👨‍💻 Author

**Divya Mashruwala**

- LinkedIn: *Add your LinkedIn profile*
- GitHub: *Add your GitHub profile*

---

## ⭐ If you found this project useful, consider giving it a star!
