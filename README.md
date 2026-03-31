# 📊 Attrition Prediction Model

A complete **end-to-end Machine Learning solution** to predict employee attrition for a fictitious organization, designed to simulate real-world enterprise scenarios.

## 🚀 Overview

This project demonstrates how data and machine learning can be used to proactively identify employees at risk of leaving.

Since real employee data was unavailable, a **synthetic dataset** was generated using realistic business assumptions around:

* Compensation
* Career growth
* Engagement
* Work patterns

The solution includes:

* Data Generation
* Feature Engineering
* Model Training & Evaluation
* Probability Calibration
* Deployment via Streamlit

---

## 🎯 Problem Statement

Employee attrition is a major business challenge impacting:

* Operational stability
* Hiring & replacement costs
* Team productivity

### Objective

Build a system that:

* Predicts **probability of attrition**
* Categorizes employees into **risk buckets (Low → Critical)**
* Enables **proactive retention strategies**

---

## 🧠 Approach

### 1️⃣ Synthetic Data Generation

A dataset of ~5000 employees was created using:

* **Demographics**: Age, Gender, Marital Status
* **Career Attributes**: Tenure, Promotions, Level Band
* **Compensation**: Salary vs Market Rate
* **Engagement**: Satisfaction, Work-Life Balance
* **Behavioral Signals**: Leaves, Overtime, Travel

Attrition probability was modeled using a **log-odds (logit) framework** inspired by HR analytics.

---

### 2️⃣ Feature Engineering

Key engineered features:

* Compensation gap (underpaid vs market)
* Stagnation index (promotion delays)
* Leave velocity & spike indicators
* Satisfaction composite score
* Interaction features (e.g., Overtime × Low Satisfaction)

---

### 3️⃣ Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* HistGradient Boosting

---

### 📏 Evaluation Metrics

* ROC-AUC (Primary metric)
* F1 Score
* Precision & Recall
* Accuracy

---

## 🏗️ Project Structure

```
attrition-project/
│
├── Micro_Attrition_Model.ipynb
├── train.py
├── app.py
├── artifacts/
│   ├── model_calibrated.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── columns.json
│   └── metadata.json
│
├── requirements.txt
└── README.md
```

---

## 📂 File Descriptions

### 1. Micro_Attrition_Model.ipynb

Core development notebook containing:

* Data generation logic
* Business assumptions
* Feature engineering
* Model training & comparison
* Calibration
* Evaluation & visualizations

---

### 2. train.py

Production-ready script that:

* Loads data
* Applies preprocessing
* Trains model
* Saves artifacts

**Outputs:**

* Trained model
* Scaler
* Encoders
* Feature metadata

---

### 3. app.py

Streamlit-based interactive application.

**Features:**

* User inputs employee details
* Applies preprocessing
* Predicts attrition probability
* Displays:

  * Probability score
  * Risk category (Low → Critical)

---

### 4. requirements.txt

Contains all dependencies required to run the project.

---

## 🌐 Live Demo

👉 [https://attritionmodel.streamlit.app/](https://attritionmodel.streamlit.app/)

---

## 🔮 Future Improvements

* Use real HR datasets
* Add SHAP explainability
* Build dashboards (Tableau / Power BI)
* Integrate with HR systems

---

## 💡 Key Highlights

* End-to-end ML pipeline (data → deployment)
* Business-driven synthetic data modeling
* Probability calibration for realistic predictions
* Production-ready architecture (train.py + app.py)
* Interactive UI for decision-making

