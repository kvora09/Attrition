# Attrition
Machine Learning Attrition model
## Overview
This project demonstrates an end-to-end machine learning solution to predict employee attrition for a fictitious organization inspired by real-world enterprise scenarios.

Since real employee data was not available, a synthetic dataset was generated using business-driven assumptions (compensation, engagement, career growth, workload, etc.) to simulate realistic attrition behavior.

The solution includes:

Data generation
Feature engineering
Model training & evaluation
Probability calibration
Deployment via Streamlit

## Problem Statement

Employee attrition is a critical business challenge impacting:

Operational stability
Hiring costs
Team productivity
### Objective:

To build a system that:

Predicts probability of attrition for each employee
Categorizes employees into risk buckets (Low → Critical)
Helps leadership take proactive retention actions

## Approach
#### 1. Synthetic Data Generation

A dataset of ~5000 employees was generated using:

Demographics (Age, Gender, Marital Status)
Career attributes (Tenure, Promotions, Band)
Compensation (Salary vs Market Rate)
Engagement (Satisfaction, Work-Life Balance)
Behavioral signals (Leaves, Overtime, Travel)

Attrition probability was modeled using a log-odds (logit) framework inspired by HR analytics.

#### 2. Feature Engineering

Key engineered features:

Compensation gap (underpaid vs market)
Stagnation index (promotion delay)
Leave velocity & spikes
Satisfaction composite score
Interaction features (e.g., Overtime × Low Satisfaction)

#### 3. Models used
i) Logistic Regression
ii) Decision Trees
iii) Random Forests
iv) Gradient Boosting
v) HistGradient Boosting

#### Evaluation Metrics
ROC-AUC (Primary metric)
F1 Score
Precision / Recall
Accuracy

#### 📁 attrition-project
│
├── 📄 Micro_Attrition_model.ipynb
├── 📄 train.py
├── 📄 app.py
├── 📁 artifacts/
│   ├── model_calibrated.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── columns.json
│   └── metadata.json
│
├── 📄 requirements.txt
└── 📄 README.md

#### File Explanations
###### 1. 📄 Micro_Attrition_Model.ipynb

This is the core notebook.

Contains:

Synthetic data generation logic
Business assumptions (coefficients, distributions)
Feature engineering
Model training & comparison
Calibration
Evaluation & visualizations

###### 2. 📄 train_models.py

Production-ready script that:

Loads dataset
Applies preprocessing
Trains selected model
Saves artifacts
Output:
Trained model
Scaler
Encoders
Feature metadata

👉 Used to prepare deployment artifacts

##### 3. 📄 app.py

Streamlit application.

Features:
User inputs employee details
Applies preprocessing
Predicts attrition probability
Displays:
Probability
Risk category

👉 This is the demo interface for stakeholders

##### 4.📄 requirements.txt

Contains all dependencies required to run the project.

## 📈 Future Improvements
Use real HR dataset
Add SHAP explainability
Build dashboard (Tableau/Power BI)
Integrate with HR systems
