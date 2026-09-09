# Swiggy Delivery Time Prediction

## 📌 Project Overview

This project focuses on predicting the **delivery time of Swiggy orders** using Machine Learning. The model uses various factors related to the delivery partner, vehicle, traffic, weather, and distance to estimate the time required to deliver an order.

## 🎯 Objective

The main objective is to build a machine learning model that can accurately predict the **estimated delivery time in minutes** based on different delivery-related features.

## 📊 Features Used

* Age of Delivery Partner
* Delivery Partner Ratings
* Vehicle Condition
* Type of Vehicle
* Weather Condition
* Traffic Condition
* Distance
* Pickup Time

### Target Variable

**Time_taken** – Actual time taken for delivery.

## 🤖 Machine Learning Model

The project uses **XGBoost Regressor**, a powerful boosting algorithm suitable for regression problems.

The data is first cleaned and preprocessed, followed by feature selection and model training.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
XGBoost Regressor
   ↓
Model Evaluation
   ↓
Delivery Time Prediction
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Jupyter Notebook

## 📁 Project Files

```text
Swiggy-Delivery-Prediction/
│
├── Swiggy Delivery Time Prediction.ipynb
├── swiggy_demographic (1).csv
├── model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## 📈 Expected Outcome

The trained model predicts the estimated delivery time based on the given order and delivery partner information. This can help understand the major factors affecting food delivery time.

## 🚀 Future Improvements

* Improve prediction accuracy through hyperparameter tuning.
* Add real-time traffic data.
* Include GPS/location-based features.
* Add more delivery-related parameters.
* Improve the model using larger and more recent datasets.

## 👨‍💻 Author

**Vallabh Kulkarni**

**Project:** Swiggy Delivery Time Prediction using Machine Learning
