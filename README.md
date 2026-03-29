# 🏋️ Customer Engagement & Attendance Prediction System for Fitness Businesses”

A web app that predicts whether a fitness club member will **attend** or **miss** their booked class, built using **Scikit-Learn** and **Streamlit**.

---

## 📋 Table of Contents

1. [Motivation](#motivation)  
2. [Project Overview](#project-overview)  
3. [Dataset](#dataset)  
4. [Modeling & Pipeline](#modeling-&-pipeline)  
5. [Streamlit App](#streamlit-app)  
6. [Usage / Demo](#usage--demo)  
7. [Project Structure](#project-structure)  
8. [Installation & Running Locally](#installation--running-locally)  
9. [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)  
10. [Future Improvements](#future-improvements)  
11. [Author & License](#author--license)

---

## 🧠 Motivation

- Fitness studios often face no-shows, resulting in resource wastage and lost revenue.  
- Predicting attendance allows opening up extra slots or notifying staff proactively.  
- This project aims to build an ML model and user interface for such predictions.

---

## 📊 Project Overview

- Use features about the member (e.g. membership duration, weight) and booking details (days before booking, class time, class category, day of week)  
- Train a classification model (Random Forest) to predict **0** = Miss / **1** = Attend  
- Deploy the model via a Streamlit app for real-time predictions  

---

## 🗃 Dataset

- CSV file (e.g. `fitness_class_2212.csv`) containing booking records  
- Key columns/features:
  - `booking_id` (identifier)  
  - `months_as_member`  
  - `weight`  
  - `days_before`  
  - `day_of_week`  
  - `time` (class time)  
  - `category` (class category: Cardio, Strength, Yoga, etc.)  
  - `attended` (target: 0 or 1)  

- Class distribution tends to be imbalanced (more misses than attends).

---

## 🔧 Modeling & Pipeline

- Preprocessing:
  - Drop or ignore `booking_id` (not predictive)  
  - One-hot encode categorical features (`day_of_week`, `time`, `category`)  
  - Optionally scale or impute numerical features  
- Model:
  - **RandomForestClassifier** (with `class_weight='balanced'`)  
  - Use cross-validation or grid search to tune hyperparameters  
- Training:
  - Split data (e.g. 80% train / 20% test, stratify on target)  
  - Evaluate using metrics: Accuracy, F1-score, ROC-AUC  
- Save the final pipeline (preprocessing + model) using `joblib`

---

## 🌐 Streamlit App

- File: `streamlit_app.py`  
- Loads the saved model (`best_fitness_rf_model.joblib`)  
- Takes user input: months_as_member, weight, days_before, day_of_week, time, category  
- Prepares DataFrame matching the model’s expected features  
- Runs `model.predict(...)` and `model.predict_proba(...)`  
- Displays:
  - Probability of attending  
  - Probability of missing  
  - Final decision (attend vs miss)  

---

## Usage / Demo

1. Launch locally:  
   ```bash
   streamlit run streamlit_app.py
