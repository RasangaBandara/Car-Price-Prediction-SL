# Group Members
Rasanga Bandara    - ITBIN-2312-0002
Mindula Deenamulla - ITBNM-2313-0074
Dilki Ishari       - ITBIN-2312-0009
Mandulee Laknara   - ITBIN-2312-0018

# 🚗 Sri Lankan Used Car Price Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rasangabandara-car-price-prediction-sl.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning web application built with **Streamlit** and a calibrated **Random Forest Regressor** to estimate secondary automobile market prices in Sri Lanka. The model accounts for local market realities, high-mileage wear curves, import ban distortions, and outlier sanitization.

🔗 **Live Web Application:** [rasangabandara-car-price-prediction-sl.streamlit.app](https://rasangabandara-car-price-prediction-sl.streamlit.app/)

---

## 📌 Project Overview
The Sri Lankan secondary automobile market faces high volatility due to economic factors and import restrictions. Traditional pricing models often fail due to non-linear relationships between vehicle age, mileage, and brand value. 

This project implements an intelligent prediction pipeline featuring:
* **Target Encoding** for high-cardinality categorical variables (`Brand`, `Model`, `Town`).
* **Quantile Outlier Filtering ($0.01 - 0.99$)** to strip out down-payment-only or invalid listings.
* **Live Market Calibration Layer** to adjust for high mileage and market caps.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.x
* **Machine Learning Library:** `scikit-learn` (Random Forest Regressor)
* **Data Processing:** `pandas`, `numpy`
* **Web Framework:** `streamlit`
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 📁 Repository Structure

```text
Car-Price-Prediction-SL/
│
├── car_price_dataset.csv      # Primary vehicle dataset
├── web_app.py                 # Main Streamlit web application
├── requirements.txt           # Python dependency manifest
├── .gitignore                 # Git ignore configuration
└── README.md                  # Project documentation
