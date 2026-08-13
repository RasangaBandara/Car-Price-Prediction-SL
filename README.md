# 🚗 Sri Lankan Used Car Price Prediction System

An end-to-end Machine Learning web application built with **Streamlit** and **Random Forest Regressor** to estimate used car market prices in Sri Lanka. The model accounts for economic market shifts, high-mileage depreciation curves, and outlier sanitization.

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

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Car-Price-Prediction-SL.git](https://github.com/YOUR_GITHUB_USERNAME/Car-Price-Prediction-SL.git)
cd Car-Price-Prediction-SL