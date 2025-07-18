# ev-adoption-forecast
project of Electric Vehicle (EV) internship focused on forecasting EV adoption using historical vehicle registration data.
# 🚗 EV Adoption Forecasting ( Internship Project)

This project is part of the **Electric Vehicle (EV) Internship**. The goal is to analyze electric vehicle registration trends and build a forecasting model to predict future EV adoption based on historical data.

---

## 📌 Problem Statement

Electric Vehicles are rapidly gaining popularity due to rising environmental concerns and government policies. This project aims to:

- Analyze EV registration trends over time
- Handle missing values and outliers in the dataset
- Build a regression model to **forecast future EV adoption**
- Visualize EV growth patterns and predictions

---

## 📊 Dataset Description

The dataset contains EV registration data across various counties in the United States, from **2017 to 2024**.

**Columns include:**
- `Date` – Registration date (monthly)
- `County`, `State`
- `Battery Electric Vehicles (BEVs)`
- `Plug-In Hybrid Electric Vehicles (PHEVs)`
- `Electric Vehicle (EV) Total`
- `Non-Electric Vehicle Total`
- `Total Vehicles`
- `Percent Electric Vehicles`

> **Source:** [Kaggle Dataset – Electric Vehicle Population Size 2024](https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population-size-2024)

---

## 🧰 Technologies & Tools Used

- Python
- Google Colab
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Joblib (for model saving)

---

## 🔁 Workflow Steps

1. **Data Cleaning**
   - Removed missing values from `County` and `State`
   - Converted string-formatted numbers (e.g., `"1,234"`) to integers
   - Capped outliers in `Percent Electric Vehicles` using IQR

2. **Data Aggregation**
   - Grouped data monthly using `Date` to track EV growth over time

3. **Visualization**
   - Line plots showing historical EV adoption

4. **Feature Engineering**
   - Created `TimeIndex` as a proxy for date in time-series forecasting

5. **Model Training**
   - Used `RandomForestRegressor` to model `EV Total` over time

6. **Forecasting**
   - Predicted EV adoption for the next 12 months

7. **Model Evaluation**
   - MAE, RMSE, and R² Score used to evaluate prediction accuracy

8. **Model Export (Optional)**
   - Saved the trained model using `joblib`

---

## 📈 Sample Forecast Plot

*Forecasting EV adoption for the next 12 months using trained model.*

![Forecast Plot](forecast_plot.png)

---

## 📂 File Structure
📁 ev-adoption-forecast-week1/
├── EV_Adoption_Forecasting_Week1.ipynb # Main notebook
├── ev_data.csv # Dataset used (if included)
├── forecast_plot.png # Visualization output
├── ev_forecast_model.pkl # Trained model (optional)
└── README.md # This file
---

## 🙋‍♀️ Author

**Syed Galib**  
Final Year B.Tech CSE Student  
AI & ML + EV Internship Trainee
