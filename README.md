-----

# 🚗 EV Adoption Forecasting: An Internship Project

This project, developed as part of an **Electric Vehicle (EV) Internship**, focuses on analyzing historical EV registration trends and building a robust forecasting model. Our primary goal is to predict future EV adoption rates based on comprehensive historical data, offering insights into the evolving landscape of electric mobility.

-----

## 📌 Problem Statement

The rapid surge in Electric Vehicle adoption, driven by growing environmental concerns and supportive government policies, necessitates accurate forecasting. This project addresses the critical need to:

  * **Analyze EV registration trends** over time to understand growth patterns.
  * **Implement robust data preprocessing** techniques, including handling missing values and outlier detection.
  * **Develop a regression model** capable of accurately forecasting future EV adoption.
  * **Visualize EV growth patterns and predictions** for clear, actionable insights.

-----

## 📊 Dataset Overview

Our analysis utilizes a comprehensive dataset containing EV registration data across various counties in the United States, spanning from **2017 to 2024**.

**Key Columns Include:**

  * **Date:** Monthly registration date.
  * **County, State:** Geographic location of registration.
  * **Battery Electric Vehicles (BEVs):** Number of BEV registrations.
  * **Plug-In Hybrid Electric Vehicles (PHEVs):** Number of PHEV registrations.
  * **Electric Vehicle (EV) Total:** Combined total of BEVs and PHEVs.
  * **Non-Electric Vehicle Total:** Number of traditional vehicle registrations.
  * **Total Vehicles:** Sum of EV and Non-EV registrations.
  * **Percent Electric Vehicles:** Percentage of EVs out of total vehicles.

> **Source:** [Kaggle Dataset – Electric Vehicle Population Size 2024](https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population-size-2024)

-----

## 🧰 Technologies & Tools Used

  * **Python:** The core programming language for data analysis and model development.
  * **Google Colab:** Cloud-based environment for interactive notebook development.
  * **Pandas & NumPy:** Essential libraries for data manipulation and numerical operations.
  * **Seaborn & Matplotlib:** Powerful tools for creating insightful data visualizations.
  * **Scikit-learn:** Comprehensive library for machine learning, used for model training and evaluation.
  * **Joblib:** Utilized for efficient saving and loading of the trained machine learning model.

-----

## 🔁 Project Workflow & Methodology

Our approach involved a structured workflow to ensure data integrity, effective modeling, and accurate forecasting:

1.  **Data Cleaning & Preprocessing:**
      * Addressed missing values in `County` and `State` columns.
      * Converted string-formatted numerical data (e.g., "1,234") to appropriate integer types.
      * Applied **Interquartile Range (IQR)** method to cap outliers in `Percent Electric Vehicles`, ensuring data robustness.
2.  **Data Aggregation:**
      * Grouped the dataset monthly by `Date` to analyze and track EV growth trends over time.
3.  **Exploratory Data Visualization:**
      * Generated **line plots** to visually represent historical EV adoption trends and patterns.
4.  **Feature Engineering:**
      * Created a `TimeIndex` feature, serving as a numerical proxy for dates to facilitate time-series forecasting.
5.  **Model Training:**
      * Utilized **RandomForestRegressor** to build a robust model for predicting `EV Total` over time, leveraging its ability to capture non-linear relationships.
6.  **Forecasting:**
      * Generated future predictions for EV adoption for the **next 12 months** based on the trained model.
7.  **Model Evaluation:**
      * Assessed the model's predictive accuracy using key metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
8.  **Model Export (Optional):**
      * Saved the trained `RandomForestRegressor` model using `joblib` for future deployment or analysis.

-----

## 📈 Forecasting Results

The trained model successfully forecasts EV adoption for the upcoming 12 months, providing a clear visual representation of predicted growth:

-----

## 📂 Project Structure

```
📁 ev-adoption-forecast/
├── EV_Adoption_Forecasting.ipynb   # Main Jupyter Notebook with code and analysis
├── ev_data.csv                     # Original dataset used for the project
├── forecast_plot.png               # Generated visualization of the forecast
├── ev_forecast_model.pkl           # Saved trained model (optional)
└── README.md                       # This comprehensive project description
```

-----

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/syedgalib4/ev-adoption-forecast.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd ev-adoption-forecast
    ```
3.  **Open the notebook:**
      * Upload `EV_Adoption_Forecasting.ipynb` to **Google Colab** and run the cells sequentially.
      * Alternatively, open it with **Jupyter Notebook** or **JupyterLab** if you have the necessary Python environment set up.

-----

## 🙋‍♀️ Author

**Syed Galib** Final Year B.Tech CSE Student  
AI & ML + EV Internship Trainee

-----
