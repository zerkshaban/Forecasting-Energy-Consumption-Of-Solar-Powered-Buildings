# Forecasting Energy Consumption of Solar Powered Buildings

## Project Overview

This repository contains the source code and research documentation for my Bachelor of Computer Science thesis, **"Forecasting Energy Consumption Of Solar Powered Buildings,"** conducted at the **National University of Computer and Emerging Sciences (FAST-NUCES)**.

The project focuses on **Time-Series Forecasting** to predict energy usage patterns in buildings dependent on solar energy. By accurately forecasting consumption, this system aids in optimizing energy management strategies and ensuring the stability of renewable energy grids.

## Objectives

* To analyze historical energy consumption data from solar-powered structures.
* To implement and compare statistical, machine learning, and deep learning approaches for time-series forecasting.
* To identify the most effective model for predicting short-term and medium-term energy loads.

## Methodology & Algorithms

This project implements a comparative analysis of three distinct algorithms to handle the volatility and seasonality of energy data:

### 1. ARIMA (Auto Regressive Integrated Moving Average)

Used as a statistical benchmark, the **ARIMA** model was implemented to capture linear trends and seasonality within the time-series data. It proved highly effective for this specific dataset due to its ability to handle non-stationary data through differencing.

### 2. Random Forest

A **Random Forest** regression model was employed to handle non-linear relationships and interactions between variables. This ensemble learning method aggregates predictions from multiple decision trees to improve accuracy and control over-fitting.

### 3. LSTM (Long Short-Term Memory)

A Recurrent Neural Network (RNN) architecture, **LSTM**, was utilized to model long-term dependencies in the sequential energy data. This deep learning approach is designed to learn patterns over extended periods, making it suitable for complex time-series forecasting.

## Key Results

The research involved a rigorous performance evaluation of the three models.

* **Conclusion:** While the LSTM and Random Forest models demonstrated strong predictive capabilities, the **ARIMA** model provided the superior accuracy for this specific dataset, effectively managing the stochastic nature of the energy consumption profiles.

## Technologies Used

* **Language:** Python
* **Libraries:**
* `Statsmodels` (for ARIMA)
* `Scikit-learn` (for Random Forest and metrics)
* `Keras` / `TensorFlow` (for LSTM)
* `Pandas` & `NumPy` (Data manipulation)
* `Matplotlib` (Visualization)



## Authors & Acknowledgements

* **Authors:** Zerk Shaban, Shaleem John, Faisal Usman
* **Supervisor:** Dr. Usman Habib
* **Institution:** National University of Computer and Emerging Sciences (FAST-NUCES)
