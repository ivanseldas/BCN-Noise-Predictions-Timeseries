# BCN Noise Predictions Time Series
This project presents a complete data science workflow applied to real-world urban sensor data. The goal is to predict short-term variations in environmental noise levels using time series modeling, robust feature engineering, and automated deployment. It simulates the development of a real-time service that could be integrated into applications involving forecasting, guest assistance, or smart city analytics.

![image](https://github.com/user-attachments/assets/b833be16-b936-4133-a35f-fc082f52df1f)

## Overview

- Build a modular and traceable pipeline for time series prediction using real data.
- Train and evaluate ml models with backtesting strategies that reflect production constraints.
- Deploy the system using FastAPI and Streamlit with cloud-native infrastructure (AWS and GCP).

     
![image](https://github.com/user-attachments/assets/0bb6c886-3ebd-4205-a828-84005ac59333)

---

## Objectives

This project analyzes and forecasts noise levels in Barcelona, focusing on two main objectives:

   1. Predict future noise levels using historical data.

![image](https://github.com/user-attachments/assets/d60656c2-34c5-4312-8e2b-05163c169e7e)

   2. Address the impact of COVID-19 lockdown on noise trends.

![image](https://github.com/user-attachments/assets/3d5b70ac-ebb1-441a-8cb6-d317e3c5c141)

---

## Key Components

### 1. Data & Feature Pipeline

- Data Source:  **[Noise Monitoring Network](https://opendata-ajuntament.barcelona.cat/data/en/dataset/xarxasoroll-equipsmonitor-dades)** provided by **[OPEN DATA BCN](https://opendata-ajuntament.barcelona.cat/en/)**, the open data portal of Barcelona City Council. 
- Input: Noise sensor data (timestamp, location, dB levels)
- Feature engineering:
  - Temporal features: hour, weekday, month, weekend flag
  - Cyclical encoding (sin/cos)
  - Lag features (1h, 24h), rolling statistics (3h, 24h)
- Data validation and cleaning to ensure valid input for modeling

### 2. Modeling & Evaluation

- Models: Random Forest, Decision Trees (extensible design)
- Baselines: Persistence (last value), seasonal (24h lag)
- Backtesting:
  - Expanding-window, one-step-ahead predictions
  - Metrics: MAE, RMSE, relative improvement
- MLflow for experiment tracking, comparison, and production model selection

### 3. Serving & Visualization

- **FastAPI**: Serves real-time predictions and backtesting summaries
- **Streamlit**: Visual interface for model forecasts and error diagnostics
- Compatible with chat-based interfaces or voice-driven assistants

### 4. Deployment & Automation

- Containerized with Docker
- CI/CD via GitHub Actions:
  - Automatically builds and pushes Docker images to DockerHub
  - Deploys to **Google Cloud Run** (cost-effective and scalable)
- Also adaptable to AWS/Azure infrastructure

---

## Results 

- **Forecasts**: Predictive insights for future noise levels.
- **COVID Impact**: Quantified noise reduction during restrictions.

---

## Tools

- **Programming**: Python (Pandas, NumPy, Statsmodels, Scikit-Learn).
- **Visualization**: Folium, Matplotlib, Seaborn.

---

## Future Work

- Extend forecasting to all city sensors
- Integrate external factors like traffic and weather.

---