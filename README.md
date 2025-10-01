# BCN Noise Predictions Time Series

**What if your ML model could hear the city and predict its next move?**  

This project uses historical noise sensor data to forecast the city noise through a machine learning pipeline ready for production. The system makes real-time predictions via FastAPI, providing visual insights through a Streamlit dashboard.

![image](https://github.com/user-attachments/assets/b833be16-b936-4133-a35f-fc082f52df1f)

### Project Overview:

- Forecasting of real-world urban noise using engineered temporal features
- Modular pipeline with MLflow for experiment tracking and model governance
- Time-aware backtesting with expanding window evaluation and baseline comparisons
- FastAPI service for real-time predictions, and Streamlit dashboard for insights
- CI/CD workflow with Docker and GitHub Actions, auto-deployed to **Google Cloud Run**
- Built for cost-efficiency, observability, and portability across cloud platforms

**Live Demo:** [Noise Forecasting App: sensor 496](https://noise-forecasting-frontend-924171883482.europe-west1.run.app/)

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

![image](https://github.com/user-attachments/assets/3ca35db1-d2ce-4f06-be47-37390870911c)
*MLFlow Experiments saved locally for cost efficiency*

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

![image](https://github.com/user-attachments/assets/93c53fd7-30ea-4295-9168-a334f7c864b3)

---

## Tools

- **Programming**: Python (Pandas, NumPy, Statsmodels, Scikit-Learn).
- **Visualization**: Folium, Matplotlib, Seaborn.

---

## Future Work

- Extend forecasting to all city sensors
- Integrate external factors like traffic and weather.

---