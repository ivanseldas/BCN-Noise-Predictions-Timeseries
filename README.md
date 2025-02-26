# BCN Noise Predictions Time Series
![image](https://github.com/user-attachments/assets/b833be16-b936-4133-a35f-fc082f52df1f)

## Overview

This project analyzes and forecasts noise levels in Barcelona, focusing on two main objectives:

   1. Predict future noise levels using historical data.
   2. Address the impact of COVID-19 lockdown on noise trends.
     
![image](https://github.com/user-attachments/assets/0bb6c886-3ebd-4205-a828-84005ac59333)

---

## Objectives

- Predict future noise trends for urban planning.
- Evaluate the impact of COVID-19 on noise levels.
- Compare pre- and post-COVID noise patterns.

![image](https://github.com/user-attachments/assets/d60656c2-34c5-4312-8e2b-05163c169e7e)

![image](https://github.com/user-attachments/assets/3d5b70ac-ebb1-441a-8cb6-d317e3c5c141)

---

## Data Source

This project utilizes data from the **[Noise Monitoring Network](https://opendata-ajuntament.barcelona.cat/data/en/dataset/xarxasoroll-equipsmonitor-dades)** provided by **[OPEN DATA BCN](https://opendata-ajuntament.barcelona.cat/en/)**, the open data portal of Barcelona City Council. The dataset is publicly available and subject to the licensing terms outlined on their website.

We acknowledge and thank OPEN DATA BCN for making this data available to the public.

---

## Data Structure

```bash 
s3://bcn-noise-project/
├── data/
│   ├── raw/                  # Datos crudos (inmutables)
│   │   ├── sensors/          # CSVs históricos de sensores
│   │   └── traffic/          # Datos de tráfico externos
│   ├── processed/            # Datos procesados (features)
│   │   ├── v1/               # Versión 1 del feature engineering
│   │   └── v2/               # Versión 2 (si hay cambios)
│   └── splits/               # Particiones train/val/test
│       ├── 2023-10-01/       # Fecha de creación
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── 2023-11-01/
│
├── models/
│   ├── experiments/          # Modelos de experimentación
│   │   ├── xgboost/
│   │   │   ├── 2023-10-01-12-00/  # Timestamp ejecución
│   │   │   │   ├── model.tar.gz    # Modelo serializado
│   │   │   │   ├── metrics.json    # RMSE, MAE, etc.
│   │   │   │   └── hyperparams.json
│   │   │   └── ...
│   │   └── randomforest/
│   ├── staging/              # Modelos candidatos a producción
│   └── production/           # Modelos en producción
│       ├── current/          # Versión activa
│       └── archive/          # Versiones anteriores
│
├── config/
│   ├── hyperparams/          # Hiperparámetros por modelo
│   │   ├── xgboost_v1.json
│   │   └── randomforest_v1.yaml
│   └── features/             # Configuración de features
│       ├── v1_features.txt
│       └── v2_features.txt
│
├── scripts/                  # Código reproducible
│   ├── preprocessing/
│   ├── training/
│   └── deployment/
│
└── logs/                     # Registros de ejecución
    ├── training/
    └── inference/
```

## Methodology

1. **Time Series Forecasting**:
   - Models: ARIMA, SARIMAX, Linear, Gradient Boosting, Random Forest, Decision Tree Regressors
   - Metrics: RMSE, MAPE, R2

---

## Results [ONGOING]

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
- Build an interactive dashboard for noise monitoring.

---

