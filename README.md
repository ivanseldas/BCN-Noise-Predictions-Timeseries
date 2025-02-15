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
barcelona-noise-prediction/
│
├── data/
│   ├── raw/
│   │   └── sensor_data.csv             # Datos crudos del sensor (original)
│   ├── processed/
│   │   └── cleaned_sensor_data.csv      # Datos limpios y normalizados
│   ├── external/
│   │   └── weather_data.csv             # Datos climáticos externos
│   ├── interim/
│   │   └── temp_features.pkl           # Características temporales
│   └── outputs/
│       └── predictions_2023.csv         # Predicciones del modelo
│
├── src/
│   ├── data_ingestion/
│   │   └── download_data.py             # Script para descargar datos de una API
│   ├── preprocessing/
│   │   └── clean_data.py                # Limpia y transforma datos crudos
│   ├── features/
│   │   └── feature_engineering.py       # Crea nuevas características
│   ├── models/
│   │   ├── train_model.py               # Entrena un modelo de ML
│   │   └── evaluate_model.py            # Evalúa el rendimiento del modelo
│   ├── visualization/
│   │   └── plot_results.py              # Genera gráficos de resultados
│   ├── utils/
│   │   └── helpers.py                   # Funciones auxiliares (ej: logger)
│   └── pipelines/
│       └── data_pipeline.py             # Flujo ETL automatizado
│
├── notebooks/
│   ├── exploratory/
│   │   └── 01_exploratory_analysis.ipynb  # Análisis inicial de datos
│   └── reports/
│       └── 02_final_report.ipynb          # Reporte con conclusiones
│
├── models/
│   ├── production/
│   │   └── best_model.pkl               # Modelo listo para producción
│   └── experiments/
│       └── experiment_001.pkl           # Modelo de prueba (hiperparámetros)
│
├── config/
│   ├── paths.yaml                       # Rutas: data/raw, models/, etc.
│   ├── model_params.yaml                # Parámetros: learning_rate=0.01, epochs=100
│   └── constants.py                     # Constantes: MAX_TEMPERATURE=100
│
├── tests/
│   ├── unit/
│   │   └── test_preprocessing.py        # Pruebas de limpieza de datos
│   └── integration/
│       └── test_pipeline.py             # Prueba del pipeline completo
│
├── docs/
│   ├── technical/
│   │   └── architecture.md              # Diseño técnico del sistema
│   └── user_guides/
│       └── how_to_train.md              # Guía para entrenar el modelo
│
├── reports/
│   ├── figures/
│   │   ├── loss_curve.png               # Gráfico de pérdida del modelo
│   │   └── feature_importance.png       # Importancia de características
│   └── presentations/
│       └── project_results.pptx         # Presentación ejecutiva
│
├── environment/
│   ├── requirements.txt                 # pandas==1.5.3, scikit-learn==1.2.2
│   ├── environment.yml                  # Entorno Conda (python=3.9)
│   └── Dockerfile                       # Configuración para Docker
│
├── references/
│   └── paper_sensor_analysis.pdf        # Artículo científico relacionado
│
├── scripts/
│   └── deploy_model.sh                  # Script para desplegar el modelo
│
├── app/
│   ├── api/
│   │   └── main.py                      # API FastAPI para predicciones
│   └── dashboard/
│       └── app.py                       # Dashboard Streamlit
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml                    # Automatización de pruebas y despliegue
```
---

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

