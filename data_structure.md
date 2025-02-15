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
│
├── .dvc/
│   └── config                           # Configuración de DVC (data versioning)
│
├── README.md                            # Descripción del proyecto, instalación
├── .gitignore                           # Ignora data/raw/, models/, etc.
└── LICENSE                              # Licencia MIT, Apache, etc.