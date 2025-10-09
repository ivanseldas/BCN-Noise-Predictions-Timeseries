# Urban Noise Forecasting - Barcelona

### What if your ML model could hear the city and predict its next move? 

This project builds a **spatio-temporal forecasting system** that predicts **urban noise** in Barcelona by combining **machine learning** with **geospatial analysis**.  
It integrates **ArcGIS, GeoPandas, and NetworkX** to engineer spatial context features (roads, parks, network centrality), allowing the model to “understand” the city’s physical structure before making predictions.


![image](https://github.com/user-attachments/assets/b833be16-b936-4133-a35f-fc082f52df1f)

### Project Overview:

**Objective:** Forecast urban noise patterns and identify potential exceedances (> 65 dB) before they occur.  
- **Approach:** Combine **temporal forecasting models** with **geospatial context layers** to improve prediction accuracy.  
- **Scale:** Over **135 M sensor records** processed and stored in a **BigQuery + Cloud Storage** data lake.  
- **Stack:** Python (GeoPandas, Shapely, ArcGIS API, NetworkX, MLflow), Docker, GitHub Actions, Google Cloud Run.  


**Live Demo:** [Noise Forecasting App: sensor 496](https://noise-forecasting-frontend-924171883482.europe-west1.run.app/)

![image](https://github.com/user-attachments/assets/0bb6c886-3ebd-4205-a828-84005ac59333)

---

## Geospatial Analysis & Feature Engineering

The **geospatial workflow** forms the core of the project, enriching each sensor with **urban context** before feeding data into the forecasting model.

### Spatial Context Layers
- Distance to **main roads**, **green areas**, and **transport corridors** using **GeoPandas** + **Shapely**  
- Street-network **betweenness centrality** with **NetworkX** + **OSMnx**  
- Local noise environment: neighborhood mean / variance within **150 m buffers**  
- Integration of all spatial features into a unified **GeoDataFrame** exported to the ML pipeline  

### GIS Processing
- Automated spatial joins and geometry operations via **ArcPy** and **ArcGIS API for Python**  
- Publication of **geospatial layers** and **forecasted exceedances** as interactive **ArcGIS Online** maps  
- Visualization of predicted hot zones across Barcelona districts  

### Hotspot & Spatial Statistics
- Detection of emerging noise clusters via **Getis-Ord Gi\*** hotspot analysis and **Moran’s I** autocorrelation in **ArcGIS**  
- Comparison of predicted vs. observed hotspots for model validation  

**ArcGIS Online Map:** [Noise Hotspot Analysis](https://arcg.is/1Du8bG3)
![bcn-noise-hotspot-map](image-1.png)
*ArcGIS Online Noise Hotspot Map*

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
  - Deploys to **Google Cloud Run** (previously deployed to **AWS Fargate (ECS)** but discontinued for cost-efficiency)
- Also adaptable to AWS/Azure infrastructure

---

## Results & Business Impact

### Model Performance (Backtesting)
- **RMSE:** 3.01 dB → ~70% lower error than the naive baseline (14.87 dB)
- **MAE:** 1.19 dB → below the human perception threshold (~3 dB)
- **sMAPE:** 1.75% | **MASE:** 0.80 → consistently better than persistence models
- **Interval accuracy:** ~93–95% coverage on 95% confidence → reliable, with room to fine-tune

![image](https://github.com/user-attachments/assets/f9915867-cdff-4484-93ae-87b0dab6c0fd)

### Business Impact
- Stable and low-error forecasts (~1.2 dB) across time, even during noise pattern shifts.
- Outperforms baselines, enabling reliable detection of exceedances (e.g., >65 dB).

![image](https://github.com/user-attachments/assets/93c53fd7-30ea-4295-9168-a334f7c864b3)

---

## Future Work

- Extend forecasting to all city sensors
- Integrate external factors like traffic and weather.

---