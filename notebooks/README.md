# Barcelona Noise Prediction Project

This project aims to predict noise levels in Barcelona using historical data from multiple sensors. The pipeline includes data collection, ETL, exploratory data analysis (EDA), predictive modeling, and visualization.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Steps](#steps)
4. [Tools and Technologies](#tools-and-technologies)
5. [How to Run](#how-to-run)
6. [Cost Optimization](#cost-optimization)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

The goal of this project is to analyze and predict noise levels in Barcelona using open-source data from multiple sensors. The project is divided into the following phases:

1. **Data Collection**: Gather raw noise data from sensors.
2. **ETL**: Clean, transform, and store data in S3.
3. **EDA**: Perform exploratory data analysis to identify patterns and missing data.
4. **Modeling**: Train predictive models using SageMaker.
5. **Visualization**: Create interactive dashboards with QuickSight.
6. **Monitoring**: Set up real-time alerts for noise anomalies.

---

## Architecture

```mermaid
graph TD
    A[1.Data Collection] --> B[2.Raw Data in S3]
    B --> C[3.ETL with AWS Glue]
    C --> D[4.Clean Data in S3]
    D --> E[5.EDA in SageMaker]
    E --> F[6.Predictive Modeling]
    F --> G[7.Visualization in QuickSight]
    G --> H[8.Monitoring & Alerts]