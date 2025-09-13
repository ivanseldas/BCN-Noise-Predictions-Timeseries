import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Noise Forecasting Dashboard", layout="wide")

st.title("Noise Forecasting in Barcelona")
st.markdown("Noise levels predicted by the model in **MLflow Production**")

# ----------------------------------------
# Section: Last 7 days predictions
# ----------------------------------------
st.subheader("Predicted Noise Levels - Last 7 Days")

try:
    response = requests.get("http://127.0.0.1:8000/predict_now/")
    if response.status_code == 200:
        result = response.json()

        # Convert predictions to DataFrame
        df_week = pd.DataFrame(result["predictions"])
        df_week["datetime"] = pd.to_datetime(df_week["datetime"])

        # Line chart with Plotly
        fig = px.line(
            df_week,
            x="datetime",
            y="prediction",
            title="Noise Level Predictions (dB) - Last 7 Days",
            labels={"datetime": "Date", "prediction": "Noise (dB)"}
        )
        fig.update_traces(mode="lines+markers")

        st.plotly_chart(fig, use_container_width=True)

        # Show summary
        st.write(f"{result['total_points']} predictions from {result['start']} to {result['end']}")

    else:
        st.error("Error calling FastAPI endpoint `/predict_now/`")

except Exception as e:
    st.error(f"Connection error with FastAPI: {e}")
