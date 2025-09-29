# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- Initial config ---
st.set_page_config(page_title="Noise Forecasting Dashboard", layout="wide")

st.title("Noise Forecasting Dashboard")

st.markdown(
    """
    This dashboard shows the **hourly noise level forecast** together with a 
    **95% confidence interval**, based on the deployed model.
    """
)

# --- Call the API ---
url = "http://localhost:8000/predict_now/"
resp = requests.get(url)

if resp.status_code != 200:
    st.error(f"API error: {resp.text}")
else:
    data = resp.json()
    df = pd.DataFrame(data["predictions"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    # --- Plotly chart ---
    fig = go.Figure()

    # Confidence Interval (shaded area)
    fig.add_traces([
        go.Scatter(
            x=df["datetime"],
            y=df["upper"],
            mode="lines",
            line=dict(width=0),
            name="Upper Bound",
            showlegend=False
        ),
        go.Scatter(
            x=df["datetime"],
            y=df["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 123, 255, 0.2)",  # light blue
            name="Confidence Interval"
        )
    ])

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["prediction"],
            mode="lines",
            line=dict(color="blue"),
            name="Forecast"
        )
    )

    # Hourly points (markers)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["prediction"],
            mode="markers",
            marker=dict(color="blue", size=6, symbol="circle"),
            name="Hourly Predictions"
        )
    )

    # Layout
    fig.update_layout(
        title="Noise Forecast with 95% Confidence Interval",
        xaxis_title="Datetime",
        yaxis_title="Noise Level (dB)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig, use_container_width=True)
