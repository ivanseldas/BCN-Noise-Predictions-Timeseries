FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (para aprovechar cache en builds futuros)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Copy the exported production model
COPY models/production /app/models/production

EXPOSE 8000 8501

CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0