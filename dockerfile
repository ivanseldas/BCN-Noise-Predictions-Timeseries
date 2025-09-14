FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Copy the exported model
COPY models/production /app/models/production

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]