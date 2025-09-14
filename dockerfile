FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Nginx and Supervisor
RUN apt-get update && apt-get install -y nginx supervisor && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Copy exported production model
COPY models/production /app/models/production

# Copy Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Supervisor config to manage FastAPI, Streamlit, and Nginx
RUN echo "[supervisord]\nnodaemon=true\n" > /etc/supervisord.conf && \
    echo "[program:fastapi]\ncommand=uvicorn api.main:app --host 0.0.0.0 --port 8001\n" >> /etc/supervisord.conf && \
    echo "[program:streamlit]\ncommand=streamlit run dashboard/app.py --server.port=8501 --server.headless=true --server.address=0.0.0.0\n" >> /etc/supervisord.conf && \
    echo "[program:nginx]\ncommand=nginx -g 'daemon off;'\n" >> /etc/supervisord.conf

EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]