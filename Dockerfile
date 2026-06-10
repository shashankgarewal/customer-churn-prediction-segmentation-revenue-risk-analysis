# python image for docker container
FROM python:3.12-slim

# container working directory
WORKDIR /app

# instant real-time logging and clean import
ENV PYTHONUNBUFFERED=1\
    PYTHONPATH=/app

# Install system dependencies required to build heavy packages like CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# copy and install dependency first to avoid package installation in every changes to repo
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
# Copy project source
COPY . .

# Explicitly pull in only the test data files exactly where they belong
COPY data/interim/transformed/test.parquet ./data/interim/transformed/test.parquet
COPY data/processed/test.parquet ./data/processed/test.parquet

# expose fastapi and streamlit port
EXPOSE 8000 8501

# Run API in background, then run UI in foreground
CMD uvicorn src.app.api:app --host 0.0.0.0 --port 8000 & \
    python -m streamlit run src/app/ui.py --server.port 8501 --server.address 0.0.0.0