FROM python:3.10-bullseye

WORKDIR /app

# Install PostgreSQL client
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    pandas \
    fastapi \
    uvicorn \
    scikit-learn \
    mlflow \
    joblib

# Copy application code
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

