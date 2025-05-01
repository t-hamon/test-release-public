FROM python:3.10-bullseye

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-client \
        curl \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY src/ ./src/

# Set env path for training and FastAPI
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

