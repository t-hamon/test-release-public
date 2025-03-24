FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y postgresql-client

RUN pip install --no-cache-dir psycopg2-binary pandas fastapi uvicorn scikit-learn

COPY src/ ./src/
# Exécution automatique faite dans l’API maintenant, donc on supprime entrypoint


EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
