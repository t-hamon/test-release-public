from fastapi import FastAPI
from .db import get_db_connection
import subprocess

# Nettoyage CSV
print("Nettoyage du CSV")
subprocess.run(["python3", "src/models/csv2db_prep.py"], check=True)

# Insertion dans Postgres
print("Insertion dans PostgreSQL")
subprocess.run([
    "psql", "host=postgres user=postgres password=root dbname=movies_db",
    "-c",
    "\\copy movies(MovieName, MovieID, Genre, UserID, Rating, Timestamp, Gender, Age, Occupation, Zipcode, age_group) FROM 'src/data/movies_cleaned.csv' DELIMITER ',' CSV HEADER;"
])
print("Données insérées !")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API de recommandation de films!"}

@app.post("/training")
def train():
    result = subprocess.run(["python3", "src/models/train_model.py"], capture_output=True, text=True)
    return {
        "status": "Training completed",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.post("/prediction")
def predict():
    result = subprocess.run(["python3", "src/models/predict_model.py"], capture_output=True, text=True)
    return {
        "status": "Prediction completed",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.get("/health")
def health():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM movies")
        count = cur.fetchone()
        cur.close()
        conn.close()
        return {"status": "ok", "movie_count": count["count"]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
