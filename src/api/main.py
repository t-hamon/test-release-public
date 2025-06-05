from fastapi import FastAPI, Query, HTTPException, UploadFile, File, APIRouter
import subprocess
from typing import Optional
import psycopg2
import pandas as pd
from pydantic import BaseModel
from typing import List
import shutil
import os

app = FastAPI()

class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]


def get_db_connection():
    return psycopg2.connect(
        host="my-postgres",
        database="movies_db",
        user="postgres",
        password="password"
    )

@app.get("/")
def root():
    return {"message": "API de recommandation de films!"}

@app.post("/training")
async def train(
    file: Optional[UploadFile] = File(
        None,
        description="Fichier CSV à Uploader. Lance l'entraînement du modèle de recommandation. Si aucun fichier n'est fourni : 'src/data/movies_cleaned.csv' est utilisé par défaut"
    )
):
    # 1. Si un fichier est fourni via Swagger ou curl
        if file is not None:
            upload_path = f"uploaded_{file.filename}"
            with open(upload_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            # 2. Sinon, utilise le fichier local par défaut
            upload_path = "src/data/movies_cleaned.csv"
            if not os.path.exists(upload_path):
                raise HTTPException(status_code=400, detail="Aucun fichier envoyé et fichier local introuvable")

        result = subprocess.run(
            ["python3", "src/models/train_model.py",
            "--input_csv", upload_path,
            "--rf_n_estimators", "100",
            "--rf_max_depth", "10",
            "--rf_min_samples_split", "5",
            "--rf_min_samples_leaf", "5",
            "--test_size", "0.3",
            "--random_state", "42"
            ],
            capture_output=True,
            text=True
        )

        if file:
            os.remove(upload_path)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")

        return {
            "status": "Training completed",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

@app.post("/prediction")
def predict():
    print("Début de la prédiction...")

    result = subprocess.run(
        ["python3", "src/models/predict_model.py"],
        capture_output=True,
        text=True
    )

    print("Fin de la prédiction")

    if result.returncode != 0:
        print(f"Erreur pendant la prédiction : {result.stderr}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {result.stderr}")

    return {
        "status": "Prediction completed",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.get("/health")
def health_check():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}



@app.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(user_id: int, n: int):
    try:
        # Load predictions
        df = pd.read_csv("src/api/predictions_rf.csv")
        df.columns = [col.lower() for col in df.columns]

        print("Colonnes disponibles :", df.columns.tolist())
        print("Extrait des données", df.head())

        if "userid" not in df.columns or "movieid" not in df.columns or "pred_rating" not in df.columns:
            raise ValueError("Required columns missing in predictions_rf.csv")

        user_recs = df[df["userid"] == user_id]

        if user_recs.empty:
            return {"user_id": user_id, "recommendations": []}

        top_recs = user_recs.sort_values(by="pred_rating", ascending=False).head(n)

        # Merge movie titles
        if "moviename" not in top_recs.columns:
            df_movies = pd.read_csv("src/data/movies.utf.csv", sep="::", engine="python",
                                    names=["movieid", "moviename", "genre"])
            df_movies.columns = [col.lower() for col in df_movies.columns]
            top_recs = top_recs.merge(df_movies[["movieid", "moviename"]], on="movieid", how="left")

        recommendations = [
            {
                "movie_id": int(row["movieid"]),
                "title": row["moviename"],
                "score": round(row["pred_rating"], 2)
            }
            for _, row in top_recs.iterrows()
        ]

        return {
            "user_id": user_id,
            "recommendations": top_recs[["movieid", "moviename", "pred_rating"]]
                .rename(columns={"movieid": "movie_id", "moviename": "title", "pred_rating": "score"})
                .to_dict(orient="records")
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Exception dans /recommendations:\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))
