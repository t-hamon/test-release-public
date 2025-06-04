from fastapi import FastAPI, Query, HTTPException, UploadFile, File
import subprocess
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
async def train(file: UploadFile = File(...)):
    upload_path = f"uploaded_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = subprocess.run(
        ["python3", "src/models/train_model.py", upload_path],
        capture_output=True,
        text=True
    )

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
    result = subprocess.run(
        ["python3", "src/models/predict_model.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
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
                .rename(columns={"movieid": "movieid", "moviename": "title", "pred_rating": "score"})
                .to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

