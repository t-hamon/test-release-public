import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load movies metadata
df_movies = pd.read_csv('src/data/movies.utf.csv', sep='::', engine='python',
                        names=['movieid', 'moviename', 'genre'])
df_movies.columns = [col.lower() for col in df_movies.columns]

print("🚀 Prediction script started", flush=True)

# Load ratings
movie_ratings = pd.read_csv('src/data/movies_db.csv')
movie_ratings.columns = [col.lower() for col in movie_ratings.columns]

# Load the trained model
rf = load('src/models/random_forest_model.joblib')

# Evaluate model performance
features = movie_ratings[['movieid', 'age', 'occupation']].values
labels = movie_ratings['rating'].values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred), flush=True)
print("Report:\n", classification_report(y_test, y_pred), flush=True)

# Predict for all users
all_user_ids = movie_ratings['userid'].unique()
all_movie_ids = movie_ratings['movieid'].unique()

predictions = []

for i, user_id in enumerate(all_user_ids):
    user_data = movie_ratings[movie_ratings['userid'] == user_id].iloc[0]
    user_age = user_data['age']
    user_occupation = user_data['occupation']
    user_rated = movie_ratings[movie_ratings['userid'] == user_id]['movieid'].unique()
    user_unrated = np.setdiff1d(all_movie_ids, user_rated)

    for movie_id in user_unrated[:150]:  # Si besoin, adapte ce seuil
        sample = np.array([[movie_id, user_age, user_occupation]])
        pred_rating = rf.predict(sample)[0]
        predictions.append((user_id, movie_id, pred_rating))

    if i % 10 == 0 or i == len(all_user_ids) - 1:
        print(f"Processed {i+1}/{len(all_user_ids)} users...", flush=True)

# Create prediction DataFrame
pred_df = pd.DataFrame(predictions, columns=['userid', 'movieid', 'pred_rating'])

# Merge with movie names
pred_df = pred_df.merge(df_movies[['movieid', 'moviename']], on='movieid', how='left')

# Normalize column names
pred_df.columns = [col.lower() for col in pred_df.columns]

print(f"✅ Writing {len(pred_df)} predictions to CSV", flush=True)

# Save to CSV
pred_df.to_csv('src/api/predictions_rf.csv', index=False)
print("✅ Saved predictions to src/api/predictions_rf.csv", flush=True)
