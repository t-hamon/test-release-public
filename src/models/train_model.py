import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib  # <-- Add joblib import
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Train model with custom hyperparameters")
parser.add_argument("--rf_n_estimators", type=int, default=100)
parser.add_argument("--rf_max_depth", type=int, default=10)
parser.add_argument("--rf_min_samples_split", type=int, default=5)
parser.add_argument("--rf_min_samples_leaf", type=int, default=5)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--input_csv", type=str, default="data/movies_cleaned.csv")

args = parser.parse_args()

# Set MLflow tracking URI and experiment name
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MovieRecommendation")

# Load CSV data file from first argument or fallback path
file_path = args.input_csv

if not os.path.exists(file_path):
    print(f"CSV file not found: {file_path}")
    sys.exit(1)

try:
    movie_ratings = pd.read_csv(file_path, sep=',')
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Prepare features and labels
features = movie_ratings[['MovieID', 'Age', 'Occupation']].values
labels = movie_ratings['Rating'].values

# Split into train and test sets
train, test, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Standardize features for applicable models
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

with mlflow.start_run(run_name="movie-rating-model-training"):
    # Log parameters
    mlflow.log_params({
        "test_size": args.test_size,
        "random_state": args.random_state,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_split": args.rf_min_samples_split,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "input_csv": os.path.basename(file_path)
    })


    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_scaled, train_labels)
    y_predict_lr = lr.predict(test_scaled)
    acc_model1 = accuracy_score(test_labels, y_predict_lr)
    classification_report_lr = classification_report(test_labels, y_predict_lr)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_split=args.rf_min_samples_split,
        min_samples_leaf=args.rf_min_samples_leaf,
        random_state=args.random_state
    )
    rf.fit(train, train_labels)
    y_pred_rf = rf.predict(test)
    acc_model2 = accuracy_score(test_labels, y_pred_rf)
    classification_report_rf = classification_report(test_labels, y_pred_rf)

    # Save the model locally as joblib file (important for predict_model.py)
    joblib_path = "src/models/random_forest_model.joblib"
    joblib.dump(rf, joblib_path)
    print(f"Random Forest model saved locally at: {joblib_path}")

    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(train_scaled, train_labels)
    y_pred_knn = knn.predict(test_scaled)
    acc_model3 = accuracy_score(test_labels, y_pred_knn)
    classification_report_knn = classification_report(test_labels, y_pred_knn)

    # Cross-validation scores
    cv_rf = cross_val_score(rf, features, labels, cv=5)
    cv_knn = cross_val_score(knn, features, labels, cv=5)

    # Log metrics
    mlflow.log_metrics({
        "lr_accuracy": acc_model1,
        "rf_accuracy": acc_model2,
        "knn_accuracy": acc_model3,
        "cv_rf_mean": cv_rf.mean(),
        "cv_knn_mean": cv_knn.mean()
    })

    # Log artifacts
    with open("classification_report_rf.txt", "w") as f:
        f.write(classification_report_rf)
    mlflow.log_artifact("classification_report_rf.txt")

    predictions_df = pd.DataFrame(y_pred_rf, columns=['Predicted_Rating'])
    predictions_df.to_csv('predictions_rf.csv', index=False)
    mlflow.log_artifact('predictions_rf.csv')

    mlflow.sklearn.log_model(rf, "random_forest_model")
    print("Model logged successfully!")
    print("Random Forest Accuracy on Test Data:", acc_model2)
