import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Optional: set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://mlflow:5000")

mlflow.set_experiment("MovieRecommendation")

# Load data
file_path = '../data/movies_db.csv'
try:
    movie_ratings = pd.read_csv(file_path, sep=',')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Features and labels
features = movie_ratings[['MovieID', 'Age', 'Occupation']].values
labels = movie_ratings['Rating'].values

# Split the data
train, test, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

with mlflow.start_run(run_name="movie-rating-model-training"):
    # Log basic parameters
    mlflow.log_params({
        "test_size": 0.3,
        "random_state": 42,
        "rf_n_estimators": 100,
        "rf_max_depth": 10,
        "rf_min_samples_split": 5,
        "rf_min_samples_leaf": 5
    })

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_scaled, train_labels)
    y_predict_lr = lr.predict(test_scaled)
    acc_model1 = accuracy_score(test_labels, y_predict_lr)
    classification_report_lr = classification_report(test_labels, y_predict_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_split=5, min_samples_leaf=5,
                                random_state=42)
    rf.fit(train, train_labels)
    y_pred_rf = rf.predict(test)
    acc_model2 = accuracy_score(test_labels, y_pred_rf)
    classification_report_rf = classification_report(test_labels, y_pred_rf)

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(train_scaled, train_labels)
    y_pred_knn = knn.predict(test_scaled)
    acc_model3 = accuracy_score(test_labels, y_pred_knn)
    classification_report_knn = classification_report(test_labels, y_pred_knn)

    # Cross-validation
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

    # Save reports
    with open("classification_report_rf.txt", "w") as f:
        f.write(classification_report_rf)
    mlflow.log_artifact("classification_report_rf.txt")

    # Save and log model
    print("Logging Random Forest model to MLflow...")
    mlflow.sklearn.log_model(rf, "random_forest_model", registered_model_name="RandomForestMovieRecoModel")
    print("Model logged successfully!")

    # Save predictions
    predictions_df = pd.DataFrame(y_pred_rf, columns=['Predicted_Rating'])
    predictions_df.to_csv('predictions_rf.csv', index=False)
    mlflow.log_artifact('predictions_rf.csv')

    print("Random Forest Accuracy on Test Data:", acc_model2)

