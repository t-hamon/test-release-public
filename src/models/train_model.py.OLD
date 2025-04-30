import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from joblib import dump

# Load data
file_path = 'src/data/movies_db.csv'  # Modify the file path if needed
try:
    movie_ratings = pd.read_csv(file_path, sep=',')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Model development and training
# Separating the feature and target variable
features = movie_ratings[['MovieID', 'Age', 'Occupation']].values
labels = movie_ratings['Rating'].values

# Splitting the dataset into training and testing data (70% training, 30% testing)
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

# Standardize the features (important for Logistic Regression and KNN)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
lr.fit(train_scaled, train_labels)
y_predict_lr = lr.predict(test_scaled)
acc_model1 = accuracy_score(test_labels, y_predict_lr)
score1 = round(lr.score(train_scaled, train_labels) * 100, 2)

# Classification Report for Logistic Regression
classification_report_lr = classification_report(test_labels, y_predict_lr)
print('Accuracy score of Logistic Regression model is:', acc_model1)
print('Score of Logistic Regression model is:', score1)
print('Classification report of Logistic Regression model is:', classification_report_lr)

# Model 2: Random Forest Classifier
#rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=5, random_state=42)

rf.fit(train, train_labels)  # No scaling for Random Forest as it's not sensitive to scaling
y_pred_rf = rf.predict(test)
acc_model2 = accuracy_score(test_labels, y_pred_rf)
score2 = round(rf.score(train, train_labels) * 100, 2)

# Classification Report for Random Forest
classification_report_rf = classification_report(test_labels, y_pred_rf)
print('Accuracy score of Random Forest model is:', acc_model2)
print('Score of Random Forest model is:', score2)
print('Classification report of Random Forest model is:', classification_report_rf)

# Model 3: K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(train_scaled, train_labels)  # Scale features for KNN
y_pred_knn = knn.predict(test_scaled)
acc_model3 = accuracy_score(test_labels, y_pred_knn)
score3 = round(knn.score(train_scaled, train_labels) * 100, 2)

# Classification Report for KNN
classification_report_knn = classification_report(test_labels, y_pred_knn)
print('Accuracy score of KNN model is:', acc_model3)
print('Score of KNN model is:', score3)
print('Classification report of KNN model is:', classification_report_knn)

# Cross-validation for Random Forest and KNN (to check for overfitting)
cv_rf = cross_val_score(rf, features, labels, cv=5)
cv_knn = cross_val_score(knn, features, labels, cv=5)
print('Random Forest Cross-validation Scores: ', cv_rf)
print('KNN Cross-validation Scores: ', cv_knn)

# Model scores in a DataFrame
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest Classifier', 'K-Nearest Neighbor'],
    'Score': [score1, score2, score3]
})

# Sort models by score
print(models.sort_values(ascending=False, by='Score'))

y_pred_rf = rf.predict(test)
acc_model2 = accuracy_score(test_labels, y_pred_rf)
print(f"Random Forest Accuracy on Test Data: {acc_model2}")


# Save predictions to a CSV file
predictions_df = pd.DataFrame(y_pred_rf, columns=['Predicted_Rating'])
predictions_df.to_csv('predictions_rf.csv', index=False)


# After training the model (rf):
dump(rf, 'random_forest_model.joblib')

