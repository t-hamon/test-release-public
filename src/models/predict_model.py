import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from joblib import load  # To load the saved model

# Load data for movie names and genres
df_movies = pd.read_csv('src/data/movies.utf.csv', sep='::', engine='python', names=['MovieID', 'MovieName', 'Genre'])

# Load ratings data
file_path = 'src/data/movies_db.csv'  # Modify the file path if needed
try:
    movie_ratings = pd.read_csv(file_path, sep=',')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Separating the feature and target variable
features = movie_ratings[['MovieID', 'Age', 'Occupation']].values
labels = movie_ratings['Rating'].values

# Splitting the dataset into training and testing data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.3)

# Load the trained Random Forest model from the .joblib file
rf = load('random_forest_model.joblib')  # Ensure the model is in the same directory or provide full path

# Using the loaded Random Forest model to predict the test data
y_pred_rf = rf.predict(test)

# Evaluating the model
acc_model2 = accuracy_score(test_labels, y_pred_rf)
classification_rep = classification_report(test_labels, y_pred_rf)

# Displaying results
print('Accuracy score of Random Forest model:', acc_model2)
print('Classification report of Random Forest model:\n', classification_rep)

# Creating a DataFrame to store model scores
models = pd.DataFrame({
    'Model': ['Random Forest Classifier'],
    'Accuracy Score': [acc_model2]
})

print(models)

# Example test case (replace with actual values)
sample_data = np.array([[123, 25, 3]])  # MovieID=123, Age=25, Occupation=3

# Ensure the input shape matches the model's expectation
sample_data = sample_data.reshape(1, -1)

# Using the loaded Random Forest model to predict new data
prediction = rf.predict(sample_data)
print("Predicted Rating for sample data:", prediction[0])

# User input for recommendations
user_age = 30  # Example age
user_occupation = 5  # Example occupation

# Get all unique MovieIDs
all_movies = movie_ratings['MovieID'].unique()

# Get movies the user has already rated
rated_movies = movie_ratings[movie_ratings['Age'] == user_age]['MovieID'].unique()

# Filter movies the user hasn't seen
unrated_movies = np.setdiff1d(all_movies, rated_movies)

recommendations = []

for movie in unrated_movies:
    sample_data = np.array([[movie, user_age, user_occupation]])  # Format the input
    predicted_rating = rf.predict(sample_data)[0]  # Predict the rating
    recommendations.append((movie, predicted_rating))  # Store the result

# Sort by rating (highest first)
recommendations.sort(key=lambda x: x[1], reverse=True)

# Get the top 5 movies
top_movies = recommendations[:5]

# Display results
print("Recommended Movies:")
for movie, rating in top_movies:
    print(f"Movie ID: {movie}, Predicted Rating: {rating}")

# Create a DataFrame with MovieID and Predicted Ratings
top_movies_df = pd.DataFrame(top_movies, columns=['MovieID', 'Predicted Rating'])

# Merge with df_movies to get Movie Titles
top_movies_df = top_movies_df.merge(df_movies[['MovieID', 'MovieName']], on='MovieID', how='left')

# Display the results
print(top_movies_df[['MovieName', 'Predicted Rating']])

