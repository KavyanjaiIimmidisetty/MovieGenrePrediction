import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Read dataset
rows = []

with open("data/train_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.split(" ::: ")

        if len(parts) >= 4:
            genre = parts[2].strip()
            plot = parts[3].strip()

            rows.append([plot, genre])

# Create DataFrame
df = pd.DataFrame(rows, columns=["plot", "genre"])

print(df.head())

# Split data
X = df["plot"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LinearSVC(
        max_iter=5000,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", accuracy)

# Save model
joblib.dump(model, "models/genre_model.pkl")

print("Model Saved Successfully!")