import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("data/clean_dataset.csv")

X = df["clean_text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))


# SAVE MODEL
joblib.dump(model, "models/logistic_model.pkl")

# SAVE VECTORIZER
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved successfully")