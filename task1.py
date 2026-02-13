import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "plot": [
        "hero saves world from aliens",
        "funny story of two friends",
        "love story between boy and girl",
        "police catches criminal",
        "ghost scares people in house"
    ],
    "genre": ["Sci-Fi", "Comedy", "Romance", "Action", "Horror"]
}

df = pd.DataFrame(data)

X = df["plot"]
y = df["genre"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sample = vectorizer.transform(["aliens attack city"])
print("Prediction:", model.predict(sample))