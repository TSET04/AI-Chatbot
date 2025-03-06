import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data (extend this for better accuracy)
intents = {
    "factual_query": ["Tell me about", "Explain", "Describe", "Why", "What is the meaning of", "Help me understand"],
    "database_query": ["Find", "Retrieve", "Get", "How many", "Names", "Ages", "Weights", "Id"],
}

# Preparing training dataset
X_train = []
y_train = []
for intent, phrases in intents.items():
    X_train.extend(phrases)
    y_train.extend([intent] * len(phrases))

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Save the trained classifier and vectorizer
with open("intent_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Intent classifier and vectorizer saved successfully!")