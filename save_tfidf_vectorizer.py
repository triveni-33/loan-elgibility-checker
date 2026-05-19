from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sample text data
documents = [
    "This is a sample document.",
    "This document is another sample.",
    "We need to create a TF-IDF vectorizer.",
    "Save this TF-IDF vectorizer to a file."
]

# Fit TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(documents)

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("TF-IDF vectorizer saved to 'tfidf_vectorizer.pkl'")