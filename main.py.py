# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:10:58 2025

@author: mmmkh
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Download and load the German language model in spaCy
spacy.cli.download("de_core_news_sm")
nlp = spacy.load("de_core_news_sm")

# Load coffee reviews dataset
file_path = 'kaffee_reviews.csv'  # Ensure the correct file path

df = pd.read_csv(file_path)

# Handle missing values
df.dropna(subset=['review', 'rating'], inplace=True)

# Remove unnecessary column if it exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Ensure 'rating' contains valid numerical values
df = df[pd.to_numeric(df['rating'], errors='coerce').notna()]
df['rating'] = df['rating'].astype(float)

# Create sentiment labels based on review ratings
df['Sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß]', ' ', str(text))  # Remove non-alphabetic characters
    text = text.lower().strip()  # Convert text to lowercase
    doc = nlp(text)  # Process text with spaCy
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])  # Remove stopwords and punctuation

# Apply text cleaning
df['Cleaned_Review'] = df['review'].apply(clean_text)

# Extract features using CountVectorizer and TF-IDF
countvectorizer = CountVectorizer(max_features=1500)
X_count = countvectorizer.fit_transform(df['Cleaned_Review']).toarray()

tfidfvectorizer = TfidfVectorizer(max_features=1500)
X_tfidf = tfidfvectorizer.fit_transform(df['Cleaned_Review']).toarray()

y = df['Sentiment'].values

# Split data into training and testing sets
X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, y, test_size=0.20, random_state=0)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.20, random_state=0)

# Build text classification model using Gaussian Naive Bayes
classifier_count = GaussianNB()
classifier_count.fit(X_train_count, y_train)

classifier_tfidf = GaussianNB()
classifier_tfidf.fit(X_train_tfidf, y_train)

# Make predictions using the classifier
y_pred_count = classifier_count.predict(X_test_count)
y_pred_tfidf = classifier_tfidf.predict(X_test_tfidf)

# Evaluate the results using a confusion matrix
cm_count = confusion_matrix(y_test, y_pred_count)
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)

print("Confusion Matrix (CountVectorizer):")
print(cm_count)

print("Confusion Matrix (TF-IDF):")
print(cm_tfidf)

# Display classification report
print("\nClassification Report (CountVectorizer):")
print(classification_report(y_test, y_pred_count))

print("\nClassification Report (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf))

# Save the model
joblib.dump(classifier_count, 'sentiment_classifier_count.pkl')
joblib.dump(classifier_tfidf, 'sentiment_classifier_tfidf.pkl')

# Display cleaned data with sentiment labels
print("\nCleaned data:")
print(df[['review', 'Cleaned_Review', 'Sentiment']].head())
