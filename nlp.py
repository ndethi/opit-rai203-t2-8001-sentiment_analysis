import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("/iphone.csv")





df = df[['reviewDescription', 'ratingScore']].dropna()


df = df[df['ratingScore'] != 3] 
df['sentiment'] = df['ratingScore'].apply(lambda x: 'positive' if x >= 4 else 'negative')


def clean_text(text):
    text = text.lower()  
    text = re.sub(f"[{string.punctuation}]", "", text)  
    text = re.sub(r'\d+', '', text)  
    return text.strip()


df['cleaned_review'] = df['reviewDescription'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


sample_review = ["The iPhone has perfect camera quality and the battery life is great!"]
cleaned_sample = [clean_text(sample_review[0])]
predicted_sentiment = pipeline.predict(cleaned_sample)

print("Predicted Sentiment:", "Positive" if predicted_sentiment[0] == 'positive' else "Negative")
