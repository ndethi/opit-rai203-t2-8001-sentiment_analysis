import pandas as pd
import spacy
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


nlp = spacy.load("en_core_web_sm")


data = pd.read_csv("/home/kev-man/Models/nlpJob/iphone.csv")


data = data[['reviewDescription', 'ratingScore']].dropna()
data = data[data['ratingScore'] != 3]  


def clean_text(text):
    text = text.lower()  
    text = re.sub(f"[{string.punctuation}]", "", text)  
    text = re.sub(r'\d+', '', text)  
    return text.strip()

data['cleaned_review'] = data['reviewDescription'].apply(clean_text)


data['sentiment'] = data['ratingScore'].apply(lambda x: 'positive' if x >= 4 else 'negative')


def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'LOC', 'PERSON']:
            entities[ent.text] = ent.label_
    return entities


data['entities'] = data['cleaned_review'].apply(extract_entities)


print(data[['cleaned_review', 'entities']].head())


X = data['cleaned_review']
y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='positive')

print("\nSentiment Analysis Model Evaluation:")
print("Accuracy:", accuracy)
print("F1-Score:", f1)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


 
