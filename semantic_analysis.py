import pandas as pd
import spacy
import re
import string


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


def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        
        if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'LOC', 'PERSON']:
            entities[ent.text] = ent.label_
    return entities


data['entities'] = data['cleaned_review'].apply(extract_entities)


print(data[['cleaned_review', 'entities']].head())


sample_review = "The iPhone has perfect camera quality and the battery life is great!"
entities_in_sample = extract_entities(sample_review)

print("\nEntities found in sample review:")
for entity, label in entities_in_sample.items():
    print(f"{entity}: {label}")
