import sqlite3
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from rake_nltk import Rake

# Ensure necessary NLTK data packages are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download the spaCy model if not already done
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Connect to the database
conn = sqlite3.connect('index.db')
cursor = conn.cursor()

# Fetch urls from the documents table
cursor.execute("SELECT content FROM documents")
rows = cursor.fetchall()

# Preprocess the text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Process each document
processed_texts = []
for row in rows:
    content = row[0]
    processed_text = preprocess_text(content)
    processed_texts.append(processed_text)

# Named Entity Recognition (NER)
entities = []
for text in processed_texts:
    doc = nlp(text)
    entities.extend([ent.text for ent in doc.ents if ent.label_ in ["GPE", "ORG", "PERSON"]])

# Debug: Print number of entities
print(f"Number of entities extracted: {len(entities)}")

# Keyword Extraction using RAKE
rake = Rake()
for text in processed_texts:
    rake.extract_keywords_from_text(text)

# Get a specified number of top keywords
num_keywords = 50  # Adjust this value to get more or fewer keywords
rake_keywords = rake.get_ranked_phrases_with_scores()

# Debug: Print number of RAKE keywords
print(f"Number of RAKE keywords extracted: {len(rake_keywords)}")

rake_keywords = sorted(rake_keywords, key=lambda x: x[0], reverse=True)
rake_keywords = [keyword for score, keyword in rake_keywords[:num_keywords]]

# Combine NER entities and RAKE keywords
combined_keywords = list(set(entities + rake_keywords))

# Debug: Print combined keywords
print(f"Combined Keywords (Total: {len(combined_keywords)}):")
for keyword in combined_keywords:
    print(keyword)

# Close the database connection
conn.close()
