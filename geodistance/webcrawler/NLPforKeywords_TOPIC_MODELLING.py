import sqlite3
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure necessary NLTK data packages are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Generate a term-document matrix using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts)
feature_names = vectorizer.get_feature_names_out()

# Apply topic modeling using LDA
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(tfidf_matrix)

# Extract top keywords for each topic
def get_top_keywords(lda_model, feature_names, n_keywords=10):
    keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-n_keywords - 1:-1]]
        keywords.extend(top_keywords)
    return list(set(keywords))

keywords = get_top_keywords(lda, feature_names)
print("Extracted Keywords:")
print(keywords)

# Close the database connection
conn.close()
