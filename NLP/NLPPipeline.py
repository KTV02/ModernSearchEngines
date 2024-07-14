import numpy as np
import nltk
import string
import requests
import json
import os
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded
# You can comment it out if you already downloaded it
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

output_file_path = 'NLPOutput.txt'

# Function to initialize the file (clear contents)
def initialize_file(file_path):
    with open(file_path, 'w') as file:
        file.write("")  # This clears the file

# Function to append output to the file
def append_to_file(file_path, text):
    try:
        with open(file_path, 'a', encoding='utf-8', errors='replace') as file:
            file.write(text + "\n")
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError: {e} for text: {text}")

#function to access database with a SQL query
def getFromDatabase(query):
    # Here instead of hostname you can also use the ip adresse Displayed in NordVPN
    url = 'http://l.kremp-everest.nord:5000/query'

    #always rename your parameters if you want to access them via their index in the output
    # e.g. here COUNT(*) renamed to count

    # Basic authentication details
    auth = ('mseproject', 'tuebingen2024')

    # Make the POST request to the Flask API with your query
    response = requests.post(url, json={'query': query}, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data
    else:
        print(f"Error executing query: {response.status_code} - {response.text}")

def clean_html_content(html_content):
    try:
        # Use Readability to extract the main content
        doc = Document(html_content)
        main_content_html = doc.summary()
        title = doc.title()

        # Parse the main content HTML using BeautifulSoup
        soup = BeautifulSoup(main_content_html, 'html.parser')

        # Remove unwanted elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # Get cleaned content
        cleaned_content = ' '.join(soup.stripped_strings)

        return {
            'title': title,
            'cleaned_content': cleaned_content
        }

    except Exception as e:
        print(f"Error while cleaning content: {e}")
        return {
            'title': None,
            'cleaned_content': None
        }

#deprecated don't use
#Can crawl a webpage
def crawl_page(url):
    try:
        response = requests.get(url, timeout=10)  # Fetch the web page
        if response.status_code != 200:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None

        cleaned_data = clean_html_content(response.text)
        print(f"Title: {cleaned_data['title']}")
        print(f"Cleaned Content: {cleaned_data['cleaned_content']}")
        return cleaned_data
    except requests.RequestException as e:
        print(f"Request exception encountered at {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected exception encountered at {url}: {e}")
        return None

def detect_language(text):
    """Detect the language of the given text."""
    try:
        return detect(text)
    except Exception as e:
        return str(e)

def remove_punctuation_and_tokenize(text):
    """Remove punctuation from the text, lowercase everything and tokenize it."""
    translator = str.maketrans('', '', string.punctuation)
    text = text.lower()
    text_no_punct = text.translate(translator)
    tokens = word_tokenize(text_no_punct)
    return tokens

def remove_stop_words(tokens, language='english'):
    """Remove stop words from the list of tokens."""
    navigation_tokens = {"back", "press", "login", "skip", "next"}
    stop_words = set(stopwords.words(language))
    stop_words.update(navigation_tokens)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def pos_tagging(tokens):
    """Perform POS tagging on the list of tokens."""
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

#Lemmatizer using wordnet
def lemmatize_tokens(tokens):
    """Lemmatize the list of tokens."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


#Use main function to create file NLPOutput.txt containing title, url and content
#processed by the NLP pipeline
#use function parse_tokens in KeywordFilter class to read the title, url and content for ranking
if __name__ == "__main__":
    data = getFromDatabase("SELECT * FROM documents")
    print('got documents')
    initialize_file(output_file_path)
    for i in tqdm(range(len(data))):
        text = data[i]['content']
        language = detect_language(text)
        tokens = remove_punctuation_and_tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        lemmatized_tokens = lemmatize_tokens(filtered_tokens)
        #output = (str(i)+'.Language: ' + language + '\n' +
        output = 'Title: '+ str(data[i]['title'])+ '\n' + 'URL: ' + str(data[i]['url'])+ '\n'+ 'Tokens: ' + ' '.join(lemmatized_tokens) + '\n'
        append_to_file(output_file_path, output)



