import string
import requests
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm

class NLP_Pipeline:
    def __init__(self, data, output_file_path='NLPOutput.txt'):
        self.output_file_path = output_file_path
        self.initialize_file(self.output_file_path)
        self.data = data

    def initialize_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write("")

    def append_to_file(self, text):
        try:
            with open(self.output_file_path, 'a', encoding='utf-8', errors='replace') as file:
                file.write(text + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e} for text: {text}")

    def getFromDatabase(self, query):
        url = 'http://l.kremp-everest.nord:5000/query'
        auth = ('mseproject', 'tuebingen2024')
        response = requests.post(url, json={'query': query}, auth=auth)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error executing query: {response.status_code} - {response.text}")

    def clean_html_content(self, html_content):
        try:
            doc = Document(html_content)
            main_content_html = doc.summary()
            title = doc.title()
            soup = BeautifulSoup(main_content_html, 'html.parser')
            for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                tag.decompose()
            cleaned_content = ' '.join(soup.stripped_strings)
            return {'title': title, 'cleaned_content': cleaned_content}
        except Exception as e:
            print(f"Error while cleaning content: {e}")
            return {'title': None, 'cleaned_content': None}

    def crawl_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to retrieve the page. Status code: {response.status_code}")
                return None
            cleaned_data = self.clean_html_content(response.text)
            print(f"Title: {cleaned_data['title']}")
            print(f"Cleaned Content: {cleaned_data['cleaned_content']}")
            return cleaned_data
        except requests.RequestException as e:
            print(f"Request exception encountered at {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected exception encountered at {url}: {e}")
            return None

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            return str(e)

    def remove_punctuation_and_tokenize(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.lower()
        text_no_punct = text.translate(translator)
        tokens = word_tokenize(text_no_punct)
        return tokens

    def remove_stop_words(self, tokens, language='english'):
        navigation_tokens = {"back", "press", "login", "skip", "next"}
        stop_words = set(stopwords.words(language))
        stop_words.update(navigation_tokens)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return filtered_tokens

    def pos_tagging(self, tokens):
        return pos_tag(tokens)

    def lemmatize_tokens(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def process_documents(self):
        for i in tqdm(range(len(self.data))):
            text = self.data[i]['content']
            language = self.detect_language(text)
            tokens = self.remove_punctuation_and_tokenize(text)
            filtered_tokens = self.remove_stop_words(tokens)
            lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
            output = 'Title: '+ str(self.data[i]['title'])+ '\n' + 'URL: ' + str(self.data[i]['url'])+ '\n'+ 'Tokens: ' + ' '.join(lemmatized_tokens) + '\n'
            self.append_to_file(output)

# Example usage:
if __name__ == "__main__":
    nlp_pipeline = NLP_Pipeline()
    nlp_pipeline.process_documents()