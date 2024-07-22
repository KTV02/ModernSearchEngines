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
from collections import Counter, defaultdict, OrderedDict
import re

# utility functions for soft deduplication
def compute_simhash(words, num_bits=64):
    hash_vector = np.zeros(num_bits)
    for word in words:
        word_hash = hash(word)
        hash_vector += np.array([1 if word_hash & (1 << i) else -1 for i in range(num_bits)])
    return ''.join(['1' if x > 0 else '0' for x in hash_vector])

def compute_tf_idf(documents):
    # Create vocabulary and document frequency
    vocab = sorted(set(word for doc in documents for word in doc))
    doc_frequency = Counter(word for doc in documents for word in set(doc))
    
    # Compute IDF
    idf = np.array([math.log(len(documents) / doc_frequency[word]) for word in vocab])
    
    # Compute TF-IDF matrix
    tfidf_matrix = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(documents):
        tf = Counter(doc)
        tfidf_matrix[i] = np.array([tf[word] for word in vocab]) * idf
    
    return tfidf_matrix, vocab

def hamming_distance(hash1, hash2):
    return np.sum(np.array(list(hash1)) != np.array(list(hash2)))

class NLP_Pipeline:
    def __init__(self, output_file_path='NLPOutput.txt', data=None, limit=None):
        self.output_file_path = output_file_path
        self.limit = limit
        self.initialize_file(self.output_file_path)
        self.data = data
        if data == None:
            self.data = self.fetch_db()

    def initialize_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write("")

    def append_to_file(self, text):
        try:
            with open(self.output_file_path, 'a', encoding='utf-8', errors='replace') as file:
                file.write(text + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e} for text: {text}")

    def fetch_db(self,
                auth = ('mseproject', 'tuebingen2024'), 
                url = 'http://l.kremp-everest.nord:5000/query',
                columns=["url", "title", "content"]) -> list[tuple]:
        """
        Query the database to select specified columns with an optional limit.
        Args:
            columns (list): List of column names to select.
            limit (int, optional): The number of results to return. Default is None.
        Returns:
            list: List of tuples containing the query results.
        """

        print("Fetching data from the database...")

        # Construct the SELECT clause of the query
        columns_str = ', '.join(columns)
        query = f'SELECT {columns_str} FROM documents'
        if self.limit is not None:
            query += f' LIMIT {self.limit}'

        try:
            response = requests.post(url, json={'query': query}, auth=auth)
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()
            if data:
                return data
            else:
                print("No data returned.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error executing query: {e}")
            return -1

    def deduplicate_hard(self, docs):
        content_map = defaultdict(list)
        dedup_docs = []
        
        for i, doc in enumerate(docs):
            content = doc['content']
            if not content_map[content]:
                dedup_docs.append(doc)
                content_map[content].append(i)
            else:
                doc['index'] = i
        
        return dedup_docs
    
    def filter_wikipedia_links(self):
        """
        Filter out Wikipedia URLs related to editing, templates, or forums from the data attribute.
        """
        if self.data is None:
            return

        filtered_data = []
        for entry in self.data:
            url = entry.get("url", "")
            if not self.is_wikipedia_edit_or_forum(url):
                filtered_data.append(entry)
        
        self.data = filtered_data

    def is_wikipedia_edit_or_forum(self, url):
        """
        Check if a Wikipedia URL is related to editing, templates, forums, Help pages, or non-English Wikipedia sites.
        Args:
            url (str): The URL to check.
        Returns:
            bool: True if the URL is related to editing, templates, forums, Help pages, or non-English Wikipedia sites, False otherwise.
        """
        wiki_edit_patterns = [
            r'/w/index.php',
            r'/wiki/Talk:',
            r'/wiki/Special:',
            r'action=edit',
            r'action=history',
            r'action=info',
            r'oldid=',
            r'&action=',
            r'/wiki/File:',
            r'/wiki/Help:',
        ]
        # Check for non-English Wikipedia sites by their subdomains
        non_english_wiki_pattern = re.compile(r'https://(?!en\.).*\.wikipedia\.org')
        wikidata_pattern = re.compile(r'https://www\.wikidata\.org')
        
        return any(re.search(pattern, url) for pattern in wiki_edit_patterns) or \
               non_english_wiki_pattern.search(url) or \
               wikidata_pattern.search(url)


    def deduplicate_soft(self, docs, similarity_threshold=0.95, simhash_threshold=2):
        content_map = defaultdict(list)
        dedup_docs = []
        
        # extract lemmatized tokens from the preprocessed docs
        preprocessed_docs = [doc['lemmatized_tokens'] for doc in docs]
        
        # Compute SimHash for all documents
        simhashes = [compute_simhash(doc) for doc in preprocessed_docs]
        
        # Compute TF-IDF matrix
        tfidf_matrix, vocab = compute_tf_idf(preprocessed_docs)
        
        for i, (doc, prep_doc, simhash) in enumerate(zip(docs, preprocessed_docs, simhashes)):
            prep_doc = ' '.join(prep_doc)
            is_duplicate = False
            
            # Check SimHash first (faster)
            for j, existing_simhash in enumerate(simhashes[:i]):
                if hamming_distance(simhash, existing_simhash) <= simhash_threshold:
                    # If SimHash is close, check cosine similarity
                    similarity = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        doc['index'] = i
                        doc['similar_to'] = j
                        break
            
            if not is_duplicate:
                dedup_docs.append(doc)
                content_map[prep_doc].append(i)
        
        return dedup_docs

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

    def process_documents(self, deduplicate_soft=False):

        # deduplicate by hard matching content strings first
        self.data = self.deduplicate_hard(self.data)

        # filter out Wikipedia links
        self.filter_wikipedia_links()

        # next process the documents with the NLP pipeline
        data = []
        for i in tqdm(range(len(self.data))):
            content = self.data[i]['content']
            cleaned_data = self.clean_html_content(content)['cleaned_content']
            #language = self.detect_language(cleaned_data)
            #if language != 'en':
            #    continue
            tokens = self.remove_punctuation_and_tokenize(cleaned_data)
            filtered_tokens = self.remove_stop_words(tokens)
            lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
            data.append({'title': self.data[i]['title'], 'url': self.data[i]['url'], 'tokens': lemmatized_tokens})

        self.data = data

        # perform fuzzy duplicates deduplication using sim hash
        if deduplicate_soft:
            self.data = self.deduplicate_soft(self.data)

        # write the processed data to the output file
        for i in range(len(self.data)):
            output = 'Title: '+ str(self.data[i]['title'])+ '\n' + 'URL: ' + str(self.data[i]['url'])+ '\n'+ 'Tokens: ' + ' '.join(self.data[i]['tokens']) + '\n'
            self.append_to_file(output)

# Example usage:
if __name__ == "__main__":
    nlp_pipeline = NLP_Pipeline()
    nlp_pipeline.process_documents()
