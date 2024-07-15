import math
import json
import re
from collections import Counter
from typing import List, Dict
from nltk.corpus import wordnet
import nltk
import numpy as np

def tokenize(text):
    """
    Example tokens.
    """
    text = text.lower()
    text = re.sub('[,\.!]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text.split(' ')

class PrecomputedDocumentFeatures:
    """
    Precompute Features for XGBoost
    """
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_count = len(documents)
        self.idf_cache_body, self.idf_max_body = self.IDF(part="body")
        self.idf_cache_title, self.idf_max_title = self.IDF(part="title")
        self.idf_cache_url, self.idf_max_url = self.IDF(part="url")
        self.precomputed_docs = []
        self._precompute_features()
    
    def IDF(self, part) -> Dict[str, float]:
        """ Estimate inverse document frequencies based on a corpus of documents. """
        idfs = {}
        for doc in self.documents:
            if part=="body":
                words = set(tokenize(f"{doc['body']}"))
            elif part=="url":
                words = set(tokenize(f"{doc['url']}"))
            elif part=="title":
                words = set(tokenize(f"{doc['title']}"))
            else:
                raise RuntimeError("IDF argument not feasible.")
            for word in words:
                if word not in idfs:
                    idfs[word] = 1
                else:
                    idfs[word] += 1
        for word in idfs:
            idfs[word] = math.log(self.doc_count / idfs[word])
        return idfs, math.log(self.doc_count)
    
    def stream_length(self, text: str) -> int:
        """ Determine the length of the text in terms of word count. """
        return len(tokenize(text))
    
    def term_frequencies(self, text: str) -> Dict[str, int]:
        """ Compute term frequencies for a given text. """
        return Counter(tokenize(text))
    
    def _precompute_features(self):
        """ Pre-compute features for all documents. """
        for doc in self.documents:
            body = doc['body'].lower()
            title = doc['title'].lower()
            url = doc['url'].lower()
            url_splitted = tokenize(url.replace('/', ' ').replace('.', ' '))
            url_terms = ' '.join(url_splitted)
            
            body_term_freq = self.term_frequencies(body)
            title_term_freq = self.term_frequencies(title)
            url_term_freq = self.term_frequencies(url_terms)
            
            body_features = {
                'stream_length': self.stream_length(body),
                'term_freq': body_term_freq
            }
            
            title_features = {
                'stream_length': self.stream_length(title),
                'term_freq': title_term_freq
            }
            
            url_features = {
                'stream_length': len(url_splitted[2]), # only the main part of the url e.g. tuebingen from https://www.tuebingen.de
                'term_freq': url_term_freq
            }
            
            # Precompute TF-IDF values
            body_tfidf = self.compute_tfidf(body_term_freq, part="body")
            title_tfidf = self.compute_tfidf(title_term_freq, part="title")
            url_tfidf = self.compute_tfidf(url_term_freq, part="url")

            # For each document: precomputed features
            self.precomputed_docs.append({
                'body': body_features,
                'title': title_features,
                'url': url_features,
                'body_tfidf': body_tfidf,
                'title_tfidf': title_tfidf,
                'url_tfidf': url_tfidf,
                'num_slashes_url': url.count('/'),
                'num_char_url': len(url)
            })

    def compute_tfidf(self, term_freqs: Dict[str, int], part) -> Dict[str, float]:
        """ Compute TF-IDF values for a given term frequency dictionary. """
        tfidf = {}
        for term, freq in term_freqs.items():
            if part == "title":
                tfidf[term] = freq * self.idf_cache_title.get(term, self.idf_max_title) # if term is not in cache use idf_max_title
            elif part == "url":
                tfidf[term] = freq * self.idf_cache_url.get(term, self.idf_max_url)
            elif part == "body":
                tfidf[term] = freq * self.idf_cache_body.get(term, self.idf_max_body)
            else:
                raise RuntimeError("TFIDF argument not feasible.")   
        return tfidf
    
    def get_synonyms(self, term):
        """ Get synonyms from Wordnet."""
        #nltk.download('wordnet')
        synonyms = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms
    
    def compute_cosine_similarity(self, query_vector, doc_vector):
        """ Compute cosine similarity between two vectors. """
        # Ensure both vectors cover the same dimensions
        terms = set(query_vector.keys()).union(set(doc_vector.keys()))
        query_vec = np.array([query_vector.get(term, 0.0) for term in terms])
        doc_vec = np.array([doc_vector.get(term, 0.0) for term in terms])

        # Compute cosine similarity
        dot_product = np.dot(query_vec, doc_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_doc = np.linalg.norm(doc_vec)
        return dot_product / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0.0

    def extract_query_features(self, query: str, debug=False) -> list:
        """ Extract query-specific features using pre-computed document features. """
        query_terms = tokenize(query)
        query_term_freq = self.term_frequencies(query)
        query_tfidf_body = self.compute_tfidf(query_term_freq, part="body")
        query_tfidf_title = self.compute_tfidf(query_term_freq, part="title")
        query_tfidf_url = self.compute_tfidf(query_term_freq, part="url")
        
        all_features = []

        for doc_idx, doc_features in enumerate(self.precomputed_docs):  # for each document
            feature_values = []
            feature_dict = {}
            for part in ['body', 'title', 'url']:
                
                # TF IDF match in Query (How many texts are covered with this word * the word frequency)
                tf_idf_values = [doc_features[f'{part}_tfidf'].get(term, 0) for term in query_terms]  # TF IDF query term (if word appears in all documents then its null!)
                sum_tf_idf = sum(tf_idf_values)
                feature_values.append(sum_tf_idf)
                if debug:
                    feature_dict[f'{part}_sum_tf_idf'] = sum_tf_idf
                
                # Term Frequencies match in Query (How many texts are covered with this word)
                term_freq = doc_features[part]['term_freq']
                sum_term_occurrences = sum(term_freq.get(term, 0) for term in query_terms)
                feature_values.append(sum_term_occurrences)
                if debug:
                    feature_dict[f'{part}_sum_term_occurrences'] = sum_term_occurrences
                
                # How many tokens after tokenization
                stream_length = doc_features[part]['stream_length']
                feature_values.append(stream_length)
                if debug:
                    feature_dict[f'{part}_stream_length'] = stream_length
                
                # Synonym match ~ semantic feature
                synonym_count = 0
                for term in query_terms:
                    synonyms = self.get_synonyms(term)
                    for synonym in synonyms:
                        synonym_count += term_freq.get(synonym, 0)
                feature_values.append(synonym_count)
                if debug:
                    feature_dict[f'{part}_synonym_count'] = synonym_count

                # Cosine similarity between TF IDF vectors of query and part
                query_tfidf_vector = query_tfidf_body if part == 'body' else (query_tfidf_title if part == 'title' else query_tfidf_url)
                doc_tfidf_vector = doc_features[f'{part}_tfidf']
                cosine_sim = self.compute_cosine_similarity(query_tfidf_vector, doc_tfidf_vector)
                feature_values.append(cosine_sim)
                if debug:
                    feature_dict[f'{part}_cosine_similarity'] = cosine_sim
            
            # Add additional document-level features
            feature_values.extend([
                doc_features['num_slashes_url'],
                doc_features['num_char_url']
            ])
            if debug:
                feature_dict['num_slashes_url'] = doc_features['num_slashes_url']
                feature_dict['num_char_url'] = doc_features['num_char_url']
            
            if debug:
                all_features.append(feature_dict)
            else:
                all_features.append(feature_values)
                
        if debug:
            return json.dumps(all_features, indent=4)
        return all_features
    
    def normalize_features(self, features: List[List[float]]) -> List[List[float]]:
        """ Normalize features to a 0-1 range. """
        features = np.array(features)
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        return ((features - min_vals) / (max_vals - min_vals)).tolist()


#pdf = PrecomputedDocumentFeatures(documents)
#query = "example body2"
#features = pdf.extract_query_features(query, debug=False)
#normalized_features = pdf.normalize_features(features)

#print(features)
#print(normalized_features)