"""
Vector Space Model (sklearn vectorizer) based on TF-IDF

The Vector Space Model represents documents and terms as vectors in a multi-dimensional space. 
Each dimension corresponds to a unique term in the entire corpus of documents.
"""

# document metadata
# Parametric indices for numbers
# Standard inverted indices for each text column => merge into a shared inverted index (zones in postings)
# Naive way:  Single term query: in the more zones it is the higher it gets scored
import math
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from scipy import spatial
import nltk

nltk.download('stopwords')

class VectorSpaceModel:
    def __init__(self):
        self.documents = []
        self.vocab = set()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, doc):
        tokens = self.tokenizer.tokenize(doc.lower())
        return [token for token in tokens if token not in self.stop_words]

    def add_document(self, doc_id, text):
        self.documents.append((doc_id, text))
    
    def _build_vocab(self):
        for doc_id, text in self.documents:
            tokens = self.tokenize(text)
            self.vocab.update(tokens)
    
    def _create_word_dicts(self):
        word_dicts = []
        for doc_id, text in self.documents:
            word_dict = dict.fromkeys(self.vocab, 0)
            tokens = self.tokenize(text)
            for token in tokens:
                word_dict[token] += 1
            word_dicts.append(word_dict)
        return word_dicts

    def compute_tf(self, word_dict, bow):
        tf_dict = {}
        bow_count = len(bow)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(bow_count)
        return tf_dict

    def compute_idf(self, doc_list):
        idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
        N = len(doc_list)
        for doc in doc_list:
            for word, val in doc.items():
                if val > 0:
                    idf_dict[word] += 1
        for word, val in idf_dict.items():
            idf_dict[word] = math.log10(N / float(val))
        return idf_dict

    def compute_tfidf(self, tf_bow, idfs):
        tfidf = {}
        for word, val in tf_bow.items():
            tfidf[word] = val * idfs.get(word, 0)
        return tfidf

    def query(self, query_text):
        query_tokens = self.tokenize(query_text)
        query_dict = dict.fromkeys(self.vocab, 0)
        for token in query_tokens:
            query_dict[token] += 1

        word_dicts = self._create_word_dicts()
        tf_docs = [self.compute_tf(word_dict, self.tokenize(text)) 
                   for word_dict, (doc_id, text) in zip(word_dicts, self.documents)]
        idfs = self.compute_idf(word_dicts)
        tfidf_docs = [self.compute_tfidf(tf, idfs) for tf in tf_docs]

        query_tf = self.compute_tf(query_dict, query_tokens)
        query_tfidf = self.compute_tfidf(query_tf, idfs)

        query_vector = [query_tfidf[word] for word in self.vocab]
        doc_vectors = [[tfidf[word] for word in self.vocab] for tfidf in tfidf_docs]

        results = {}
        for doc_id, doc_vector in zip([doc_id for doc_id, text in self.documents], doc_vectors):
            cosine_sim = 1 - spatial.distance.cosine(query_vector, doc_vector)
            if not math.isnan(cosine_sim):
                results[doc_id] = cosine_sim

        return results

vsm = VectorSpaceModel()

# Add documents
vsm.add_document(0, "The iPhone revolutionized the mobile phone industry")
vsm.add_document(1, "Tesla is known for electric cars and innovation")
vsm.add_document(2, "Elon Musk founded SpaceX and Tesla")
vsm.add_document(3, "Steve Jobs was the co-founder of Apple")
vsm.add_document(4, "Artificial Intelligence is transforming many industries")

# Build vocabulary
vsm._build_vocab()

# Query the model
query_text = "Tesla innovation"
results = vsm.query(query_text)

# Print results
print("Query Results for terms '{}':".format(query_text))
for doc_id, score in results.items():
    print(f"Document {doc_id} has score {score:.4f}")
