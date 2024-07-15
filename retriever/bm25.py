"""
Lucene Implementation of BM25.

This implementation follows the Lucene (accurate) variant of the BM25 algorithm, 
which uses exact document lengths instead of lossy compressed lengths.

References:
- BM25 Literature Review: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
"""
import math
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


class BM25S:
    def __init__(self, corpus, k1=1.2, b=0.75, num_threads=4):
        """
        Initialize BM25S with the given corpuses and parameters.

        Parameters:
        - corpus: List of documents, where each document is a list of terms.
        - k1: BM25 parameter for term frequency scaling and non-linear term frequency normalization
        - b: BM25 parameter for document length tf values normalization.
        - num_threads: Number of threads for parallelization.
        """
        if k1 < 0:
            raise ValueError("k1 must be greater or equal to 0")
        if num_threads <= 0:
            raise ValueError("num threads must be greater than 1")
        if not (0 <= b <= 1):
            raise ValueError(f"b must be between 0 and 1, got {b}")
        self.k1 = k1
        self.b = b
        self.num_threads = num_threads
        if not isinstance(corpus[0], list):
            raise ValueError("Corpus is not tokenized properly.")
        self.corpus = corpus
        self.N = len(corpus)
        self._initialize()

    def _initialize(self):
        """
        Initialize the term frequency (TF) and inverse document frequency (IDF)
        matrices for the corpus.
        """
        term_frequencies = {}  # To store term frequencies
        doc_freqs = {}   # To store document frequencies
        total_terms = 0

        for doc in tqdm(self.corpus, desc="Processing corpus"):
            total_terms += len(doc)
            unique_words_in_doc = set()  # Track unique words in the current document
            for word in doc:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                    doc_freqs[word] = 0  # Initialize document frequency for new words
                term_frequencies[word] += 1
                unique_words_in_doc.add(word)
        
            # Update document frequencies
            for word in unique_words_in_doc:
                doc_freqs[word] += 1

        self.idf = {word: self._idf(doc_freq, self.N) for word, doc_freq in doc_freqs.items()}
        self.tf = sp.dok_matrix((self.N, len(term_frequencies)), dtype=np.float32)
        self.word_index = {word: idx for idx, word in enumerate(term_frequencies)}

        for doc_idx, doc in tqdm(enumerate(self.corpus), desc="Calculating TF", total=self.N):
            for word in doc:
                self.tf[doc_idx, self.word_index[word]] += 1

        self.tf = self.tf.tocsr()
        self.doc_len = np.array(self.tf.sum(axis=1)).flatten()

        # sum of all term frequencies divided by the number of documents
        self.avgdl = total_terms / self.N

    def get_corpus(self):
        return self.corpus
    
    def _get_topk_results(self, query_scores: np.ndarray, k: int, sorted: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently retrieve the top-k elements from a numpy array.
        
        Args:
            query_scores (np.ndarray): Array of scores.
            k (int): Number of top elements to retrieve.
            sorted (bool): If True, returns the top-k elements in sorted order.
        
        Returns:
            tuple: Top-k scores and their corresponding indices.
        """
        # Efficiently get the indices of the top-k elements
        topk_indices = np.argpartition(query_scores, -k)[-k:]

        # Retrieve the top-k scores
        topk_scores = query_scores[topk_indices]

        if sorted:
            # If sorting is required, sort the top-k scores and their indices
            sort_indices = np.argsort(topk_scores)[::-1]
            topk_indices = topk_indices[sort_indices]
            topk_scores = topk_scores[sort_indices]

        return topk_scores, topk_indices

    def _get_top_k_results(self, query_tokens_single: List[str], k: int = 1000, sorted: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        scores_q = self._score(query_tokens_single)
        return self._get_topk_results(scores_q, k=k, sorted=sorted)

    def _score(self, query: List[str]):
        """
        Compute BM25 scores for all documents in the corpus against a query.

        Parameters:
        - query: List of terms in the query.

        Returns:
        - scores: Array of BM25 scores for all documents.
        """
        scores = np.zeros(self.N)
        for word in query:
            if word in self.word_index:
                word_idx = self.word_index[word]
                idf = self.idf[word]
                for doc_idx in range(self.N):
                    freq = self.tf[doc_idx, word_idx]
                    if freq == 0:
                        continue
                    scores[doc_idx] += idf * self._compute_tf_component(freq, doc_idx)
        return scores
    
    def retrieve(self, query_tokens: List[List[str]], k: int = 50, sorted: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the top-k BM25 scores and their corresponding document indices for a list of queries.
        
        Parameters:
        - query_tokens: List of tokenized queries.
        - k: Number of top elements to retrieve for each query.
        - sorted: If True, returns the top-k elements in sorted order.

        Returns:
        - Tuple of numpy arrays: scores and indices of the top-k elements.
        """
        topk_fn = partial(self._get_top_k_results, k=k, sorted=sorted)

        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(tqdm(executor.map(topk_fn, query_tokens), total=len(query_tokens), desc="Scoring queries"))
        else:
            results = list(tqdm(map(topk_fn, query_tokens), total=len(query_tokens), desc="Scoring queries"))

        scores, indices = zip(*results)
        return np.array(scores), np.array(indices)


    def _compute_tf_component(self, freq, doc_idx):
        """
        Compute the term frequency component of the BM25 score.

        Formula based on Lucene (accurate)

        Parameters:
        - freq: Term frequency in the document.
        - doc_idx: Document index.

        Returns:
        - TF component of the BM25 score.
        """
        doc_len = self.doc_len[doc_idx]
        norm = self.k1 * ((1 - self.b) + self.b * doc_len / self.avgdl)
        return freq / (freq + norm)

    def _idf(self, doc_freq, doc_count):
        """
        Compute the inverse document frequency component.

        Formula based on Lucene (accurate).

        Parameters:
        - doc_freq: df_t = Document frequency of the term.
        - doc_count: N = Total number of documents.

        Returns:
        - IDF value.
        """
        return float(math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))

# Example Usage
if __name__ == "__main__":
    corpus = [
        ["hello", "world", "ddifern"],
        ["foo", "bar", "baz"],
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ["hello", "foo"],
        [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        "this", "is", "a", "long", "sentence", "to", "test", "the", "bm25", "implementation",
        "we", "need", "to", "include", "many", "different", "words", "to", "ensure",
        "that", "the", "algorithm", "can", "handle", "a", "larger", "vocabulary",
        "and", "compute", "the", "scores", "correctly", "even", "with", "a", "long",
        "document", "like", "this", "one", "which", "contains", "contains", "contains", "multiple", "unique",
        "terms", "and", "repeated", "terms", "to", "make", "sure", "that", "document",
        "frequencies", "and", "term", "frequencies", "are", "calculated", "properly"
        ]
    ]
    bm25 = BM25S(corpus, num_threads=4)
    query = [["hello", "term"]]
    x, y = bm25.retrieve(query, k=5)
    print(x)

    query_many = [["hello", "term"], ["hello", "foo"], ["make", "ipsum"], ["long", "long", "long"]]
    x, y = bm25.retrieve(query_many, k=5)
    print(x)
    #scores = bm25.score(query)
    #print(scores)
