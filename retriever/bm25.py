"""
Implement BM25 (BM25L or BM25+) with parallelization. 
"""
import math
from typing import List, Dict, Callable, Union
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import scipy.sparse as sp
from multiprocessing import Pool

class BM25S:
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1.0, method="bm25+", num_threads=4):
        """
        Initialize BM25S with the given corpuses and parameters.

        Parameters:
        - corpus: List of documents, where each document is a list of terms.
        - k1: BM25 parameter for term frequency scaling.
        - b: BM25 parameter for document length normalization.
        - delta: BM25 parameter for BM25+ and BM25L variants.
        - method: Scoring method to use ("bm25l", "bm25+").
        - num_threads: Number of threads for parallelization.
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.method = method
        self.num_threads = num_threads

        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N
        self.doc_len = np.array([len(doc) for doc in corpus])

        self._initialize()

    def _initialize(self):
        """
        Initialize the term frequency (TF) and inverse document frequency (IDF)
        matrices for the corpus.
        """
        word_count = {}
        for doc in self.corpus:
            for word in doc:
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

        self.idf = {}
        for word, freq in word_count.items():
            self.idf[word] = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        self.tf = sp.dok_matrix((self.N, len(word_count)), dtype=np.float32)
        self.word_index = {word: idx for idx, word in enumerate(word_count)}

        for doc_idx, doc in enumerate(self.corpus):
            for word in doc:
                self.tf[doc_idx, self.word_index[word]] += 1

        self.tf = self.tf.tocsr()

    def _score_tfc_bm25plus(self, tf_array, l_d, l_avg, k1, b, delta):
        """
        Computes the term frequency component of the BM25 score using BM25+ variant.
        
        Parameters:
        - tf_array: Term frequency array.
        - l_d: Length of the document.
        - l_avg: Average document length in the corpus.
        - k1: BM25 parameter for term frequency scaling.
        - b: BM25 parameter for document length normalization.
        - delta: BM25+ parameter.
        
        Returns:
        - tfc: Term frequency component for BM25+.
        """
        num = (k1 + 1) * tf_array
        den = k1 * (1 - b + b * l_d / l_avg) + tf_array
        return (num / den) + delta

    def _score_idf_bm25plus(self, df, num_documents):
        """
        Computes the inverse document frequency component of the BM25 score using BM25+ variant.
        
        Parameters:
        - df: Document frequency of the token.
        - N: Total number of documents in the corpus.
        
        Returns:
        - idf: Inverse document frequency for the token.
        """
        return math.log((num_documents + 1) / df)

    def _score_tfc_bm25l(self, tf_array, l_d, l_avg, k1, b, delta):
        """
        Computes the term frequency component of the BM25 score using BM25L variant.
        
        Parameters:
        - tf_array: Term frequency array.
        - l_d: Length of the document.
        - l_avg: Average document length in the corpus.
        - k1: BM25 parameter for term frequency scaling.
        - b: BM25 parameter for document length normalization.
        - delta: BM25L parameter.
        
        Returns:
        - tfc: Term frequency component for BM25L.
        """
        c_array = tf_array / (1 - b + b * l_d / l_avg)
        return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)

    def _score_idf_bm25l(self, df, num_documents):
        """
        Computes the inverse document frequency component of the BM25 score using BM25L variant.
        
        Parameters:
        - df: Document frequency of the token.
        - N: Total number of documents in the corpus.
        
        Returns:
        - idf: Inverse document frequency for the token.
        """
        return math.log((num_documents + 1) / (df + 0.5))

    def _select_tfc_scorer(self, method) -> Callable:
        """
        Select the term frequency component scorer based on the method.

        Parameters:
        - method: The BM25 variant method.

        Returns:
        - A function to compute the term frequency component.
        """
        if method == "bm25l":
            return self._score_tfc_bm25l
        elif method == "bm25+":
            return self._score_tfc_bm25plus
        else:
            raise ValueError(f"Invalid scoring method: {method}. Choose from 'bm25l', 'bm25+'.")

    def _select_idf_scorer(self, method) -> Callable:
        """
        Select the inverse document frequency component scorer based on the method.

        Parameters:
        - method: The BM25 variant method.

        Returns:
        - A function to compute the inverse document frequency component.
        """
        if method == "bm25l":
            return self._score_idf_bm25l
        elif method == "bm25+":
            return self._score_idf_bm25plus
        else:
            raise ValueError(f"Invalid scoring method: {method}. Choose from 'bm25l', 'bm25+'.")

    def get_scores(self, query):
        """
        Compute BM25 scores for all documents in the corpus against a query.

        Parameters:
        - query: List of terms in the query.

        Returns:
        - scores: Array of BM25 scores for all documents.
        """
        doc_indices = list(range(self.N))
        if self.num_threads > 1:
            with Pool(self.num_threads) as pool:
                results = pool.starmap(self._parallel_score, 
                                       [(query, doc_indices[i::self.num_threads]) for i in range(self.num_threads)])
                scores = [score for sublist in results for score in sublist]
        else:
            scores = self._parallel_score(query, doc_indices)

        return np.array(scores)

    def _parallel_score(self, query, doc_indices):
        """
        Compute BM25 scores for a subset of documents.

        Parameters:
        - query: List of terms in the query.
        - doc_indices: List of document indices to score.

        Returns:
        - scores: List of BM25 scores for the documents.
        """
        scores = []
        for doc_idx in doc_indices:
            scores.append(self._score(query, doc_idx))
        return scores

    def _score(self, query, doc_idx):
        """
        Compute the BM25 score for a single document against a query.

        Parameters:
        - query: List of terms in the query.
        - doc_idx: Index of the document to score.

        Returns:
        - score: BM25 score for the document.
        """
        score = 0.0
        doc_vector = self.tf[doc_idx]
        doc_len = self.doc_len[doc_idx]

        tfc_scorer = self._select_tfc_scorer(self.method)
        idf_scorer = self._select_idf_scorer(self.method)

        for word in query:
            if word in self.word_index:
                word_idx = self.word_index[word]
                freq = doc_vector[0, word_idx]

                if freq == 0:
                    continue

                idf = idf_scorer(df=self.tf[:, word_idx].sum(), num_documents=self.N)
                tfc = tfc_scorer(tf_array=freq, l_d=doc_len, l_avg=self.avgdl, 
                                 k1=self.k1, b=self.b, delta=self.delta)

                score += idf * tfc

        return score


# Example Usage
if __name__ == "__main__":
    corpus = [
        ["hello", "world"],
        ["foo", "bar", "baz"],
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ["hello", "foo"]
    ]
    bm25 = BM25S(corpus, num_threads=4, method="bm25+")
    query = ["hello", "foo", "ablenkung", "andereAblenkung"]
    scores = bm25.get_scores(query)
    print(scores)

