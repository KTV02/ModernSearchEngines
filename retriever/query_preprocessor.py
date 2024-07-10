import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import torch

class QueryPreprocessor:
    def __init__(self, query_string: str):
        """
        Initializes the QueryPreprocessor with a query string.
        """
        self.query_string = query_string
        self.word_vectors = api.load("glove-wiki-gigaword-100")  # Load pre-trained GloVe embeddings

    def compute_embedding(self, model_id: str):
        """
        Computes embeddings for specific language model
        Args:
            model_id: huggingface model ID
        Returns:
            embeddings (numpy.Array): latent space embeddings of query
        """

        model = SentenceTransformer(model_id)
        embeddings = model.encode(self.query_string)

        return embeddings
    
    def find_similar_keywords(self, top_n: int = 5) -> dict:
        """
        Finds similar keywords to those that appear in the query using word embeddings.
        Args:
            top_n (int): Number of similar keywords to return for each query word.
        Returns:
            dict: A dictionary with query words as keys and lists of similar keywords as values.
        """
        keywords = self.query_string.split()
        similar_keywords = {}

        for keyword in keywords:
            if keyword in self.word_vectors:
                similar_keywords[keyword] = self.word_vectors.most_similar(keyword, topn=top_n)
            else:
                similar_keywords[keyword] = []

        return similar_keywords

# Example usage
query = "cheap apartments tuebingen"
q_preprocessor = QueryPreprocessor(query)

model_id = 'sentence-transformers/all-MiniLM-L6-v2'
emb = q_preprocessor.compute_embedding(model_id)
print(emb.shape)

rel_keywords = q_preprocessor.find_similar_keywords()
print(rel_keywords)