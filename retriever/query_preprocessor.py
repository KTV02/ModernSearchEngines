import requests
import json
import ast
from langchain_community.llms import Ollama
import gensim.downloader as api

class QueryPreprocessor:
    def __init__(self, query_string: str, model="gemma2"):
        """
        Initializes the QueryPreprocessor with a query string.
        """
        self.model = model
        self.query_string = query_string
    
    def _is_list_response(self, response_text):
        try:
            # Find the first occurrence of '[' and the last occurrence of ']'
            start_index = response_text.find('[')
            end_index = response_text.rfind(']') + 1
            
            # Extract the substring that should be the list
            if start_index != -1 and end_index != -1:
                list_str = response_text[start_index:end_index]
                
                # Ensure the response text is a properly encoded string
                list_str = list_str.encode('latin1').decode('utf-8')
                
                # Safely evaluate the substring as a Python expression
                response_list = ast.literal_eval(list_str)
                return response_list if isinstance(response_list, list) else False
            else:
                return False
        except (ValueError, SyntaxError, UnicodeDecodeError):
            return False

    def generate_search_queries_ollama(self, retries=3):
        """
        Please install ollama and run ollama pull gemma2

        """
        prompt = f"""Rephrase this search engines query 5 times: {self.query_string}. Output as python, comma-separated list."""
        
        for attempt in range(retries):
            try:
                llm = Ollama(model=self.model)  # assuming you have Ollama installed and have gemma2 model pulled

                result = llm.invoke(prompt)

                result_list = self._is_list_response(result)
                
                if result_list:
                    return result_list
                else:
                    raise ValueError(f"Response {result} is not a list")
            except (requests.RequestException, ValueError) as e:
                print(f"Error encountered: {e}. Is Ollama running?")
                if attempt == retries - 1:
                    raise
                print("Retrying...")



    def generate_search_queries_gensim(self, top_n: int = 5) -> dict:
        """
        Finds similar keywords to those that appear in the query using word embeddings.
        Args:
            top_n (int): Number of similar keywords to return for each query word.
        Returns:
            dict: A dictionary with query words as keys and lists of similar keywords as values.
        """
        self.word_vectors = api.load("glove-wiki-gigaword-100")  # Load pre-trained GloVe embeddings
        keywords = self.query_string.split()
        similar_keywords = {}

        for keyword in keywords:
            if keyword in self.word_vectors:
                similar_keywords[keyword] = self.word_vectors.most_similar(keyword, topn=top_n)
            else:
                similar_keywords[keyword] = []

        return similar_keywords

