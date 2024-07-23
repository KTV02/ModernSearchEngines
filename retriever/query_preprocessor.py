import requests
import json
import ast
from langchain_community.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

class QueryPreprocessor:
    """
    Query Processing with implemented query summarisation and Query Expansion.

    Created by an LLM running locally with Ollama.
    """
    def __init__(self, query_string: str, model="gemma2"):
        """
        Initializes the QueryPreprocessor with a query string.
        """
        self.model = model
        self.query_string = query_string
        self.is_summarized = False

    def generate_search_queries_ollama(self, retries=3):
        """
        Please install ollama and run ollama pull gemma2

        """
        llm = Ollama(model=self.model)  # assuming you have Ollama installed and have gemma2 model pulled

        if len(self.query_string) >= 50 and not self.is_summarized:
            # summarize 
            prompt = PromptTemplate(
                    template="Summarize this search query: {query}. Directly return this without explanation.",
                    input_variables=["query"]
            )

            chain = prompt | llm | output_parser
            self.query_string = chain.invoke({"query": self.query_string})
            self.is_summarized = True

        #prompt = f"""Rephrase this search engines query 5 times, without changing the meaning: {self.query_string}. Queries must be city of Tübingen related. Output as python, comma-separated list."""
        
        for attempt in range(retries):
            try:

                output_parser = CommaSeparatedListOutputParser()
                format_instructions = output_parser.get_format_instructions()
                prompt = PromptTemplate(
                    template="Rephrase this search engines query 8 times, without changing the meaning: {query}. Queries must be city of Tübingen related and only output the queries without explanation.\n{format_instructions}",
                    input_variables=["query"],
                    partial_variables={"format_instructions": format_instructions},
                )

                chain = prompt | llm | output_parser

                result = chain.invoke({"query": self.query_string})

                if result:
                    return result
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

        Currently not in use.

        Args:
            top_n (int): Number of similar keywords to return for each query word.
        Returns:
            dict: A dictionary with query words as keys and lists of similar keywords as values.
        """
        import gensim.downloader as api

        self.word_vectors = api.load("glove-wiki-gigaword-100")  # Load pre-trained GloVe embeddings
        keywords = self.query_string.split()
        similar_keywords = {}

        for keyword in keywords:
            if keyword in self.word_vectors:
                similar_keywords[keyword] = self.word_vectors.most_similar(keyword, topn=top_n)
            else:
                similar_keywords[keyword] = []

        return similar_keywords

