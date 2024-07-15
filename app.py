import os
import time
import pickle
import requests
from multiprocessing import cpu_count
from retriever.bm25 import BM25S
#from retriever.query_preprocessor import QueryPreprocessor
from retriever.compute_features import tokenize, PrecomputedDocumentFeatures
import bm25s
from NLP.KeywordFilter import parse_tokens
import NLP.KeywordFilter as kwf
import numpy as np
import nltk
import xgboost as xgb
import numpy as np
#nltk.download("wordnet")

# set globals
BENCHMARK_WITH_BM25S = False
LOAD_CACHE = False
CACHE_PATH = "./retriever/bm25_cache_new.pkl"
SAVE_MODEL = False
SAVE_PATH = "./retriever/bm25_cache_new.pkl"
XGB_TOP_K = 10

def input_query():
    user_input = input("Please type your query and press Enter: ")
    return user_input

def input_model():
    user_input = input("Please type the model you want to use and press Enter:")
    return user_input

def get_top_n_results(results, titles, urls, content, n):
    top_n_indices = np.argsort(results)[-n:][::-1]
    top_n_results = [{'title': titles[idx], 'url': urls[idx], 'score': results[idx], 'body': content[idx]} for idx in top_n_indices]
    return top_n_results

def extract_features(group_docs, query):

    pdf = PrecomputedDocumentFeatures(group_docs)
    group_features = pdf.extract_query_features(query)
    normalized_features = pdf.normalize_features(group_features)
    return normalized_features


def query_db(columns: list, limit: int=None, auth = ('mseproject', 'tuebingen2024'), url = 'http://l.kremp-everest.nord:5000/query') -> list[tuple]:
    """
    Query the database to select specified columns with an optional limit.
    Args:
        columns (list): List of column names to select.
        limit (int, optional): The number of results to return. Default is None.
    Returns:
        list: List of tuples containing the query results.
    """

    # Construct the SELECT clause of the query
    columns_str = ', '.join(columns)
    query = f'SELECT {columns_str} FROM documents'
    if limit is not None:
        query += f' LIMIT {limit}'

    try:
        response = requests.post(url, json={'query': query}, auth=auth)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()
        if data:
            return data
        else:
            print("No data returned.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error executing query: {e}")
        return -1

def main():
    """
    print("Parsing corpus...")
    corpus, titles, urls = kwf.parse_tokens("./NLP/NLPOutput10000.txt")  # NOTE: parse_tokens returns list of tokens that needs to be lowercased
    corpus = [' '.join([token.lower() for token in doc]) for doc in corpus]    
    print('Corpus parsed.')
    """
    data = query_db(columns=["url", "title", "content"])
    titles, urls = [doc['title'] for doc in data], [doc['url'] for doc in data]
    print("Tokenizing corpus...")
    corpus = [tokenize(doc["content"]) for doc in data]
    print("Corpus tokenized.")

    # Create or load the BM25 model and index the corpus
    if LOAD_CACHE and os.path.exists(CACHE_PATH):
        print("Loading BM25 model from cache...")
        with open(CACHE_PATH, 'rb') as f:
            retriever = pickle.load(f)
        print("BM25 model loaded from cache.")

    else:
        print("Creating BM25 model...")
        retriever = BM25S(corpus, num_threads=cpu_count())
        if SAVE_MODEL:
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            with open(SAVE_PATH, 'wb') as f:
                pickle.dump(retriever, f)
                print("BM25 model saved to cache.")
        print("BM25 model created.")

    # if the bm25 model should be benchmarked with the ready-to-use version from bm25s
    if BENCHMARK_WITH_BM25S:
        retriever_bm = bm25s.BM25(corpus=corpus)
        retriever_bm.index(bm25s.tokenize(corpus))
        print("Benchmark BM25 initialized.")

    query = input_query()
    start = time.time()
    tok_query = tokenize(query)

    assert type(tok_query) == list, "Query is not tokenized properly."

    # Preprocess the query (not used yet)
    #query_preprocessor = QueryPreprocessor(query)
    #query_embedding = query_preprocessor.compute_embedding('sentence-transformers/all-MiniLM-L6-v2')
    #related_keywords = query_preprocessor.find_similar_keywords()

    #model = input_model()  # not used yet

    # Query the corpus and get top-k results
    print("Retrieving results...")
    scores = retriever.get_scores(tok_query)  # BUG: Scores are mostly negative 
    print(np.where(scores > 0).shape)
    print(np.where(scores == 0).shape)
    print(np.where(scores < 0).shape)
    print(scores)
    bm25_time = start - time.time()
    top_50_scores_indices = scores.argsort()[-50:][::-1]
    top_50_scores = scores[top_50_scores_indices]
    print(top_50_scores)
    if BENCHMARK_WITH_BM25S:
        results_bm, scores_bm = retriever_bm.retrieve(bm25s.tokenize(query), k=10)
        results_bm = [res for res in results_bm.reshape(-1)]
        scores_bm = [score for score in scores_bm.reshape(-1)]

    documents = [data[index] for index in top_50_scores_indices]
    for doc in documents:
        doc["body"] = doc.pop("content")

    # get top 50 results for XGBoost and get XGB predictions
    #top_n_results = get_top_n_results(results, titles, urls, corpus, 50)
    extracted_features = extract_features(documents, query)
    ranker = xgb.XGBRanker()

    # Load the model from the file
    ranker.load_model('./retriever/xgb_ranker_model.json')
    y_pred = ranker.predict(extracted_features)
    top_n_indices = np.argsort(y_pred)[-XGB_TOP_K:][::-1]

    results = [
        {"title": titles[i], "url": urls[i], "score": y_pred[i], "body": corpus[i]} for i in top_n_indices
    ]
    # Print the top n documents
    print("Top n documents:")
    for doc in results:
        print(f"URL: {doc['url']} ({doc['score']}) \nTitle: {doc['title']}\n")

    print(f"+-------- {XGB_TOP_K} results in {start - time.time()} seconds ({bm25_time} + {start - bm25_time}) --------+")
    return

    # save the results
    with open(f"./retriever/results/{query}_results.txt", "w") as f:
        for i, res in enumerate(top_n_results):
            f.write(f"{i+1}. {res[1]} ({res[0]}) - {res[2]}\n")

    if BENCHMARK_WITH_BM25S:
        with open(f"./retriever/results/{query}_results_bm.txt", "w") as f:
            for i, res in enumerate(zip(scores_bm, results_bm)):
                f.write(f"{i+1}. {res[0]} ({res[1]})\n")

if __name__ == '__main__':
    main()