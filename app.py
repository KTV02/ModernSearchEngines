import os
import time
import pickle
import requests
from multiprocessing import cpu_count
from retriever.bm25 import OurBM25
from retriever.query_preprocessor import QueryPreprocessor  
from retriever.compute_features import tokenize, PrecomputedDocumentFeatures
from NLP.KeywordFilter import parse_tokens
from NLP.NLPPipeline import NLP_Pipeline
import NLP.KeywordFilter as kwf
import numpy as np
import nltk
import xgboost as xgb
import numpy as np

#NOTE download necessary resources
#nltk.download("wordnet")
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

# set globals
READ_INDEX = False  # True: index will be read from DB and NLP pipeline performed on it, False: Use NLP_Output.txt
NLP_OUTPUT_PATH = "./NLP/NLPOutput.txt"
LOAD_BM25 = False
BM25_PATH = "./retriever/bm25_cache.pkl"
SAVE_MODEL = True
XGB_TOP_K = 10 
OLLAMA_AVAILABLE = True # you need to install that.
DEBUG = False # if ollama unavailable and you want to research multiple queries

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
    # for READ_INDEX == True: get most recent set of documents from the database and run the NLP pipeline
    if READ_INDEX:
        print("Reading index from database...")
        data = query_db(columns=["url", "title", "content"])
        print(len(data))
        print("...and running NLP pipeline...")
        nlp_pipeline = NLP_Pipeline(data, output_file_path=NLP_OUTPUT_PATH)
        nlp_pipeline.process_documents()
        print("NLP pipeline completed.")
    
    corpus, titles, urls = kwf.parse_tokens(NLP_OUTPUT_PATH)
    print("Doc count: " + str(len(corpus)))
    
    # Create or load the BM25 model and index the corpus
    if LOAD_BM25 and os.path.exists(BM25_PATH):
        print("Loading BM25 model from cache...")
        retriever = OurBM25.load_from_pkl(BM25_PATH)
        print("BM25 model loaded from cache.")
    else:
        print("Creating BM25 model...")
        retriever = OurBM25(corpus, num_threads=cpu_count())
        if SAVE_MODEL:
            os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
            retriever.save_to_pkl(BM25_PATH)
            print("BM25 model saved to cache.")
        print("BM25 model created.")

    query = "food and drinks"

    if OLLAMA_AVAILABLE: 
        print("Generating queries using LLM...")
        # takes a few seconds 
        q_preprocessor = QueryPreprocessor(query)
        five_queries = q_preprocessor.generate_search_queries_ollama()
        six_queries = five_queries + [query]
        tok_query = [tokenize(i) for i in six_queries]
        print(tok_query)
        # we could also add the words with the most similiar Glove Embedding here
    elif DEBUG:
        tok_query = [['food', '&', 'beverages'], ['eating', '&', 'drinking'], ['cuisine', '&', 'cocktails'], ['restaurants', '&', 'cafes'], ['meals', '&', 'refreshments'], ['food', 'and', 'drinks']] + [tokenize(query)]
    else:
        print("Not using Query processing.")
        tok_query = [tokenize(query)]

    start_total = time.time()
    assert isinstance(tok_query, list), "Query is not tokenized properly."

    # Query the corpus and get top-k results
    print("Retrieving results...")
    start_bm25 = time.time()
    x, y = retriever.retrieve(tok_query, k=50, sorted=True)
    print(y)
    bm25_time = time.time() - start_bm25

    sum_dict = {i: 0 for i in range(len(corpus))}

    # Iterate over the score and index arrays to calculate sums
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            index = y[i, j]
            sum_dict[index] += x[i, j] # get the score from x at the exact position 

    sorted_sums = sorted(sum_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_indices = [item[0] for item in sorted_sums]
    sorted_sums_values = [item[1] for item in sorted_sums]
    top_n = min(50, len(sorted_indices))

    # get the top index from the overall map 
    top_indices = sorted_indices[:top_n]

    documents = []
    for index in top_indices:
        documents.append({"body": " ".join(corpus[index]), "title": titles[index], "url": urls[index]})

    # get top 50 results for XGBoost and get XGB predictions
    start_xgb = time.time()
    extracted_features = extract_features(documents, query)
    ranker = xgb.XGBRanker()
    # Load the model from the file
    ranker.load_model('./retriever/xgb_ranker_model.json')
    y_pred = ranker.predict(extracted_features)
    XGB_top_indices = np.argsort(y_pred)[-XGB_TOP_K:][::-1]
    xgb_time = time.time() - start_xgb
    total_time = time.time() - start_total

    XGB_results = [
        {"index": top_indices[i], "title": titles[top_indices[i]], 
         "url": urls[top_indices[i]], "score": y_pred[i], 
         "body": corpus[top_indices[i]]} for i in XGB_top_indices
    ]
    
    BM25_results = [
        {"index": i, "title": titles[i], "url": urls[i], "score": sum_dict[i], 
         "body": corpus[i]} for i in top_indices[:XGB_TOP_K]
    ]

    print(f"+-------- {XGB_TOP_K} results in {total_time:.2f} seconds (BM25: {bm25_time:.2f}s + XGBoost: {xgb_time:.2f}s) --------+")

    # save the results
    with open(f"./retriever/results/{query}_XGB_results.txt", "w") as f:
        for i, res in enumerate(XGB_results):
            f.write(f"{i+1}. Title: {res['title']} (Score: {res['score']:.4f}) (Document No. {res['index']})\n   URL: {res['url']}\n\n")

    with open(f"./retriever/results/{query}_BM25_results.txt", "w") as f:
        for i, res in enumerate(BM25_results):
            f.write(f"{i+1}. Title: {res['title']} (Score: {res['score']:.4f}) (Document No. {res['index']})\n   URL: {res['url']}\n\n")


if __name__ == '__main__':
    main()