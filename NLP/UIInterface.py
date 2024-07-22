import sys
import os
import io
import time
import requests
sys.path.append(os.path.abspath('../retriever'))
sys.path.append(os.path.abspath('../UI'))
sys.path.append(os.path.abspath('../'))
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from multiprocessing import cpu_count
import numpy as np
import xgboost as xgb
from bm25 import OurBM25
from query_preprocessor import QueryPreprocessor
from compute_features import tokenize, PrecomputedDocumentFeatures
import KeywordFilter as kwf
from NLPPipeline import NLP_Pipeline
import topicmodelling as tm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import topicTree as tree

app = Flask(__name__)
CORS(app)

# Declare the global variables at the top
retriever = None
READ_INDEX = False
LOAD_FROM_DATABASE = False
NLP_OUTPUT_PATH = "NLPOutput.txt"
LOAD_BM25 = True
BM25_PATH = "./retriever/bm25_cache.pkl"
SAVE_MODEL = True
OLLAMA_AVAILABLE = False
XGB_TOP_K = 50
USE_XGB = True
titles = []
urls = []
corpus = []
topicArray = []


@app.route('/rank_batch', methods=['POST'])
def rank_batch():
    """retrieves documents for a batch file of queries
        passed via the import function of the UI"""
    file = request.files['file']
    lines = file.read().decode('utf-8').splitlines()
    results = []
    output = io.StringIO()
    
    for line in lines:
        query_number, query = line.split('\t', 1)
        relevantTitles, relevantUrls, relevanceScores = retrieval(query)
        for rank, (title, url, score) in enumerate(zip(relevantTitles, relevantUrls, relevanceScores), start=1):
            output.write(f"{query_number}\t{rank}\t{url}\t{score}\n")
    
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/plain', as_attachment=True, download_name='processed_file.txt')


@app.route('/rank', methods=['POST'])
def rank():
    """retrieves documents for the query passen by the user via the UI"""
    data = request.json
    query = data.get('query')
    relevantTitles, relevantUrls, _ = retrieval(query)
    return jsonify({'relevantTitles': relevantTitles, 'relevantUrls': relevantUrls})


@app.route('/get_links', methods=['GET'])
def get_links():
    """passes links of grouped in 5 different topics for display purposes in the UI"""
    if topicArray:
        links = [
            {"name": "Topic 1", "urls": [{"title": topicArray[0][0][0], "url": topicArray[0][0][1]}, {"title": topicArray[0][1][0], "url": topicArray[0][1][1]}, {"title": topicArray[0][2][0], "url": topicArray[0][2][1]}, {"title": topicArray[0][3][0], "url": topicArray[0][3][1]}, {"title": topicArray[0][4][0], "url": topicArray[0][4][1]}], "color": "#f8d7da"},
            {"name": "Topic 2", "urls": [{"title": topicArray[1][0][0], "url": topicArray[1][0][1]}, {"title": topicArray[1][1][0], "url": topicArray[1][1][1]}, {"title": topicArray[1][2][0], "url": topicArray[1][2][1]}, {"title": topicArray[1][3][0], "url": topicArray[1][3][1]}, {"title": topicArray[1][4][0], "url": topicArray[1][4][1]}], "color": "#d4edda"},
            {"name": "Topic 3", "urls": [{"title": topicArray[2][0][0], "url": topicArray[2][0][1]}, {"title": topicArray[2][1][0], "url": topicArray[2][1][1]}, {"title": topicArray[2][2][0], "url": topicArray[2][2][1]}, {"title": topicArray[2][3][0], "url": topicArray[2][3][1]}, {"title": topicArray[2][4][0], "url": topicArray[2][4][1]}], "color": "#d1ecf1"},
            {"name": "Topic 4", "urls": [{"title": topicArray[3][0][0], "url": topicArray[3][0][1]}, {"title": topicArray[3][1][0], "url": topicArray[3][1][1]}, {"title": topicArray[3][2][0], "url": topicArray[3][2][1]}, {"title": topicArray[3][3][0], "url": topicArray[3][3][1]}, {"title": topicArray[3][4][0], "url": topicArray[3][4][1]}], "color": "#fff3cd"},
            {"name": "Topic 5", "urls": [{"title": topicArray[4][0][0], "url": topicArray[4][0][1]}, {"title": topicArray[4][1][0], "url": topicArray[4][1][1]}, {"title": topicArray[4][2][0], "url": topicArray[4][2][1]}, {"title": topicArray[4][3][0], "url": topicArray[4][3][1]}, {"title": topicArray[4][4][0], "url": topicArray[4][4][1]}], "color": "#e8b3f5"}
        ]
    else:
        links = [{"name": "INFO", "urls": [{"title": "please run a query first", "url": None}], "color": "#f8d7da"}]
    return jsonify(links)


def retrieval(query, k=100):
    """retrieves k documents for query with BM25 and reranks the results with XGBoost,
    returning XGB_TOP_K results for topic modeling.
    Topic modeling results get included in the top results for the user to guarantee diversity
    :param query: query to be ranked
    :param k: number of documents to be pre-ranked by BM25
    """
    global retriever
    global corpus
    global titles
    global urls
    global topicArray

    if not retriever:
        print('ohoh')
        return jsonify({'error': 'Retriever not yet initialized'}), 500
    tok_query = ollamaProcess(query)
    start_total = time.time()
    assert isinstance(tok_query, list), "Query is not tokenized properly."

    # Query the corpus and get top-k results
    print("Retrieving results...")
    start_bm25 = time.time()
    x, y = retriever.retrieve(tok_query, k=k, sorted=True)
    bm25_time = time.time() - start_bm25
    sum_dict = {i: 0 for i in range(len(corpus))}

    # Iterate over the score and index arrays to calculate sums
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            index = y[i, j]
            sum_dict[index] += x[i, j]  # get the score from x at the exact position

    sorted_sums = sorted(sum_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_indices = [item[0] for item in sorted_sums]
    sorted_sums_values = [item[1] for item in sorted_sums]
    top_n = min(k, len(sorted_indices))
    # get the top index from the overall map
    top_indices = sorted_indices[:top_n]
    documents = []
    relevantTitlesBM25 = []
    relevantUrlsBM25 = []
    relevantContentBM25 = []
    for index in top_indices:
        documents.append({"body": " ".join(corpus[index]), "title": titles[index], "url": urls[index]})
        relevantTitlesBM25.append(titles[index])
        relevantUrlsBM25.append(urls[index])
        relevantContentBM25.append(corpus[index])
    if USE_XGB:
        start_xgb = time.time()
        extracted_features = extract_features(documents, query)
        ranker = xgb.XGBRanker()
        ranker.load_model('../retriever/xgb_ranker_model.json')
        y_pred = ranker.predict(extracted_features)
        XGB_top_indices = np.argsort(y_pred)[-XGB_TOP_K:][::-1]
        y_pred.sort()
        y_pred = y_pred[::-1]
        xgb_time = time.time() - start_xgb
        total_time = time.time() - start_total
        relevantTitles = []
        relevantUrls = []
        relevantContent = []
        for index in XGB_top_indices:
            relevantTitles.append(titles[top_indices[index]])
            relevantUrls.append(urls[top_indices[index]])
            relevantContent.append(corpus[top_indices[index]])
        print(f"+-------- {XGB_TOP_K} results in {total_time:.2f} seconds (BM25: {bm25_time:.2f}s + XGBoost: {xgb_time:.2f}s) --------+")
        topicModelingOutput = []
        for i in range(XGB_TOP_K):
            topicModelingOutput.append([i, relevantTitles[i], relevantUrls[i], relevantContent[i], y_pred[i]])
        try:
            #write results of XGBoost in file to be processed by topic modeling
            with open("topicmodelingoutput.txt", 'w') as file:
                file.write("")
            with open("topicmodelingoutput.txt", 'a', encoding='utf-8', errors='replace') as file:
                for sentence in topicModelingOutput:
                    file.write(str(sentence) + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e} for text: {text}")
        print(tm.perform_calculations("topicmodelingoutput.txt"))
        searchResults = tm.get_search_results()
        relevantTitles = list(searchResults["title"])
        relevantUrls = list(searchResults["url"])
        accuracy = list(searchResults["accuracy"])
        topicArray = tm.get_topic_arrays()
        return relevantTitles, relevantUrls, accuracy
    else:
        print(f"+-------- {k} results in {bm25_time:.2f} seconds using BM25 --------+")
        return relevantTitlesBM25, relevantUrlsBM25

def ollamaProcess(query):
    """Generates similar queries based on the user query with ollama language model if it's available
    to achieve better results with BM25 and XGBoost.
    :param query: query to be ranked
    :return: tokenized queries based on the user query with ollama language model+"""
    tok_query = []
    if OLLAMA_AVAILABLE:
        print("Generating queries using LLM...")
        q_preprocessor = QueryPreprocessor(query)
        five_queries = q_preprocessor.generate_search_queries_ollama()
        six_queries = five_queries + [query]
        tok_query = [tokenize(i) for i in six_queries]
        print(tok_query)
    else:
        print("Not using Query processing.")
        tok_query = [tokenize(query)]
    return tok_query

def load_data_from_file(file_path: str) -> list[tuple]:
    """Loads JSON data (typically if you have an index.json) then returns it as a list of tuples"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
        return None

def initialize_retriever():
    """initializes the BM25 ranker either by computing the weights or loading them from cache
    uses the NLP processed documents or calls the NLPPipeline to process documents directly from the index"""
    global retriever
    global corpus
    global titles
    global urls
    global embedding_model 
    if READ_INDEX:
        if LOAD_FROM_DATABASE:
            print("Reading index from database...")
            try:
                data = query_db(columns=["url", "title", "content"])
                if data is None:
                    raise ValueError("No data returned from the database.")
            except Exception as e:
                print(f"Error initializing retriever: {e}")
                return
        else: 
            print("Loading index from local checkpoint.... Else: LOAD_FROM_DATABASE=True")
            data = load_data_from_file("index.json")
        print("...and running NLP pipeline...")
        nlp_pipeline = NLP_Pipeline(data=data, output_file_path=NLP_OUTPUT_PATH)
        nlp_pipeline.process_documents()
        print("NLP pipeline completed.")
    corpus, titles, urls = kwf.parse_tokens(NLP_OUTPUT_PATH)
    print("Doc count: " + str(len(corpus)))
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
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')


def query_db(columns: list, limit: int=None, auth=('mseproject', 'tuebingen2024'), url='http://l.kremp-everest.nord:5000/query') -> list[tuple]:
    """This function stream loads the data of the index from a database to a local data object in memory.
    is able to query the Database directly if there is no output from the actual NLPPipeline directly availiable.
    
    Returns in-memory data object of all the texts and urls."""
    columns_str = ', '.join(columns)
    query = f'SELECT {columns_str} FROM documents'
    if limit is not None:
        query += f' LIMIT {limit}'
    
    json_payload = {'query': query}
    
    try:
        response = requests.post(url, json=json_payload, auth=auth, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024  # Size of each chunk in bytes
        data = []
        output_file = "index.json"

        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        data.append(chunk)
                        f.write(chunk)
                        pbar.update(len(chunk))

        data = b''.join(data).decode('utf-8')
        data = json.loads(data)
        
        if data:
            return data
        else:
            print("No data returned.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error executing query: {e}")
        return None


def extract_features(group_docs, query):
    """extracts features for the XGBoost model"""
    pdf = PrecomputedDocumentFeatures(group_docs)
    group_features = pdf.extract_query_features(query)
    normalized_features = pdf.normalize_features(group_features)
    return normalized_features

if __name__ == '__main__':
    with app.app_context():
        initialize_retriever()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
