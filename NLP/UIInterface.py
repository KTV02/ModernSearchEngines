import sys
import os
import io
import time
sys.path.append(os.path.abspath('../retriever'))
sys.path.append(os.path.abspath('../UI'))
sys.path.append(os.path.abspath('../'))
import KeywordFilter as kwf
import topicTree as tree
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from multiprocessing import cpu_count
from bm25 import OurBM25
from query_preprocessor import QueryPreprocessor
import xgboost as xgb
#from query_preprocessor import QueryPreprocessor
from compute_features import tokenize, PrecomputedDocumentFeatures

app = Flask(__name__)
CORS(app)

# Declare the global variables at the top
retriever = None
READ_INDEX = False
#HACK: Path is relative to where bm25 is sitting
NLP_OUTPUT_PATH = "NLPOutput.txt"
LOAD_BM25 = True
BM25_PATH = "./retriever/bm25_cache.pkl"
SAVE_MODEL = True
OLLAMA_AVAILABLE = False
XGB_TOP_K = 10
USE_XGB = True
titles = []
urls = []
corpus = []

@app.route('/rank_batch', methods=['POST'])
def rank_batch():
    file = request.files['file']
    lines = file.read().decode('utf-8').splitlines()
    results = []
    output = io.StringIO()
    for line in lines:
        relevantTitles, relevantUrls = retrieval(line)
        results.append({
            'query': line,
            'relevantTitles': relevantTitles,
            'relevantUrls': relevantUrls
        })
        output.write(f"Query: {line}\n")
        output.write(f"Results: {', '.join(relevantTitles)}\n\n")
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/plain', as_attachment=True, download_name='processed_file.txt')

@app.route('/rank', methods=['POST'])
def rank():
    data = request.json
    query = data.get('query')
    relevantTitles, relevantUrls = retrieval(query)
    return jsonify({'relevantTitles': relevantTitles, 'relevantUrls': relevantUrls})

@app.route('/get_tree', methods=['GET'])
def makeTree():
    print("hello")
    dtree = tree.get_tree()
    print(dtree)
    return dtree
#k influences how many documents are retrieved by bm25 -> preranking
def retrieval(query, k=50):
    global retriever
    global corpus
    global titles
    global urls

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
    top_n = min(50, len(sorted_indices))

    # get the top index from the overall map
    top_indices = sorted_indices[:top_n]

    documents = []
    #HACK: titles and urls spererately for testing
    relevantTitlesBM25 = []
    relevantUrlsBM25 = []
    relevantContentBM25 = []
    for index in top_indices:
        documents.append({"body": " ".join(corpus[index]), "title": titles[index], "url": urls[index]})
        relevantTitlesBM25.append(titles[index])
        relevantUrlsBM25.append(urls[index])
        relevantContentBM25.append(corpus[index])
    if USE_XGB:
        #wie bekomm ich den score von xgb
        start_xgb = time.time()
        extracted_features = extract_features(documents, query)
        ranker = xgb.XGBRanker()
        # Load the model from the file
        ranker.load_model('../retriever/xgb_ranker_model.json')
        y_pred = ranker.predict(extracted_features)
        XGB_top_indices = np.argsort(y_pred)[-XGB_TOP_K:][::-1]
        xgb_time = time.time() - start_xgb
        total_time = time.time() - start_total
        relevantTitles = []
        relevantUrls = []
        for index in XGB_top_indices:
            relevantTitles.append(titles[top_indices[index]])
            relevantUrls.append(urls[top_indices[index]])
        print(f"+-------- {XGB_TOP_K} results in {total_time:.2f} seconds (BM25: {bm25_time:.2f}s + XGBoost: {xgb_time:.2f}s) --------+")
        #print only to be able to directly compare bm25 with xgboost
        topicModelingOutput = []
        for i in range(k):
            #HACK: prints content
            topicModelingOutput.append([i, relevantTitlesBM25[i], relevantUrlsBM25[i], relevantContentBM25[i], x[0][i]])
            #print(i, relevantTitlesBM25[i], relevantUrlsBM25[i], relevantContentBM25[i], x[0][i])
        #HACK file output for topic modeling
        #try:
        #    with open("topicmodelingoutput.txt", 'a', encoding='utf-8', errors='replace') as file:
        #        for sentence in topicModelingOutput:
        #            file.write(str(sentence) + "\n")
        #except UnicodeEncodeError as e:
        #    print(f"UnicodeEncodeError: {e} for text: {text}")
        return relevantTitles, relevantUrls
    else:
        print(f"+-------- {k} results in {bm25_time:.2f} seconds using BM25 --------+")
        return relevantTitlesBM25, relevantUrlsBM25

def ollamaProcess(query):
    tok_query = []
    if OLLAMA_AVAILABLE:
        print("Generating queries using LLM...")
        # takes a few seconds
        q_preprocessor = QueryPreprocessor(query)
        five_queries = q_preprocessor.generate_search_queries_ollama()
        six_queries = five_queries + [query]
        tok_query = [tokenize(i) for i in six_queries]
        print(tok_query)
        # we could also add the words with the most similiar Glove Embedding here
    #elif DEBUG:
    #    tok_query = [['food', '&', 'beverages'], ['eating', '&', 'drinking'], ['cuisine', '&', 'cocktails'], ['restaurants', '&', 'cafes'], ['meals', '&', 'refreshments'], ['food', 'and', 'drinks']] + [tokenize(query)]
    else:
        print("Not using Query processing.")
        tok_query = [tokenize(query)]
    return tok_query


def initialize_retriever():
    global retriever
    global corpus
    global titles
    global urls
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
    except request.exceptions.RequestException as e:
        print(f"Error executing query: {e}")
        return -1

def extract_features(group_docs, query):

    pdf = PrecomputedDocumentFeatures(group_docs)
    group_features = pdf.extract_query_features(query)
    normalized_features = pdf.normalize_features(group_features)
    return normalized_features



if __name__ == '__main__':
    initialize_retriever()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)


