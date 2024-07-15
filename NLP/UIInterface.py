import sys
import os

sys.path.append(os.path.abspath('../retriever'))
import bm25s
import KeywordFilter as kwf

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Declare the global variables at the top
retriever = None
titles = []
urls = []

#TODO: when rank is called the main function also get's executed again which breaks shit
@app.route('/rank', methods=['POST'])
def rank():
    data = request.json
    query = data.get('query')
    # HACK: modified bm25s to return indices instead of documents as results
    relevantTitles, relevantUrls = retrieval(query)
    return jsonify({'relevantTitles': relevantTitles, 'relevantUrls': relevantUrls})

def retrieval(query, k=10):
    global retriever
    global titles
    global urls

    if not retriever:
        print('ohoh')
        return jsonify({'error': 'Retriever not yet initialized'}), 500

    results, scores = retriever.retrieve(bm25s.tokenize(query), k=k)
    relevantTitles = [titles[i] for i in results[0]]
    relevantUrls = [urls[i] for i in results[0]]
    print(len(relevantTitles), len(relevantUrls))
    #return jsonify({'relevantTitles': relevantTitles, 'relevantUrls': relevantUrls})
    return relevantTitles, relevantUrls

def initialize_retriever():
    global retriever
    global titles
    global urls

    corpus, titles, urls = kwf.parse_tokens("NLPOutput.txt")
    corpusJoined = [' '.join(tokens) for tokens in corpus]
    print('Corpus parsed')
    retriever = bm25s.BM25(corpus=corpusJoined, method="bm25+")
    retriever.index(corpus)
    print(retriever)
    print('Retriever indexed')


if __name__ == '__main__':
   if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
       print("Skipping reinitialization on reloader")
   else:
       initialize_retriever()
   app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)


