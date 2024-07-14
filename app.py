import os
import pickle
from retriever.bm25 import BM25S
import bm25s
from NLP.KeywordFilter import parse_tokens
import NLP.KeywordFilter as kwf
import numpy as np

# set globals
BENCHMARK_WITH_BM25S = True
CACHE = True
CACHE_PATH = "./retriever/bm25_cache.pkl"
TOP_K = 10

def input_query():
    user_input = input("Please type your query and press Enter: ")
    user_input = user_input.split()
    return user_input

def input_model():
    user_input = input("Please type the model you want to use and press Enter:")
    return user_input

def get_top_n_results(results, titles, urls, n):
    top_n_indices = np.argsort(results)[-n:][::-1]
    top_n_results = [(titles[idx], urls[idx], results[idx]) for idx in top_n_indices]
    return top_n_results

def main():
    print("Parsing corpus...")
    corpus, titles, urls = kwf.parse_tokens("./NLP/NLPOutput10000.txt")
    corpus = [' '.join(tokens) for tokens in corpus]
    print('Corpus parsed.')

    # Create or load the BM25 model and index the corpus
    if CACHE and os.path.exists(CACHE_PATH):
        print("Loading BM25 model from cache...")
        with open(CACHE_PATH, 'rb') as f:
            retriever = pickle.load(f)
        print("BM25 model loaded from cache.")
    else:
        print("Creating BM25 model...")
        retriever = BM25S(corpus=corpus)
        print("BM25 model created.")
        if CACHE:
            print("Caching BM25 model...")
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump(retriever, f)
            print("BM25 model cached.")

    # if the bm25 model should be benchmarked with the ready-to-use version from bm25s
    if BENCHMARK_WITH_BM25S:
        retriever_bm = bm25s.BM25(corpus=corpus)
        retriever_bm.index(bm25s.tokenize(corpus))
        print("Benchmark BM25 initialized.")

    query = input_query()
    #model = input_model()  # not used yet

    # Query the corpus and get top-k results
    print("Retrieving results...")
    results = retriever.get_scores(query)
    if BENCHMARK_WITH_BM25S:
        results_bm, scores_bm = retriever_bm.retrieve(bm25s.tokenize(query), k=10)
        results_bm = [res for res in results_bm.reshape(-1)]
        scores_bm = [score for score in scores_bm.reshape(-1)]
    print("Results retrieved.")

    # get top k results
    top_n_results = get_top_n_results(results, titles, urls, TOP_K)

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