import KeywordFilter as kwf
import importlib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../retriever'))
from bm25 import BM25S
importlib.reload(kwf)

def get_user_input():
    user_input = input("Please type your query and press Enter: ")
    user_input = user_input.split()
    return user_input



if __name__ == "__main__":
    corpus, titles, urls = kwf.parse_tokens("NLPOutput.txt")
    print('Corpus parsed')
    print(len(corpus), len(titles), len(urls))
    bm25 = BM25S(corpus[:1000], num_threads=8, method="bm25+")
    print('BM25 loaded')
    query = get_user_input()
    scores = bm25.get_scores(query)
    sorted_indexes = np.argsort(scores)[::-1]
    relevantTitles = [titles[i] for i in sorted_indexes[:10]]
    relevantUrls = [urls[i] for i in sorted_indexes[:10]]
    for i in range(len(relevantTitles)):
        print(relevantTitles[i], 'Score: ', scores[sorted_indexes[i]])
        print(relevantUrls[i])
