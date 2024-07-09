import KeywordFilter as kwf
import importlib
import numpy as np
import sys
import os
import bm25s
sys.path.append(os.path.abspath('../retriever'))
from bm25 import BM25S
importlib.reload(kwf)

def get_user_input():
    user_input = input("Please type your query and press Enter: ")
    user_input = user_input.split()
    return user_input



# if __name__ == "__main__":
#     corpus, titles, urls = kwf.parse_tokens("NLPOutput.txt")
#     print('Corpus parsed')
#     print(len(corpus), len(titles), len(urls))
#     bm25 = BM25S(corpus, num_threads=8, method="bm25l")
#     print('BM25 loaded')
#     try:
#         while True:
#             query = get_user_input()
#             scores = bm25.get_scores(query)
#             sorted_indexes = np.argsort(scores)[::-1]
#             relevantTitles = [titles[i] for i in sorted_indexes[:10]]
#             relevantUrls = [urls[i] for i in sorted_indexes[:10]]
#             for i in range(len(relevantTitles)):
#                 print(relevantTitles[i], 'Score: ', scores[sorted_indexes[i]])
#                 print(relevantUrls[i])
#     except KeyboardInterrupt:
#         print('interrupted!')


if __name__ == "__main__":
    #Quelle: https://github.com/xhluca/bm25s
    corpus, titles, urls = kwf.parse_tokens("NLPOutput.txt")
    corpus = [' '.join(tokens) for tokens in corpus]
    print('Corpus parsed')

    #technically working, not sure how to get the title -> indexes?
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))

    # Query the corpus and get top-k results
    query = get_user_input()
    #HACK: modified bm25s to return indices instead of documents as results
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=10)

    # Let's see what we got!
    #doc, score = results[0, 0], scores[0, 0]
    print(results)
    relevantTitles = [titles[i] for i in results[0]]
    relevantUrls = [urls[i] for i in results[0]]
    print(len(relevantTitles), len(relevantUrls))
    for i in range(len(relevantTitles)):
          print(relevantTitles[i], 'Score: ', scores[0][i])
          print(relevantUrls[i])