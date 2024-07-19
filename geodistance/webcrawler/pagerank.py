import sqlite3
import requests
from bs4 import BeautifulSoup
import datetime
from langdetect import detect, LangDetectException
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse





import sqlite3
import numpy as np
from tqdm import tqdm

def get_web_graph(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT url, outgoing_links FROM documents")
    rows = cursor.fetchall()
    conn.close()

    web_graph = {}
    for url, outgoing_links in rows:
        if outgoing_links:
            web_graph[url] = outgoing_links.split(',')
        else:
            web_graph[url] = []
    return web_graph

def initialize_pagerank(urls):
    N = len(urls)
    pagerank = {url: 1/N for url in urls}
    return pagerank
#100 already takes a long time on just 1000 urls
def compute_pagerank(web_graph, damping_factor=0.85, max_iterations=10, tol=1.0e-6):
    urls = list(web_graph.keys())
    N = len(urls)
    pagerank = initialize_pagerank(urls)
    new_pagerank = pagerank.copy()

    for iteration in tqdm(range(max_iterations), desc="Computing PageRank"):
        for url in urls:
            rank_sum = 0.0
            for u in urls:
                if url in web_graph[u]:
                    rank_sum += pagerank[u] / len(web_graph[u])
            new_pagerank[url] = (1 - damping_factor) / N + damping_factor * rank_sum

        # Normalize the PageRank values
        norm_factor = sum(new_pagerank.values())
        for url in new_pagerank:
            new_pagerank[url] /= norm_factor

        # Check for convergence
        if all(abs(new_pagerank[url] - pagerank[url]) < tol for url in urls):
            print(f"Converged after {iteration + 1} iterations.")
            break

        pagerank = new_pagerank.copy()

    return pagerank

def save_pagerank(db_path, pagerank):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN pagerank REAL")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    for url, rank in pagerank.items():
        cursor.execute("UPDATE documents SET pagerank = ? WHERE url = ?", (rank, url))
    conn.commit()
    conn.close()


def pagerank_statistics(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Get the top 10 websites by PageRank
    cursor.execute("SELECT title, pagerank FROM documents ORDER BY pagerank DESC LIMIT 10;")
    top_rows = cursor.fetchall()

    # Get the sum of PageRank values
    cursor.execute("SELECT SUM(pagerank) FROM documents;")
    total_pagerank = cursor.fetchone()[0]

    # Get the worst PageRank
    cursor.execute("SELECT MIN(pagerank) FROM documents;")
    worst_pagerank = cursor.fetchone()[0]

    # Get the best PageRank
    cursor.execute("SELECT MAX(pagerank) FROM documents;")
    best_pagerank = cursor.fetchone()[0]

    # Get all PageRank values for mean and median calculation
    cursor.execute("SELECT pagerank FROM documents;")
    all_pageranks = [row[0] for row in cursor.fetchall()]
    
    mean_pagerank = np.mean(all_pageranks)
    median_pagerank = np.median(all_pageranks)

    print("Top 10 Websites by PageRank:")
    for row in top_rows:
        print(f"Title: {row[0]}, PageRank: {row[1]}")

    print(f"\nTotal PageRank (should be ~1): {total_pagerank}")
    print(f"Worst PageRank: {worst_pagerank}")
    print(f"Best PageRank: {best_pagerank}")
    print(f"Mean PageRank: {mean_pagerank}")
    print(f"Median PageRank: {median_pagerank}")

    conn.close()


def run_pagerank(db_path):
    web_graph = get_web_graph(db_path)
    pagerank = compute_pagerank(web_graph)
    save_pagerank(db_path, pagerank)
    print("PageRank values have been computed and stored in the database.")


db_path = 'index.db'
run_pagerank(db_path)
pagerank_statistics(db_path)





