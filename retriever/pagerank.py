import sqlite3
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

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

def build_sparse_matrix(web_graph, urls):
    N = len(urls)
    url_index = {url: idx for idx, url in enumerate(urls)}

    row_ind = []
    col_ind = []
    data = []

    for url, outgoing_links in web_graph.items():
        if outgoing_links:
            src_idx = url_index[url]
            for outgoing in outgoing_links:
                if outgoing in url_index:
                    dest_idx = url_index[outgoing]
                    row_ind.append(dest_idx)
                    col_ind.append(src_idx)
                    data.append(1 / len(outgoing_links))

    M = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))
    return M

def compute_pagerank(web_graph, damping_factor=0.85, max_iterations=100, tol=1.0e-6):
    urls = list(web_graph.keys())
    N = len(urls)
    M = build_sparse_matrix(web_graph, urls)
    
    # Initialize pagerank vector
    pagerank = np.ones(N) / N
    
    for iteration in tqdm(range(max_iterations), desc="Computing PageRank"):
        new_pagerank = (1 - damping_factor) / N + damping_factor * M @ pagerank
        if np.linalg.norm(new_pagerank - pagerank, ord=1) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        pagerank = new_pagerank
    
    return {url: pagerank[idx] for url, idx in zip(urls, range(N))}

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

def get_pagerank(urls):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT url, pagerank FROM documents WHERE url IN ({seq})".format(
        seq=','.join(['?']*len(urls))), urls)
    rows = cursor.fetchall()
    conn.close()
    return {url: rank for url, rank in rows}

db_path = 'index.db'
run_pagerank(db_path)
pagerank_statistics(db_path)

# Example usage of get_pagerank
urls_to_check = ['http://example.com/page1', 'http://example.com/page2']
pagerank_scores = get_pagerank(urls_to_check)
print("PageRank scores for specified URLs:")
print(pagerank_scores)
