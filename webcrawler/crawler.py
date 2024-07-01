import sqlite3
import requests
from bs4 import BeautifulSoup
import datetime
from langdetect import detect, LangDetectException
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse, urlsplit
from urllib.request import urlopen
import re

# define globals
DB_NAME = "index.db"
NUM_WORKERS = 10
FILTER_CONTENT = True
TIMEOUT = 15
TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen', 't%c3%bcbingen']

### --- DATABASE HELPER FUNCTIONS --- ###

def setup_database(db_name=DB_NAME, drop_existing=False):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # option to reset the database
    if drop_existing:
        cursor.execute("DROP TABLE IF EXISTS frontier")
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("DROP TABLE IF EXISTS incoming_links")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frontier (
        url TEXT PRIMARY KEY,
        crawled INTEGER DEFAULT 0
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        url TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        outgoing_links TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()


def index_doc(doc, index_path):
    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR IGNORE INTO documents (url, title, content, outgoing_links, timestamp)
    VALUES (?, ?, ?, ?, ?)
    ''', (doc['url'], doc['title'], doc['content'], ','.join(doc['outgoing_links']), doc['timestamp']))
    conn.commit()


def count_remaining_frontier(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT count(*) FROM frontier WHERE crawled = 0")
    total_count = cursor.fetchone()
    conn.close()

    return total_count[0]

def get_total_indexed_docs(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT count(*) FROM documents")
    index_tot = cursor.fetchone()
    conn.close()

    return index_tot[0]

### --- CRAWLER FUNCTIONS --- ###

def get_links(url, keywords=None):
    external_links = set()
    internal_links = set()
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            # Check if the link is external or internal
            if urlparse(full_url).netloc == urlparse(url).netloc:
                internal_links.add(full_url)
            else:
                external_links.add(full_url)
    
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")

    # if keywords are passed, keep only those links that contain any of the keywords
    if keywords:
        internal_links = [link for link in internal_links if any(keyword in link for keyword in keywords)]
        external_links = [link for link in external_links if any(keyword in link for keyword in keywords)]
    
    return external_links, internal_links


def crawl_page(url):
    try:

        response = requests.get(url, timeout=TIMEOUT)  # Fetch the web page
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')  # Parse the HTML content
        title = soup.title.string if soup.title else "N/A"
        content = ' '.join(soup.stripped_strings)  # Using stripped_strings to clean up the text

        # Filter out pages that do not contain "tuebingen" in their content or are not in English
        if FILTER_CONTENT:
            if not any(word in content.lower() for word in ['tübingen', 'tubingen', 'tuebingen']):
                return None

            try:
                if detect(content) != 'en':
                    return None
            except LangDetectException:
                return None

        ext_links, int_links = get_links(url, keywords=TUEBINGEN_KEYWORDS)

        doc = {
            'url': url,
            'title': title,
            'content': content,
            'outgoing_links': list(set(ext_links + int_links)),
            'timestamp': datetime.datetime.now().isoformat()
        }

    except requests.RequestException as e:
        print(f"Request exception encountered at {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected exception encountered at {url}: {e}")
        return None

    return doc

def crawl(index_path):
    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()

    # Calculate total number of URLs to be crawled
    cursor.execute("SELECT COUNT(*) FROM frontier WHERE crawled = 0")
    total_to_crawl = cursor.fetchone()[0]

    with tqdm(total=total_to_crawl, desc="Crawling Progress", unit="page") as pbar:
        while True:
            cursor.execute("SELECT url FROM frontier WHERE crawled = 0 LIMIT 10")
            rows = cursor.fetchall()
            if not rows:
                break

            urls = [row[0] for row in rows]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results = executor.map(lambda url: crawl_page(url), urls)

            for url, doc in zip(urls, results):

                if doc:
                    for link in doc['outgoing_links']: # Add outgoing links to frontier
                        cursor.execute("INSERT OR IGNORE INTO frontier (url) VALUES (?)", (link,))
                        conn.commit()

                    # Mark the URL as crawled
                    cursor.execute("UPDATE frontier SET crawled = 1 WHERE url = ?", (doc['url'],))
                    conn.commit()

                    # Index the document if it is not already in the database
                    cursor.execute("SELECT 1 FROM documents WHERE url = ? LIMIT 1", (doc['url'],))
                    if cursor.fetchone() is None:
                        index_doc(doc, index_path)
                        pbar.update(1)

                else:
                    # Mark the URL as crawled if it is not a valid document
                    cursor.execute("UPDATE frontier SET crawled = 1 WHERE url = ?", (url,))
                    conn.commit()
                    pbar.update(1)
    
    conn.close()

def initialize_frontier(initial_urls, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for url in initial_urls:
        cursor.execute("INSERT OR IGNORE INTO frontier (url) VALUES (?)", (url,))
    conn.commit()
    conn.close()

    return None

def calculate_incoming_links(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a temporary table to store incoming links
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS incoming_links (
        url TEXT PRIMARY KEY,
        incoming_count INTEGER DEFAULT 0
    )''')

    # Clear any existing data
    cursor.execute("DELETE FROM incoming_links")

    # Retrieve all documents and their outgoing links
    cursor.execute("SELECT url, outgoing_links FROM documents")
    rows = cursor.fetchall()

    for row in rows:
        url, outgoing_links = row
        outgoing_links_list = outgoing_links.split(',')

        for link in outgoing_links_list:
            cursor.execute('''
            INSERT INTO incoming_links (url, incoming_count)
            VALUES (?, 1)
            ON CONFLICT(url) DO UPDATE SET incoming_count = incoming_count + 1
            ''', (link,))

    conn.commit()
    conn.close()

def main():

    try:
        setup_database(drop_existing=False)
        initialize_frontier(initial_urls + additional_urls) 
        crawl(DB_NAME)
        calculate_incoming_links()

    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed.")

        # show number of indexed documents and remaining URLs in the frontier
        index_tot = get_total_indexed_docs()
        frontier_tot = count_remaining_frontier()

        print(f"Total indexed documents: {index_tot}")
        print(f"Remaining URLs in frontier: {frontier_tot}")


# Setup initial URLs and call main
initial_urls = [
    "https://www.tuebingen.de/en/",
    "https://en.wikipedia.org/wiki/T%C3%BCbingen",
    "https://www.uni-tuebingen.de/en.html"
]

additional_urls = [
    "https://historicgermany.travel/historic-germany/tubingen/",
    "https://www.germansights.com/tubingen/",
    "https://tuebingenresearchcampus.com/en/tuebingen/general-information/local-infos",
    "https://www.germany.travel/en/cities-culture/tuebingen.html",
    "https://uni-tuebingen.de/en/forschung/zentren-und-institute/brasilien-und-lateinamerika-zentrum/german-brazilian-symposium-2024/about-tuebingen/welcome-to-tuebingen/"
]

if __name__ == "__main__":
    main()