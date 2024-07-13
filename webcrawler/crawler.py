# This Python file uses the following encoding: utf-8
import requests
from bs4 import BeautifulSoup
import datetime
from langdetect import detect, LangDetectException
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import re

# define globals
API_URL = "http://l.kremp-everest.nord:5000"  # Replace with your Flask API URL
NUM_WORKERS = 10
FILTER_CONTENT = True
TIMEOUT = 15
TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen','tuebing','tübing', 't%c3%bcbingen','eberhard','karls']

### --- DATABASE HELPER FUNCTIONS --- ###

def execute_query(api_url, query, params=None):
    url = f"{api_url}/query"
    payload = {'query': query}
    if params:
        payload['params'] = params
    response = requests.post(url, json=payload, auth=('mseproject', 'tuebingen2024'))
    if response.status_code == 200:
        return response.json()
    else:
        print("Error executing query:", response.text)
        return None

def setup_database(api_url, drop_existing=False):
    if drop_existing:
        execute_query(api_url, "DROP TABLE IF EXISTS frontier")
        execute_query(api_url, "DROP TABLE IF EXISTS documents")

    execute_query(api_url, '''
    CREATE TABLE IF NOT EXISTS frontier (
        url TEXT PRIMARY KEY,
        crawled INTEGER DEFAULT 0
    )''')
    execute_query(api_url, '''
    CREATE TABLE IF NOT EXISTS documents (
        url TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        outgoing_links TEXT,
        timestamp TEXT
    )''')
    print("Database setup completed.")

def index_doc(doc, api_url):
    query = '''
    INSERT OR IGNORE INTO documents (url, title, content, outgoing_links, timestamp)
    VALUES (?, ?, ?, ?, ?)
    '''
    params = (doc['url'], doc['title'], doc['content'], ','.join(doc['outgoing_links']), doc['timestamp'])
    execute_query(api_url, query, params)
    print(f"Document indexed: {doc['url']}")

def count_remaining_frontier(api_url):
    query = "SELECT count(*) AS count FROM frontier WHERE crawled = 0"
    result = execute_query(api_url, query)
    print("Count remaining frontier result:", result)
    if result:
        return result[0]['count']
    return 0

def get_total_indexed_docs(api_url):
    query = "SELECT count(*) AS count FROM documents"
    result = execute_query(api_url, query)
    if result:
        return result[0]['count']
    return 0

def initialize_frontier(initial_urls, api_url):
    for url in initial_urls:
        query = "INSERT OR IGNORE INTO frontier (url) VALUES (?)"
        params = (url,)
        execute_query(api_url, query, params)
    print("Frontier initialized.")

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
            #ensure that urls that point to different sections of a website get treated as the same url 
            full_url=full_url.split("#")[0]
            
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

def crawl_page(url, keywords=None):
    try:
        response = requests.get(url, timeout=TIMEOUT)  # Fetch the web page
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')  # Parse the HTML content
        title = soup.title.string if soup.title else "N/A"
        content = ' '.join(soup.stripped_strings)  # Using stripped_strings to clean up the text

        # Filter out pages that do not contain relevant keywords in their content or are not in English
        if FILTER_CONTENT and keywords:
            if not any(word in content.lower() for word in keywords):
                return None

            try:
                if detect(content) != 'en':
                    return None
            except LangDetectException:
                return None
        #CHANGED: URL is not being keyword filtered anymore => now only content is checked
        ext_links, int_links = get_links(url)

        doc = {
            'url': url,
            'title': title,
            'content': content,
            'outgoing_links': list(ext_links | int_links),  # Using union operator to combine lists instead of "+"
            'timestamp': datetime.datetime.now().isoformat()
        }

    except requests.RequestException as e:
        print(f"Request exception encountered at {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected exception encountered at {url}: {e}")
        return None

    return doc

def crawl(api_url):
    total_to_crawl = count_remaining_frontier(api_url)

    with tqdm(total=total_to_crawl, desc="Crawling Progress", unit="page") as pbar:
        while True:
            query = "SELECT url FROM frontier WHERE crawled = 0 LIMIT 10"
            result = execute_query(api_url, query)
            if not result:
                break

            urls = [row['url'] for row in result]
            if not urls:
                break

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results = executor.map(lambda url: crawl_page(url,TUEBINGEN_KEYWORDS), urls)

            for url, doc in zip(urls, results):
                if doc:
                    index_doc(doc, api_url)
                    for link in doc['outgoing_links']:  # Add outgoing links to frontier
                        query = "INSERT OR IGNORE INTO frontier (url) VALUES (?)"
                        params = (link,)
                        execute_query(api_url, query, params)
                    query = "UPDATE frontier SET crawled = 1 WHERE url = ?"
                    params = (doc['url'],)
                    execute_query(api_url, query, params)
                    pbar.update(1)
                else:
                    query = "UPDATE frontier SET crawled = 1 WHERE url = ?"
                    params = (url,)
                    execute_query(api_url, query, params)
                    pbar.update(1)


def is_relevant_content(text, keywords, threshold=0.05):
    '''
    Use this method to determine if new webpage is relevant and
    if the webcrawler should continue crawling from it
    :param text: Content of the webpage
    :param keywords: Keywords determined by e.g. Tfidf to represent english content related to tuebingen
    :param threshold: We have to tune this once we have the keywords
    :return: if webpage is pseudo-relevant (based on cheap techniques like TF-IDF => real relevance calc. in retrieval)
    '''
    # Preprocess the text
    processed_text = preprocess(text)
    word_count = len(processed_text.split())
    
    # Count keyword occurrences
    keyword_hits = sum(processed_text.count(keyword) for keyword in keywords)
    
    # Calculate relevance score
    relevance_score = keyword_hits / word_count
    
    return relevance_score >= threshold

def main():
    try:
        setup_database(API_URL, drop_existing=False)
        initialize_frontier(initial_urls + additional_urls, API_URL)
        crawl(API_URL)

    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        
    finally:
        # show number of indexed documents and remaining URLs in the frontier
        index_tot = get_total_indexed_docs(API_URL)
        frontier_tot = count_remaining_frontier(API_URL)

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
