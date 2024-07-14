# This Python file uses the following encoding: utf-8
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import datetime
from langdetect import detect, LangDetectException
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import requests
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# define globals
API_URL = "http://l.kremp-everest.nord:5000"  # Replace with your Flask API URL
NUM_WORKERS = 20  # Increased number of workers for better concurrency
FILTER_CONTENT = True
TIMEOUT = 15
TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen', 'eberhard', 'karls',"Wurmlingen","Wolfenhausen","Wilhelmshöhe","Wendelsheim","Weitenburg","Weilheim","Weiler","Wankheim","Waldhörnle","Waldhof","Waldhausen","Wachendorf","Unterjesingen","Landkreis Tübingen","Tübingen","Talheim","Sulzau","Sülchen","Streimberg","Stockach","Westliche Steingart","Steinenberg","Seebronn","Schwärzloch","Schwalldorf","Schönbuchspitz","Naturpark Schönbuch","Schönberger Kopf","Schloßlesberg","Schloßbuckel","Schadenweilerhof","Saurücken","Rottenburg","Rosenau","Reusten","Remmingsheim","Rappenberg","Poltringen","Pfrondorf","Pfäffingen","Pfaffenberg","Österberg","Öschingen","Ofterdinger Berg","Ofterdingen","Odenburg","Oberwörthaus","Oberndorf","Obernau","Oberhausen","Neuhaus","Nellingsheim","Nehren","Mössingen","Mähringen","Lustnau","Lausbühl","Kusterdingen","Kreuzberg","Kreßbach","Kirchkopf","Kirchentellinsfurt","Kilchberg","Kiebingen","Jettenburg","Immenhausen","Hornkopf","Horn","Hohenstöffel","Schloss Hohenentringen","Hochburg","Hirschkopf","Hirschau","Hirrlingen","Hinterweiler","Heubergerhof","Heuberg","Heuberg","Hennental","Hemmendorf","Härtlesberg","Hailfingen","Hagelloch","Günzberg","Gomaringen","Geißhalde","Galgenberg","Frommenhausen","Firstberg","Filsenberg","Felldorf","Farrenberg","Bahnhof Eyach","Ergenzingen","Erdmannsbach","Ammerbuch","Einsiedel","Eichenfirst","Eichenberg","Ehingen","Eckenweiler","Höhe","Dußlingen","Dürrenberg","Dickenberg","Dettingen","Dettenhausen","Derendingen","Denzenberg","Buß","Burg","Buhlbachsaue","Bühl","Bühl","Bühl","Bromberg","Breitenholz","Börstingen","Bodelshausen","Bläsiberg","Bläsibad","Bierlingen","Bieringen","Belsen","Bei der Zeitungseiche","Bebenhausen","Baisingen","Bad Sebastiansweiler","Bad Niedernau","Ammern","Ammerbuch","Altstadt","Altingen","Alter Berg","Flugplatz Poltringen Ammerbuch","Starzach","Neustetten","Hotel Krone Tubingen","Hotel Katharina Garni","Bodelshausen","Dettenhausen","Dußlingen","Gomaringen","Hirrlingen","Kirchentellinsfurt","Kusterdingen","Nehren","Ofterdingen","Mössingen","Rottenburg am Neckar","Tübingen, Universitätsstadt","Golfclub Schloß Weitenburg","Siebenlinden","Steinenbertturm","Best Western Hotel Convita","Bebenhausen Abbey","Schloss Bebenhausen","Burgstall","Rafnachberg","Östliche Steingart","Kirnberg","Burgstall","Großer Spitzberg","Kleiner Spitzberg","Kapellenberg","Tannenrain","Grabhügel","Hemmendörfer Käpfle","Kornberg","Rotenberg","Weilerburg","Martinsberg","Eckberg","Entringen","Ofterdingen, Rathaus","Randelrain","Wahlhau","Unnamed Point","Spundgraben","University Library Tübingen","Tübingen Hbf","Bad Niedernau","Bieringen","Kiebingen","Unterjesingen Mitte","Unterjesingen Sandäcker","Entringen","Ergenzingen","Kirchentellinsfurt","Mössingen","Pfäffingen","Rottenburg (Neckar)","Tübingen West","Tübingen-Lustnau","Altingen (Württ)","Bad Sebastiansweiler-Belsen","Dußlingen","Bodelshausen","Nehren","Tübingen-Derendingen","Dettenhausen"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error("Error executing query: %s", response.text)
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
    logger.info("Database setup completed.")

def index_doc_batch(docs, api_url):
    query = '''
    INSERT OR IGNORE INTO documents (url, title, content, outgoing_links, timestamp)
    VALUES (?, ?, ?, ?, ?)
    '''
    for doc in docs:
        params = (doc['url'], doc['title'], doc['content'], ','.join(doc['outgoing_links']), doc['timestamp'])
        execute_query(api_url, query, params)
    logger.info("Batch of %d documents indexed.", len(docs))

def update_frontier_status_batch(urls, api_url):
    query = '''
    UPDATE frontier SET crawled = 1 WHERE url = ?
    '''
    for url in urls:
        params = (url,)
        execute_query(api_url, query, params)
    logger.info("Batch of %d URLs updated in frontier.", len(urls))

def count_remaining_frontier(api_url):
    query = "SELECT count(*) AS count FROM frontier WHERE crawled = 0"
    result = execute_query(api_url, query)
    logger.info("Count remaining frontier result: %s", result)
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
    logger.info("Frontier initialized.")

### --- CRAWLER FUNCTIONS --- ###

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def fetch_url(session, url):
    async with session.get(url, timeout=TIMEOUT) as response:
        response.raise_for_status()
        return await response.text()

async def get_links(session, url, keywords=None):
    external_links = set()
    internal_links = set()
    
    try:
        html = await fetch_url(session, url)
        soup = BeautifulSoup(html, "html.parser")
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            full_url = full_url.split("#")[0]
            
            if urlparse(full_url).netloc == urlparse(url).netloc:
                internal_links.add(full_url)
            else:
                external_links.add(full_url)
    
    except Exception as e:
        logger.error("Error fetching %s: %s", url, e)

    if keywords:
        internal_links = [link for link in internal_links if any(keyword in link for keyword in keywords)]
        external_links = [link for link in external_links if any(keyword in link for keyword in keywords)]
    
    return external_links, internal_links

async def crawl_page(session, url, keywords=None):
    try:
        html = await fetch_url(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else "N/A"
        content = ' '.join(soup.stripped_strings)

        if FILTER_CONTENT and keywords:
            if not any(word in content.lower() for word in keywords):
                return None

            try:
                if detect(content) != 'en':
                    return None
            except LangDetectException:
                return None

        ext_links, int_links = await get_links(session, url)

        doc = {
            'url': url,
            'title': title,
            'content': content,
            'outgoing_links': list(ext_links | int_links),
            'timestamp': datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Exception encountered at %s: %s", url, e)
        return None

    return doc

async def crawl(api_url):
    total_to_crawl = count_remaining_frontier(api_url)

    async with aiohttp.ClientSession() as session:
        with tqdm(total=total_to_crawl, desc="Crawling Progress", unit="page") as pbar:
            while True:
                query = "SELECT url FROM frontier WHERE crawled = 0 LIMIT 100"
                result = execute_query(api_url, query)
                if not result:
                    break

                urls = [row['url'] for row in result]
                if not urls:
                    break

                tasks = [crawl_page(session, url, TUEBINGEN_KEYWORDS) for url in urls]
                pages = await asyncio.gather(*tasks, return_exceptions=True)

                docs = [page for page in pages if page is not None]
                index_doc_batch(docs, api_url)

                update_frontier_status_batch(urls, api_url)
                pbar.update(len(urls))

def is_relevant_content(text, keywords, threshold=0.05):
    processed_text = preprocess(text)
    word_count = len(processed_text.split())
    keyword_hits = sum(processed_text.count(keyword) for keyword in keywords)
    relevance_score = keyword_hits / word_count
    return relevance_score >= threshold

def main():
    try:
        setup_database(API_URL, drop_existing=False)
        initialize_frontier(initial_urls + additional_urls, API_URL)
        asyncio.run(crawl(API_URL))

    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting...")
        
    finally:
        index_tot = get_total_indexed_docs(API_URL)
        frontier_tot = count_remaining_frontier(API_URL)
        logger.info("Total indexed documents: %d", index_tot)
        logger.info("Remaining URLs in frontier: %d", frontier_tot)

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
