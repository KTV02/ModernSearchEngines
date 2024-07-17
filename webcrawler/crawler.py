# This Python file uses the following encoding: utf-8
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import datetime
from langdetect import detect, LangDetectException
from tqdm.asyncio import tqdm_asyncio
from urllib.parse import urljoin, urlparse
import requests
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# define globals
API_URL = "http://l.kremp-everest.nord:5000"
NUM_WORKERS = 20  
FILTER_CONTENT = True
TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen',  # Add more keywords if needed
                      'wurmlingen', 'wolfenhausen', 'wilhelmshöhe', 'wendelsheim', 'weitenburg', 'weilheim',
                      'wankheim', 'waldhörnle', 'waldhausen', 'wachendorf', 'unterjesingen', 'landkreis tübingen',
                      'tübingen', 'talheim', 'sulzau', 'sülchen', 'streimberg', 'stockach', 'westliche steingart', 'steinenberg',
                      'seebronn', 'schwärzloch', 'schwalldorf', 'schönbuchspitz', 'naturpark schönbuch', 'schönberger kopf',
                      'schloßlesberg', 'schloßbuckel', 'schadenweilerhof', 'saurücken', 'rottenburg', 'rosenau', 'reusten',
                      'remmingsheim', 'rappenberg', 'poltringen', 'pfrondorf', 'pfäffingen', 'pfaffenberg', 'österberg',
                      'öschingen', 'ofterdinger berg', 'ofterdingen', 'odenburg', 'oberwörthaus', 'oberndorf', 'obernau',
                      'oberhausen', 'neuhaus', 'nellingsheim', 'nehren', 'mössingen', 'mähringen', 'lustnau', 'lausbühl',
                      'kusterdingen', 'kreuzberg', 'kreßbach', 'kirchkopf', 'kirchentellinsfurt', 'kilchberg', 'kiebingen',
                      'jettenburg', 'immenhausen', 'hornkopf', 'horn', 'hohenstöffel', 'schloss hohenentringen', 'hochburg',
                      'hirschkopf', 'hirschau', 'hirrlingen', 'hinterweiler', 'heubergerhof', 'heuberg', 'heuberg',
                      'hennental', 'hemmendorf', 'härtlesberg', 'hailfingen', 'hagelloch', 'günzberg', 'gomaringen',
                      'geißhalde', 'galgenberg', 'frommenhausen', 'firstberg', 'filsenberg', 'felldorf', 'farrenberg',
                      'bahnhof eyach', 'ergenzingen', 'erdmannsbach', 'ammerbuch', 'einsiedel', 'eichenfirst', 'eichenberg',
                      'ehingen', 'eckenweiler', 'höhe', 'dußlingen', 'dürrenberg', 'dickenberg', 'dettingen', 'dettenhausen',
                      'derendingen', 'denzenberg', 'buß', 'burg', 'buhlbachsaue', 'bühl', 'bühl', 'bühl', 'bromberg',
                      'breitenholz', 'börstingen', 'bodelshausen', 'bläsiberg', 'bläsibad', 'bierlingen', 'bieringen',
                      'belsen', 'bei der zeitungseiche', 'bebenhausen', 'baisingen', 'bad sebastiansweiler', 'bad niedernau',
                      'ammern', 'ammerbuch', 'altstadt', 'altingen', 'alter berg', 'flugplatz poltringen ammerbuch', 'starzach',
                      'neustetten', 'hotel krone tubingen', 'hotel katharina garni', 'bodelshausen', 'dettenhausen',
                      'dußlingen', 'gomaringen', 'hirrlingen', 'kirchentellinsfurt', 'kusterdingen', 'nehren', 'ofterdingen',
                      'mössingen', 'rottenburg am neckar', 'tübingen, universitätsstadt', 'golfclub schloß weitenburg',
                      'siebenlinden', 'steinenbertturm', 'best western hotel convita', 'bebenhausen abbey', 'schloss bebenhausen',
                      'burgstall', 'rafnachberg', 'östliche steingart', 'kirnberg', 'burgstall', 'großer spitzberg', 'kleiner spitzberg',
                      'kapellenberg', 'tannenrain', 'grabhügel', 'hemmendörfer käpfle', 'kornberg', 'rotenberg', 'weilerburg',
                      'martinsberg', 'eckberg', 'entringen', 'ofterdingen, rathaus', 'randelrain', 'wahlhau', 'unnamed point',
                      'spundgraben', 'university library tübingen', 'tübingen hbf', 'bad niedernau', 'bieringen', 'kiebingen',
                      'unterjesingen mitte', 'unterjesingen sandäcker', 'entringen', 'ergenzingen', 'kirchentellinsfurt',
                      'mössingen', 'pfäffingen', 'rottenburg (neckar)', 'tübingen west', 'tübingen-lustnau', 'altingen (württ)',
                      'bad sebastiansweiler-belsen', 'dußlingen', 'bodelshausen', 'nehren', 'tübingen-derendingen', 'dettenhausen']

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
        crawled INTEGER DEFAULT 0,
        error INTEGER DEFAULT 0
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

def mark_frontier_error(url, api_url):
    query = '''
    UPDATE frontier SET error = 1 WHERE url = ?
    '''
    params = (url,)
    execute_query(api_url, query, params)
    logger.info("URL marked with error: %s", url)

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

def check_if_url_exists(url, api_url):
    query = "SELECT 1 FROM frontier WHERE url = ? UNION SELECT 1 FROM documents WHERE url = ?"
    params = (url, url)
    result = execute_query(api_url, query, params)
    return result is not None and len(result) > 0

def initialize_frontier(initial_urls, api_url):
    for url in initial_urls:
        if not check_if_url_exists(url, api_url):
            query = "INSERT OR IGNORE INTO frontier (url) VALUES (?)"
            params = (url,)
            execute_query(api_url, query, params)
            logger.info("URL added to frontier: %s", url)  # Log each URL being added to frontier
        else:
            logger.info("URL already exists in frontier or documents: %s", url)  # Log URLs that already exist

### --- CRAWLER FUNCTIONS --- ###

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def fetch_url(session, url):
    headers = {"User-Agent": USER_AGENT}
    async with session.get(url, headers=headers, timeout=TIMEOUT) as response:
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

        logger.info("Crawling URL: %s", url)  # Log URL being crawled
        logger.info("Page content length: %d", len(content))  # Log content length

        if FILTER_CONTENT and keywords:
            if not any(word in content.lower() for word in keywords):
                logger.info("Content filtered out based on keywords: %s", url)  # Log filtering action
                return None

            try:
                if detect(content) != 'en':
                    logger.info("Content filtered out based on language: %s", url)  # Log filtering action
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

        logger.info("Document would be added: %s", doc['url'])  # Log document details
        return doc

    except Exception as e:
        if isinstance(e, aiohttp.ClientResponseError):
            logger.error("Exception encountered at %s: %s %s", url, e.status, e.message)
        else:
            logger.error("Exception encountered at %s: %s", url, e)
        mark_frontier_error(url, API_URL)
        return None

async def crawl(api_url):
    total_to_crawl = count_remaining_frontier(api_url)

    async with aiohttp.ClientSession() as session:
        with tqdm_asyncio(total=total_to_crawl, desc="Crawling Progress", unit="page") as pbar:
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

                relevant_urls = [doc['url'] for doc in docs]
                outgoing_links = set()
                for doc in docs:
                    outgoing_links.update(doc['outgoing_links'])

                # Add outgoing links of relevant pages to the frontier
                for link in outgoing_links:
                    if not check_if_url_exists(link, api_url):
                        query = "INSERT OR IGNORE INTO frontier (url) VALUES (?)"
                        params = (link,)
                        execute_query(api_url, query, params)

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
    "https://www.eventbrite.de/d/germany--t%C3%BCbingen/events/",
    "https://www.ubereats.com/de-en/city/t%C3%BCbingen-bw",
    "https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html",
    "https://www.uni-tuebingen.de/en.html"
]

additional_urls = [
    "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
    "https://en.wikipedia.org/wiki/T%C3%BCbingen",
    "https://historicgermany.travel/historic-germany/tubingen/",
    "https://www.mygermanuniversity.com/universities/University-of-Tuebingen",
    "https://www.germansights.com/tubingen/",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
    "https://tuebingenresearchcampus.com/en/tuebingen/general-information/local-infos",
    "https://www.germany.travel/en/cities-culture/tuebingen.html",
    "https://uni-tuebingen.de/en/forschung/zentren-und-institute/brasilien-und-lateinamerika-zentrum/german-brazilian-symposium-2024/about-tuebingen/welcome-to-tuebingen/"
]

if __name__ == "__main__":
    main()
