import aiohttp
import asyncio
import async_timeout
import sqlite3
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from datetime import datetime
from langdetect import detect, LangDetectException
from urllib.parse import urlparse, urlunparse, urljoin
import logging

# Initialize database connection
DATABASE_URL = 'new.db'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_urls = [
    "https://www.tuebingen.de/en/",
    "https://www.eventbrite.de/d/germany--t%C3%BCbingen/events/",
    "https://www.ubereats.com/de-en/city/t%C3%BCbingen-bw",
    "https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html",
    "https://www.uni-tuebingen.de/en.html",
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

TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen']
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
MAX_RETRIES = 3
TIMEOUT = 10
CONCURRENCY = 5  # Reduced concurrency for debugging
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10 MB

def setup_database(api_url, drop_existing=False):
    conn = sqlite3.connect(api_url)
    cursor = conn.cursor()
    if drop_existing:
        cursor.execute("DROP TABLE IF EXISTS frontier")
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("DROP TABLE IF EXISTS sent_documents")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frontier (
        url TEXT PRIMARY KEY,
        crawled INTEGER DEFAULT 0,
        error INTEGER DEFAULT 0
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        url TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        outgoing_links TEXT,
        timestamp TEXT
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sent_documents (
        url TEXT PRIMARY KEY
    )''')
    conn.commit()
    conn.close()

async def fetch(session, url):
    headers = {"User-Agent": USER_AGENT}
    for _ in range(MAX_RETRIES):
        try:
            async with async_timeout.timeout(TIMEOUT):
                async with session.get(url, headers=headers) as response:
                    content = await response.read()
                    logger.info(f"Fetched {url} with status {response.status} and size {len(content)} bytes")
                    if len(content) > MAX_CONTENT_SIZE:
                        logger.warning(f"Skipping {url} as it exceeds the maximum allowed size.")
                        return None, url
                    return content.decode('utf-8'), url
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
    return None, url

def is_relevant(content):
    return any(keyword in content.lower() for keyword in TUEBINGEN_KEYWORDS)

def is_english(content):
    try:
        return detect(content) == 'en'
    except LangDetectException:
        return False

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
    text = ' '.join(element.get_text() for element in text_elements)
    return text

def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = [urljoin(base_url, a.get('href')) for a in soup.find_all('a', href=True)]
    return links

def normalize_url(url):
    parsed_url = urlparse(url)
    return urlunparse(parsed_url._replace(fragment=''))

def dont_crawl():
    return [
        '%wikipedia%',
        '%wikimedia%',
        '%.jpg%',
        '%.png%',
        '%.pdf%'
    ]

async def crawl(url, session, db_url):
    conn = sqlite3.connect(db_url)
    cursor = conn.cursor()

    normalized_url = normalize_url(url)
    html, fetched_url = await fetch(session, normalized_url)
    if not html:
        cursor.execute("UPDATE frontier SET error = 1, crawled = 1 WHERE url = ?", (normalized_url,))
        conn.commit()
        conn.close()
        return

    text_content = extract_text_from_html(html)
    if is_english(text_content) and is_relevant(text_content):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        timestamp = datetime.now().isoformat()
        links = extract_links(html, url)
        
        cursor.execute("INSERT OR REPLACE INTO documents (url, title, content, outgoing_links, timestamp) VALUES (?, ?, ?, ?, ?)",
                       (normalized_url, title, html, ','.join(links), timestamp))
        for link in links:
            cursor.execute("INSERT OR IGNORE INTO frontier (url) VALUES (?)", (normalize_url(link),))
        logger.info(f"Added {normalized_url} to documents.")
    cursor.execute("UPDATE frontier SET crawled = 1 WHERE url = ?", (normalized_url,))
    conn.commit()
    conn.close()

async def crawl_urls(session):
    while True:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        exclude_conditions = " AND ".join([f"url NOT LIKE '{pattern}'" for pattern in dont_crawl()])
        query = f"SELECT url FROM frontier WHERE crawled = 0 AND error = 0 AND {exclude_conditions} LIMIT ?"
        cursor.execute(query, (CONCURRENCY,))
        urls_to_crawl = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not urls_to_crawl:
            break

        tasks = [crawl(url, session, DATABASE_URL) for url in urls_to_crawl]
        await asyncio.gather(*tasks)

        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        documents_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM frontier")
        frontier_count = cursor.fetchone()[0]
        conn.close()
        logger.info(f"Documents: {documents_count}, Frontier: {frontier_count}")

async def main():
    setup_database(DATABASE_URL)
    
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.executemany("INSERT OR IGNORE INTO frontier (url) VALUES (?)", [(url,) for url in initial_urls])
    conn.commit()
    conn.close()

    async with ClientSession() as session:
        await crawl_urls(session)

if __name__ == "__main__":
    asyncio.run(main())
