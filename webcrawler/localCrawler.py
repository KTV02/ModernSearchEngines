import aiohttp
import asyncio
import async_timeout
import sqlite3
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from datetime import datetime
from langdetect import detect, LangDetectException
from urllib.parse import urlparse, urlunparse, urljoin
import sync_database  # Import your sync script

# Initialize database connection
DATABASE_URL = 'new.db'

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

TUEBINGEN_KEYWORDS = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen', 'wurmlingen', 'wolfenhausen', 'wilhelmshöhe', 'wendelsheim', 'weitenburg', 'weilheim', 'wankheim', 'waldhörnle', 'waldhausen', 'wachendorf', 'unterjesingen', 'talheim', 'sulzau', 'sülchen', 'streimberg', 'stockach', 'westliche steingart', 'steinenberg', 'seebronn', 'schwärzloch', 'schwalldorf', 'schönbuchspitz', 'naturpark schönbuch', 'schönberger kopf', 'schloßlesberg', 'schloßbuckel', 'schadenweilerhof', 'saurücken', 'rottenburg', 'rosenau', 'reusten', 'remmingsheim', 'rappenberg', 'poltringen', 'pfrondorf', 'pfäffingen', 'pfaffenberg', 'österberg', 'öschingen', 'ofterdinger berg', 'ofterdingen', 'odenburg', 'oberwörthaus', 'oberndorf', 'obernau', 'oberhausen', 'neuhaus', 'nellingsheim', 'nehren', 'mössingen', 'mähringen', 'lustnau', 'lausbühl', 'kusterdingen', 'kreuzberg', 'kreßbach', 'kirchkopf', 'kirchentellinsfurt', 'kilchberg', 'kiebingen', 'jettenburg', 'immenhausen', 'hornkopf', 'horn', 'hohenstöffel', 'schloss hohenentringen', 'hirschkopf', 'hirschau', 'hirrlingen', 'hinterweiler', 'heubergerhof', 'heuberg', 'hennental', 'hemmendorf', 'härtlesberg', 'hailfingen', 'hagelloch', 'günzberg', 'gomaringen', 'geißhalde', 'galgenberg', 'frommenhausen', 'firstberg', 'filsenberg', 'felldorf', 'farrenberg', 'bahnhof eyach', 'ergenzingen', 'erdmannsbach', 'ammerbuch', 'einsiedel', 'eichenfirst', 'eichenberg', 'ehingen', 'eckenweiler','dußlingen', 'dürrenberg', 'dickenberg', 'dettingen', 'dettenhausen', 'derendingen', 'denzenberg', 'buhlbachsaue','bromberg', 'breitenholz', 'börstingen', 'bodelshausen', 'bläsiberg', 'bläsibad', 'bierlingen', 'bieringen', 'belsen', 'bei der zeitungseiche', 'bebenhausen', 'baisingen', 'bad sebastiansweiler', 'bad niedernau', 'ammern', 'ammerbuch', 'altingen', 'alter berg', 'flugplatz poltringen ammerbuch', 'starzach', 'neustetten', 'hotel krone tubingen', 'hotel katharina garni', 'bodelshausen', 'dettenhausen', 'dußlingen', 'gomaringen', 'hirrlingen', 'kirchentellinsfurt', 'kusterdingen', 'nehren', 'ofterdingen', 'mössingen', 'rottenburg am neckar', 'tübingen, universitätsstadt', 'golfclub schloß weitenburg', 'siebenlinden', 'steinenbertturm', 'best western hotel convita', 'bebenhausen abbey', 'schloss bebenhausen', 'burgstall', 'rafnachberg', 'östliche steingart', 'kirnberg', 'burgstall', 'großer spitzberg', 'kleiner spitzberg', 'kapellenberg', 'tannenrain', 'grabhügel', 'hemmendörfer käpfle', 'kornberg', 'rotenberg', 'weilerburg', 'martinsberg', 'eckberg', 'entringen', 'ofterdingen, rathaus', 'randelrain', 'wahlhau', 'spundgraben', 'university library tübingen', 'tübingen hbf', 'bad niedernau', 'bieringen', 'kiebingen', 'unterjesingen mitte', 'unterjesingen sandäcker', 'entringen', 'ergenzingen', 'kirchentellinsfurt', 'mössingen', 'pfäffingen', 'rottenburg (neckar)', 'tübingen west', 'tübingen-lustnau', 'altingen (württ)', 'bad sebastiansweiler-belsen', 'dußlingen', 'bodelshausen', 'nehren', 'tübingen-derendingen', 'dettenhausen']

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
MAX_RETRIES = 3
TIMEOUT = 10
CONCURRENCY = 10

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
            with async_timeout.timeout(TIMEOUT):
                async with session.get(url, headers=headers) as response:
                    return await response.text(), url
        except Exception as e:
            print(f"Error fetching {url}: {e}")
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
        print(f"Added {normalized_url} to documents.")
    cursor.execute("UPDATE frontier SET crawled = 1 WHERE url = ?", (normalized_url,))
    conn.commit()
    conn.close()

async def run_sync():
    while True:
        await asyncio.sleep(1800)  # Wait for 30 minutes
        print("Starting synchronization...")
        await asyncio.to_thread(sync_database.synchronize)
        print("Synchronization complete.")

async def crawl_urls(session):
    while True:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM frontier WHERE crawled = 0 AND error = 0 LIMIT ?", (CONCURRENCY,))
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
        print(f"Documents: {documents_count}, Frontier: {frontier_count}")

async def main():
    setup_database(DATABASE_URL)
    
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.executemany("INSERT OR IGNORE INTO frontier (url) VALUES (?)", [(url,) for url in initial_urls])
    conn.commit()
    conn.close()

    async with ClientSession() as session:
        # Start the synchronization task
        asyncio.create_task(run_sync())
        
        # Continue with the crawling process
        await crawl_urls(session)

if __name__ == "__main__":
    asyncio.run(main())
