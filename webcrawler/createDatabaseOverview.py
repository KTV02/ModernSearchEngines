import sqlite3
from urllib.parse import urlparse
from collections import Counter

# Connect to the SQLite database
db_path = 'uberEatsStartingPoint.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query to select all URLs from the documents table
query = "SELECT url FROM documents"
cursor.execute(query)

# Fetch all URLs
urls = cursor.fetchall()

# Function to extract the base domain from a URL
def get_base_domain(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return None

# Extract base domains and count occurrences
base_domains = [get_base_domain(url[0]) for url in urls]
domain_counter = Counter(base_domains)

# Print out a summary of the most common base domains
print("Base Domain Overview:")
for domain, count in domain_counter.most_common():
    print(f"{domain}: {count}")

# Close the database connection
conn.close()
