#This script syncs the local database filled by the local Crawler with the remote database that is used by everyone 
import sqlite3
import requests
import json
import time
import os
from requests.auth import HTTPBasicAuth

# Configuration
LOCAL_DB_PATH = 'new.db'  # path to your local database
REMOTE_DB_URL = 'http://l.kremp-everest.nord:5000/query'

#security is less of an issue here (hence credentials in plain-text) because the remote database
#like we set it up is not accessible from the internet, but only if you are in the local network
#This was achieved using the free meshnet functionality provided by nordvpn
#This is also possible using a standard VPN into the local network 
AUTH = ('mseproject', 'tuebingen2024')  # username and password for remote database


BATCH_SIZE = 100  # Number of records to send in each batch
RETRY_COUNT = 5  # Number of retries for database access
RETRY_DELAY = 5  # Delay between retries in seconds

# Setup remote database if not already present
#only documents table is relevant for team members working with index
def setup_remote_database(api_url):
    queries = [
        '''
        CREATE TABLE IF NOT EXISTS documents (
            url TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            outgoing_links TEXT,
            timestamp TEXT
        )'''
    ]

    for query in queries:
        payload = {
            'query': query
        }
        try:
            response = requests.post(api_url, json=payload, auth=AUTH)
            response.raise_for_status()
            print(f"Executed query: {query}")
        except requests.exceptions.RequestException as e:
            print(f"Error executing query: {query} - {e}")

# Function to fetch new data from local database with retry mechanism
def fetch_new_data():
    for attempt in range(RETRY_COUNT):
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            
            # Fetch new or updated documents that haven't been sent
            cursor.execute('''
                SELECT * FROM documents
                WHERE url NOT IN (SELECT url FROM sent_documents)
            ''')
            new_documents = cursor.fetchall()
            
            conn.close()
            return new_documents
        except sqlite3.OperationalError as e:
            print(f"Database is locked. Attempt {attempt + 1} of {RETRY_COUNT}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    raise sqlite3.OperationalError(f"Failed to access the database after {RETRY_COUNT} attempts.")

# Function to send data to remote database in batches
def send_data_to_remote(table_name, data):
    if not data:
        return True

    total_records = len(data)
    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)
        batch = data[start:end]
        
        placeholders = ','.join(['?' for _ in range(len(batch[0]))])
        update_placeholders = ','.join([f'{col}=excluded.{col}' for col in ['title', 'content', 'outgoing_links', 'timestamp']])
        
        query = f'INSERT INTO {table_name} (url, title, content, outgoing_links, timestamp) VALUES ({placeholders}) ON CONFLICT(url) DO UPDATE SET {update_placeholders}'
        
        payload = {
            'query': query,
            'params': [tuple(item) for item in batch]
        }
        
        try:
            response = requests.post(REMOTE_DB_URL, json=payload, auth=AUTH)
            response.raise_for_status()
            print(f"Successfully sent batch of {len(batch)} records to {table_name}")
            
            # Mark these documents as sent
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.executemany('INSERT OR IGNORE INTO sent_documents (url) VALUES (?)', [(item[0],) for item in batch])
            conn.commit()
            conn.close()
        except requests.exceptions.RequestException as e:
            print(f"Error sending data: {e}")
            return False
    
    return True

# Main synchronization function
def synchronize():
    setup_remote_database(REMOTE_DB_URL)  # Ensure the remote database is set up
    
    new_documents = fetch_new_data()
    
    if not new_documents:
        print("No new data to sync.")
        return
    
    success = send_data_to_remote('documents', new_documents)
    if success:
        print("Documents synchronized successfully.")
        print("Data synchronized successfully.")
    else:
        print("Data synchronization failed.")

if __name__ == "__main__":
    synchronize()
