import sqlite3
import requests
import json
import time
import os
from requests.auth import HTTPBasicAuth

# Configuration
LOCAL_DB_PATH = 'new.db'  # Update this with the path to your local database
REMOTE_DB_URL = 'http://l.kremp-everest.nord:5000/query'
SYNC_FILE_PATH = 'sync_status.json'
AUTH = ('mseproject', 'tuebingen2024')  # Update this with your actual username and password
BATCH_SIZE = 100  # Number of records to send in each batch
RETRY_COUNT = 5  # Number of retries for database access
RETRY_DELAY = 5  # Delay between retries in seconds

# Setup remote database function
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

# Initialize sync status file if it doesn't exist
if not os.path.exists(SYNC_FILE_PATH):
    with open(SYNC_FILE_PATH, 'w') as f:
        json.dump({'last_sync': 0}, f)

# Function to load sync status
def load_sync_status():
    with open(SYNC_FILE_PATH, 'r') as f:
        return json.load(f)

# Function to save sync status
def save_sync_status(status):
    with open(SYNC_FILE_PATH, 'w') as f:
        json.dump(status, f)

# Function to fetch new data from local database with retry mechanism
def fetch_new_data(last_sync_time):
    for attempt in range(RETRY_COUNT):
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            
            # Fetch new or updated documents that haven't been sent
            cursor.execute('''
                SELECT * FROM documents
                WHERE timestamp > ? AND url NOT IN (SELECT url FROM sent_documents)
            ''', (last_sync_time,))
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
    
    sync_status = load_sync_status()
    last_sync_time = sync_status['last_sync']
    
    new_documents = fetch_new_data(last_sync_time)
    
    if not new_documents:
        print("No new data to sync.")
        return
    
    success = send_data_to_remote('documents', new_documents)
    if success:
        print("Documents synchronized successfully.")
        sync_status['last_sync'] = time.time()
        save_sync_status(sync_status)
        print("Data synchronized successfully.")
    else:
        print("Data synchronization failed.")

if __name__ == "__main__":
    synchronize()
