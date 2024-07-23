import subprocess
import time
import platform

WEB_CRAWLER_SCRIPT = 'localCrawler.py'
SYNC_SCRIPT = 'sync_database.py'
CRAWL_DURATION = 60 * 60  # 60 minutes

def start_webcrawler():
    print("Starting webcrawler...")
    return subprocess.Popen(['python', WEB_CRAWLER_SCRIPT])

def stop_webcrawler(process):
    print("Stopping webcrawler...")
    if platform.system() == "Windows":
        process.terminate()
    else:
        process.send_signal(signal.SIGINT)
    process.wait()

def run_sync_script():
    print("Running synchronization script...")
    subprocess.run(['python', SYNC_SCRIPT])

def main():
    while True:
        # Start the webcrawler
        webcrawler_process = start_webcrawler()
        
        # Wait for the specified duration
        time.sleep(CRAWL_DURATION)
        
        # Stop the webcrawler
        stop_webcrawler(webcrawler_process)
        
        # Run the sync script
        run_sync_script()

if __name__ == "__main__":
    main()
