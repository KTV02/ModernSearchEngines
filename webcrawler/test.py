import requests

# Use the internal IP address of your Raspberry Pi
url = 'http://l.kremp-everest.nord:5000/query'
query = 'SELECT COUNT(*) AS count FROM frontier where crawled=0'

# Basic authentication details
auth = ('mseproject', 'tuebingen2024')

# Make the POST request to the Flask API with the query
response = requests.post(url, json={'query': query}, auth=auth)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    if data:
        count = data[0]['count']
        print(f"Total documents count: {count}")
    else:
        print("No data returned.")
else:
    print(f"Error executing query: {response.status_code} - {response.text}")
