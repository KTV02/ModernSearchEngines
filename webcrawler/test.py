import requests

# Use the internal IP address of your Raspberry Pi
url = 'http://100.113.220.63:5000/query'
query = 'SELECT * FROM documents LIMIT 10'

response = requests.post(url, json={'query': query})
data = response.json()
print(data)
