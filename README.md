# Remote Database Guide

## Steps to Setup Meshnet

1. Create a free NordVPN Account [here](https://nordvpn.com) (you don't even need the 30-day free trial, you just need the free account).
2. [Download](https://nordvpn.com/download/) the app on the device you want to use (For Linux, there is also a CLI version).
3. Enable Meshnet:
   ![Enable Meshnet](readme/readme1.png)
4. Click on External Device > Link External Device:
   ![Link External Device](readme/readme2.png)
5. Enter email address: l.kremp@gmx.de
6. Allow the first and last options (sending and receiving):
   ![Allow Options](readme/readme3.png)
7. Send the invitation AND TELL ME! I need to accept this invitation.
8. Now you can see the IP address / hostname of the server under the tab "External Devices".
   ![IP_Adress](readme/readme4.png)


## Use the IP Address or Hostname to send requests to database
Its best to use the hostname, because this will stay the same for every team member 

```python
import requests

# Here instead of hostname you can also use the ip adresse Displayed in NordVPN
url = 'http://l.kremp-everest.nord:5000/query'

#always rename your parameters if you want to access them via their index in the output
# e.g. here COUNT(*) renamed to count
query = 'SELECT COUNT(*) AS count FROM documents'

# Basic authentication details
auth = ('mseproject', 'tuebingen2024')

# Make the POST request to the Flask API with your query
response = requests.post(url, json={'query': query}, auth=auth)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    if data:
        #here i can access the result via the index "count" because i renamed it ^
        count = data[0]['count']
        print(f"Total documents count: {count}")
    else:
        print("No data returned.")
else:
    print(f"Error executing query: {response.status_code} - {response.text}")

