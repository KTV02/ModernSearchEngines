# Remote Database Guide

## Steps to Setup Meshnet

1. Create a free NordVPN Account [here](https://nordvpn.com) (you don't even need the 30-day free trial, you just need the free account).
2. Download the app on the PC you want to use (For Linux, there is also a CLI version).
3. Enable Meshnet:
   ![Enable Meshnet](readme/readme1.png)
4. Click on External Device > Link External Device:
   ![Link External Device](readme/readme2.png)
5. Enter email address: l.kremp@gmx.de
6. Allow the first and last options (sending and receiving):
   ![Allow Options](readme/readme3.png)
7. Send the invitation AND TELL ME! I need to accept this invitation.
8. Now you can see the IP address of the server under the tab "External Devices".
   ![IP_Adress](readme/readme4.png)


## Use the IP Address to send requests to database

```python
import requests

# Use the IP address displayed in NordVPN meshnet
url = 'http://100.113.220.63:5000/query'
query = 'SELECT * FROM documents LIMIT 10'

response = requests.post(url, json={'query': query})
data = response.json()
print(data)
