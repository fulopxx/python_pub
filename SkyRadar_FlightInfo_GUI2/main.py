import requests
import json

# Replace 'your-username' and 'your-password' with your OpenSky API username and password
username = 'your-username'
password = 'your-password'
url = 'https://openskynetwork.org/api/states/all'

response = requests.get(url, auth=(username, password))

# Parse the response to JSON
data = response.json()

# Print the data
print(json.dumps(data, indent=4))
