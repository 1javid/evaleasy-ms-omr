import requests
import os
path = os.path.abspath('images/test7_marked.jpg')
url = 'http://127.0.0.1:5000/process_image'
files = {'file': open(path, 'rb')}
response = requests.post(url, files=files)

# Print the JSON response
print(response.json())
