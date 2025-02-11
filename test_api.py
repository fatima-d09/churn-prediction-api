import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [5, 1, 1, 1200.50, 1127, 453, 240.10, 0.002203]}

response = requests.post(url, json=data)
print(response.json())