import requests


url = "http://127.0.0.1:5000/api/chat"

data = {
    "question": "What B.Tech programs are offered at VIT-AP?"
}

response = requests.post(
    url,
    json=data
)

print(response.json())