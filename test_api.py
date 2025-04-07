import requests

url = "http://127.0.0.1:8000/analyze"

payload = {
    "coordinates": [
        [37.7749, -122.4194],
        [37.7759, -122.4194],
        [37.7759, -122.4184],
        [37.7749, -122.4184]
    ]
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())