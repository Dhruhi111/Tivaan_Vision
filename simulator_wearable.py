import requests, time
URL = "http://127.0.0.1:8501/api/iot"

while True:
    payload = {
        "sensor": "wearable",
        "distance_cm": 0,
        "alert": True,
        "gps": "22.3011,73.1925"
    }

    print("Wearable alert:", payload)
    requests.post(URL, json=payload)

    time.sleep(10)
