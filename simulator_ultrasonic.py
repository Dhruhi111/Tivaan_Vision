import requests, time, random

URL = "http://127.0.0.1:8501/api/iot"

while True:
    dist = random.uniform(5, 150)
    alert = dist < 30
    gps = "22.3011,73.1925"

    payload = {
        "sensor": "ultrasonic",
        "distance_cm": round(dist, 2),
        "alert": alert,
        "gps": gps
    }

    print("Sending:", payload)
    requests.post(URL, json=payload)

    time.sleep(5)
