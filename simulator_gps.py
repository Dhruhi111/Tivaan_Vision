import requests, time, random

URL = "http://127.0.0.1:8501/api/iot"

lat, lon = 22.3011, 73.1925

while True:
    lat += random.uniform(-0.0003, 0.0003)
    lon += random.uniform(-0.0003, 0.0003)

    payload = {
        "sensor": "gps",
        "distance_cm": 0,
        "alert": False,
        "gps": f"{lat:.6f},{lon:.6f}"
    }

    print("GPS:", payload)
    requests.post(URL, json=payload)

    time.sleep(4)
