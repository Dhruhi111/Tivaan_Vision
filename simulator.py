# simulator.py
import requests, time, random

API = "http://127.0.0.1:8501/api/iot"  # same port used by backend.py
while True:
    dist = random.uniform(5, 120)  # cm
    alert = dist < 30
    gps = "22.3011,73.1925"
    payload = {"sensor":"ultrasonic", "distance_cm": round(dist,2), "alert": alert, "gps": gps}
    r = requests.post(API, json=payload)
    print("sent:", payload, "resp:", r.status_code)
    time.sleep(6)  # every 6 seconds
