import requests
import json
url = "http://127.0.0.1:8000/api/analyze"
file_path = "c:/Arafath_A/visioncare/outputs/scans/cfe8336c21054f48bed60a5ef9f5820a.png"
with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"user_id": "1", "patient_name": "Test"}
    try:
        response = requests.post(url, data=data, files=files)
        print("Status code:", response.status_code)
        if response.status_code != 200:
            print("Response:", response.text)
    except Exception as e:
        print("Error:", str(e))
