import requests

# Deployment URL and API key
url = "https://predict-6a51ad6034f53f473ae8-dproatj77a-df.a.run.app/predict"
api_key = "ul_5769423c635872a50187d62b7c380800a67a91dc"

# Optional inference parameters (conf, iou, imgsz)
args = {"conf": 0.25, "iou": 0.7, "imgsz": 640}

with open("video.mp4", "rb") as f:
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=args,
        files={"file": f},
    )

print(response.json())