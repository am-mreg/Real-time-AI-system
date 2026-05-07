import argparse, os, urllib.request

YOLO_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
}

def download(dest, name):
    url = YOLO_URLS.get(name)
    if not url:
        raise ValueError("Unknown model name")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {name} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="yolov8")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--out", default="weights")
    args = p.parse_args()
    out_path = os.path.join(args.out, args.model)
    download(out_path, args.model)