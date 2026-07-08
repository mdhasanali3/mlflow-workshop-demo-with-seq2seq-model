import os
import urllib.request
import zipfile

def download_data():
    url = "https://www.manythings.org/anki/fra-eng.zip"
    os.makedirs("data", exist_ok=True)
    zip_path = "data/fra-eng.zip"
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/fra-eng")
    print("Data ready.")

if __name__ == "__main__":
    download_data()
