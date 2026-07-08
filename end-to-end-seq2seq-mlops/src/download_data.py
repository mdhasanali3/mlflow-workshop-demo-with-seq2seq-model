import os
import urllib.request
import zipfile

def download_data():
    url = "https://www.manythings.org/anki/fra-eng.zip"
    os.makedirs("data", exist_ok=True)
    zip_path = "data/fra-eng.zip"
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/fra-eng")
    print("Data ready.")

if __name__ == "__main__":
    download_data()
