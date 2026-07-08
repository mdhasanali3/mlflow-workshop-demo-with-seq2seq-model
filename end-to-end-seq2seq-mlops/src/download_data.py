from pathlib import Path
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fra-eng"
DATA_FILE = DATA_DIR / "fra.txt"
ZIP_PATH = PROJECT_ROOT / "data" / "fra-eng.zip"


def download_data():
    url = "https://www.manythings.org/anki/fra-eng.zip"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, ZIP_PATH)

    if not DATA_FILE.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset archive did not contain {DATA_FILE.name}")

    print("Data ready.")

if __name__ == "__main__":
    download_data()
