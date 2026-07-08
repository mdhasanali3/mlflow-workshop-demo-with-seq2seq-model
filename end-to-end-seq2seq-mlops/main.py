# import subprocess
# import os

# if __name__ == "__main__":
#     print("Downloading data...")
#     subprocess.run(["python", "src/download_data.py"])
#     print("Training LSTM...")
#     subprocess.run(["python", "src/trainer/train_lstm.py"])
#     print("Training Transformer...")
#     subprocess.run(["python", "src/trainer/train_transformer.py"])
#     print("Project ready. Run 'mlflow ui' to view experiments.")


import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    print("Downloading data...")
    subprocess.run([sys.executable, "-m", "src.download_data"], check=True, cwd=PROJECT_ROOT)

    print("Training LSTM...")
    subprocess.run([sys.executable, "-m", "src.trainer.train_lstm"], check=True, cwd=PROJECT_ROOT)

    print("Training Transformer...")
    subprocess.run([sys.executable, "-m", "src.trainer.train_transformer"], check=True, cwd=PROJECT_ROOT)

    print("Project ready.")
    print("View runs with: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
