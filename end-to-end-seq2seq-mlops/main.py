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

if __name__ == "__main__":
    print("Downloading data...")
    subprocess.run([sys.executable, "-m", "src.download_data"], check=True)

    print("Training LSTM...")
    subprocess.run([sys.executable, "-m", "src.trainer.train_lstm"], check=True)

    print("Training Transformer...")
    subprocess.run([sys.executable, "-m", "src.trainer.train_transformer"], check=True)

    print("Project ready. Run 'mlflow ui' to view experiments.")