# Seq2Seq MLOps with MLflow

## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run
python main.py

## MLflow
mlflow ui

## Docker
docker build -t seq2seq .
docker run -p 8000:8000 seq2seq

## Kubernetes
kubectl apply -f k8s/
