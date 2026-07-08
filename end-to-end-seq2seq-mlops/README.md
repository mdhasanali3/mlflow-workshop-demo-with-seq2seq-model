# Seq2Seq MLOps with MLflow

## Setup and training

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Training writes to the `seq2seq-translation` experiment in `mlflow.db`. Each run records hyperparameters, dataset lineage and fingerprint, samples, vocabularies, model architecture and weights, epoch metrics, runtime details, Git metadata, and deployment configuration.

## MLflow UI

Open the same backend used by training:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then visit http://localhost:5000 and select **seq2seq-translation**. To use a remote server instead, set `MLFLOW_TRACKING_URI` before training.

## Docker

```powershell
docker compose up --build
```

This exposes MLflow at http://localhost:5000 and the inference API at http://localhost:8000. The API health check is `/health`.

## CI/CD

`.github/workflows/ci-cd.yml` performs syntax checks and model smoke tests on pull requests. On `main`, it also builds and pushes a versioned image to GitHub Container Registry.

Kubernetes deployment is deliberately gated. Set the repository variable `ENABLE_K8S_DEPLOY=true` and secret `KUBE_CONFIG` to enable it. The workflow replaces the manifest image with the immutable commit-tagged image before applying the manifests.
