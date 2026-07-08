import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DATASET_SOURCE = "https://www.manythings.org/anki/fra-eng.zip"


def configure_mlflow() -> str:
    """Use the same backend for training and the local MLflow UI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "seq2seq-translation"))
    return tracking_uri


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_run_context(model_type: str) -> None:
    mlflow.set_tags(
        {
            "project": "end-to-end-seq2seq-mlops",
            "task": "machine-translation",
            "source_language": "English",
            "target_language": "French",
            "model_family": model_type,
            "git.commit": _git_value("rev-parse", "HEAD"),
            "git.branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
            "deployment.target": os.getenv("DEPLOYMENT_TARGET", "kubernetes"),
        }
    )
    mlflow.log_params(
        {
            "runtime.python": platform.python_version(),
            "runtime.platform": platform.platform(),
            "runtime.torch": torch.__version__,
            "runtime.mlflow": mlflow.__version__,
            "runtime.device": "cuda" if torch.cuda.is_available() else "cpu",
            "deployment.image": os.getenv("DEPLOYMENT_IMAGE", "seq2seq:latest"),
            "deployment.replicas": os.getenv("DEPLOYMENT_REPLICAS", "1"),
            "deployment.port": os.getenv("DEPLOYMENT_PORT", "8000"),
            "deployment.health_endpoint": "/health",
            "deployment.prediction_endpoint": "/predict",
        }
    )


def log_dataset(pairs: list[list[str]], subset_size: int) -> tuple[list[str], list[str]]:
    data_file = PROJECT_ROOT / "data" / "fra-eng" / "fra.txt"
    eng = [pair[0] for pair in pairs]
    fra = [pair[1] for pair in pairs]
    frame = pd.DataFrame({"english": eng, "french": fra})

    dataset = mlflow.data.from_pandas(
        frame,
        source=DATASET_SOURCE,
        name="manythings-fra-eng",
    )
    mlflow.log_input(dataset, context="training")
    mlflow.log_params(
        {
            "data.source": DATASET_SOURCE,
            "data.file": str(data_file.relative_to(PROJECT_ROOT)),
            "data.sha256": file_sha256(data_file),
            "data.file_bytes": data_file.stat().st_size,
            "data.training_rows": len(frame),
            "data.subset_limit": subset_size,
            "data.columns": ",".join(frame.columns),
            "data.preprocessing": "unicode_to_ascii+lowercase+punctuation_normalization",
        }
    )
    mlflow.log_metrics(
        {
            "data.avg_source_tokens": sum(map(lambda x: len(x.split()), eng)) / len(eng),
            "data.avg_target_tokens": sum(map(lambda x: len(x.split()), fra)) / len(fra),
            "data.max_source_tokens": max(map(lambda x: len(x.split()), eng)),
            "data.max_target_tokens": max(map(lambda x: len(x.split()), fra)),
        }
    )
    mlflow.log_table(frame.head(100), artifact_file="data/sample_pairs.json")
    return eng, fra


def log_model_info(model: torch.nn.Module, src_vocab: dict, tgt_vocab: dict) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    mlflow.log_params(
        {
            "model.class": type(model).__name__,
            "model.total_parameters": total,
            "model.trainable_parameters": trainable,
            "model.source_vocab_size": len(src_vocab),
            "model.target_vocab_size": len(tgt_vocab),
            "model.serialization": "pickle",
        }
    )
    mlflow.log_text(str(model), "model/architecture.txt")
    mlflow.log_dict(src_vocab, "model/source_vocabulary.json")
    mlflow.log_dict(tgt_vocab, "model/target_vocabulary.json")

    manifests = PROJECT_ROOT / "k8s"
    for manifest in ("deployment.yaml", "service.yaml"):
        path = manifests / manifest
        if path.exists():
            mlflow.log_artifact(path, artifact_path="deployment/kubernetes")
    dockerfile = PROJECT_ROOT / "Dockerfile"
    if dockerfile.exists():
        mlflow.log_artifact(dockerfile, artifact_path="deployment")


def write_run_summary(model_type: str, tracking_uri: str, extra: dict) -> None:
    summary = {
        "model_type": model_type,
        "tracking_uri": tracking_uri,
        "run_id": mlflow.active_run().info.run_id,
        "experiment_id": mlflow.active_run().info.experiment_id,
        **extra,
    }
    mlflow.log_dict(summary, "run_summary.json")
