from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import mlflow

from src.config.config import Config, load_config


def _ensure_tracking_uri(cfg: Config) -> None:
    uri = f"file://{cfg.paths.mlruns}"
    Path(cfg.paths.mlruns).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)


@contextmanager
def start_run(experiment: str, run_name: Optional[str] = None, config: Optional[Config] = None) -> Iterator[mlflow.ActiveRun]:
    cfg = config or load_config()
    _ensure_tracking_uri(cfg)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: Dict) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: Dict, step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path, artifact_path: Optional[str] = None) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
