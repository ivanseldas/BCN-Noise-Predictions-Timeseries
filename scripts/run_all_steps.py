from __future__ import annotations

from src.config.config import load_config
from data.load_data import run as load_raw
from src.data.validate import run as validate
from src.data.preprocess import run as preprocess
from src.features.build_features import run as build_features
from src.models.tune import tune
from src.models.train import train
from src.models.evaluate import evaluate


def main() -> None:
    cfg = load_config()
    raw_path = load_raw(cfg)
    validate(raw_path, cfg)
    proc_path = preprocess(raw_path, cfg)
    feat_path = build_features(proc_path, cfg)
    tune(feat_path, cfg, n_iter=10, cv=3)
    train(feat_path, cfg)
    evaluate(feat_path, cfg.paths.artifacts / "model.joblib", cfg)


if __name__ == "__main__":
    main()
