from __future__ import annotations

from src.config.config import load_config
from data.load_data import run as load_raw
from src.data.preprocess import run as preprocess
from src.data.validate import run as validate
from src.features.build_features import run as build_features
from src.models.train import TrainParams, train
from src.models.evaluate import evaluate


def main() -> None:
    cfg = load_config()
    raw_path = load_raw(cfg)
    # Validate freshly saved raw data
    validate(raw_path, cfg)
    proc_path = preprocess(raw_path, cfg)
    feat_path = build_features(proc_path, cfg)
    metrics = train(feat_path, cfg)
    eval_metrics = evaluate(feat_path, config=cfg)
    print({"train": metrics, "eval": eval_metrics})


if __name__ == "__main__":
    main()
