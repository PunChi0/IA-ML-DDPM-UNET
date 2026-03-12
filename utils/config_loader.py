import yaml
import torch


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg