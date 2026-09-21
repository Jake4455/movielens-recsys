from pathlib import Path

import yaml


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent.parent)
    return cfg


def get(cfg, dotted_key, default=None):
    node = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def path_of(cfg, dotted_key):
    value = get(cfg, dotted_key)
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
