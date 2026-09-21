"""Shared helpers: config, paths, panel loading, time-split."""
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "churn_model" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg, key):
    return ROOT / cfg["paths"][key]


def out_dir(cfg):
    p = path(cfg, "outputs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def month_end(cfg, idx):
    start = pd.Timestamp(cfg["features"]["data_start"])
    return start + pd.DateOffset(months=idx + 1) - pd.Timedelta(days=1)


def hash_id(values, salt=None):
    """Salted SHA-256 of an identifier. The salt lives in CHURN_HASH_SALT, never in the repo."""
    import hashlib
    salt = salt or os.environ.get("CHURN_HASH_SALT")
    if not salt:
        raise RuntimeError("Set the CHURN_HASH_SALT environment variable (kept out of the repo).")
    return [hashlib.sha256((salt + str(v)).encode()).hexdigest()[:16] for v in values]


def load_panel(cfg):
    return pd.read_parquet(out_dir(cfg) / "churn_panel.parquet")


def split(panel, cfg):
    td = cfg["time_design"]
    return (panel[panel["cutoff"].isin(td["train"])],
            panel[panel["cutoff"].isin(td["validation"])],
            panel[panel["cutoff"].isin(td["test"])])
