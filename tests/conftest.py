"""
Shared pytest fixtures for the Telecom Churn Pipeline output tests.

All fixtures are session-scoped: manifests and Parquet files are read once
per test run.  Tests in test_pipeline_output.py consume these fixtures
directly — no Spark or pyiceberg dependency, just pandas + stdlib json.

Prerequisite: run the full pipeline at least once before running pytest.
    .\run_pipeline.ps1          (Windows)
    pytest tests/               (then run tests)
"""

import glob
import json
import os

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC          = os.path.join(_PROJECT_ROOT, "src")
_CATALOG      = os.path.join(_SRC, "data", "catalog")
_WAREHOUSE    = os.path.join(_SRC, "data", "processed", "warehouse")
_MART         = os.path.join(_SRC, "data", "processed", "mart")
_MONITORING   = os.path.join(_SRC, "07_monitoring")


def _latest(pattern: str) -> str:
    """Return the most recent file matching a glob pattern, or fail clearly."""
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.fail(
            f"No file matching {pattern!r} found.\n"
            "Run the pipeline first:  .\\run_pipeline.ps1"
        )
    return files[-1]


# ---------------------------------------------------------------------------
# Manifest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def phase2_manifest() -> dict:
    path = _latest(os.path.join(_CATALOG, "raw_manifest_*.json"))
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def phase4_manifest() -> dict:
    path = _latest(os.path.join(_CATALOG, "root_cause_manifest_*.json"))
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def phase5_manifest() -> dict:
    path = _latest(os.path.join(_CATALOG, "phase5_manifest_*.json"))
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def phase6_manifest() -> dict:
    path = _latest(os.path.join(_CATALOG, "phase6_manifest_*.json"))
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def phase7_report() -> dict:
    path = _latest(os.path.join(_MONITORING, "pipeline_health_report_*.json"))
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data fixtures  (read warehouse Parquet directly with pandas)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fact_churn() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_WAREHOUSE, "fact_customer_churn_snapshot"))


@pytest.fixture(scope="session")
def fact_tickets() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_WAREHOUSE, "fact_network_tickets"))


@pytest.fixture(scope="session")
def dim_customer() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_WAREHOUSE, "dim_customer"))


@pytest.fixture(scope="session")
def mart_customer_360() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_MART, "mart_customer_360.parquet"))


@pytest.fixture(scope="session")
def mart_root_cause() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_MART, "mart_root_cause_summary.parquet"))
