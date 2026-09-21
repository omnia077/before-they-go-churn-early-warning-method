"""
Parity test: PySpark as-of features must match the pandas reference features in
src/01_generate_data/phase1b_acceptance_test.py for the same cutoff.

Requires the v2 data: src/data/raw (Phase 2 output) and src/data/sftp_drop +
src/data/ground_truth (the pandas reference reads these). Skipped if missing.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "01_generate_data"))

if Path("C:/hadoop/bin").exists():                      # PySpark on Windows
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

DATA = ROOT / "src" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "raw").exists() or not (DATA / "ground_truth").exists(),
                                reason="v2 data not generated")

SHARED = ["use_30d", "use_prev60d", "use_trend", "data_mb_30d", "days_since_use",
          "rech_30d", "rech_90d", "rech_amt_90d", "days_since_rech", "tix_30d",
          "tix_90d", "unpaid_last3", "tenure_days", "is_prepaid", "monthly_arpu_sdg"]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(ROOT / "churn_model" / "config.yaml"))


@pytest.fixture(scope="module")
def spark_tables():
    from churn_model.as_of_features import get_spark, load_raw
    spark = get_spark("as_of_parity")
    spark.sparkContext.setLogLevel("ERROR")
    yield load_raw(spark, str(DATA / "raw"))
    spark.stop()


@pytest.fixture(scope="module")
def pandas_tables():
    rd = lambda n: pd.read_parquet(DATA / "sftp_drop" / f"{n}.parquet")
    T = {"cust": rd("customers"), "usage": rd("usage_events"), "rech": rd("recharge_records"),
         "tix": rd("network_tickets"), "bill": rd("billing_records")}
    T["cust"]["activation_date"] = pd.to_datetime(T["cust"]["activation_date"])
    T["usage"]["event_date"] = pd.to_datetime(T["usage"]["event_date"])
    T["rech"]["recharge_date"] = pd.to_datetime(T["rech"]["recharge_date"])
    T["tix"]["ticket_date"] = pd.to_datetime(T["tix"]["ticket_date"])
    lab = pd.read_parquet(DATA / "ground_truth" / "churn_labels.parquet")
    lab["churn_date"] = pd.to_datetime(lab["churn_date"])
    T["lab"] = lab[["customer_id", "churn_date"]]
    T["lat"] = pd.read_parquet(DATA / "ground_truth" / "latent_state.parquet")
    return T


@pytest.mark.parametrize("cutoff_idx", [5, 8])       # Jun and Sep month-ends
def test_spark_matches_pandas(cutoff_idx, cfg, spark_tables, pandas_tables):
    from phase1b_acceptance_test import build, month_end
    from churn_model.as_of_features import build_as_of_features

    as_of = month_end(cutoff_idx).strftime("%Y-%m-%d")
    ref = build(str(DATA), cutoff_idx, pandas_tables).set_index("customer_id")
    sp = build_as_of_features(spark_tables, as_of, cfg).toPandas().set_index("customer_id")

    # counts: Spark scores everyone activated by the cutoff; pandas has already
    # dropped subscribers who churned before it, so Spark must be a superset.
    assert sp.index.is_unique
    assert set(ref.index) <= set(sp.index)
    already_churned = len(sp) - len(ref)
    assert already_churned >= 0
    print(f"\ncutoff {as_of}: spark={len(sp):,} pandas={len(ref):,} "
          f"(already churned, dropped in panel: {already_churned:,})")

    sp = sp.loc[ref.index]
    for col in SHARED:
        diff = np.abs(sp[col].astype(float).values - ref[col].astype(float).values)
        assert diff.max() < 1e-6, f"{col}: max abs diff {diff.max():.6g} at {as_of}"


def test_no_leakage_columns(cfg, spark_tables):
    from churn_model.as_of_features import FEATURE_COLUMNS, build_as_of_features
    out = build_as_of_features(spark_tables, "2024-06-30", cfg)
    banned = {"persona", "is_churned", "root_cause", "classification_reason", "churn_date",
              "days_silent", "tickets_last_30d_before_silence", "msisdn", "gender", "age"}
    assert not banned & set(out.columns)
    assert not banned & set(FEATURE_COLUMNS)
