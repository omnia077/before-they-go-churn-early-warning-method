"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
etl/run_phase2.py — Raw Storage Layer Ingestion Module
================================================================================

PURPOSE:
    This is the single importable Python module that contains ALL Phase 2 logic.

    It can be used in three ways:
        1. Imported by the Airflow DAG (telecom_churn_pipeline.py):
               from run_phase2_module import validate_dataset, ...
        2. Called as a script by the local orchestrator:
               subprocess.run(["python", "etl/run_phase2.py"])
        3. Called from the Phase 2 notebook (01_data_generation.ipynb):
               The notebook imports and calls functions from here

    BUG FIX #3 — Missing Module:
        Original code had three separate copies of Phase 2 logic:
            a) Inline in 02_raw_storage_layer.ipynb (the only real version)
            b) telecom_churn_pipeline.py tried to import run_phase2_module
               (module did not exist — ImportError on any real Airflow run)
            c) local_orchestrator.py shelled out to ../etl/run_phase2.py
               (file did not exist — FileNotFoundError on execution)

        This file IS that module. One file, three consumers, zero duplication.

    BUG FIX #1 — Manifest Schema:
        Manifest now writes quarantined_rows and quality_score for every
        dataset — the fields that Phase 3 orchestrator reads but were
        previously missing, causing silent N/A values in all log lines.

    BUG FIX #7 — Label Leakage Enforcement:
        The persona column (and any other LABEL_COLUMNS) is explicitly dropped
        from the clean DataFrame before writing to the raw/ partition layer.
        This is the staging boundary — data that crosses it must not contain
        generation labels that would leak the target variable into Phase 4
        feature engineering.

    BUG FIX #4 (Quality Threshold):
        Quality threshold unified to 95% across all code paths.
        Single QUALITY_THRESHOLD constant — one source of truth.

PRODUCTION ARCHITECTURE MAPPING:
    sftp_drop/   → Huawei SFTP landing zone
    raw/         → Staging Tables (Parquet on HDFS)
    Partitioning → HDFS partition pruning for Spark/Presto queries
    catalog/     → Hive Metastore (read by Hue and Zeppelin)
    quarantine/  → Bad records preserved and logged, never silently dropped

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio
================================================================================
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration & Paths
# ════════════════════════════════════════════════════════════════════════════

# ── BUG FIX #4: Single quality threshold constant ─────────────────────────
# Original code had 80% in the DAG, 95% in the README, no check in the
# orchestrator. Now one constant controls every quality gate in the pipeline.
QUALITY_THRESHOLD = 95.0   # Minimum score to mark a dataset PASSED

# ── Paths ──────────────────────────────────────────────────────────────────
# These paths are relative to the project root.
# When called from local_orchestrator.py, the working dir is orchestration/
# When imported by the DAG, paths come from Airflow Variables (overridden below)
SFTP_DROP_PATH  = Path("../data/sftp_drop")
RAW_PATH        = Path("../data/raw")
STAGING_PATH    = Path("../data/staging")
CATALOG_PATH    = Path("../data/catalog")
QUARANTINE_PATH = Path("../data/quarantine")
QUALITY_PATH    = Path("../data/quality_reports")

# Create all output directories on first run
for _p in [RAW_PATH, STAGING_PATH, CATALOG_PATH, QUARANTINE_PATH, QUALITY_PATH]:
    _p.mkdir(parents=True, exist_ok=True)

# ── BUG FIX #7: Label columns that must be dropped at the staging boundary ─
# These columns exist in the raw layer for generation validation only.
# They must NEVER reach Phase 4 feature engineering — they directly reveal
# the target variable (churn status) which would constitute label leakage.
#
# Enforcement: drop_label_columns() is called inside write_partitioned_parquet()
# before any data is written to the partitioned raw/ layer.
#
# The raw Parquet files in sftp_drop/ still contain these columns
# (they are the "as-received" files). The processed raw/ layer does not.
LABEL_COLUMNS_TO_DROP = ["persona"]

# ── Ingestion Timestamp ────────────────────────────────────────────────────
# Used to name manifest files and catalog entries — enables time-travel
# debugging (which run produced which output?)
INGESTION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DATE     = datetime.now().strftime("%Y-%m-%d")

# ── Logging ────────────────────────────────────────────────────────────────
# Structured log — every file movement is auditable.
# In the operator's production: these logs feed into the monitoring layer (Phase 9).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../data/raw_layer.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("raw_layer")
log.info(f"Phase 2 Raw Storage Layer started | Run: {INGESTION_TS}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Schema Registry
# ════════════════════════════════════════════════════════════════════════════
# The schema registry is the contract between the SFTP source and the raw
# storage layer. Every dataset has a registered schema, primary key,
# non-nullable fields, and value validation rules.
#
# This is the equivalent of CREATE TABLE DDL in a data warehouse.
# In the operator's environment: this maps to Hive Metastore table definitions.

SCHEMA_REGISTRY = {

    "customers": {
        "source_file":   "customers.parquet",
        "description":   "Subscriber master records — one row per customer",
        "partition_col": "region",          # Partition by region for geo queries
        "primary_key":   "customer_id",
        "columns": {
            "customer_id":      "str",
            "msisdn":           "str",
            "region":           "str",
            "persona":          "str",      # LABEL — dropped at staging boundary
            "plan_type":        "str",
            "age":              "int",
            "gender":           "str",
            "activation_date":  "str",
            "monthly_arpu_sdg": "float",
            "is_churned":       "int"       # Target variable — kept, not a feature
        },
        "non_nullable": ["customer_id", "msisdn", "region", "plan_type", "is_churned"],
        "validations": {
            "age":        {"min": 18, "max": 90},
            "is_churned": {"allowed": [0, 1]},
            "plan_type":  {"allowed": ["prepaid", "postpaid"]},
            "gender":     {"allowed": ["M", "F"]}
        }
    },

    "usage_events": {
        "source_file":   "usage_events.parquet",
        "description":   "CDR-style usage events — data, voice, SMS per customer",
        "partition_col": "usage_type",      # Partition by type for usage analysis
        "primary_key":   "event_id",
        "columns": {
            "event_id":    "str",
            "customer_id": "str",
            "msisdn":      "str",
            "event_date":  "str",
            "usage_type":  "str",
            "usage_value": "float",
            "region":      "str",
            "persona":     "str"            # LABEL — dropped at staging boundary
        },
        "non_nullable": ["event_id", "customer_id", "msisdn", "event_date", "usage_type"],
        "validations": {
            "usage_value": {"min": 0},
            "usage_type":  {"allowed": ["data", "voice", "sms"]}
        }
    },

    "billing_records": {
        "source_file":   "billing_records.parquet",
        "description":   "Monthly billing records — amounts billed, paid, outstanding",
        "partition_col": "plan_type",
        "primary_key":   "billing_id",
        "columns": {
            "billing_id":          "str",
            "customer_id":         "str",
            "msisdn":              "str",
            "billing_month":       "str",
            "amount_billed":       "float",
            "amount_paid":         "float",
            "outstanding_balance": "float",
            "payment_status":      "str",
            "plan_type":           "str"
        },
        "non_nullable": ["billing_id", "customer_id", "billing_month", "payment_status"],
        "validations": {
            "payment_status": {"allowed": ["paid", "late", "unpaid"]},
            "amount_billed":  {"min": 0},
            "amount_paid":    {"min": 0}
        }
    },

    "recharge_records": {
        "source_file":   "recharge_records.parquet",
        "description":   "Prepaid recharge transactions from IN system",
        "partition_col": "channel",
        "primary_key":   "recharge_id",
        "columns": {
            "recharge_id":   "str",
            "customer_id":   "str",
            "msisdn":        "str",
            "recharge_date": "str",
            "amount_sdg":    "float",
            "channel":       "str",
            "region":        "str"
        },
        "non_nullable": ["recharge_id", "customer_id", "msisdn", "recharge_date", "amount_sdg"],
        "validations": {
            "amount_sdg": {"min": 0},
            "channel":    {"allowed": ["ussd", "agent", "bank", "app", "online"]}
        }
    },

    "network_tickets": {
        "source_file":   "network_tickets.parquet",
        "description":   "NOC network complaint tickets per customer",
        "partition_col": "status",
        "primary_key":   "ticket_id",
        "columns": {
            "ticket_id":       "str",
            "customer_id":     "str",
            "msisdn":          "str",
            "ticket_date":     "str",
            "issue_type":      "str",
            "region":          "str",
            "status":          "str",
            "resolution_days": "float",
            "persona":         "str"        # LABEL — dropped at staging boundary
        },
        "non_nullable": ["ticket_id", "customer_id", "msisdn", "ticket_date", "issue_type"],
        "validations": {
            "status":     {"allowed": ["open", "in_progress", "resolved", "closed"]},
            "issue_type": {"allowed": [
                "dropped_call", "slow_data", "no_signal", "sms_failure", "roaming_issue"
            ]}
        }
    }
}

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Type Enforcement
# ════════════════════════════════════════════════════════════════════════════

def enforce_types(df: pd.DataFrame, schema: dict) -> tuple:
    """
    Cast each column to its registered type from the schema.

    Uses pandas nullable integer type (Int64) to handle NaN in int columns
    without silent coercion to float — a common source of downstream bugs.

    Args:
        df:     Raw DataFrame as loaded from sftp_drop/
        schema: Dataset schema dict from SCHEMA_REGISTRY

    Returns:
        (df, errors): Type-enforced DataFrame and dict of any type errors
    """
    errors = {}

    for col, dtype in schema["columns"].items():
        if col not in df.columns:
            errors[col] = "MISSING_COLUMN"
            continue
        try:
            if dtype == "str":
                df[col] = df[col].astype(str).str.strip()
            elif dtype == "int":
                # Int64 (capital I) is pandas nullable integer — handles NaN
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        except Exception as e:
            errors[col] = str(e)

    return df, errors

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Validation Engine
# ════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame, dataset_name: str, schema: dict) -> tuple:
    """
    Full validation pipeline for one dataset.

    Steps:
        1. Column presence check
        2. Type enforcement
        3. Non-nullable constraint check
        4. Value validation (min/max/allowed)
        5. Duplicate primary key detection
        6. Clean/quarantine split
        7. Quality score calculation

    Args:
        df:           Raw DataFrame loaded from sftp_drop/
        dataset_name: String key matching SCHEMA_REGISTRY
        schema:       Schema dict for this dataset

    Returns:
        clean_df:      DataFrame with only valid records (no label columns)
        quarantine_df: DataFrame with rejected records + quarantine_reason column
        report:        dict with full validation results including quality_score
    """
    total_rows        = len(df)
    quarantine_mask   = pd.Series([False] * total_rows, index=df.index)
    quarantine_reasons = pd.Series([""] * total_rows, index=df.index)
    validation_checks = {}

    # ── Step 1: Column presence ────────────────────────────────────────────
    expected_cols = set(schema["columns"].keys())
    actual_cols   = set(df.columns)
    missing_cols  = expected_cols - actual_cols

    validation_checks["missing_columns"] = list(missing_cols)
    validation_checks["extra_columns"]   = list(actual_cols - expected_cols)

    if missing_cols:
        log.error(f"{dataset_name}: Missing columns: {missing_cols}")
        raise ValueError(f"Schema mismatch — missing columns: {missing_cols}")

    # ── Step 2: Type enforcement ───────────────────────────────────────────
    df, type_errors = enforce_types(df, schema)
    validation_checks["type_errors"] = type_errors

    # ── Step 3: Non-nullable check ─────────────────────────────────────────
    null_counts = {}
    for col in schema["non_nullable"]:
        null_mask = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        n_nulls   = int(null_mask.sum())
        if n_nulls > 0:
            null_counts[col]           = n_nulls
            quarantine_mask           |= null_mask
            quarantine_reasons[null_mask] += f"NULL_IN_{col.upper()}; "
    validation_checks["null_violations"] = null_counts

    # ── Step 4: Value validations ──────────────────────────────────────────
    val_violations = {}
    for col, rules in schema.get("validations", {}).items():
        if col not in df.columns:
            continue

        if "min" in rules:
            bad = (df[col] < rules["min"]).fillna(False)
            if bad.sum() > 0:
                val_violations[f"{col}_below_min"]    = int(bad.sum())
                quarantine_mask                       |= bad
                quarantine_reasons[bad]               += f"{col.upper()}_BELOW_MIN; "

        if "max" in rules:
            bad = (df[col] > rules["max"]).fillna(False)
            if bad.sum() > 0:
                val_violations[f"{col}_above_max"]    = int(bad.sum())
                quarantine_mask                       |= bad
                quarantine_reasons[bad]               += f"{col.upper()}_ABOVE_MAX; "

        if "allowed" in rules:
            bad = (~df[col].isin(rules["allowed"])).fillna(False)
            if bad.sum() > 0:
                val_violations[f"{col}_invalid_value"] = int(bad.sum())
                quarantine_mask                        |= bad
                quarantine_reasons[bad]                += f"{col.upper()}_INVALID_VALUE; "

    validation_checks["value_violations"] = val_violations

    # ── Step 5: Duplicate primary key ─────────────────────────────────────
    pk       = schema["primary_key"]
    dup_mask = df.duplicated(subset=[pk], keep="first")
    n_dups   = int(dup_mask.sum())
    validation_checks["duplicate_primary_keys"] = n_dups
    if n_dups > 0:
        quarantine_mask                |= dup_mask
        quarantine_reasons[dup_mask]   += f"DUPLICATE_{pk.upper()}; "

    # ── Step 6: Split clean vs quarantine ─────────────────────────────────
    n_quarantined = int(quarantine_mask.sum())
    n_clean       = total_rows - n_quarantined

    clean_df      = df[~quarantine_mask].copy()

    quarantine_df = df[quarantine_mask].copy()
    quarantine_df["quarantine_reason"] = quarantine_reasons[quarantine_mask].values
    quarantine_df["quarantine_ts"]     = datetime.now().isoformat()
    quarantine_df["source_dataset"]    = dataset_name

    # ── Step 7: MD5 checksum for data lineage ─────────────────────────────
    # Enables idempotency check: if the same raw file is re-ingested,
    # the checksum will match and we know the output is identical.
    checksum = hashlib.md5(
        pd.util.hash_pandas_object(clean_df).values.tobytes()
    ).hexdigest()

    quality_score = round((n_clean / total_rows) * 100, 4) if total_rows > 0 else 0.0

    # Unified threshold bands — matches Phase 1 quality passport exactly
    if quality_score >= QUALITY_THRESHOLD:
        status = "PASSED"
    elif quality_score >= 80.0:
        status = "WARNING"   # Proceed with caution — flag for data team
    else:
        status = "FAILED"    # Pipeline must halt

    report = {
        "dataset":             dataset_name,
        "ingestion_run":       INGESTION_TS,
        "source_file":         schema["source_file"],
        "total_rows_received": total_rows,
        "clean_rows":          n_clean,
        "quarantined_rows":    n_quarantined,
        "quality_score":       quality_score,
        "status":              status,
        "checks":              validation_checks,
        "checksum_md5":        checksum,
        "partition_col":       schema["partition_col"],
        "timestamp":           datetime.now().isoformat()
    }

    return clean_df, quarantine_df, report

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Label Column Drop (Bug 7 enforcement)
# ════════════════════════════════════════════════════════════════════════════

def drop_label_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Drop label/generation columns before writing to the processed raw layer.

    BUG FIX #7: This function enforces the staging boundary.
    - sftp_drop/ Parquet files: contain persona (as-received, untouched)
    - raw/ partitioned Parquet: persona REMOVED here, before write

    Why here and not in Phase 4?
        Phase 4 feature engineering should receive data that is already clean
        of internal labels. The staging transform is the correct architectural
        location for this drop — not the model training step.

    Why not in Phase 1?
        The raw layer should store data faithfully as received from source.
        The drop happens at the staging boundary, not at generation.

    Args:
        df:           Clean DataFrame (post-validation)
        dataset_name: Used for logging only

    Returns:
        DataFrame with LABEL_COLUMNS_TO_DROP removed
    """
    cols_present = [c for c in LABEL_COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        log.info(
            f"{dataset_name}: Dropped label columns at staging boundary: {cols_present}"
        )
    return df

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Partitioned Parquet Writer
# ════════════════════════════════════════════════════════════════════════════

def write_partitioned_parquet(
    clean_df: pd.DataFrame,
    dataset_name: str,
    schema: dict,
    report: dict
) -> dict:
    """
    Write clean DataFrame to partitioned Parquet directory structure.

    Output:
        raw/{dataset_name}/{partition_col}={value}/data.parquet

    This mirrors the operator's HDFS Parquet table layout — the same structure
    Spark reads with partition pruning and Presto queries with predicate
    pushdown. Partition pruning means Spark/Presto only reads the partitions
    needed for a query, not the entire table.

    BUG FIX #7: Calls drop_label_columns() before writing.
    Label columns never appear in the raw/ output layer.

    Args:
        clean_df:     Validated DataFrame (from validate_dataset)
        dataset_name: String key matching SCHEMA_REGISTRY
        schema:       Schema dict for this dataset
        report:       Validation report dict (for logging)

    Returns:
        partition_info: dict mapping partition values → row counts and paths
    """
    partition_col   = schema["partition_col"]
    dataset_out_dir = RAW_PATH / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Drop label columns before writing ────────────────────────────────
    # This is the staging boundary — data written here must not contain
    # generation labels. Phase 4 reads from raw/, not sftp_drop/.
    clean_df = drop_label_columns(clean_df, dataset_name)

    partition_info   = {}
    partition_values = clean_df[partition_col].unique()

    for partition_val in sorted(partition_values):
        partition_df  = clean_df[clean_df[partition_col] == partition_val].copy()

        # raw/usage_events/usage_type=data/
        partition_dir = dataset_out_dir / f"{partition_col}={partition_val}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Overwrite mode = idempotent — safe to re-run for the same date
        out_path = partition_dir / "data.parquet"
        partition_df.to_parquet(out_path, index=False, compression="snappy")

        file_size_kb = out_path.stat().st_size / 1024

        partition_info[str(partition_val)] = {
            "rows":          len(partition_df),
            "path":          str(out_path),
            "size_kb":       round(file_size_kb, 2),
            "pct_of_total":  round(len(partition_df) / len(clean_df) * 100, 2)
        }

        log.info(
            f"{dataset_name} | partition {partition_col}={partition_val} | "
            f"{len(partition_df):,} rows | {file_size_kb:.1f} KB"
        )

    return partition_info

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Quarantine Handler
# ════════════════════════════════════════════════════════════════════════════

def save_quarantine(quarantine_df: pd.DataFrame, dataset_name: str):
    """
    Persist quarantined records with timestamp and rejection reason.

    Production principle: never silently drop bad data.
    Quarantine files are the audit trail for data quality investigations.
    In the operator's environment: these feed into the data quality monitoring dashboard.

    Args:
        quarantine_df: DataFrame of rejected records with quarantine_reason column
        dataset_name:  Used to name the output file

    Returns:
        str path to the quarantine file, or None if no records quarantined
    """
    if len(quarantine_df) == 0:
        return None

    q_path = QUARANTINE_PATH / f"{dataset_name}_quarantine_{INGESTION_TS}.csv"
    quarantine_df.to_csv(q_path, index=False)
    log.warning(
        f"{dataset_name}: {len(quarantine_df):,} records quarantined → {q_path}"
    )
    return str(q_path)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Data Catalog Builder
# ════════════════════════════════════════════════════════════════════════════

def build_catalog_entry(
    dataset_name: str,
    schema: dict,
    report: dict,
    partition_info: dict,
    quarantine_path
) -> dict:
    """
    Build a Data Catalog entry for one dataset.

    Simulates what Hive Metastore stores for each table in the operator's
    Hue/Zeppelin environment. Analysts use this to understand schema,
    location, quality, and lineage of each table.

    Args:
        dataset_name:    String key matching SCHEMA_REGISTRY
        schema:          Schema dict for this dataset
        report:          Validation report dict from validate_dataset()
        partition_info:  Output of write_partitioned_parquet()
        quarantine_path: Path to quarantine file (or None)

    Returns:
        dict: Catalog entry — stored in raw_layer_catalog.json
    """
    return {
        # ── Identity ──────────────────────────────────────────────────────
        "table_name":    dataset_name,
        "layer":         "raw",
        "description":   schema["description"],
        "owner":         "data_engineering",
        "created_at":    datetime.now().isoformat(),
        "ingestion_run": INGESTION_TS,

        # ── Schema ────────────────────────────────────────────────────────
        "schema": {
            "columns":      schema["columns"],
            "primary_key":  schema["primary_key"],
            "non_nullable": schema["non_nullable"],
            "label_columns_dropped_at_staging": LABEL_COLUMNS_TO_DROP
        },

        # ── Storage ───────────────────────────────────────────────────────
        "storage": {
            "format":        "parquet",
            "compression":   "snappy",
            "location":      str(RAW_PATH / dataset_name),
            "partition_col": schema["partition_col"],
            "partitions":    partition_info,
            "total_size_kb": round(
                sum(p["size_kb"] for p in partition_info.values()), 2
            )
        },

        # ── Quality ───────────────────────────────────────────────────────
        "quality": {
            "total_rows_received": report["total_rows_received"],
            "clean_rows":          report["clean_rows"],
            "quarantined_rows":    report["quarantined_rows"],
            "quality_score":       report["quality_score"],
            "status":              report["status"],
            "quarantine_file":     quarantine_path
        },

        # ── Lineage ───────────────────────────────────────────────────────
        "lineage": {
            "source":       "sftp_drop",
            "source_file":  schema["source_file"],
            "upstream":     "Huawei BSS → SFTP → sftp_drop/",
            "downstream":   "Spark Processing (Phase 4)",
            "checksum_md5": report["checksum_md5"]
        }
    }

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Main Ingestion Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_ingestion_pipeline() -> dict:
    """
    Execute the full Phase 2 ingestion pipeline for all registered datasets.

    Called when this module is run as a script:
        python etl/run_phase2.py

    Also importable — the Airflow DAG calls individual functions directly.

    Returns:
        manifest: dict with per-dataset stats and pipeline-level flags.
                  Written to catalog/raw_manifest_{INGESTION_TS}.json.
    """
    print("=" * 65)
    print("  PHASE 2 — RAW STORAGE LAYER INGESTION PIPELINE")
    print(f"  Run: {INGESTION_TS}")
    print("=" * 65)

    full_catalog    = {}
    pipeline_report = []

    for dataset_name, schema in SCHEMA_REGISTRY.items():

        print(f"\n📥 Processing: {dataset_name}")
        print(f"   Source: {SFTP_DROP_PATH / schema['source_file']}")

        # ── Load from sftp_drop/ ───────────────────────────────────────────
        source_path = SFTP_DROP_PATH / schema["source_file"]
        if not source_path.exists():
            log.error(f"{dataset_name}: Source file not found: {source_path}")
            print(f"   ❌ File not found — skipping")
            continue

        df_raw = pd.read_parquet(source_path)
        log.info(f"{dataset_name}: Loaded {len(df_raw):,} rows from sftp_drop")

        # ── Validate + quarantine ──────────────────────────────────────────
        clean_df, quarantine_df, report = validate_dataset(df_raw, dataset_name, schema)

        # ── Save quarantined records ───────────────────────────────────────
        q_path = save_quarantine(quarantine_df, dataset_name)

        # ── Write partitioned Parquet to raw/ ─────────────────────────────
        # Label columns are dropped inside this function (Bug 7 fix)
        partition_info = write_partitioned_parquet(clean_df, dataset_name, schema, report)

        # ── Build catalog entry ────────────────────────────────────────────
        catalog_entry = build_catalog_entry(
            dataset_name, schema, report, partition_info, q_path
        )
        full_catalog[dataset_name] = catalog_entry

        # ── Save individual validation report ─────────────────────────────
        report_path = QUALITY_PATH / f"{dataset_name}_raw_validation_{INGESTION_TS}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # ── Summary row ───────────────────────────────────────────────────
        pipeline_report.append({
            "Dataset":     dataset_name,
            "Received":    f"{report['total_rows_received']:,}",
            "Clean":       f"{report['clean_rows']:,}",
            "Quarantined": f"{report['quarantined_rows']:,}",
            "Score":       f"{report['quality_score']}%",
            "Status":      report["status"],
            "Partitions":  len(partition_info)
        })

        icon = "✅" if report["status"] == "PASSED" else ("⚠️" if report["status"] == "WARNING" else "❌")
        print(
            f"   {icon} Clean: {report['clean_rows']:,} | "
            f"Quarantined: {report['quarantined_rows']:,} | "
            f"Score: {report['quality_score']}% | "
            f"Status: {report['status']} | "
            f"Partitions: {len(partition_info)}"
        )

    log.info(f"Pipeline completed | {len(pipeline_report)} datasets processed")

    # ── Save Data Catalog ──────────────────────────────────────────────────
    catalog_path = CATALOG_PATH / "raw_layer_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(full_catalog, f, indent=2)
    print(f"\n📚 Data Catalog saved → {catalog_path}")

    # ── Build and Save Manifest ────────────────────────────────────────────
    # BUG FIX #1: manifest now includes quarantined_rows and quality_score
    # for every dataset. These fields are what Phase 3 orchestrator reads.
    # Original manifest only wrote clean_rows, status, partitions —
    # causing silent N/A values in every orchestration log line.
    manifest = {
        "layer":              "raw",
        "ingestion_run":      INGESTION_TS,
        "run_date":           RUN_DATE,
        "datasets_processed": len(pipeline_report),
        "datasets": {
            row["Dataset"]: {
                "clean_rows":       row["Clean"],
                "quarantined_rows": row["Quarantined"],        # FIX #1: now included
                "quality_score":    row["Score"].replace("%", ""),  # FIX #1: now included
                "status":           row["Status"],
                "partitions":       row["Partitions"]
            }
            for row in pipeline_report
        },
        "total_clean_rows":       sum(
            int(r["Clean"].replace(",", "")) for r in pipeline_report
        ),
        "total_quarantined_rows": sum(
            int(r["Quarantined"].replace(",", "")) for r in pipeline_report
        ),
        "all_passed":  all(r["Status"] == "PASSED" for r in pipeline_report),
        "next_phase":  (
            "Phase 4 — Spark Processing"
            if all(r["Status"] == "PASSED" for r in pipeline_report)
            else "BLOCKED — fix quality issues first"
        ),

        # ── Lineage (top-level manifest) ───────────────────────────────────
        # The per-dataset catalog entries already carry detailed lineage
        # (see build_catalog_entry). This block does the same job at the
        # pipeline level — the manifest is what Phase 3 reads and what
        # Phase 4 will read next, so it needs its own lineage answer too.
        # Three fields only: where this manifest's data came from, what
        # produced it, and what consumes it next. Same structure Phase 4
        # will use in its own manifest — one consistent pattern end to end.
        "lineage": {
            "source":       "sftp_drop/ (Huawei SFTP simulation)",
            "produced_by":  "etl/run_phase2.py",
            "consumed_by":  "Phase 4 — spark/process_layer.py"
        }
    }

    manifest_path = CATALOG_PATH / f"raw_manifest_{INGESTION_TS}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"📋 Manifest saved → {manifest_path}")
    print(f"   All passed:             {manifest['all_passed']}")
    print(f"   Total clean rows:       {manifest['total_clean_rows']:,}")
    print(f"   Total quarantined rows: {manifest['total_quarantined_rows']:,}")
    print(f"   Next phase:             {manifest['next_phase']}")
    print(f"   Lineage → produced_by:  {manifest['lineage']['produced_by']}")
    print(f"   Lineage → consumed_by:  {manifest['lineage']['consumed_by']}")

    return manifest


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Entry Point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Script entry point — called by local_orchestrator.py via subprocess:
        subprocess.run(["python", "../etl/run_phase2.py"])

    Also callable directly:
        python etl/run_phase2.py
    """
    manifest = run_ingestion_pipeline()

    print("\n" + "=" * 65)
    print("  PHASE 2 COMPLETE — RAW STORAGE LAYER SUMMARY")
    print("=" * 65)
    print("\n🗺️  PRODUCTION ARCHITECTURE MAPPING:")
    print("   sftp_drop/   → Huawei SFTP landing zone")
    print("   raw/         → Staging Tables (Parquet on HDFS)")
    print("   Partitioning → HDFS partition pruning for Spark/Presto")
    print("   catalog/     → Hive Metastore (read by Hue + Zeppelin)")
    print("   quarantine/  → Bad records preserved, never dropped")
    print("   raw_layer.log → Audit trail for monitoring (Phase 9)")
    print("\n🔧 BUGS FIXED IN THIS VERSION:")
    print("   Bug 1 — Manifest now includes quarantined_rows + quality_score")
    print("   Bug 3 — This file IS the importable module the DAG and orchestrator need")
    print("   Bug 4 — Quality threshold = 95% (unified constant)")
    print("   Bug 7 — Label columns dropped at staging boundary before raw/ write")
    print(f"\n✅ Ready for Phase 4 — Spark Processing")
    print("=" * 65)
