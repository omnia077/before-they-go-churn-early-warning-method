"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
spark/process_layer.py — Spark Feature Engineering Layer
================================================================================

PURPOSE:
    This is the Phase 4 entry point. Loads partitioned Parquet from raw/
    (written by Phase 2), reconstructs each customer's journey, and computes
    the behavioral and network features that root_cause_engine.py classifies.

    In production: this is the spark-submit job that Phase 3's DAG triggers
    after the quality gate passes.

WHAT THIS FILE DOES NOT DO:
    It does not classify root cause — that logic lives in root_cause_engine.py,
    kept separate so the decision tree can be read, tested, and explained on
    its own, independent of the feature engineering plumbing around it.

    It does not write Iceberg — that lives in iceberg_writer.py. This file's
    job ends at "produce a clean feature DataFrame", the same separation of
    concerns the operator's pipeline uses: compute, then persist, are different steps.

FEATURES COMPUTED (map directly to the root cause decision tree):
    days_silent                       — days since last usage event
    unpaid_months_recent              — unpaid billing months in last 3 cycles
    arpu_drop_pct                     — % drop vs customer's own 3-month avg ARPU
    tickets_last_30d_before_silence   — network tickets in the 30 days
                                         immediately before the customer went quiet

PRODUCTION ARCHITECTURE MAPPING:
    raw/        → input  (Staging Tables / HDFS Parquet, written by Phase 2)
    Spark       → this file (Apache Spark — same engine operators run)
    processed/  → output (handed to iceberg_writer.py for Iceberg persistence)

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio
================================================================================
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration
# ════════════════════════════════════════════════════════════════════════════

RAW_PATH       = Path("../data/raw")
PROCESSED_PATH = Path("../data/processed")
CATALOG_PATH   = Path("../data/catalog")

for _p in [PROCESSED_PATH, CATALOG_PATH]:
    _p.mkdir(parents=True, exist_ok=True)

# Reference date for "days since" calculations.
# In production this is the Airflow execution date ({{ ds }}).
# Hardcoded here to match the simulation window from Phase 1.
REFERENCE_DATE = "2024-12-31"

# Quality threshold — same constant used in Phase 2 and Phase 3.
# A feature pipeline should not silently process a dataset that failed
# the upstream quality gate, even if Phase 3 already checked it once.
QUALITY_THRESHOLD = 95.0

INGESTION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DATE     = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../data/spark_processing.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("spark_layer")


def get_spark_session() -> SparkSession:
    """
    Build the Spark session for local execution.

    In production: master would be "yarn" and config would include
    executor counts and memory, matching the operator's cluster sizing.
    Here: local[2] is enough for the portfolio's data volume (2.4M events
    is well within what a 2-core local Spark handles in seconds).
    """
    return (
        SparkSession.builder
        .appName("telecom_churn_feature_engineering")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")  # Small for local mode
        .getOrCreate()
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Load Raw Layer
# ════════════════════════════════════════════════════════════════════════════

def load_raw_datasets(spark: SparkSession) -> dict:
    """
    Load all 5 partitioned datasets from raw/ — the output of Phase 2.

    Why read from raw/ and not sftp_drop/?
        sftp_drop/ still has the persona label column and is unvalidated.
        raw/ is post-quarantine, post-validation, and (per Bug 7 fix in
        Phase 2) has the persona column already dropped. This is the
        staging boundary — Phase 4 should never see generation labels.

    Returns:
        dict of {dataset_name: Spark DataFrame}
    """
    datasets = {}
    dataset_names = ["customers", "usage_events", "billing_records",
                      "recharge_records", "network_tickets"]

    for name in dataset_names:
        dataset_dir = RAW_PATH / name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"raw/{name} not found. Run Phase 2 (etl/run_phase2.py) first."
            )
        # Spark reads the whole partitioned directory tree in one call —
        # partition columns become DataFrame columns automatically.
        df = spark.read.parquet(str(dataset_dir))
        datasets[name] = df
        log.info(f"Loaded raw/{name}: {df.count():,} rows")

    # Sanity check: persona must NOT be present (Bug 7 enforcement check)
    for name in ["usage_events", "network_tickets"]:
        if "persona" in datasets[name].columns:
            raise ValueError(
                f"{name} still has 'persona' column in raw/ — Bug 7 fix in "
                f"etl/run_phase2.py was not applied. Re-run Phase 2 before Phase 4."
            )

    log.info("✅ Label leakage check passed — persona column absent from raw/")
    return datasets


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Engineering
# ════════════════════════════════════════════════════════════════════════════

def _as_of_filter(df, date_col: str, as_of):
    """Keep rows dated on or before `as_of` (None = no filter, legacy behaviour)."""
    if as_of is None:
        return df
    return df.filter(F.col(date_col).cast("date") <= F.lit(as_of).cast("date"))


def _as_of_billing(billing_df, as_of):
    """Billing rows for months up to and including the month of `as_of`."""
    if as_of is None:
        return billing_df
    return billing_df.filter(F.col("billing_month") <= as_of[:7])


def compute_days_silent(usage_events_df, reference_date: str, as_of: str = None):
    """
    Feature: days since each customer's last usage event.

    This is the primary behavioral churn signal. A customer who has used
    the network recently is, by definition, not behaviorally churned —
    regardless of what their billing or persona history shows.

    as_of: if given, only events on or before that date are visible and
    `reference_date` should equal it (ML cutoff). None keeps Phase 4 behaviour.
    """
    usage_events_df = _as_of_filter(usage_events_df, "event_date", as_of)
    last_usage = usage_events_df.groupBy("customer_id").agg(
        F.max("event_date").alias("last_event_date")
    )
    return last_usage.withColumn(
        "days_silent",
        F.datediff(F.lit(reference_date), F.col("last_event_date").cast("date"))
    )


def compute_unpaid_months(billing_df, lookback_months: int = 3, as_of: str = None):
    """
    Feature: count of unpaid billing months in the most recent lookback window.

    Why a rolling window and not all-time unpaid count?
        A customer who was unpaid once in January but has paid every month
        since is not currently at billing risk. Recent payment behavior is
        what signals active disengagement — this mirrors how a real
        collections team would prioritize accounts.
    """
    from pyspark.sql.window import Window
    billing_df = _as_of_billing(billing_df, as_of)
    w = Window.partitionBy("customer_id").orderBy(F.col("billing_month").desc())
    ranked = billing_df.withColumn("rn", F.row_number().over(w))
    recent_only = ranked.filter(F.col("rn") <= lookback_months)

    return recent_only.groupBy("customer_id").agg(
        F.sum(F.when(F.col("payment_status") == "unpaid", 1).otherwise(0))
         .alias("unpaid_months_recent")
    )


def compute_arpu_drop(billing_df, as_of: str = None):
    """
    Feature: % drop in a customer's most recent bill vs their own 3-month average.

    Why compare to the customer's OWN average, not a global average?
        ARPU varies enormously by persona — a loyal customer's "normal" is
        a declining customer's "high". Comparing each customer only to their
        own history detects the SHAPE of decline, not the absolute level.
        This is the same logic a real billing anomaly detector would use.
    """
    from pyspark.sql.window import Window
    billing_df = _as_of_billing(billing_df, as_of)
    w = Window.partitionBy("customer_id").orderBy(F.col("billing_month").desc())
    ranked = billing_df.withColumn("rn", F.row_number().over(w))

    most_recent = ranked.filter(F.col("rn") == 1).select(
        "customer_id", F.col("amount_billed").alias("latest_bill")
    )
    avg_last_3 = ranked.filter(F.col("rn") <= 3).groupBy("customer_id").agg(
        F.avg("amount_billed").alias("avg_bill_3mo")
    )

    arpu = most_recent.join(avg_last_3, "customer_id")
    return arpu.withColumn(
        "arpu_drop_pct",
        F.when(
            F.col("avg_bill_3mo") > 0,
            (1 - (F.col("latest_bill") / F.col("avg_bill_3mo"))) * 100
        ).otherwise(0.0)
    ).select("customer_id", "arpu_drop_pct")


def compute_tickets_before_silence(tickets_df, last_usage_df, window_days: int = 30,
                                   as_of: str = None):
    """
    Feature: network tickets filed in the N days immediately BEFORE
    a customer's last usage event.

    This is the most important feature for separating network-driven churn
    from behavioral churn. Counting ALL tickets ever would conflate a
    resolved issue from 8 months ago with an active, unresolved problem
    right before the customer left — only tickets in the pre-silence window
    are causally relevant.
    """
    tickets_df = _as_of_filter(tickets_df, "ticket_date", as_of)
    joined = tickets_df.join(
        last_usage_df.select("customer_id", "last_event_date"),
        "customer_id"
    )
    joined = joined.withColumn(
        "days_before_silence",
        F.datediff(F.col("last_event_date").cast("date"), F.col("ticket_date").cast("date"))
    )
    # Only tickets that occurred BEFORE the silence (>=0) and WITHIN the window
    relevant = joined.filter(
        (F.col("days_before_silence") >= 0) &
        (F.col("days_before_silence") <= window_days)
    )
    return relevant.groupBy("customer_id").agg(
        F.count("*").alias("tickets_last_30d_before_silence")
    )


def build_feature_table(datasets: dict, reference_date: str, as_of: str = None):
    """
    Join all computed features onto the customer master table.

    as_of: optional cutoff date. When set, every feature only sees data up to
    that date (used for point-in-time ML features). Default None preserves the
    Phase 4 root-cause behaviour exactly.

    Returns:
        Spark DataFrame: one row per customer, with all churn-relevant
        features needed by root_cause_engine.py.
    """
    customers = datasets["customers"]
    usage     = datasets["usage_events"]
    billing   = datasets["billing_records"]
    tickets   = datasets["network_tickets"]

    log.info("Computing days_silent...")
    last_usage_df = compute_days_silent(usage, reference_date, as_of)

    log.info("Computing unpaid_months_recent...")
    unpaid_df = compute_unpaid_months(billing, as_of=as_of)

    log.info("Computing arpu_drop_pct...")
    arpu_df = compute_arpu_drop(billing, as_of=as_of)

    log.info("Computing tickets_last_30d_before_silence...")
    tickets_df = compute_tickets_before_silence(tickets, last_usage_df, as_of=as_of)

    # ── Join everything onto the customer master table ────────────────────
    features = (
        customers
        .join(last_usage_df.select("customer_id", "last_event_date", "days_silent"),
              "customer_id", "left")
        .join(unpaid_df, "customer_id", "left")
        .join(arpu_df, "customer_id", "left")
        .join(tickets_df, "customer_id", "left")
        # A customer with NO usage events ever, NO billing issues, NO tickets
        # gets nulls from the left joins — fillna makes the decision tree
        # in root_cause_engine.py safe to evaluate without null-handling
        # logic scattered through every rule.
        .fillna({
            "days_silent":                     999,  # Treat missing as "very silent"
            "unpaid_months_recent":            0,
            "arpu_drop_pct":                    0.0,
            "tickets_last_30d_before_silence":  0
        })
        # BUG FIX (lineage) — stamp the business date this feature table was
        # computed against directly onto every row. Previously REFERENCE_DATE
        # only existed as a constant in this file's code, never as a column
        # in the output. That meant a downstream consumer reading the
        # Iceberg table this eventually feeds into had no way to tell WHICH
        # date a given row's classification represents, once the table has
        # more than one run's worth of history in it. Kept as a string, same
        # serialization convention as every other date column in this
        # pipeline (activation_date, event_date, ticket_date) — see
        # iceberg_writer.py's own note on why string dates avoid timezone
        # ambiguity in pyiceberg's stricter schema inference.
        .withColumn("reference_date", F.lit(reference_date))
    )

    return features


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Data Reconciliation
# ════════════════════════════════════════════════════════════════════════════

def reconcile_row_counts(datasets: dict, feature_df) -> dict:
    """
    Prevent silent data loss: verify every customer in the master table
    appears exactly once in the final feature table.

    Why this matters: a bad join (inner instead of left, or a key mismatch)
    silently drops rows — the pipeline still "succeeds" but with fewer
    customers than it started with. This check catches that immediately
    instead of letting bad row counts flow into Phase 5-6 undetected.
    """
    customer_count = datasets["customers"].count()
    feature_count  = feature_df.count()

    reconciliation = {
        "customers_in":     customer_count,
        "customers_out":    feature_count,
        "rows_lost":        customer_count - feature_count,
        "reconciled":       customer_count == feature_count
    }

    if not reconciliation["reconciled"]:
        log.error(
            f"Row count mismatch: {customer_count} customers in, "
            f"{feature_count} in feature table. {reconciliation['rows_lost']} lost."
        )
    else:
        log.info(f"✅ Reconciliation passed: {customer_count} customers, no rows lost")

    return reconciliation


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Manifest & Lineage
# ════════════════════════════════════════════════════════════════════════════

def build_phase4_manifest(feature_df, reconciliation: dict) -> dict:
    """
    Build the Phase 4 manifest — same pattern as Phase 2's manifest,
    continuing the lineage chain link by link.

    LINEAGE: this manifest's source is Phase 2's raw/ output. Its
    produced_by is this file. Its consumed_by is Phase 4's next stage —
    root_cause_engine.py reads the feature table this manifest describes.
    """
    feature_count = feature_df.count()

    manifest = {
        "layer":               "processed",
        "stage":                "feature_engineering",
        "processing_run":       INGESTION_TS,
        "run_date":             RUN_DATE,
        "reference_date":       REFERENCE_DATE,
        "rows_processed":       feature_count,
        "reconciliation":       reconciliation,
        "features_computed": [
            "days_silent",
            "unpaid_months_recent",
            "arpu_drop_pct",
            "tickets_last_30d_before_silence"
        ],
        "lineage_columns_added": [
            "reference_date  — business date this run's classification represents"
        ],
        "quality_passed":       reconciliation["reconciled"],
        "next_stage": (
            "root_cause_engine.py"
            if reconciliation["reconciled"]
            else "BLOCKED — row count mismatch, investigate before classifying"
        ),

        # ── Lineage — same 3-field pattern as Phase 2's manifest ──────────
        "lineage": {
            "source":      "raw/ (Phase 2 output via etl/run_phase2.py)",
            "produced_by": "spark/process_layer.py",
            "consumed_by": "spark/root_cause_engine.py"
        }
    }

    manifest_path = CATALOG_PATH / f"phase4_feature_manifest_{INGESTION_TS}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest saved → {manifest_path}")
    return manifest


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

def run_feature_engineering():
    """
    Full Phase 4 feature engineering pipeline.

    Called by:
        - 04_spark_processing.ipynb (interactive walkthrough)
        - root_cause_engine.py (imports and calls this directly)
        - Airflow DAG (Phase 4 task, once added to telecom_churn_pipeline.py)

    Returns:
        (spark, feature_df, manifest): the live SparkSession (caller must
        stop it), the feature DataFrame, and the manifest dict.
    """
    print("=" * 65)
    print("  PHASE 4 — SPARK FEATURE ENGINEERING")
    print(f"  Run: {INGESTION_TS}")
    print("=" * 65)

    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("\n📥 Loading raw layer datasets...")
    datasets = load_raw_datasets(spark)

    print("\n🔧 Computing churn features...")
    feature_df = build_feature_table(datasets, REFERENCE_DATE)

    print("\n🔍 Reconciling row counts...")
    reconciliation = reconcile_row_counts(datasets, feature_df)

    print("\n📋 Building manifest...")
    manifest = build_phase4_manifest(feature_df, reconciliation)

    # Cache for downstream use — feature_df gets read again by
    # root_cause_engine.py and we don't want to recompute joins twice
    feature_df = feature_df.cache()
    feature_df.count()  # Force materialization

    print(f"\n✅ Feature engineering complete")
    print(f"   Rows: {manifest['rows_processed']:,}")
    print(f"   Reconciled: {manifest['quality_passed']}")
    print(f"   Lineage → produced_by: {manifest['lineage']['produced_by']}")
    print(f"   Lineage → consumed_by: {manifest['lineage']['consumed_by']}")
    print("=" * 65)

    return spark, feature_df, manifest


if __name__ == "__main__":
    spark, feature_df, manifest = run_feature_engineering()
    feature_df.show(10, truncate=False)
    spark.stop()
