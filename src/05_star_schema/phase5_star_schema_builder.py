"""
================================================================================
PHASE 5 — TELECOM DATA WAREHOUSE MODELING (STAR SCHEMA)
================================================================================
Project   : Telecom Customer Journey & Churn Root Cause Analysis Platform
Phase     : 5 of 8
Author    : <your name>
Maps to   : a Sudanese operator's Apache Spark + Iceberg + HDFS warehouse layer

REWRITE NOTE: this version replaces an earlier draft that guessed at Phase 4's
output schema (assumed a "segment" column, plan-level pricing, a daily
snapshot_date, and a separate ticket-grain Phase 4 output — none of which
exist). This version is built and tested against your REAL Phase 1, 2, and 4
code, run end to end. See the schema notes below for what's actually there.

WHAT THIS SCRIPT DOES
----------------------
Reads the real Iceberg table your Phase 4 pipeline writes
(telecom_churn.churn_classified, via iceberg_writer.py) and the raw-layer
network_tickets Parquet (via Phase 2), and reshapes both into a star schema:
small descriptive DIMENSION tables surrounding large, narrow FACT tables —
the same modeling pattern the operator's real warehouse uses on Iceberg/HDFS.

CONFIRMED SOURCE SCHEMA (read directly off your real pipeline's output —
not assumed)
--------------------------------------------------------------------------
telecom_churn.churn_classified (Iceberg table, one row per customer):
    customer_id, msisdn, region, plan_type, age, gender, activation_date,
    monthly_arpu_sdg, is_churned, last_event_date, days_silent,
    unpaid_months_recent, arpu_drop_pct, tickets_last_30d_before_silence,
    root_cause ('active'|'network'|'mixed'|'behavioral'), classification_reason

    NOTE: plan_type is just 'prepaid'/'postpaid' — there is no plan-level
    price tier and no 'segment' column anywhere in your generator. ARPU is
    a per-customer figure (monthly_arpu_sdg), not a plan attribute. The
    dimension design below reflects that — no invented columns.

    NOTE: there is no snapshot_date. Phase 4 computes ONE classification per
    customer, anchored at the REFERENCE_DATE constant in process_layer.py
    (currently "2024-12-31"). That is why the fact table below is named
    fact_customer_churn_snapshot, not "_daily" — calling it daily would
    overstate a granularity the pipeline doesn't actually produce yet.
    Re-running Phase 4 with a different REFERENCE_DATE and appending to the
    Iceberg table is what would eventually make this a real time series.

data/raw/network_tickets (Parquet, Phase 2 output, ticket grain):
    ticket_id, customer_id, msisdn, ticket_date, issue_type, region, status,
    resolution_days (nullable — only set for resolved/closed tickets)

    Phase 4's classified_df only carries an AGGREGATE ticket count
    (tickets_last_30d_before_silence). Ticket-grain detail for network
    root-cause drill-down has to come from the raw layer directly — this is
    a deliberate, explainable modeling choice, not an oversight: the two
    fact tables in this warehouse read from two different layers of the
    same pipeline because they answer different questions.

WHY A STAR SCHEMA
-------------------
1. Storage efficiency: a customer's region/plan/segment-equivalent data is
   stored ONCE in dim_customer instead of repeated across every ticket.
2. Query performance: Presto (DuckDB locally) prunes partitions and pushes
   filters into dimension joins instead of scanning wide denormalized rows.
3. Conformed dimensions: dim_customer, dim_date, dim_region are reused by
   BOTH fact tables — one definition of "what is a customer" warehouse-wide.
4. SCD Type 2 on dim_customer: region and plan_type can change between
   pipeline runs. Overwriting the dimension would let a later plan change
   silently rewrite the attribution of an earlier classification. SCD2
   preserves history so facts join to the dimension state that was TRUE
   at the time the fact happened.

PRODUCTION MAPPING
------------
  This script (Spark)              -> Spark ETL jobs orchestrated by Airflow
  pyiceberg read of churn_classified -> the same Iceberg table Presto/Superset
                                         would query in the operator's real warehouse
  Parquet fact/dim output, partitioned -> Iceberg tables on HDFS, partitioned
  Hash-based surrogate keys         -> Same pattern used for Iceberg MERGE INTO

WINDOWS DEVELOPMENT ENVIRONMENT — WORKAROUNDS IN THIS FILE
-----------------------------------------------------------
Three patterns in this file exist specifically because PySpark on Windows
(local mode, OneDrive-synced project directory) has constraints that do not
apply in a Linux/HDFS production environment:

1. spark.range(0).select(...) for empty DataFrames (build_dim_customer_scd2)
   Production equivalent: spark.createDataFrame([], schema=schema)
   Why here: on Windows, even an empty Python-backed DataFrame (created via
   createDataFrame) causes Python worker processes to be spawned during
   shuffle tasks. Those workers crash with EOFException on this machine.
   spark.range(0) is a pure-JVM operation that produces an empty relation
   without touching the Python worker infrastructure.

2. pandas → Parquet → spark.read.parquet() bridges for small DataFrames
   (load_classified_customers, build_dim_root_cause)
   Production equivalent: spark.createDataFrame(pdf) or direct Iceberg read
   via the Spark-Iceberg connector (not pyiceberg's Python API).
   Why here: pyiceberg returns arrow-backed pandas dtypes that PySpark's
   Python worker serializer cannot pickle. The Parquet round-trip strips the
   arrow backing and gives Spark a clean JVM Parquet source with no Python
   worker dependency.

3. All Spark writes go to tempfile.gettempdir() (TEMP_WAREHOUSE_DIR), then
   copied to the permanent warehouse with shutil.copytree at the end of main()
   Production equivalent: df.write.parquet(hdfs_path) directly.
   Why here: writing a Parquet file into an OneDrive-synced directory and
   reading it back immediately caused Hadoop's local FileSystem to get a sync
   lock (UNABLE_TO_INFER_SCHEMA). The system temp directory is not synced by
   OneDrive. In production on HDFS or S3, this pattern does not exist.
================================================================================
"""

import os
import json
import sys
import shutil
import tempfile
from datetime import date, timedelta, datetime

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, DateType, LongType
)

# ==============================================================================
# CONFIG
# ==============================================================================

SPARK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SPARK_DIR)

# Re-use the EXACT catalog connector from iceberg_writer.py instead of
# duplicating the SqlCatalog config here. One source of truth for how to
# reach the Iceberg table — if the catalog config ever changes, it changes
# in one place, not two files that can silently drift apart.
# iceberg_writer.py lives in src/04_classify_spark/, not src/05_star_schema/
PHASE4_DIR = os.path.join(BASE_DIR, "04_classify_spark")
sys.path.insert(0, PHASE4_DIR)
from iceberg_writer import get_iceberg_catalog, NAMESPACE, TABLE_NAME  # noqa: E402

# Raw-layer input (Phase 2 output) — only network_tickets is read from here.
# Customer attributes come from the Iceberg table; reading customers twice
# from two different layers would risk the two copies disagreeing.
RAW_NETWORK_TICKETS_PATH = os.path.join(BASE_DIR, "data", "raw", "network_tickets")

# Permanent warehouse location (recorded in manifests, read by Phase 6 DuckDB).
# DuckDB reads Parquet with its own file reader and handles spaces in paths fine.
REAL_WAREHOUSE_DIR = os.path.join(BASE_DIR, "data", "processed", "warehouse")

# Spark writes and reads must go through a temp directory outside OneDrive.
# Writing a fresh Parquet file into an OneDrive-synced directory and immediately
# reading it back causes Spark's Hadoop reader to get a sync lock, producing
# the same UNABLE_TO_INFER_SCHEMA error that broke the classified_df bridge.
# After all Spark operations finish, main() copies temp → REAL_WAREHOUSE_DIR
# so Phase 6 finds the files at the expected permanent path.
TEMP_WAREHOUSE_DIR = os.path.join(tempfile.gettempdir(), "telecom_p5_warehouse")

WAREHOUSE_DIR = TEMP_WAREHOUSE_DIR  # alias used in write_phase5_manifest

DIM_DATE_PATH         = os.path.join(TEMP_WAREHOUSE_DIR, "dim_date")
DIM_CUSTOMER_PATH     = os.path.join(TEMP_WAREHOUSE_DIR, "dim_customer")
DIM_PLAN_PATH         = os.path.join(TEMP_WAREHOUSE_DIR, "dim_plan")
DIM_REGION_PATH       = os.path.join(TEMP_WAREHOUSE_DIR, "dim_region")
DIM_ROOT_CAUSE_PATH   = os.path.join(TEMP_WAREHOUSE_DIR, "dim_root_cause")
FACT_CHURN_SNAPSHOT_PATH  = os.path.join(TEMP_WAREHOUSE_DIR, "fact_customer_churn_snapshot")
FACT_NETWORK_TICKETS_PATH = os.path.join(TEMP_WAREHOUSE_DIR, "fact_network_tickets")

CATALOG_DIR = os.path.join(BASE_DIR, "data", "catalog")

# NOTE: REFERENCE_DATE used to be a hardcoded constant here, manually kept
# in sync with process_layer.py. That gap is now closed — process_layer.py
# stamps a real reference_date column onto every row, so this script reads
# the business date directly from the data instead of assuming it. See
# build_fact_customer_churn_snapshot() below.

# Date dimension range — covers the simulation window (2024) with margin
# for re-runs at a later REFERENCE_DATE.
DATE_DIM_START = date(2023, 1, 1)
DATE_DIM_END = date(2026, 12, 31)

# Anchor date for the FIRST version of every customer on the initial
# dim_customer build. Must be earlier than the earliest fact row expected —
# tickets can land as early as SIM_START (2024-01-01) in Phase 1's generator.
BACKFILL_EPOCH = date(2000, 1, 1)

# The load date for THIS Phase 5 run (when dim_customer was last refreshed) —
# distinct from REFERENCE_DATE (what business date the churn facts describe).
# Worth keeping these two concepts separate in the interview: load_date is a
# pipeline/technical timestamp, REFERENCE_DATE is a business-as-of date.
LOAD_DATE = date.today()

# root_cause vocabulary — confirmed directly from root_cause_engine.py's
# classify_root_cause(): exactly these four values, 'active' for non-churned,
# no 'none' category.
ROOT_CAUSE_CATEGORIES = [
    ("active", "Not churned — usage within normal range"),
    ("network", "Churned with 3+ network tickets in the 30 days before going silent"),
    ("mixed", "Churned with both network tickets AND 2+ unpaid months recently"),
    ("behavioral", "Churned due to silence and/or non-payment, no dominant network signal"),
]


# ==============================================================================
# SPARK SESSION
# ==============================================================================

def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Phase5_StarSchemaBuilder")
        .config("spark.sql.shuffle.partitions", "8")
        .master("local[*]")
        .getOrCreate()
    )


# ==============================================================================
# SOURCE READERS
# ==============================================================================

def load_classified_customers(spark: SparkSession) -> DataFrame:
    """
    Read telecom_churn.churn_classified back through the REAL Iceberg API —
    not a Parquet-folder shortcut. This is the same proof point
    iceberg_writer.verify_iceberg_read() makes: the table is genuinely
    queryable through pyiceberg, with real snapshot history behind it.

    Bridge: pyiceberg -> pandas -> Parquet file -> Spark.
    Writing to a Parquet file first and reading it back via Spark's JVM
    Parquet reader avoids Python-worker serialization issues that arise on
    Windows when PySpark tries to pickle pyiceberg's arrow-backed dtypes
    during createDataFrame(). The data round-trips cleanly through Parquet.
    """
    catalog = get_iceberg_catalog()
    full_table_name = f"{NAMESPACE}.{TABLE_NAME}"
    iceberg_table = catalog.load_table(full_table_name)
    pdf = iceberg_table.scan().to_pandas()

    # Convert arrow-backed dtypes to plain numpy/object before writing Parquet
    for col in pdf.columns:
        if hasattr(pdf[col].dtype, "pyarrow_dtype"):
            pdf[col] = pdf[col].astype(object)

    # Write to Windows temp dir (no spaces in path — Spark can't handle
    # paths with spaces on Windows when using spark.read.parquet()).
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "p5_classified_bridge.parquet")
    pdf.to_parquet(temp_path, index=False)
    return spark.read.parquet(temp_path)


def load_raw_network_tickets(spark: SparkSession) -> DataFrame:
    """
    Ticket-grain detail comes from the RAW layer (Phase 2), not Phase 4 —
    classified_df only has the pre-aggregated
    tickets_last_30d_before_silence count. Reading raw/ here mirrors how a
    real Presto analyst drilling into "show me every ticket for this
    network-churned customer" would have to join back to raw CDR/ticket
    detail rather than stopping at an aggregated feature table.
    """
    if not os.path.exists(RAW_NETWORK_TICKETS_PATH):
        raise FileNotFoundError(
            f"{RAW_NETWORK_TICKETS_PATH} not found. Run Phase 2 "
            f"(etl/run_phase2.py) before Phase 5."
        )
    return spark.read.parquet(RAW_NETWORK_TICKETS_PATH)


# ==============================================================================
# DIMENSION BUILDERS
# ==============================================================================

def build_dim_date(spark: SparkSession) -> DataFrame:
    """
    Standard date dimension, one row per calendar day.

    Built via Spark SQL sequence() instead of spark.createDataFrame(python_list)
    to avoid Python worker serialization crashes on Windows: a DataFrame built
    from a Python list of date objects requires Spark to call Python workers
    when writing to Parquet. Using sequence() keeps the entire plan in the JVM.
    """
    start_str = DATE_DIM_START.strftime("%Y-%m-%d")
    end_str   = DATE_DIM_END.strftime("%Y-%m-%d")
    return spark.sql(f"""
        SELECT
            CAST(date_format(d, 'yyyyMMdd') AS INT)         AS date_key,
            d                                                AS full_date,
            year(d)                                          AS year,
            quarter(d)                                       AS quarter,
            month(d)                                         AS month,
            date_format(d, 'MMMM')                           AS month_name,
            day(d)                                           AS day,
            dayofweek(d)                                     AS day_of_week,
            date_format(d, 'EEEE')                           AS day_name,
            (dayofweek(d) IN (1, 7))                         AS is_weekend,
            weekofyear(d)                                    AS week_of_year
        FROM (
            SELECT explode(
                sequence(date '{start_str}', date '{end_str}', interval 1 day)
            ) AS d
        )
    """)


def build_dim_region(classified_df: DataFrame, tickets_df: DataFrame) -> DataFrame:
    """
    Conformed region dimension from the union of regions seen in both
    source feeds. Confirmed values (REGIONS in run_phase1.py): Khartoum,
    Omdurman, Kassala, Port Sudan, Wad Madani, El Obeid, Atbara, Nyala.

    Hash-based surrogate key (xxhash64), not monotonically_increasing_id():
    deterministic across reruns, which matters once Phase 4 starts
    appending new Iceberg snapshots on a schedule and Phase 5 needs to
    incrementally re-key against an existing dimension without shifting
    every previously-loaded fact's foreign key.
    """
    regions = (
        classified_df.select("region")
        .union(tickets_df.select("region"))
        .distinct()
        .withColumnRenamed("region", "region_name")
    )
    return regions.withColumn(
        "region_key", F.abs(F.xxhash64("region_name")).cast(LongType())
    ).select("region_key", "region_name")


def build_dim_plan(classified_df: DataFrame) -> DataFrame:
    """
    plan_type has exactly two values in the real data: 'prepaid', 'postpaid'.
    No price tier or plan category exists upstream — this dimension is kept
    deliberately small and honest rather than inventing a richer hierarchy
    the source data doesn't support. It still earns its place as a
    conformed dimension because plan_type is SCD2-tracked on dim_customer
    and Phase 6 will want to slice tickets/billing by it independently.
    """
    plans = classified_df.select("plan_type").distinct()
    return plans.withColumn(
        "plan_key", F.abs(F.xxhash64("plan_type")).cast(LongType())
    ).select("plan_key", "plan_type")


def build_dim_root_cause(spark: SparkSession) -> DataFrame:
    """
    Static, business-defined dimension — the four-value vocabulary
    classify_root_cause() actually emits, confirmed against root_cause_engine.py.

    Built via pandas → Parquet → Spark like load_classified_customers, because
    even string-typed spark.createDataFrame(python_list) creates a Python-RDD-
    backed DataFrame whose write tasks require Python workers (which crash on
    this Windows environment).
    """
    import tempfile
    import pandas as pd

    pdf = pd.DataFrame(ROOT_CAUSE_CATEGORIES, columns=["root_cause_category", "description"])
    temp_path = os.path.join(tempfile.gettempdir(), "p5_root_cause_dim.parquet")
    pdf.to_parquet(temp_path, index=False)
    base = spark.read.parquet(temp_path)
    return base.withColumn(
        "root_cause_key", F.abs(F.xxhash64("root_cause_category")).cast(LongType())
    ).select("root_cause_key", "root_cause_category", "description")


_DIM_CUSTOMER_COLS = [
    "customer_key", "customer_id", "msisdn", "region", "plan_type",
    "age", "gender", "activation_date", "monthly_arpu_sdg",
    "effective_date", "end_date", "is_current"
]


def _dim_customer_schema() -> StructType:
    return StructType([
        StructField("customer_key", LongType()),
        StructField("customer_id", StringType()),
        StructField("msisdn", StringType()),
        StructField("region", StringType()),
        StructField("plan_type", StringType()),
        StructField("age", IntegerType()),
        StructField("gender", StringType()),
        StructField("activation_date", DateType()),
        StructField("monthly_arpu_sdg", DoubleType()),
        StructField("effective_date", DateType()),
        StructField("end_date", DateType()),
        StructField("is_current", BooleanType()),
    ])


def build_dim_customer_scd2(
    spark: SparkSession,
    classified_df: DataFrame,
    load_date: date,
) -> DataFrame:
    """
    SCD Type 2 customer dimension. Tracks changes in region and plan_type
    only — the two attributes that are genuinely plausible to change
    between pipeline runs. age, gender, msisdn, and monthly_arpu_sdg are
    carried on every version as Type-1 (always-current) attributes: age
    and gender don't change in this dataset's generation logic, msisdn is
    tied to the SIM and shouldn't change, and monthly_arpu_sdg is a
    persona-driven generation artifact rather than a real monthly-varying
    figure here — tracking it as SCD2 would add complexity the source data
    doesn't actually exercise. (A real production version, with genuinely
    monthly-varying ARPU, would likely move ARPU to a fact measure instead
    of a dimension attribute entirely — flagged here as a deliberate
    simplification, not an oversight.)

    LOGIC (unchanged from the original design, re-validated against the
    real schema): new customer -> insert; tracked attribute changed ->
    expire old version (end_date = load_date - 1) and insert a new one;
    unchanged -> keep as is.

    SURROGATE KEY: hash(customer_id + effective_date), unique per VERSION
    of a customer.
    """
    incoming = (
        classified_df
        .select("customer_id", "msisdn", "region", "plan_type",
                "age", "gender", "activation_date", "monthly_arpu_sdg")
        .withColumn("activation_date", F.to_date("activation_date", "yyyy-MM-dd"))
        .dropDuplicates(["customer_id"])
    )

    tracked_cols = ["region", "plan_type"]
    carried_cols = ["msisdn", "age", "gender", "activation_date", "monthly_arpu_sdg"]

    # Check the REAL warehouse (not the temp write path) for an existing
    # dim_customer from a previous run. On re-runs, copy it into temp so
    # Spark can read it without hitting OneDrive sync locks.
    _real_dc = os.path.join(REAL_WAREHOUSE_DIR, "dim_customer")
    is_initial_load = not os.path.exists(_real_dc)
    if not is_initial_load:
        _temp_dc = os.path.join(tempfile.gettempdir(), "p5_existing_dim_customer")
        if os.path.exists(_temp_dc):
            shutil.rmtree(_temp_dc)
        shutil.copytree(_real_dc, _temp_dc)
        existing = spark.read.parquet(_temp_dc)
        current = existing.filter(F.col("is_current") == True)  # noqa: E712
        history = existing.filter(F.col("is_current") == False)  # noqa: E712
    else:
        # spark.range(0) is a pure-JVM operation — unlike createDataFrame([], schema),
        # it does not create a Python-RDD-backed relation. Python workers are only
        # spawned when there is a Python source in the lineage; an empty Python RDD
        # still triggers worker creation during the shuffle in the outer join below,
        # causing the same crash as non-empty Python-backed DataFrames.
        _empty = spark.range(0).select(
            F.lit(None).cast(LongType()).alias("customer_key"),
            F.lit(None).cast(StringType()).alias("customer_id"),
            F.lit(None).cast(StringType()).alias("msisdn"),
            F.lit(None).cast(StringType()).alias("region"),
            F.lit(None).cast(StringType()).alias("plan_type"),
            F.lit(None).cast(IntegerType()).alias("age"),
            F.lit(None).cast(StringType()).alias("gender"),
            F.lit(None).cast(DateType()).alias("activation_date"),
            F.lit(None).cast(DoubleType()).alias("monthly_arpu_sdg"),
            F.lit(None).cast(DateType()).alias("effective_date"),
            F.lit(None).cast(DateType()).alias("end_date"),
            F.lit(None).cast(BooleanType()).alias("is_current"),
        )
        current = _empty
        history = _empty

    # See the matching note in the original design: on the very first run
    # there is no prior dimension state, so new rows must be anchored
    # BEFORE the earliest fact date (BACKFILL_EPOCH), not at today's
    # load_date — otherwise the point-in-time fact join below returns NULL
    # customer_keys for every historical row. Only genuinely new customers
    # appearing on a LATER incremental run get effective_date = load_date.
    new_row_effective_date = BACKFILL_EPOCH if is_initial_load else load_date

    joined = incoming.alias("inc").join(
        current.alias("cur"), on="customer_id", how="full_outer"
    )

    change_condition = F.lit(False)
    for c in tracked_cols:
        change_condition = change_condition | (F.col(f"inc.{c}") != F.col(f"cur.{c}"))

    is_new = F.col("cur.customer_id").isNull()
    is_changed = (~is_new) & change_condition
    is_unchanged = (~is_new) & (~change_condition)
    is_dropped = F.col("inc.customer_id").isNull()

    select_incoming_cols = [F.col("inc.customer_id").alias("customer_id")] + \
        [F.col(f"inc.{c}").alias(c) for c in tracked_cols + carried_cols]

    new_versions = (
        joined.filter(is_new | is_changed)
        .select(*select_incoming_cols)
        .withColumn("effective_date", F.lit(new_row_effective_date))
        .withColumn("end_date", F.lit(None).cast(DateType()))
        .withColumn("is_current", F.lit(True))
        .withColumn(
            "customer_key",
            F.abs(F.xxhash64("customer_id", "effective_date")).cast(LongType())
        )
        .select(_DIM_CUSTOMER_COLS)
    )

    select_cur_cols = [
        F.col("cur.customer_key").alias("customer_key"),
        F.col("cur.customer_id").alias("customer_id"),
    ] + [F.col(f"cur.{c}").alias(c) for c in carried_cols + tracked_cols]

    expired_versions = (
        joined.filter(is_changed)
        .select(
            *select_cur_cols,
            F.col("cur.effective_date").alias("effective_date"),
            F.date_sub(F.lit(load_date), 1).alias("end_date"),
            F.lit(False).alias("is_current"),
        )
        .select(_DIM_CUSTOMER_COLS)
    )

    unchanged_versions = (
        joined.filter(is_unchanged | is_dropped)
        .select(
            *select_cur_cols,
            F.col("cur.effective_date").alias("effective_date"),
            F.col("cur.end_date").alias("end_date"),
            F.col("cur.is_current").alias("is_current"),
        )
        .select(_DIM_CUSTOMER_COLS)
    )

    return new_versions.unionByName(expired_versions).unionByName(unchanged_versions).unionByName(history)


# ==============================================================================
# FACT BUILDERS
# ==============================================================================

def build_fact_customer_churn_snapshot(
    classified_df: DataFrame,
    dim_customer: DataFrame,
    dim_plan: DataFrame,
    dim_region: DataFrame,
    dim_root_cause: DataFrame,
) -> DataFrame:
    """
    Grain: one row per customer per reference_date.

    UPDATED: reference_date is now read directly from classified_df's own
    reference_date column (added in process_layer.py's lineage fix) instead
    of being passed in as a constant. This is the real fix for the gap
    flagged earlier — previously Phase 5 had to assume a date that matched
    whatever was hardcoded in process_layer.py, with no way to detect drift.
    Now it reads the date the data itself claims to represent, per row,
    which is also what makes this genuinely become a time series the
    moment Phase 4 runs more than once with different reference_dates: no
    schema or code change needed here, the join below already handles it
    row by row.

    POINT-IN-TIME DIMENSION LOOKUP: joins to the dim_customer VERSION whose
    [effective_date, end_date) window contains EACH ROW'S OWN
    reference_date, not a single global date — important once multiple
    reference_dates coexist in the table.
    """
    dc = dim_customer.alias("dc")
    cf = (
        classified_df
        .withColumn("ref_date_parsed", F.to_date("reference_date", "yyyy-MM-dd"))
        .alias("cf")
    )

    customer_joined = cf.join(
        dc,
        (F.col("cf.customer_id") == F.col("dc.customer_id")) &
        (F.col("cf.ref_date_parsed") >= F.col("dc.effective_date")) &
        (F.col("dc.end_date").isNull() | (F.col("cf.ref_date_parsed") < F.col("dc.end_date"))),
        how="left"
    )

    fact = (
        customer_joined
        .withColumn("date_key", F.date_format("cf.ref_date_parsed", "yyyyMMdd").cast(IntegerType()))
        .join(dim_plan, on="plan_type", how="left")
        .join(dim_region, customer_joined["cf.region"] == dim_region["region_name"], how="left")
        .join(dim_root_cause, customer_joined["cf.root_cause"] == dim_root_cause["root_cause_category"], how="left")
        .select(
            F.col("date_key"),
            F.col("dc.customer_key").alias("customer_key"),
            F.col("plan_key"),
            F.col("region_key"),
            F.col("root_cause_key"),
            F.col("cf.is_churned").alias("is_churned"),
            F.col("cf.days_silent").alias("days_silent"),
            F.col("cf.unpaid_months_recent").alias("unpaid_months_recent"),
            F.col("cf.arpu_drop_pct").alias("arpu_drop_pct"),
            F.col("cf.tickets_last_30d_before_silence").alias("tickets_last_30d_before_silence"),
            F.col("cf.classification_reason").alias("classification_reason"),
        )
        .withColumn(
            "fact_key",
            F.abs(F.xxhash64("date_key", "customer_key")).cast(LongType())
        )
        .withColumn("load_timestamp", F.current_timestamp())
    )

    cols = ["fact_key", "date_key", "customer_key", "plan_key", "region_key", "root_cause_key",
            "is_churned", "days_silent", "unpaid_months_recent", "arpu_drop_pct",
            "tickets_last_30d_before_silence", "classification_reason", "load_timestamp"]
    return fact.select(cols)


def build_fact_network_tickets(
    tickets: DataFrame,
    dim_customer: DataFrame,
    dim_region: DataFrame,
) -> DataFrame:
    """
    Grain: one row per network ticket, sourced from the RAW layer (see
    load_raw_network_tickets). Kept separate from the churn-snapshot fact
    because the grains genuinely differ — a customer can have zero, one,
    or several tickets against exactly one churn-snapshot row. ticket_date
    arrives as a 'yyyy-MM-dd' string (confirmed from the real raw Parquet,
    same serialization Phase 1 applies to every date column) and is parsed
    before the point-in-time dimension join.
    """
    t = tickets.withColumn("ticket_date", F.to_date("ticket_date", "yyyy-MM-dd")).alias("t")
    dc = dim_customer.alias("dc")

    joined = t.join(
        dc,
        (F.col("t.customer_id") == F.col("dc.customer_id")) &
        (F.col("t.ticket_date") >= F.col("dc.effective_date")) &
        (F.col("dc.end_date").isNull() | (F.col("t.ticket_date") < F.col("dc.end_date"))),
        how="left"
    ).join(
        dim_region, F.col("t.region") == dim_region["region_name"], how="left"
    )

    fact = (
        joined
        .withColumn("date_key", F.date_format("t.ticket_date", "yyyyMMdd").cast(IntegerType()))
        .select(
            F.col("date_key"),
            F.col("dc.customer_key").alias("customer_key"),
            F.col("region_key"),
            F.col("t.ticket_id").alias("ticket_id"),
            F.col("t.issue_type").alias("issue_type"),
            F.col("t.status").alias("status"),
            F.col("t.resolution_days").alias("resolution_days"),
        )
        .withColumn("fact_key", F.abs(F.xxhash64("ticket_id")).cast(LongType()))
        .withColumn("load_timestamp", F.current_timestamp())
    )

    cols = ["fact_key", "date_key", "customer_key", "region_key", "ticket_id",
            "issue_type", "status", "resolution_days", "load_timestamp"]
    return fact.select(cols)


# ==============================================================================
# DATA QUALITY CHECKS — run before any write
# ==============================================================================

class DataQualityError(Exception):
    pass


def check_no_orphan_keys(fact: DataFrame, fk_col: str, fact_name: str) -> None:
    orphan_count = fact.filter(F.col(fk_col).isNull()).count()
    if orphan_count > 0:
        raise DataQualityError(
            f"[{fact_name}] {orphan_count} rows have NULL {fk_col} — "
            f"dimension lookup failed for these rows. Check for join-key "
            f"mismatches before this table is loaded."
        )
    print(f"[DQ PASS] {fact_name}.{fk_col} — no orphan keys")


def check_grain_uniqueness(fact: DataFrame, key_cols: list, fact_name: str) -> None:
    total = fact.count()
    distinct = fact.select(*key_cols).distinct().count()
    if total != distinct:
        raise DataQualityError(
            f"[{fact_name}] grain violation: {total} rows but only "
            f"{distinct} distinct {key_cols} combinations. "
            f"{total - distinct} duplicate rows would inflate every "
            f"aggregate built on this table. (iceberg_writer.py now writes "
            f"idempotently per reference_date, so this shouldn't happen from "
            f"a Phase 4 rerun — if it does, check for a regression there.)"
        )
    print(f"[DQ PASS] {fact_name} grain is unique on {key_cols} ({total} rows)")


def check_row_count_not_lost(source: DataFrame, fact: DataFrame, fact_name: str) -> None:
    src_count, fact_count = source.count(), fact.count()
    if fact_count < src_count:
        raise DataQualityError(
            f"[{fact_name}] row loss: source had {src_count} rows, "
            f"fact has {fact_count}. A join likely dropped unmatched rows."
        )
    print(f"[DQ PASS] {fact_name} row count OK ({fact_count} >= source {src_count})")


# ==============================================================================
# WRITE HELPERS
# ==============================================================================

def write_dimension(df: DataFrame, path: str, name: str) -> None:
    df.write.mode("overwrite").parquet(path)
    print(f"[WRITE] {name} -> {path} ({df.count()} rows)")


def write_fact(df: DataFrame, path: str, name: str) -> None:
    df_partitioned = df.withColumn("load_year", (F.col("date_key") / 10000).cast(IntegerType())) \
                        .withColumn("load_month", ((F.col("date_key") / 100).cast(IntegerType()) % 100))
    df_partitioned.write.mode("overwrite").partitionBy("load_year", "load_month").parquet(path)
    print(f"[WRITE] {name} -> {path} ({df.count()} rows, partitioned by load_year/load_month)")


def write_phase5_manifest(
    classified_row_count: int,
    ticket_row_count: int,
    tables_written: list,
    dq_checks_passed: int,
) -> None:
    """
    Write a Phase 5 completion manifest to data/catalog/ — the same
    pattern every other phase in this pipeline uses. Phase 7 monitoring
    reads these manifests to confirm each phase ran successfully and
    to surface row counts and DQ results without re-running the pipeline.
    """
    os.makedirs(CATALOG_DIR, exist_ok=True)
    manifest = {
        "phase":              "5_star_schema",
        "run_timestamp":      datetime.now().isoformat(),
        "warehouse_path":     REAL_WAREHOUSE_DIR,
        "sources": {
            "classified_customers_rows": classified_row_count,
            "raw_network_ticket_rows":   ticket_row_count,
        },
        "tables_written":     tables_written,
        "dq_checks_passed":   dq_checks_passed,
        "status":             "SUCCESS",
        "lineage": {
            "produced_by":  "spark/phase5_star_schema_builder.py",
            "consumed_by":  "Phase 6 — Analytics Data Mart",
        },
    }
    path = os.path.join(
        CATALOG_DIR,
        f"phase5_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Phase 5 manifest saved -> {path}")


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def main():
    """
    Run order: Phase 1 -> Phase 2 -> process_layer.py -> root_cause_engine.py
    -> iceberg_writer.py must all have completed before this script runs.
    It reads telecom_churn.churn_classified through the real Iceberg catalog
    and data/raw/network_tickets through Spark's Parquet reader — no mock
    data, no fallback generator. If either source is missing, the read
    functions above raise a clear, specific error telling you which earlier
    phase to run.

    LINEAGE GAP CLOSED: classified_df now carries its own reference_date
    column (added in process_layer.py), and iceberg_writer.py writes
    idempotently per reference_date (overwrite, not blind append). This
    script no longer hardcodes or assumes the business date — it reads it
    from the data, per row, in build_fact_customer_churn_snapshot().
    """
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 80)
    print("PHASE 5 — STAR SCHEMA BUILDER")
    print("=" * 80)

    print(f"\n[READ] Loading {NAMESPACE}.{TABLE_NAME} via pyiceberg...")
    classified_df = load_classified_customers(spark)
    classified_df.cache()
    print(f"[READ] classified_df: {classified_df.count()} rows")

    print("[READ] Loading data/raw/network_tickets...")
    tickets = load_raw_network_tickets(spark)
    print(f"[READ] tickets: {tickets.count()} rows")

    # ---- Dimensions ----
    print("\n--- Building dimensions ---")
    dim_date = build_dim_date(spark)
    dim_region = build_dim_region(classified_df, tickets)
    dim_plan = build_dim_plan(classified_df)
    dim_root_cause = build_dim_root_cause(spark)
    dim_customer = build_dim_customer_scd2(spark, classified_df, LOAD_DATE)

    write_dimension(dim_date, DIM_DATE_PATH, "dim_date")
    write_dimension(dim_region, DIM_REGION_PATH, "dim_region")
    write_dimension(dim_plan, DIM_PLAN_PATH, "dim_plan")
    write_dimension(dim_root_cause, DIM_ROOT_CAUSE_PATH, "dim_root_cause")
    write_dimension(dim_customer, DIM_CUSTOMER_PATH, "dim_customer")

    # Re-read post-write so fact builders join against what's actually on
    # disk, not an in-memory lazy plan that could re-evaluate differently.
    dim_customer = spark.read.parquet(DIM_CUSTOMER_PATH)

    # ---- Facts ----
    print("\n--- Building facts ---")
    fact_churn_snapshot = build_fact_customer_churn_snapshot(
        classified_df, dim_customer, dim_plan, dim_region, dim_root_cause
    )
    fact_tickets = build_fact_network_tickets(tickets, dim_customer, dim_region)

    # ---- Data quality gate ----
    print("\n--- Running data quality checks ---")
    check_no_orphan_keys(fact_churn_snapshot, "customer_key", "fact_customer_churn_snapshot")
    check_no_orphan_keys(fact_churn_snapshot, "plan_key", "fact_customer_churn_snapshot")
    check_no_orphan_keys(fact_churn_snapshot, "region_key", "fact_customer_churn_snapshot")
    check_no_orphan_keys(fact_churn_snapshot, "root_cause_key", "fact_customer_churn_snapshot")
    check_grain_uniqueness(fact_churn_snapshot, ["date_key", "customer_key"], "fact_customer_churn_snapshot")
    check_row_count_not_lost(classified_df, fact_churn_snapshot, "fact_customer_churn_snapshot")

    check_no_orphan_keys(fact_tickets, "customer_key", "fact_network_tickets")
    check_grain_uniqueness(fact_tickets, ["ticket_id"], "fact_network_tickets")
    check_row_count_not_lost(tickets, fact_tickets, "fact_network_tickets")

    # ---- Write facts ----
    print("\n--- Writing facts ---")
    write_fact(fact_churn_snapshot, FACT_CHURN_SNAPSHOT_PATH, "fact_customer_churn_snapshot")
    write_fact(fact_tickets, FACT_NETWORK_TICKETS_PATH, "fact_network_tickets")

    # ---- Manifest ----
    write_phase5_manifest(
        classified_row_count=classified_df.count(),
        ticket_row_count=tickets.count(),
        tables_written=[
            "dim_date", "dim_region", "dim_plan", "dim_root_cause",
            "dim_customer", "fact_customer_churn_snapshot", "fact_network_tickets"
        ],
        dq_checks_passed=7,   # 6 fact checks + 1 ticket grain check
    )

    # Copy the warehouse from temp to the permanent location so Phase 6
    # (DuckDB) and Phase 7 (monitoring) find it at the expected path.
    print("\n--- Copying warehouse from temp to permanent location ---")
    if os.path.exists(REAL_WAREHOUSE_DIR):
        shutil.rmtree(REAL_WAREHOUSE_DIR)
    shutil.copytree(TEMP_WAREHOUSE_DIR, REAL_WAREHOUSE_DIR)
    print(f"[COPY] {TEMP_WAREHOUSE_DIR} -> {REAL_WAREHOUSE_DIR}")

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE — star schema written to:", REAL_WAREHOUSE_DIR)
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
