"""
================================================================================
PHASE 6 — ANALYTICS DATA MART
================================================================================
Project   : Telecom Customer Journey & Churn Root Cause Analysis Platform
Phase     : 6 of 8
Author    : <your name>
Maps to   : a Sudanese operator's Presto query engine, sitting on top of Iceberg/HDFS

WHAT A "MART" IS, AND WHY IT'S A SEPARATE PHASE FROM PHASE 5
----------------------------------------------------------------
Phase 5 built the WAREHOUSE: a star schema, normalized into fact and
dimension tables, optimized for storage efficiency and flexibility. Every
query against it needs a join — that's correct and intentional, but it's
not what a dashboard tool wants to run a hundred times a day.

The MART is different on purpose:
    - DENORMALIZED — joins are done ONCE, here, not on every dashboard
      refresh. mart_customer_360 below is one flat table a BI tool can
      filter and group directly, no join logic required on the consumer
      side.
    - PRE-AGGREGATED — mart_churn_by_region, mart_root_cause_summary, etc.
      answer specific business questions directly, instead of making
      Superset compute a GROUP BY over millions of rows on every page load.
    - BUSINESS-FACING NAMES — region_name, plan_type, root_cause_category,
      not region_key, plan_key, root_cause_key. A dashboard builder should
      never need to know what a surrogate key is.

This is the same warehouse-vs-mart split the operator's stack makes: Iceberg/HDFS
holds the normalized warehouse, Presto queries it, and the result a
dashboard actually points at is a curated, denormalized mart view —
exactly what this script builds, with DuckDB standing in for Presto
locally (same SQL engine role: query an external table format directly,
no data copy required to start querying).

SOURCE (read directly off Phase 5's parquet output, no copy):
    data/processed/warehouse/{dim_*, fact_*}

OUTPUT:
    data/processed/analytics_mart.duckdb  — queryable DB file (Superset can
        connect to this directly via a DuckDB SQLAlchemy connector)
    data/processed/mart/*.parquet         — same mart tables, also exported
        as Parquet for portability / inspection without DuckDB installed

PRODUCTION MAPPING
------------
  DuckDB (this script)        -> Presto, querying Iceberg tables on HDFS
  CREATE VIEW over Parquet     -> Presto querying Iceberg without a data copy
  CREATE TABLE (materialized)  -> Presto CTAS into a curated mart schema
  mart_customer_360            -> the kind of table Superset/Power BI connects to
================================================================================
"""

import os
import json
from datetime import datetime
from pathlib import Path

import duckdb

# ==============================================================================
# CONFIG
# ==============================================================================

ANALYTICS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ANALYTICS_DIR)

WAREHOUSE_DIR = os.path.join(BASE_DIR, "data", "processed", "warehouse")
MART_DB_PATH = os.path.join(BASE_DIR, "data", "processed", "analytics_mart.duckdb")
MART_EXPORT_DIR = os.path.join(BASE_DIR, "data", "processed", "mart")
CATALOG_DIR = os.path.join(BASE_DIR, "data", "catalog")

os.makedirs(MART_EXPORT_DIR, exist_ok=True)

WAREHOUSE_TABLES = [
    "dim_date", "dim_customer", "dim_plan", "dim_region", "dim_root_cause",
    "fact_customer_churn_snapshot", "fact_network_tickets",
]

# Tables built by this script, in dependency order (mart_customer_360 first
# since several of the others are aggregates built FROM it, not from raw
# warehouse tables directly — querying the mart instead of re-joining the
# warehouse every time is the whole point of having a mart).
MART_TABLES = [
    "mart_customer_360",
    "mart_churn_by_region",
    "mart_churn_by_plan",
    "mart_root_cause_summary",
    "mart_network_tickets_summary",
    "mart_network_root_cause_detail",
]


# ==============================================================================
# CONNECTION & WAREHOUSE REGISTRATION
# ==============================================================================

def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Single persistent DuckDB file, not an in-memory connection. A real
    analyst (or Superset) needs to be able to connect to this file and
    query it AFTER this script has finished running — an in-memory
    connection would vanish the moment the Python process exits.
    """
    return duckdb.connect(MART_DB_PATH)


def register_warehouse_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create a VIEW over each Phase 5 Parquet output — not a copy. This
    mirrors how Presto queries Iceberg tables on HDFS: the query engine
    reads the files where they live, on demand, rather than duplicating
    data into the query engine's own storage first.

    fact_* tables are Hive-partitioned (load_year/load_month, from Phase
    5's write_fact()) — hive_partitioning=true tells DuckDB to recover
    those as real columns from the directory structure, the same way
    Presto's Hive connector does partition pruning. dim_* tables have no
    partitioning, hive_partitioning=true is harmless there (no partition
    directories to find, so it's a no-op) — using it everywhere keeps this
    function uniform instead of branching per table.
    """
    for table_name in WAREHOUSE_TABLES:
        path = os.path.join(WAREHOUSE_DIR, table_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run Phase 5 "
                f"(spark/phase5_star_schema_builder.py) before Phase 6."
            )
        con.execute(f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_parquet(
                '{path}/**/*.parquet', hive_partitioning=true
            )
        """)
    print(f"[REGISTERED] {len(WAREHOUSE_TABLES)} warehouse views over {WAREHOUSE_DIR}")


# ==============================================================================
# MART BUILDERS
# ==============================================================================

def build_mart_customer_360(con: duckdb.DuckDBPyConnection) -> None:
    """
    The core mart table: one row per (customer, snapshot date), fully
    denormalized. Every dimension is joined in ONCE here — a dashboard
    filtering by region or plan_type just does a WHERE clause against
    this table, no join required on the consumer side.

    customer_key is kept (alongside the natural customer_id) specifically
    so mart_network_root_cause_detail below can join to fact_network_tickets
    on the surrogate key — joining on customer_id instead would risk a
    fan-out, since dim_customer can hold multiple SCD2 versions of the same
    customer_id (see Phase 5's SCD2 design). Keeping the surrogate key
    around in the mart avoids re-deriving that join correctness here.
    """
    con.execute("""
        CREATE OR REPLACE TABLE mart_customer_360 AS
        SELECT
            f.fact_key,
            f.date_key,
            d.full_date                            AS snapshot_date,
            c.customer_key,
            c.customer_id,
            c.msisdn,
            c.age,
            c.gender,
            c.activation_date,
            c.monthly_arpu_sdg,
            r.region_name,
            p.plan_type,
            f.is_churned,
            f.days_silent,
            f.unpaid_months_recent,
            f.arpu_drop_pct,
            f.tickets_last_30d_before_silence,
            rc.root_cause_category,
            f.classification_reason
        FROM fact_customer_churn_snapshot f
        JOIN dim_date        d  ON f.date_key       = d.date_key
        JOIN dim_customer    c  ON f.customer_key    = c.customer_key
        JOIN dim_region      r  ON f.region_key      = r.region_key
        JOIN dim_plan        p  ON f.plan_key        = p.plan_key
        JOIN dim_root_cause  rc ON f.root_cause_key  = rc.root_cause_key
    """)


def build_mart_churn_by_region(con: duckdb.DuckDBPyConnection) -> None:
    """Answers: 'where is churn worst, and is it tied to higher-value customers?'
    — a direct Superset bar-chart source, no GROUP BY needed at dashboard time."""
    con.execute("""
        CREATE OR REPLACE TABLE mart_churn_by_region AS
        SELECT
            region_name,
            COUNT(*)                                              AS total_customers,
            SUM(is_churned)                                       AS churned_customers,
            ROUND(100.0 * SUM(is_churned) / COUNT(*), 2)          AS churn_rate_pct,
            ROUND(AVG(monthly_arpu_sdg), 2)                       AS avg_monthly_arpu_sdg,
            ROUND(AVG(CASE WHEN is_churned = 1
                           THEN days_silent END), 1)              AS avg_days_silent_churned
        FROM mart_customer_360
        GROUP BY region_name
        ORDER BY churn_rate_pct DESC
    """)


def build_mart_churn_by_plan(con: duckdb.DuckDBPyConnection) -> None:
    """Answers: 'does plan type predict churn?' — Sudan is ~72% prepaid
    (see run_phase1.py's PLAN_WEIGHTS), so this is a real business question,
    not a generic cut for its own sake."""
    con.execute("""
        CREATE OR REPLACE TABLE mart_churn_by_plan AS
        SELECT
            plan_type,
            COUNT(*)                                     AS total_customers,
            SUM(is_churned)                               AS churned_customers,
            ROUND(100.0 * SUM(is_churned) / COUNT(*), 2)  AS churn_rate_pct,
            ROUND(AVG(monthly_arpu_sdg), 2)               AS avg_monthly_arpu_sdg
        FROM mart_customer_360
        GROUP BY plan_type
        ORDER BY churn_rate_pct DESC
    """)


def build_mart_root_cause_summary(con: duckdb.DuckDBPyConnection) -> None:
    """
    Answers: 'of the customers we lost, why did we lose them, and in what
    proportion?' — this is the single most important chart in the whole
    platform; it's the number a churn-save team budgets against (e.g. "81
    network churns this period -> is that worth a tower investment?").

    pct_of_churned_only is intentionally NULL for the 'active' row — dividing
    a non-churn count by the churned total would be a meaningless figure,
    and a silently wrong number is worse than a visible NULL.
    """
    con.execute("""
        CREATE OR REPLACE TABLE mart_root_cause_summary AS
        WITH totals AS (
            SELECT
                COUNT(*)                                                 AS total_customers,
                SUM(CASE WHEN root_cause_category != 'active'
                         THEN 1 ELSE 0 END)                              AS total_churned
            FROM mart_customer_360
        )
        SELECT
            m.root_cause_category,
            COUNT(*)                                                     AS customer_count,
            ROUND(100.0 * COUNT(*) / t.total_customers, 2)               AS pct_of_all_customers,
            CASE WHEN m.root_cause_category = 'active' THEN NULL
                 ELSE ROUND(100.0 * COUNT(*) / NULLIF(t.total_churned, 0), 2)
            END                                                          AS pct_of_churned_only
        FROM mart_customer_360 m
        CROSS JOIN totals t
        GROUP BY m.root_cause_category, t.total_customers, t.total_churned
        ORDER BY customer_count DESC
    """)


def build_mart_network_tickets_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Answers: 'which region and issue type combination is generating the
    most unresolved tickets?' — operational view for the NOC team, distinct
    from the churn-outcome views above; this one is about ticket handling
    quality regardless of whether the customer ultimately churned."""
    con.execute("""
        CREATE OR REPLACE TABLE mart_network_tickets_summary AS
        SELECT
            r.region_name,
            t.issue_type,
            COUNT(*)                                                     AS ticket_count,
            SUM(CASE WHEN t.status IN ('resolved','closed')
                     THEN 1 ELSE 0 END)                                  AS resolved_count,
            ROUND(100.0 * SUM(CASE WHEN t.status IN ('resolved','closed')
                              THEN 1 ELSE 0 END) / COUNT(*), 2)          AS resolution_rate_pct,
            ROUND(AVG(t.resolution_days), 1)                             AS avg_resolution_days
        FROM fact_network_tickets t
        JOIN dim_region r ON t.region_key = r.region_key
        GROUP BY r.region_name, t.issue_type
        ORDER BY ticket_count DESC
    """)


def build_mart_network_root_cause_detail(con: duckdb.DuckDBPyConnection) -> None:
    """
    Drill-down view: for every customer classified 'network' or 'mixed',
    show the ACTUAL tickets behind that classification. This is the
    explainability payoff promised in root_cause_engine.py's own docstring
    ("a churn-save team needs to know WHY to act") taken one level further —
    not just a label, but the literal ticket records a NOC analyst could
    open and investigate.

    Joins fact-to-fact on customer_key (the surrogate key), not customer_id,
    for the same fan-out-avoidance reason noted in mart_customer_360 above.
    """
    con.execute("""
        CREATE OR REPLACE TABLE mart_network_root_cause_detail AS
        SELECT
            c.customer_id,
            r.region_name,
            rc.root_cause_category,
            f.classification_reason,
            f.tickets_last_30d_before_silence,
            t.ticket_id,
            t.issue_type,
            t.status,
            t.resolution_days,
            td.full_date AS ticket_date
        FROM fact_customer_churn_snapshot f
        JOIN dim_root_cause rc ON f.root_cause_key = rc.root_cause_key
        JOIN dim_customer   c  ON f.customer_key    = c.customer_key
        JOIN dim_region     r  ON f.region_key      = r.region_key
        JOIN fact_network_tickets t ON t.customer_key = f.customer_key
        JOIN dim_date td ON t.date_key = td.date_key
        WHERE rc.root_cause_category IN ('network', 'mixed')
        ORDER BY c.customer_id, ticket_date
    """)


# ==============================================================================
# DATA QUALITY CHECKS — same gate-before-trust pattern as Phase 5
# ==============================================================================

class MartQualityError(Exception):
    pass


def check_no_fanout(con: duckdb.DuckDBPyConnection) -> None:
    """
    mart_customer_360 joins five dimensions onto one fact table. If any
    dimension join isn't truly 1:1 against the fact's foreign key (a
    dimension bug letting a duplicate surrogate key slip through), the row
    count would silently grow past the source fact table's count. This is
    the cheapest possible check for that: row counts must match exactly,
    not just be "close."
    """
    fact_count = con.execute("SELECT COUNT(*) FROM fact_customer_churn_snapshot").fetchone()[0]
    mart_count = con.execute("SELECT COUNT(*) FROM mart_customer_360").fetchone()[0]
    if mart_count != fact_count:
        raise MartQualityError(
            f"mart_customer_360 has {mart_count} rows but "
            f"fact_customer_churn_snapshot has {fact_count}. A dimension "
            f"join produced a fan-out — check for duplicate surrogate keys "
            f"in one of the five dimension tables."
        )
    print(f"[DQ PASS] mart_customer_360 row count matches fact exactly ({mart_count} rows)")


def check_aggregates_reconcile(con: duckdb.DuckDBPyConnection) -> None:
    """The sum of customers across every region in mart_churn_by_region must
    equal mart_customer_360's total — and the same for root cause categories.
    Catches an accidental WHERE filter or a dropped GROUP BY key silently
    losing rows in one of the aggregate marts."""
    total = con.execute("SELECT COUNT(*) FROM mart_customer_360").fetchone()[0]

    region_sum = con.execute("SELECT SUM(total_customers) FROM mart_churn_by_region").fetchone()[0]
    if region_sum != total:
        raise MartQualityError(
            f"mart_churn_by_region sums to {region_sum} customers, "
            f"mart_customer_360 has {total}. Rows are missing from the "
            f"region aggregate."
        )
    print(f"[DQ PASS] mart_churn_by_region reconciles to mart_customer_360 ({region_sum} customers)")

    rc_sum = con.execute("SELECT SUM(customer_count) FROM mart_root_cause_summary").fetchone()[0]
    if rc_sum != total:
        raise MartQualityError(
            f"mart_root_cause_summary sums to {rc_sum} customers, "
            f"mart_customer_360 has {total}. Rows are missing from the "
            f"root cause aggregate."
        )
    print(f"[DQ PASS] mart_root_cause_summary reconciles to mart_customer_360 ({rc_sum} customers)")


# ==============================================================================
# EXPORT — Parquet copies for portability outside DuckDB
# ==============================================================================

def export_mart_to_parquet(con: duckdb.DuckDBPyConnection) -> None:
    """
    Exports every mart table to Parquet in addition to leaving it queryable
    inside the .duckdb file. Two consumption paths, same data: Superset (or
    anyone) can connect to analytics_mart.duckdb directly with a DuckDB
    connector, OR read the Parquet files directly if DuckDB isn't available
    in the consuming environment — mirrors how the operator's mart layer would be
    queryable from Presto AND readable as plain Parquet by Spark/Hue/Zeppelin
    at the same time, since Iceberg tables are Parquet underneath.
    """
    for table in MART_TABLES:
        out_path = os.path.join(MART_EXPORT_DIR, f"{table}.parquet")
        con.execute(f"COPY {table} TO '{out_path}' (FORMAT PARQUET)")
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[EXPORT] {table} -> {out_path} ({row_count} rows)")


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_mart_summary(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "-" * 80)
    print("ROOT CAUSE SUMMARY (mart_root_cause_summary)")
    print("-" * 80)
    print(con.execute("SELECT * FROM mart_root_cause_summary").df().to_string(index=False))

    print("\n" + "-" * 80)
    print("CHURN BY REGION (mart_churn_by_region)")
    print("-" * 80)
    print(con.execute("SELECT * FROM mart_churn_by_region").df().to_string(index=False))

    print("\n" + "-" * 80)
    print("CHURN BY PLAN (mart_churn_by_plan)")
    print("-" * 80)
    print(con.execute("SELECT * FROM mart_churn_by_plan").df().to_string(index=False))


def write_phase6_manifest(con: duckdb.DuckDBPyConnection) -> None:
    """
    Write a Phase 6 completion manifest to data/catalog/ — same pattern
    as every other phase. Includes row counts per mart table so Phase 7
    monitoring can confirm completeness without re-querying the mart.
    """
    os.makedirs(CATALOG_DIR, exist_ok=True)
    table_counts = {}
    for table in MART_TABLES:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        table_counts[table] = count

    manifest = {
        "phase":          "6_analytics_mart",
        "run_timestamp":  datetime.now().isoformat(),
        "mart_db_path":   MART_DB_PATH,
        "mart_tables":    table_counts,
        "dq_checks_passed": 2,   # no_fanout + aggregates_reconcile
        "status":         "SUCCESS",
        "lineage": {
            "produced_by":  "analytics/build_mart.py",
            "consumed_by":  "Phase 8 — Dashboard (Superset / Power BI)",
        },
    }
    path = os.path.join(
        CATALOG_DIR,
        f"phase6_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Phase 6 manifest saved -> {path}")


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def main():
    print("=" * 80)
    print("PHASE 6 — ANALYTICS DATA MART")
    print("=" * 80)

    con = get_duckdb_connection()

    print("\n--- Registering warehouse views ---")
    register_warehouse_views(con)

    print("\n--- Building mart tables ---")
    build_mart_customer_360(con)
    print("[BUILD] mart_customer_360")
    build_mart_churn_by_region(con)
    print("[BUILD] mart_churn_by_region")
    build_mart_churn_by_plan(con)
    print("[BUILD] mart_churn_by_plan")
    build_mart_root_cause_summary(con)
    print("[BUILD] mart_root_cause_summary")
    build_mart_network_tickets_summary(con)
    print("[BUILD] mart_network_tickets_summary")
    build_mart_network_root_cause_detail(con)
    print("[BUILD] mart_network_root_cause_detail")

    print("\n--- Running data quality checks ---")
    check_no_fanout(con)
    check_aggregates_reconcile(con)

    print("\n--- Exporting mart to Parquet ---")
    export_mart_to_parquet(con)

    print_mart_summary(con)

    write_phase6_manifest(con)

    con.close()

    print("\n" + "=" * 80)
    print(f"PHASE 6 COMPLETE — mart DB: {MART_DB_PATH}")
    print(f"                    Parquet exports: {MART_EXPORT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
