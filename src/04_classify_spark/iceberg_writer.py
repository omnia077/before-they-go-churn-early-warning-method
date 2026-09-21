"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
spark/iceberg_writer.py — Iceberg Table Writer
================================================================================

PURPOSE:
    Persists the root-cause-classified churn table as a real Apache Iceberg
    table — not simulated, not just Parquet-with-extra-metadata. This uses
    pyiceberg with a SQLite-backed catalog, which produces actual Iceberg
    metadata.json files and real snapshot history on disk.

WHY ICEBERG (not just Parquet) FOR THIS LAYER:
    The job spec lists "Spark, HDFS, Parquet, Iceberg" explicitly. Iceberg's
    job is NOT primarily to make reads faster — its job is to make a table
    SAFE TO READ WHILE IT'S BEING WRITTEN, and to remember every version:

    1. Snapshot isolation — if Spark is writing new partitions while Presto
       (or Superset) is reading the table, the reader always sees one
       complete, consistent snapshot. Never a half-written, corrupted state.
    2. Time travel — every write creates a new snapshot. You can query the
       table as it looked yesterday, even after ten overwrites since.
    3. Schema evolution — columns can be added later without rewriting
       every existing Parquet file.

    Plain Parquet partitions (what Phase 2 writes to raw/) give none of
    this — an overwrite destroys the old version completely, and a reader
    mid-write can see a corrupted partial state.

WHY SQLite CATALOG (not Hive Metastore) FOR THIS PORTFOLIO:
    A real Hive Metastore needs a running service. SQLite is a single local
    file — zero setup, but still a REAL pyiceberg catalog backend (not a
    mock). The metadata.json files this produces are genuine Iceberg
    metadata, readable by any Iceberg-compatible engine. In the operator's actual
    environment, this SqlCatalog config would be swapped for a Hive or
    Glue catalog — the rest of this file's logic does not change.

PRODUCTION ARCHITECTURE MAPPING:
    Input:  classified_df from root_cause_engine.py
    Output: processed/iceberg_warehouse/ — real Iceberg table on disk
    In production: this table lives on HDFS, registered in Hive Metastore,
    queryable from Presto and Superset exactly as it is here.

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio
================================================================================
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.expressions import EqualTo

PROCESSED_PATH  = Path("../data/processed")
CATALOG_PATH    = Path("../data/catalog")
WAREHOUSE_PATH  = PROCESSED_PATH / "iceberg_warehouse"
CATALOG_DB_PATH = PROCESSED_PATH / "iceberg_catalog.db"

for _p in [PROCESSED_PATH, CATALOG_PATH, WAREHOUSE_PATH]:
    _p.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("iceberg_writer")

NAMESPACE  = "telecom_churn"
TABLE_NAME = "churn_classified"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Catalog Setup
# ════════════════════════════════════════════════════════════════════════════

def get_iceberg_catalog() -> SqlCatalog:
    """
    Build (or connect to) the local SQLite-backed Iceberg catalog.

    The catalog tracks WHICH tables exist and WHERE their current metadata
    lives — this is the same role Hive Metastore plays in the operator's production
    stack. SqlCatalog is a real pyiceberg catalog implementation; swapping
    this for HiveCatalog or GlueCatalog in production changes only this
    function, nothing downstream.
    """
    catalog = SqlCatalog(
        "telecom_local",
        **{
            "uri": f"sqlite:///{CATALOG_DB_PATH.resolve()}",
            "warehouse": f"file://{WAREHOUSE_PATH.resolve()}",
        },
    )

    # Namespaces are like databases/schemas — create once, reuse after
    existing_namespaces = [ns[0] for ns in catalog.list_namespaces()]
    if NAMESPACE not in existing_namespaces:
        catalog.create_namespace(NAMESPACE)
        log.info(f"Created Iceberg namespace: {NAMESPACE}")
    else:
        log.info(f"Using existing Iceberg namespace: {NAMESPACE}")

    return catalog


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Spark DataFrame to Iceberg Bridge
# ════════════════════════════════════════════════════════════════════════════

def spark_df_to_arrow(spark_df) -> pa.Table:
    """
    Convert a Spark DataFrame to a PyArrow Table — the format pyiceberg
    requires for writes.

    Why go through Pandas? PySpark's toPandas() collects the DataFrame to
    the driver, which is fine at this data volume (one row per customer,
    ~50K rows max) but would NOT scale to the full 2.4M-row usage_events
    table. This is acceptable here because root cause classification
    output is one row per CUSTOMER, not per event — already aggregated
    down from millions of rows to tens of thousands by process_layer.py.

    Datetime columns are cast to string before the Arrow conversion —
    pyiceberg's schema inference is stricter about timestamp types than
    plain Parquet writes, and string dates avoid timezone ambiguity.
    """
    pdf = spark_df.toPandas()

    for col in pdf.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        pdf[col] = pdf[col].astype(str)

    return pa.Table.from_pandas(pdf, preserve_index=False)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Write to Iceberg
# ════════════════════════════════════════════════════════════════════════════

def write_to_iceberg(classified_spark_df, catalog: SqlCatalog) -> dict:
    """
    Write the classified churn DataFrame as an Iceberg table.

    First write: creates the table and writes snapshot #1.
    Subsequent writes for a NEW reference_date: appends a new snapshot —
    old data is NOT lost, it remains queryable via time travel
    (table.history()).
    Subsequent writes for the SAME reference_date (a rerun or an Airflow
    retry): OVERWRITES just that date's rows, leaving every other date's
    history untouched.

    BUG FIX (idempotency): the original version always called
    iceberg_table.append(arrow_table) unconditionally. That meant rerunning
    this script — including an Airflow task retry, the most common reason
    a Spark job runs twice — silently duplicated every row in the table.
    Two retries of the same day's classification produced 10,000 rows for
    5,000 customers, not a refreshed 5,000. The DQ checks in Phase 5 would
    have caught this downstream (a grain-uniqueness failure on
    (date_key, customer_key)), but the right place to prevent it is here,
    at the source, not several phases later.

    The fix uses pyiceberg's overwrite() with a partial filter on
    reference_date (added to classified_df in process_layer.py) instead of
    append(). This is exactly the MERGE-style pattern this file's own
    docstring promises Iceberg enables — it just wasn't actually wired up
    before. A brand-new table still uses append() for its first write,
    since there's no prior data for that reference_date to conflict with.

    Returns:
        dict with table identifier, row count, and snapshot info
    """
    arrow_table = spark_df_to_arrow(classified_spark_df)
    full_table_name = f"{NAMESPACE}.{TABLE_NAME}"

    if "reference_date" not in arrow_table.column_names:
        raise ValueError(
            "classified_spark_df has no 'reference_date' column. This write "
            "function requires it to make writes idempotent per run — make "
            "sure process_layer.py's build_feature_table() has the "
            "reference_date column added (see its lineage bug-fix note)."
        )

    ref_dates = arrow_table.column("reference_date").unique().to_pylist()
    if len(ref_dates) != 1:
        raise ValueError(
            f"Expected exactly one reference_date in a single write, found "
            f"{ref_dates}. This write function assumes one classification "
            f"run = one reference_date — mixing dates in one write would "
            f"make the overwrite filter below ambiguous."
        )
    reference_date = ref_dates[0]

    if catalog.table_exists(full_table_name):
        iceberg_table = catalog.load_table(full_table_name)
        log.info(
            f"Table exists — overwriting reference_date={reference_date} "
            f"in {full_table_name} (idempotent: safe to rerun)"
        )
        iceberg_table.overwrite(
            arrow_table,
            overwrite_filter=EqualTo("reference_date", reference_date)
        )
        write_mode = "overwrite_partition"
    else:
        iceberg_table = catalog.create_table(full_table_name, schema=arrow_table.schema)
        log.info(f"Created new Iceberg table: {full_table_name}")
        iceberg_table.append(arrow_table)
        write_mode = "append_initial"

    snapshots = list(iceberg_table.history())
    current_snapshot = snapshots[-1] if snapshots else None

    result = {
        "table_name":       full_table_name,
        "reference_date":   str(reference_date),
        "write_mode":       write_mode,
        "rows_written":     arrow_table.num_rows,
        "total_snapshots":  len(snapshots),
        "current_snapshot_id": str(current_snapshot.snapshot_id) if current_snapshot else None,
        "warehouse_path":   str(WAREHOUSE_PATH.resolve()),
        "catalog_db_path":  str(CATALOG_DB_PATH.resolve())
    }

    log.info(
        f"✅ Wrote {result['rows_written']:,} rows | mode={write_mode} | "
        f"Snapshot {result['current_snapshot_id']} | "
        f"Total snapshots: {result['total_snapshots']}"
    )

    return result


def verify_iceberg_read(catalog: SqlCatalog) -> dict:
    """
    Read the table back via pyiceberg to prove the write is real and queryable.

    This is the proof point for the interview: not "I wrote files that look
    like Iceberg" but "I can read this table back through the Iceberg API,
    and here's the snapshot history to prove time travel works."
    """
    full_table_name = f"{NAMESPACE}.{TABLE_NAME}"
    iceberg_table = catalog.load_table(full_table_name)

    readback = iceberg_table.scan().to_arrow()

    return {
        "rows_read_back":   readback.num_rows,
        "columns":          readback.schema.names,
        "snapshot_count":   len(list(iceberg_table.history()))
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Manifest
# ════════════════════════════════════════════════════════════════════════════

def build_iceberg_manifest(write_result: dict, verify_result: dict) -> dict:
    """
    Build the final Phase 4 manifest — the last link in the lineage chain
    for this phase. consumed_by points to Phase 5 (Star Schema Builder),
    continuing the pattern established in every prior phase's manifest.
    """
    manifest = {
        "layer":             "processed",
        "stage":              "iceberg_persistence",
        "write_timestamp":    datetime.now().isoformat(),
        "table":              write_result["table_name"],
        "reference_date":     write_result["reference_date"],
        "write_mode":         write_result["write_mode"],
        "rows_written":       write_result["rows_written"],
        "total_snapshots":    write_result["total_snapshots"],
        "current_snapshot_id": write_result["current_snapshot_id"],
        "warehouse_path":     write_result["warehouse_path"],
        "read_verification": {
            "rows_read_back": verify_result["rows_read_back"],
            "matches_write":  verify_result["rows_read_back"] == write_result["rows_written"]
            if write_result["total_snapshots"] == 1 else None,  # N/A after multiple appends
            "columns":        verify_result["columns"]
        },
        "iceberg_features_demonstrated": [
            "ACID snapshot isolation (SqlCatalog-backed writes)",
            "Time travel (snapshot history queryable via table.history())",
            "Real metadata.json files on disk (not simulated)",
            "Idempotent partial overwrite by reference_date (safe reruns/retries)"
        ],

        # Final lineage link in the Phase 4 chain
        "lineage": {
            "source":      "spark/root_cause_engine.py (classified_df)",
            "produced_by": "spark/iceberg_writer.py",
            "consumed_by": "Phase 5 — Star Schema Builder"
        }
    }

    manifest_path = CATALOG_PATH / f"iceberg_write_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Iceberg write manifest saved -> {manifest_path}")
    return manifest


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

def run_iceberg_write(classified_spark_df) -> dict:
    """
    Full Iceberg persistence pipeline: catalog setup, write, verify, manifest.

    Args:
        classified_spark_df: output of root_cause_engine.run_root_cause_classification()

    Returns:
        manifest dict summarizing the write
    """
    print("=" * 65)
    print("  PHASE 4 — ICEBERG TABLE WRITE")
    print("=" * 65)

    print("\n🧊 Setting up Iceberg catalog (SQLite-backed)...")
    catalog = get_iceberg_catalog()

    print("\n💾 Writing classified data to Iceberg table...")
    write_result = write_to_iceberg(classified_spark_df, catalog)

    print("\n🔍 Verifying read-back via Iceberg API...")
    verify_result = verify_iceberg_read(catalog)

    print("\n📋 Building manifest...")
    manifest = build_iceberg_manifest(write_result, verify_result)

    print(f"\n✅ Iceberg write complete")
    print(f"   Table:           {manifest['table']}")
    print(f"   Reference date:  {write_result['reference_date']}")
    print(f"   Write mode:      {write_result['write_mode']}")
    print(f"   Rows written:    {manifest['rows_written']:,}")
    print(f"   Total snapshots: {manifest['total_snapshots']}")
    print(f"   Snapshot ID:     {manifest['current_snapshot_id']}")
    print(f"   Warehouse:       {manifest['warehouse_path']}")
    print(f"   Lineage -> consumed_by: {manifest['lineage']['consumed_by']}")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from process_layer import run_feature_engineering
    from root_cause_engine import run_root_cause_classification

    spark, feature_df, _ = run_feature_engineering()
    classified_df, _ = run_root_cause_classification(feature_df)
    manifest = run_iceberg_write(classified_df)

    spark.stop()
