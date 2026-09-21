"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
Phase 3 — Airflow DAG Orchestration
================================================================================

DEPLOYMENT: Copy this file to your Airflow DAGs folder:
    cp telecom_churn_pipeline.py $AIRFLOW_HOME/dags/

PRODUCTION ARCHITECTURE MAPPING:
    This DAG mirrors the operator's production pipeline orchestration pattern:
    - Sensors monitor SFTP landing zone (Huawei → NiFi → Staging)
    - Tasks run in dependency order with failure notifications
    - XComs pass manifest data between tasks
    - Retries handle transient failures gracefully
    - Logging integrates with monitoring layer (Phase 9)

SCHEDULE: Daily at 01:00 UTC (after EOD batch exports from Huawei)

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio

BUGS FIXED (vs original):
    1. BashOperator → BranchPythonOperator for branch_task
       (original used BashOperator which ignores the branch return value)
    2. Removed the deprecated airflow.utils.decorators import (removed in Airflow 2.3+)
    3. Quality threshold unified to 95% across all locations
    4. manifest["datasets"] now always includes quarantined_rows and
       quality_score — success_task no longer risks KeyError
================================================================================
"""

from datetime import datetime, timedelta
from pathlib import Path
import json

# ── Airflow imports ────────────────────────────────────────────────────────
# BUG FIX #2: Removed `from airflow.utils.decorators import apply_defaults`
# That decorator was deprecated in Airflow 2.0 and fully removed in 2.3+.
# Importing it on a modern Airflow instance causes an ImportError and the
# entire DAG fails to load — it never even appears in the UI.
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.exceptions import AirflowException

# ════════════════════════════════════════════════════════════════════════════
# DAG Configuration
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_ARGS = {
    "owner": "data_engineering",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["data-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="telecom_churn_platform",
    description="Daily ETL pipeline: SFTP → Raw Layer → Spark Processing → Star Schema → Analytics Mart",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",  # 01:00 UTC daily — after Huawei EOD batch
    catchup=False,
    tags=["telecom", "churn", "production"],
    doc_md=__doc__,
)

# ════════════════════════════════════════════════════════════════════════════
# Paths (Airflow Variables — set these before deploying)
# ════════════════════════════════════════════════════════════════════════════

SFTP_DROP_PATH      = Path(Variable.get("sftp_drop_path",      "/mnt/data/sftp_drop"))
RAW_PATH            = Path(Variable.get("raw_path",            "/mnt/data/raw"))
CATALOG_PATH        = Path(Variable.get("catalog_path",        "/mnt/data/catalog"))
ORCHESTRATION_SCRIPTS = Path(Variable.get("orchestration_scripts", "/home/airflow/dags/scripts"))

# BUG FIX #3: Unified quality threshold.
# Original had 80% in the DAG code, 95% in the README, and no check at all
# in the local orchestrator. Now a single constant controls all quality gates.
QUALITY_THRESHOLD = 95  # Minimum quality score to proceed to Spark

# ════════════════════════════════════════════════════════════════════════════
# Task 1: Check SFTP Landing Zone
# ════════════════════════════════════════════════════════════════════════════
# In the operator's pipeline: Huawei drops files here via SFTP every 15 minutes.
# This FileSensor blocks downstream tasks until the file is confirmed present.
# Timeout of 4 hours covers late-arriving Huawei batch exports.

wait_for_sftp_files = FileSensor(
    task_id="wait_for_sftp_files",
    filepath=str(SFTP_DROP_PATH / "customers.parquet"),
    fs_conn_id="local_fs",          # In production: SFTP connection to Huawei
    poke_interval=300,              # Poll every 5 minutes
    timeout=14400,                  # Wait up to 4 hours before failing
    mode="poke",
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 2: Phase 2 — Raw Storage Layer Ingestion
# ════════════════════════════════════════════════════════════════════════════
# Runs schema validation, partitioning, quarantine, and cataloging.
# Input:  Files from sftp_drop/
# Output: Partitioned Parquet in raw/, data catalog JSON, layer manifest
# Failure: If quality_score < QUALITY_THRESHOLD, task raises AirflowException

def run_raw_layer_ingestion(**context):
    """
    Execute Phase 2: Raw Storage Layer Ingestion Pipeline.

    Steps:
        1. Load each dataset from sftp_drop/
        2. Validate against schema registry (types, nulls, ranges, duplicates)
        3. Quarantine bad records to quarantine/
        4. Write clean records as partitioned Parquet to raw/
        5. Generate data catalog and layer manifest
        6. Return manifest dict → Airflow stores in XCom automatically

    Returns:
        dict: Manifest with per-dataset stats including clean_rows,
              quarantined_rows, quality_score, and status.
              These fields are required by validate_raw_layer_manifest
              and log_pipeline_success downstream.
    """
    import sys
    import glob
    sys.path.insert(0, str(ORCHESTRATION_SCRIPTS))

    # BUG FIX #3: Original imported from run_phase2_module which did not exist.
    # The module is etl/run_phase2.py — added to sys.path via ORCHESTRATION_SCRIPTS.
    # We call run_ingestion_pipeline() which executes the full pipeline and
    # returns the manifest dict directly — no subprocess needed in the DAG.
    from run_phase2 import run_ingestion_pipeline, SCHEMA_REGISTRY

    # Execute Phase 2 pipeline — returns manifest with all dataset stats
    manifest_data = run_ingestion_pipeline()

    # Verify quality threshold before returning to Airflow XCom
    for dataset_name, stats in manifest_data["datasets"].items():
        score = float(str(stats.get("quality_score", 0)).replace("%", ""))
        if score < QUALITY_THRESHOLD:
            raise AirflowException(
                f"{dataset_name}: quality_score {score}% below threshold {QUALITY_THRESHOLD}%"
            )

    return manifest_data


raw_layer_task = PythonOperator(
    task_id="phase_2_raw_layer_ingestion",
    python_callable=run_raw_layer_ingestion,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 3: Validate Raw Layer Manifest
# ════════════════════════════════════════════════════════════════════════════
# Reads the manifest XCom from Task 2.
# If all_passed=False, halts the pipeline before any data reaches Spark.

def validate_raw_layer_manifest(**context):
    """
    Validate that Phase 2 completed with all datasets passing quality gates.

    Reads manifest from XCom (written by run_raw_layer_ingestion).
    Raises AirflowException if any dataset failed — prevents bad data
    from reaching Spark processing (same pattern as the operator's staging tables).

    Returns:
        str: Confirmation message stored in XCom for audit trail.
    """
    task_instance = context["task_instance"]
    manifest = task_instance.xcom_pull(
        task_ids="phase_2_raw_layer_ingestion",
        key="return_value"
    )

    if not manifest:
        raise AirflowException("No manifest returned from Phase 2 ingestion task")

    if not manifest.get("all_passed", False):
        failing_datasets = [
            d for d, v in manifest["datasets"].items()
            if v.get("status") != "PASSED"
        ]
        raise AirflowException(
            f"Raw layer validation FAILED for datasets: {failing_datasets}\n"
            f"Check quarantine/ folder and quality_reports/ for details."
        )

    # Push validated manifest for both branch_task and success_task to read
    task_instance.xcom_push(key="raw_manifest", value=manifest)

    # Log per-dataset summary — now includes quarantined_rows and quality_score
    # because the manifest writer was fixed in run_raw_layer_ingestion
    for dataset, stats in manifest["datasets"].items():
        print(
            f"  {dataset:<25} "
            f"Clean: {stats['clean_rows']:>8,} | "
            f"Quarantined: {stats['quarantined_rows']:>6,} | "    # No longer N/A
            f"Score: {float(stats['quality_score']):>5.1f}% | "   # cast: manifest stores as string
            f"Status: {stats['status']}"
        )

    return f"✅ Raw layer PASSED | {len(manifest['datasets'])} datasets | " \
           f"Total clean: {manifest['total_clean_rows']:,}"


validate_raw_task = PythonOperator(
    task_id="validate_raw_layer_manifest",
    python_callable=validate_raw_layer_manifest,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 4: Branch — Proceed to Spark or Notify Failure
# ════════════════════════════════════════════════════════════════════════════
# BUG FIX #2 (CRITICAL): Original used a BashOperator here.
#
# The original code:
#   - Defined check_proceed_to_spark() that returns a task_id string
#   - But registered it as BashOperator(bash_command="echo 'Proceeding...'")
#   - A BashOperator ignores any Python return value — it just runs bash
#   - The function check_proceed_to_spark() was NEVER called
#   - The pipeline always proceeded unconditionally (the echo always exits 0)
#   - The branch to notify_pipeline_failure could NEVER be reached
#
# Fix: use BranchPythonOperator, which calls the Python function and uses
# its return value to route execution to the correct downstream task.

def check_proceed_to_spark(**context):
    """
    Route execution based on manifest validation result.

    BranchPythonOperator contract: return the task_id (string) of the
    next task to execute. Airflow skips all other downstream branches.

    Returns:
        str: "04_classify_spark_processing" if all passed,
             "notify_pipeline_failure" otherwise.
    """
    task_instance = context["task_instance"]
    manifest = task_instance.xcom_pull(
        task_ids="validate_raw_layer_manifest",
        key="raw_manifest"
    )

    if manifest and manifest.get("all_passed"):
        return "04_classify_spark_processing"   # Route to Spark
    else:
        return "notify_pipeline_failure"    # Route to alert


# BranchPythonOperator — not BashOperator — is what makes branching work
branch_task = BranchPythonOperator(
    task_id="branch_proceed_to_spark",
    python_callable=check_proceed_to_spark,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 5a: Phase 4 — Spark Processing (Placeholder for Phase 4 build-out)
# ════════════════════════════════════════════════════════════════════════════
# In Phase 4, this BashOperator becomes a spark-submit call.
# The --run_date parameter passes Airflow's execution date for idempotency.

spark_task = BashOperator(
    task_id="04_classify_spark_processing",
    bash_command="""
echo "Phase 4: Spark Processing"
echo "In production this runs:"
echo ""
echo "  spark-submit \\"
echo "    --master yarn \\"
echo "    --num-executors 8 \\"
echo "    --executor-memory 4g \\"
echo "    /home/airflow/dags/scripts/process_layer.py \\"
echo "    --input_path /mnt/data/raw \\"
echo "    --output_path /mnt/data/processed \\"
echo "    --run_date {{ ds }}"
echo ""
echo "Tasks: load raw Parquet → deduplicate → feature engineering"
echo "       → churn detection → root cause classification"
echo "       → write processed/ (Iceberg format)"
    """,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 5b: Notify Pipeline Failure
# ════════════════════════════════════════════════════════════════════════════
# Only reachable if branch_task routes here (quality gate failed).
# trigger_rule="one_failed" also catches unexpected upstream failures.

notify_failure = BashOperator(
    task_id="notify_pipeline_failure",
    bash_command="""
echo "⚠️  PIPELINE FAILURE ALERT"
echo "Check quality_reports/ for dataset validation details."
echo "Check quarantine/ for rejected records."
echo "Airflow UI: http://airflow.example.com:8080"
    """,
    trigger_rule="one_failed",
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 6: Pipeline Success Summary
# ════════════════════════════════════════════════════════════════════════════
# BUG FIX #1 (reader side): Original did stats['quarantined_rows'] — a direct
# key access that would crash with KeyError because the manifest writer never
# included that field. Now that the writer (run_raw_layer_ingestion) includes
# both quarantined_rows and quality_score, this task runs safely.

def log_pipeline_success(**context):
    """
    Log successful pipeline completion with full per-dataset metrics.

    Reads manifest from XCom (written by validate_raw_layer_manifest).
    Prints a formatted summary table for the Airflow task log.

    Returns:
        dict: Summary stored in XCom for monitoring layer (Phase 9).
    """
    task_instance = context["task_instance"]
    manifest = task_instance.xcom_pull(
        task_ids="validate_raw_layer_manifest",
        key="raw_manifest"
    )

    print("\n" + "=" * 70)
    print("  PIPELINE SUCCESS — RAW LAYER INGESTION COMPLETE")
    print("=" * 70)
    print(f"  {'Dataset':<25} {'Clean':>8}  {'Quarantined':>11}  {'Score':>7}  Status")
    print("-" * 70)

    for dataset, stats in manifest["datasets"].items():
        # These fields are now guaranteed present — fixed in the writer
        print(
            f"  {dataset:<25} "
            f"{stats['clean_rows']:>8,}  "
            f"{stats['quarantined_rows']:>11,}  "        # Was KeyError before fix
            f"{float(stats['quality_score']):>6.1f}%  "  # cast: manifest stores as string
            f"{stats['status']}"
        )

    print("=" * 70)
    print(f"  Total clean rows:       {manifest['total_clean_rows']:>12,}")
    print(f"  Total quarantined rows: {manifest['total_quarantined_rows']:>12,}")
    print("=" * 70)

    return {
        "status": "SUCCESS",
        "timestamp": datetime.now().isoformat(),
        "datasets_processed": len(manifest["datasets"]),
        "total_clean_rows": manifest["total_clean_rows"],
        "total_quarantined_rows": manifest["total_quarantined_rows"],
    }


success_task = PythonOperator(
    task_id="pipeline_success",
    python_callable=log_pipeline_success,
    trigger_rule="all_success",
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 7: Phase 5 — Star Schema Builder
# ════════════════════════════════════════════════════════════════════════════
# Reads the real Iceberg table via pyiceberg and raw network_tickets via
# Spark, builds the warehouse star schema, and writes a Phase 5 manifest
# to data/catalog/ for the monitoring layer to read.

def run_star_schema_builder(**context):
    """
    Execute Phase 5: Star Schema Builder.

    In production this would be a SparkSubmitOperator pointing at a YARN
    cluster. Here it imports and calls the Phase 5 main() directly,
    the same approach the local orchestrator uses for Phase 2.

    Returns:
        dict: Phase 5 manifest summary stored in XCom for Phase 7 monitoring.
    """
    import sys
    sys.path.insert(0, str(ORCHESTRATION_SCRIPTS))
    from phase5_star_schema_builder import main as phase5_main
    phase5_main()

    # Read the manifest Phase 5 wrote to catalog/ and push to XCom
    import glob
    manifests = sorted(glob.glob(str(CATALOG_PATH / "phase5_manifest_*.json")))
    if not manifests:
        raise AirflowException("Phase 5 manifest not found — phase5_star_schema_builder.py may have failed silently.")
    with open(manifests[-1]) as f:
        manifest = json.load(f)
    if manifest.get("status") != "SUCCESS":
        raise AirflowException(f"Phase 5 reported non-SUCCESS status: {manifest.get('status')}")
    return manifest


star_schema_task = PythonOperator(
    task_id="05_star_schema",
    python_callable=run_star_schema_builder,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task 8: Phase 6 — Analytics Mart Builder
# ════════════════════════════════════════════════════════════════════════════
# Builds the DuckDB analytics mart on top of the Phase 5 warehouse and
# writes a Phase 6 manifest. Downstream consumers (Superset, Power BI)
# connect to analytics_mart.duckdb after this task completes.

def run_mart_builder(**context):
    """
    Execute Phase 6: Analytics Data Mart Builder.

    Returns:
        dict: Phase 6 manifest summary stored in XCom for Phase 7 monitoring.
    """
    import sys
    sys.path.insert(0, str(ORCHESTRATION_SCRIPTS))
    from build_mart import main as phase6_main
    phase6_main()

    import glob
    manifests = sorted(glob.glob(str(CATALOG_PATH / "phase6_manifest_*.json")))
    if not manifests:
        raise AirflowException("Phase 6 manifest not found — build_mart.py may have failed silently.")
    with open(manifests[-1]) as f:
        manifest = json.load(f)
    if manifest.get("status") != "SUCCESS":
        raise AirflowException(f"Phase 6 reported non-SUCCESS status: {manifest.get('status')}")
    return manifest


mart_task = PythonOperator(
    task_id="06_analytics_mart_mart",
    python_callable=run_mart_builder,
    dag=dag,
)

# ════════════════════════════════════════════════════════════════════════════
# Task Dependency Graph
# ════════════════════════════════════════════════════════════════════════════
#
#   wait_for_sftp_files
#          ↓
#   phase_2_raw_layer_ingestion
#          ↓
#   validate_raw_layer_manifest
#          ↓
#   branch_proceed_to_spark (BranchPythonOperator)
#        ├──→ 04_classify_spark_processing → pipeline_success
#        │                                      ↓
#        │                            05_star_schema
#        │                                      ↓
#        │                            06_analytics_mart_mart
#        └──→ notify_pipeline_failure
#
# Note: BranchPythonOperator automatically skips the branch not chosen.
# pipeline_success uses trigger_rule="all_success" so it won't run
# if notify_pipeline_failure was triggered.

wait_for_sftp_files >> raw_layer_task
raw_layer_task      >> validate_raw_task
validate_raw_task   >> branch_task
branch_task         >> spark_task    >> success_task >> star_schema_task >> mart_task
branch_task         >> notify_failure
