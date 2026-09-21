"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
Phase 3 — Local Task Orchestrator (Airflow Simulator)
================================================================================

PURPOSE:
    Simulates Airflow DAG execution locally without a running Airflow instance.
    Demonstrates task execution order, dependency checking, XCom-style data
    passing, manifest validation, and error handling — the same patterns
    used in the production DAG (telecom_churn_pipeline.py).

BUGS FIXED vs original:
    Bug 1 (reader side) — task_validate_raw_layer_manifest now correctly reads
        quarantined_rows and quality_score from the manifest. These were always
        present in the manifest file but never in the manifest DICT because the
        writer (Phase 2 Section 7) didn't include them. With etl/run_phase2.py
        as the single source of truth, both writer and reader now agree.

    Bug 3 — Original called subprocess.run(["python", "../etl/run_phase2.py"])
        but that file did not exist. Phase 2 logic only lived inside the notebook.
        This orchestrator now shells out to etl/run_phase2.py which IS a real
        file — the same module the Airflow DAG imports from.

    Bug 4 — Quality threshold: original had no numeric check in the orchestrator
        at all — it only checked the all_passed flag. Now validates that every
        dataset's quality_score meets QUALITY_THRESHOLD (95%) before proceeding.
        This matches the DAG, Phase 1 quality passport, and Phase 2 validation.

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio
================================================================================
"""

import os
import json
import time
import logging
import sys
from pathlib import Path
from datetime import datetime
from enum import Enum

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration
# ════════════════════════════════════════════════════════════════════════════

# BUG FIX #4: Single quality threshold — matches DAG, Phase 1, and Phase 2
QUALITY_THRESHOLD = 95.0

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Task Status Enum
# ════════════════════════════════════════════════════════════════════════════

class TaskStatus(Enum):
    """Mirrors Airflow's task instance states."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Local Orchestrator Class
# ════════════════════════════════════════════════════════════════════════════

class LocalOrchestrator:
    """
    Simulates Airflow DAG execution on a single machine.

    Provides:
        - Task dependency resolution (run in correct order)
        - XCom-style data passing between tasks (dict-based)
        - Structured logging to file and console
        - Execution summary with timing and status per task

    Maps to the operator's production architecture:
        LocalOrchestrator  ↔  Airflow Scheduler
        run_task()         ↔  Airflow TaskInstance execution
        xcom_push/pull()   ↔  Airflow XCom store
        task_status dict   ↔  Airflow task_instance.state
    """

    def __init__(self, dag_id: str, run_date: datetime, log_dir: str):
        self.dag_id      = dag_id
        self.run_date    = run_date
        self.log_dir     = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.xcom_store  = {}   # {task_id: {key: value}}
        self.task_status = {}   # {task_id: TaskStatus}
        self.task_times  = {}   # {task_id: elapsed_seconds}

        self._setup_logging()

    def _setup_logging(self):
        """Configure dual logging: file (DEBUG) + console (INFO)."""
        self.log_path = (
            self.log_dir
            / f"orchestration_{self.run_date.strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger = logging.getLogger(self.dag_id)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(task_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_task(self, task_id: str, level: int, message: str):
        """Log with task context injected — mirrors Airflow task log format."""
        self.logger.log(level, message, extra={"task_id": task_id})

    def register_task(self, task_id: str, dependencies=None):
        """Register a task before execution."""
        if task_id not in self.task_status:
            self.task_status[task_id] = TaskStatus.PENDING
            self.xcom_store[task_id]  = {}

    def run_task(self, task_id: str, task_callable, dependencies=None):
        """
        Execute a task after verifying all dependencies succeeded.

        Args:
            task_id:       Unique task identifier
            task_callable: Function to execute — signature: fn(orchestrator, task_id)
            dependencies:  List of upstream task_ids that must be SUCCESS

        Returns:
            (TaskStatus, result): Status enum and task return value
        """
        dependencies = dependencies or []

        # ── Dependency check ───────────────────────────────────────────────
        for dep in dependencies:
            if self.task_status.get(dep) != TaskStatus.SUCCESS:
                self.log_task(
                    task_id, logging.WARNING,
                    f"Skipped — dependency '{dep}' did not succeed"
                )
                self.task_status[task_id] = TaskStatus.SKIPPED
                return TaskStatus.SKIPPED, None

        # ── Execute ───────────────────────────────────────────────────────
        self.task_status[task_id] = TaskStatus.RUNNING
        start_time = time.time()
        self.log_task(task_id, logging.INFO, "Starting task execution")

        try:
            result                    = task_callable(self, task_id)
            elapsed                   = time.time() - start_time
            self.task_times[task_id]  = elapsed
            self.task_status[task_id] = TaskStatus.SUCCESS
            self.log_task(
                task_id, logging.INFO,
                f"Completed successfully ({elapsed:.2f}s)"
            )
            return TaskStatus.SUCCESS, result

        except Exception as e:
            elapsed                   = time.time() - start_time
            self.task_times[task_id]  = elapsed
            self.task_status[task_id] = TaskStatus.FAILED
            self.log_task(task_id, logging.ERROR, f"Failed: {str(e)}")
            raise

    def xcom_push(self, task_id: str, key: str, value):
        """Push data to XCom store — available to all downstream tasks."""
        self.xcom_store[task_id][key] = value
        self.log_task(task_id, logging.DEBUG, f"XCom push: {key}")

    def xcom_pull(self, upstream_task_id: str, key: str):
        """Pull data from an upstream task's XCom store."""
        return self.xcom_store.get(upstream_task_id, {}).get(key)

    def get_summary(self) -> dict:
        """Generate execution summary — mirrors Airflow's DAG run report."""
        total_tasks = len(self.task_status)
        success     = sum(1 for s in self.task_status.values() if s == TaskStatus.SUCCESS)
        failed      = sum(1 for s in self.task_status.values() if s == TaskStatus.FAILED)
        skipped     = sum(1 for s in self.task_status.values() if s == TaskStatus.SKIPPED)
        total_time  = sum(self.task_times.values())

        return {
            "dag_id":             self.dag_id,
            "run_date":           self.run_date.isoformat(),
            "total_tasks":        total_tasks,
            "success":            success,
            "failed":             failed,
            "skipped":            skipped,
            "total_time_seconds": round(total_time, 2),
            "task_status":        {k: v.value for k, v in self.task_status.items()},
            "task_times":         self.task_times,
            "log_file":           str(self.log_path)
        }

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Task Implementations
# ════════════════════════════════════════════════════════════════════════════

def task_wait_for_sftp_files(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Verify all required files exist in the SFTP drop zone.

    Maps to the operator's FileSensor watching the NiFi/SFTP landing directory.
    In production: blocks until Huawei's batch export arrives.
    """
    sftp_drop = Path("../data/sftp_drop")

    required_files = [
        "customers.parquet",
        "usage_events.parquet",
        "billing_records.parquet",
        "recharge_records.parquet",
        "network_tickets.parquet"
    ]

    missing = [f for f in required_files if not (sftp_drop / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing SFTP files: {missing}\n"
            f"Run run_phase1.py first to generate source data."
        )

    orchestrator.log_task(
        task_id, logging.INFO,
        f"✅ All {len(required_files)} SFTP files present"
    )
    return {"files_found": len(required_files)}


def task_phase_2_raw_layer_ingestion(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Run Phase 2 — Raw Storage Layer ingestion pipeline.

    BUG FIX #3: Original called subprocess.run(["python", "../etl/run_phase2.py"])
    but that file did not exist — the logic only lived inside the notebook.
    Now shells out to etl/run_phase2.py which IS a real importable module.

    The script writes the manifest JSON to catalog/.
    This task reads that manifest and pushes it to XCom.
    """
    import subprocess
    import glob

    orchestrator.log_task(task_id, logging.INFO, "Launching Phase 2 pipeline...")

    # BUG FIX #3: correct path — 02_raw_ingest/run_phase2.py is the real module
    result = subprocess.run(
        ["python", "../02_raw_ingest/run_phase2.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(
            f"Phase 2 failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # ── Load the manifest written by run_phase2.py ─────────────────────────
    manifest_files = glob.glob("../data/catalog/raw_manifest_*.json")
    if not manifest_files:
        raise Exception(
            "No manifest file found in data/catalog/ after Phase 2 run. "
            "Check that etl/run_phase2.py is writing to the correct path."
        )

    manifest_path = sorted(manifest_files)[-1]   # Most recent run
    with open(manifest_path) as f:
        manifest = json.load(f)

    orchestrator.log_task(
        task_id, logging.INFO,
        f"Phase 2 complete | Datasets: {len(manifest['datasets'])}"
    )

    # Push manifest to XCom — downstream tasks pull from here
    orchestrator.xcom_push(task_id, "manifest", manifest)
    return manifest


def task_validate_raw_layer_manifest(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Validate the Phase 2 manifest before allowing Spark to run.

    BUG FIX #1 (reader side): Now correctly reads quarantined_rows and
    quality_score from the manifest. These fields are guaranteed present
    because etl/run_phase2.py (the writer) includes them in every manifest.

    BUG FIX #4: Added explicit numeric quality_score check per dataset.
    Original only checked the all_passed boolean flag, which is set by
    Phase 2 based on its own threshold. Now the orchestrator independently
    verifies every dataset meets QUALITY_THRESHOLD (95%) — a second gate.
    """
    manifest = orchestrator.xcom_pull(
        upstream_task_id="phase_2_raw_layer_ingestion",
        key="manifest"
    )

    if not manifest:
        raise Exception(
            "No manifest received from phase_2_raw_layer_ingestion. "
            "Check XCom push in that task."
        )

    # ── Gate 1: all_passed flag ────────────────────────────────────────────
    if not manifest.get("all_passed", False):
        failing = [
            d for d, v in manifest["datasets"].items()
            if v.get("status") != "PASSED"
        ]
        raise Exception(
            f"Raw layer validation FAILED for: {failing}\n"
            f"Check quarantine/ and quality_reports/ for details."
        )

    # ── Gate 2: per-dataset quality score check ────────────────────────────
    # BUG FIX #4: independent numeric check — doesn't rely solely on
    # the all_passed flag which Phase 2 sets based on its own logic.
    below_threshold = []
    for dataset, stats in manifest["datasets"].items():
        score_raw = stats.get("quality_score", "0")
        # quality_score is stored as string "97.8" (% stripped by writer)
        try:
            score = float(str(score_raw).replace("%", ""))
        except (ValueError, TypeError):
            score = 0.0
        if score < QUALITY_THRESHOLD:
            below_threshold.append(f"{dataset} ({score:.1f}%)")

    if below_threshold:
        raise Exception(
            f"Quality threshold {QUALITY_THRESHOLD}% not met for: {below_threshold}\n"
            f"Pipeline halted — bad data must not reach Spark processing."
        )

    # ── Log per-dataset summary ────────────────────────────────────────────
    orchestrator.log_task(task_id, logging.INFO, "✅ Raw layer validation PASSED")
    orchestrator.log_task(task_id, logging.INFO, "Quality summary:")

    for dataset, stats in manifest["datasets"].items():
        # BUG FIX #1: these fields now exist in the manifest
        clean       = str(stats.get("clean_rows",       "N/A")).replace(",", "")
        quarantined = str(stats.get("quarantined_rows", "N/A")).replace(",", "")
        score       = stats.get("quality_score", "N/A")
        status      = stats.get("status", "UNKNOWN")

        orchestrator.log_task(
            task_id, logging.INFO,
            f"  {dataset:<25} Status: {status:<8} | "
            f"Clean: {clean:>8} | "
            f"Quarantined: {quarantined:>6} | "    # No longer N/A
            f"Score: {score:>6}%"                   # No longer N/A
        )

    # Push for downstream tasks
    orchestrator.xcom_push(task_id, "validated_manifest", manifest)
    return manifest


def task_branch_proceed_to_spark(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Decision point — proceed to Phase 4 or halt.

    In production Airflow DAG: this is a BranchPythonOperator.
    Here: the return value is logged; the orchestrator always proceeds
    to 04_classify_spark_processing because validation already passed.
    If validation failed, this task is never reached (exception in
    task_validate_raw_layer_manifest halts the pipeline).
    """
    manifest = orchestrator.xcom_pull(
        upstream_task_id="validate_raw_layer_manifest",
        key="validated_manifest"
    )

    if manifest and manifest.get("all_passed"):
        orchestrator.log_task(
            task_id, logging.INFO,
            "✅ All conditions met — proceeding to Phase 4 (Spark Processing)"
        )
        return "04_classify_spark_processing"
    else:
        orchestrator.log_task(
            task_id, logging.WARNING,
            "⚠️  Halting — quality checks failed"
        )
        return "notify_pipeline_failure"


def task_04_classify_spark_processing(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Phase 4 — runs the real Spark feature engineering, classification,
    and Iceberg write chain (process_layer → root_cause_engine → iceberg_writer).
    """
    import sys
    import os
    # Resolve the project root from this file's location regardless of which
    # subdirectory the orchestrator is run from.
    THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)              # src/
    SPARK_DIR    = os.path.join(PROJECT_ROOT, "04_classify_spark")
    sys.path.insert(0, SPARK_DIR)

    orchestrator.log_task(task_id, logging.INFO, "Starting Phase 4 Spark chain")

    from process_layer import run_feature_engineering
    from root_cause_engine import run_root_cause_classification
    from iceberg_writer import run_iceberg_write

    orchestrator.log_task(task_id, logging.INFO, "Running process_layer feature engineering...")
    spark, feature_df, feature_manifest = run_feature_engineering()

    orchestrator.log_task(task_id, logging.INFO, "Running root_cause_engine classification...")
    classified_df, rc_manifest = run_root_cause_classification(feature_df)

    orchestrator.log_task(task_id, logging.INFO, "Running iceberg_writer persistence...")
    iceberg_manifest = run_iceberg_write(classified_df)

    spark.stop()

    result = {
        "phase": "spark",
        "total_customers": rc_manifest.get("total_customers"),
        "total_churned": rc_manifest.get("total_churned"),
        "root_cause_distribution": rc_manifest.get("root_cause_distribution"),
        "iceberg_snapshot_id": iceberg_manifest.get("current_snapshot_id"),
        "reference_date": iceberg_manifest.get("reference_date"),
    }
    orchestrator.xcom_push(task_id, "phase4_result", result)
    orchestrator.log_task(
        task_id, logging.INFO,
        f"Phase 4 complete — {result['total_customers']:,} customers classified, "
        f"reference_date={result['reference_date']}"
    )
    return result


def task_05_star_schema(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Phase 5 — builds the star schema (5 dim tables + 2 fact tables)
    over the Iceberg-classified data. Reads phase5_manifest_*.json from
    data/catalog/ after the run to confirm DQ checks passed.

    Maps to the operator's stack: this is the Spark ETL job that writes the
    warehouse layer (Iceberg tables on HDFS in production).
    """
    import sys
    import os
    import glob
    THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)              # src/
    SPARK_DIR    = os.path.join(PROJECT_ROOT, "05_star_schema")
    sys.path.insert(0, SPARK_DIR)

    orchestrator.log_task(task_id, logging.INFO, "Starting Phase 5 star schema builder...")
    from phase5_star_schema_builder import main as phase5_main
    phase5_main()

    # Read back the manifest to confirm DQ passed and surface row counts
    catalog_dir = os.path.join(PROJECT_ROOT, "data", "catalog")
    manifests = sorted(glob.glob(os.path.join(catalog_dir, "phase5_manifest_*.json")))
    if not manifests:
        raise RuntimeError("Phase 5 ran but no phase5_manifest_*.json found in data/catalog/")

    with open(manifests[-1]) as f:
        manifest = json.load(f)

    if manifest.get("status") != "SUCCESS":
        raise RuntimeError(
            f"Phase 5 manifest status={manifest.get('status')} — star schema may be incomplete"
        )

    orchestrator.xcom_push(task_id, "phase5_manifest", manifest)
    orchestrator.log_task(
        task_id, logging.INFO,
        f"Phase 5 complete — {len(manifest.get('tables_written', []))} tables written, "
        f"{manifest.get('dq_checks_passed')} DQ checks passed"
    )
    return manifest


def task_06_analytics_mart_mart(orchestrator: LocalOrchestrator, task_id: str):
    """
    Task: Phase 6 — builds DuckDB analytics mart tables over the Phase 5
    warehouse and exports them to Parquet. Reads phase6_manifest_*.json
    after the run to confirm all mart tables were built and reconciled.

    Maps to the operator's stack: this is the Presto CTAS step that materialises
    the curated mart layer Superset/Power BI connects to.
    """
    import sys
    import os
    import glob
    THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT  = os.path.dirname(THIS_DIR)              # src/
    ANALYTICS_DIR = os.path.join(PROJECT_ROOT, "06_analytics_mart")
    sys.path.insert(0, ANALYTICS_DIR)

    orchestrator.log_task(task_id, logging.INFO, "Starting Phase 6 analytics mart builder...")
    from build_mart import main as mart_main
    mart_main()

    catalog_dir = os.path.join(PROJECT_ROOT, "data", "catalog")
    manifests = sorted(glob.glob(os.path.join(catalog_dir, "phase6_manifest_*.json")))
    if not manifests:
        raise RuntimeError("Phase 6 ran but no phase6_manifest_*.json found in data/catalog/")

    with open(manifests[-1]) as f:
        manifest = json.load(f)

    if manifest.get("status") != "SUCCESS":
        raise RuntimeError(
            f"Phase 6 manifest status={manifest.get('status')} — mart may be incomplete"
        )

    orchestrator.xcom_push(task_id, "phase6_manifest", manifest)
    orchestrator.log_task(
        task_id, logging.INFO,
        f"Phase 6 complete — {len(manifest.get('mart_tables', []))} mart tables built, "
        f"DQ checks passed"
    )
    return manifest


def task_pipeline_success(orchestrator: LocalOrchestrator, task_id: str):
    """Task: Log full pipeline completion — summarises all phase manifests."""
    raw_manifest  = orchestrator.xcom_pull("validate_raw_layer_manifest", "validated_manifest")
    phase4_result = orchestrator.xcom_pull("04_classify_spark_processing",    "phase4_result")
    phase5_mf     = orchestrator.xcom_pull("05_star_schema",          "phase5_manifest")
    phase6_mf     = orchestrator.xcom_pull("06_analytics_mart_mart",       "phase6_manifest")

    orchestrator.log_task(task_id, logging.INFO, "🎉 Full pipeline completed successfully")

    if raw_manifest:
        orchestrator.log_task(task_id, logging.INFO,
            f"Phase 2 — clean rows: {raw_manifest.get('total_clean_rows', 'N/A'):,}")
    if phase4_result:
        orchestrator.log_task(task_id, logging.INFO,
            f"Phase 4 — {phase4_result.get('total_customers', 'N/A'):,} customers classified "
            f"({phase4_result.get('total_churned', 'N/A')} churned), "
            f"reference_date={phase4_result.get('reference_date')}")
    if phase5_mf:
        orchestrator.log_task(task_id, logging.INFO,
            f"Phase 5 — {len(phase5_mf.get('tables_written', []))} warehouse tables, "
            f"{phase5_mf.get('dq_checks_passed')} DQ checks passed")
    if phase6_mf:
        orchestrator.log_task(task_id, logging.INFO,
            f"Phase 6 — {len(phase6_mf.get('mart_tables', []))} mart tables built")

    return {"status": "success"}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Execution
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("  TELECOM CHURN PLATFORM — LOCAL TASK ORCHESTRATOR")
    print("  Simulating Airflow DAG Execution")
    print("=" * 80 + "\n")

    orchestrator = LocalOrchestrator(
        dag_id   = "telecom_churn_platform",
        run_date = datetime.now(),
        log_dir  = "../data/orchestration_logs"
    )

    orchestrator.log_task("__main__", logging.INFO, "Starting orchestration run")

    # ── Task DAG definition ────────────────────────────────────────────────
    # Mirrors the Airflow DAG in telecom_churn_pipeline.py exactly:
    # same task names, same dependency order, same XCom keys.
    tasks = {
        "wait_for_sftp_files": {
            "callable":     task_wait_for_sftp_files,
            "dependencies": []
        },
        "phase_2_raw_layer_ingestion": {
            "callable":     task_phase_2_raw_layer_ingestion,
            "dependencies": ["wait_for_sftp_files"]
        },
        "validate_raw_layer_manifest": {
            "callable":     task_validate_raw_layer_manifest,
            "dependencies": ["phase_2_raw_layer_ingestion"]
        },
        "branch_proceed_to_spark": {
            "callable":     task_branch_proceed_to_spark,
            "dependencies": ["validate_raw_layer_manifest"]
        },
        "04_classify_spark_processing": {
            "callable":     task_04_classify_spark_processing,
            "dependencies": ["branch_proceed_to_spark"]
        },
        "05_star_schema": {
            "callable":     task_05_star_schema,
            "dependencies": ["04_classify_spark_processing"]
        },
        "06_analytics_mart_mart": {
            "callable":     task_06_analytics_mart_mart,
            "dependencies": ["05_star_schema"]
        },
        "pipeline_success": {
            "callable":     task_pipeline_success,
            "dependencies": ["06_analytics_mart_mart"]
        }
    }

    # ── Execute all tasks ──────────────────────────────────────────────────
    for task_id, config in tasks.items():
        try:
            orchestrator.register_task(task_id, config["dependencies"])
            orchestrator.run_task(
                task_id,
                config["callable"],
                dependencies=config["dependencies"]
            )
        except Exception as e:
            orchestrator.log_task(
                "__main__", logging.ERROR,
                f"Pipeline halted at task: {task_id}"
            )
            orchestrator.log_task("__main__", logging.ERROR, str(e))
            break

    # ── Execution Summary ──────────────────────────────────────────────────
    summary = orchestrator.get_summary()

    print("\n" + "=" * 80)
    print("  ORCHESTRATION EXECUTION SUMMARY")
    print("=" * 80)
    print(f"DAG:                    {summary['dag_id']}")
    print(f"Run Date:               {summary['run_date']}")
    print(f"Total Tasks:            {summary['total_tasks']}")
    print(f"  ✅ Success:           {summary['success']}")
    print(f"  ❌ Failed:            {summary['failed']}")
    print(f"  ⊘  Skipped:           {summary['skipped']}")
    print(f"Total Execution Time:   {summary['total_time_seconds']:.2f}s")
    print(f"Log File:               {summary['log_file']}")

    print("\n" + "-" * 80)
    print("  TASK EXECUTION DETAILS")
    print("-" * 80)
    print(f"{'Task ID':<40} {'Status':<12} {'Time (s)':<12}")
    print("-" * 80)

    for task_id, status in summary["task_status"].items():
        exec_time   = summary["task_times"].get(task_id, 0)
        status_icon = {
            "success": "✅",
            "failed":  "❌",
            "pending": "⏳",
            "skipped": "⊘"
        }.get(status, "?")
        print(f"{task_id:<40} {status_icon} {status:<10} {exec_time:>10.2f}s")

    print("-" * 80)

    # ── Save summary JSON ──────────────────────────────────────────────────
    summary_dir  = Path("../data/orchestration_logs")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved → {summary_path}")

    # ── Architecture Mapping ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PRODUCTION ARCHITECTURE MAPPING")
    print("=" * 80)
    print("""
Pipeline Flow:
  sftp_drop/ (Huawei batch files)
      ↓ [wait_for_sftp_files]   — FileSensor equivalent
  etl/run_phase2.py
      ↓ [phase_2_raw_layer_ingestion]  — Schema validation + partitioned write
  raw_manifest_{ts}.json
      ↓ [validate_raw_layer_manifest]  — Quality gate (must be ≥95%)
  XCom: validated_manifest
      ↓ [branch_proceed_to_spark]      — BranchPythonOperator equivalent
  spark-submit process_layer.py
      ↓ [04_classify_spark_processing]     — Feature eng + churn detection
  processed/ (Iceberg format)
      ↓ [pipeline_success]             — Log metrics, notify monitoring

Production equivalent:
  Deploy telecom_churn_pipeline.py → $AIRFLOW_HOME/dags/
  Airflow scheduler triggers daily at 01:00 UTC
  Same task logic, same XCom pattern, same quality gates
    """)

    print("=" * 80 + "\n")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
