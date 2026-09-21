"""
================================================================================
PHASE 7 — PIPELINE MONITORING & ALERTING
================================================================================
Project   : Telecom Customer Journey & Churn Root Cause Analysis Platform

WHAT THIS SCRIPT DOES
----------------------
Reads every manifest written to data/catalog/ by the previous phases:
    Phase 2  → raw_manifest_*.json
    Phase 4  → phase4_feature_manifest_*.json
              root_cause_manifest_*.json
              iceberg_write_manifest_*.json
    Phase 5  → phase5_manifest_*.json
    Phase 6  → phase6_manifest_*.json

For each phase it checks:
    1. STATUS CHECK  — did the phase complete with status SUCCESS?
    2. ROW COUNT     — did key tables have expected minimum row counts?
    3. QUALITY SCORE — did raw datasets meet the 95% threshold?
    4. DQ CHECKS     — did the expected number of DQ checks pass?
    5. DRIFT CHECK   — did churn rate shift significantly from
                       the previous run? (Phase 4 only)

Then it:
    - Prints a colour-coded terminal health report
    - Writes monitoring/pipeline_health_report_<ts>.json
    - Writes monitoring/alerts_<ts>.json if any CRITICAL alerts fired
    - Exits with code 1 if any CRITICAL alert fired — Airflow reads
      the exit code and triggers email_on_failure

WHY MONITORING IS PHASE 7, NOT BUILT INTO EACH PHASE
------------------------------------------------------
Each phase already does "local" DQ: orphan key checks, grain uniqueness,
reconciliation totals. Those checks guard against structural corruption
at write time — they fail fast and stop bad data from landing.

Phase 7 monitors the PIPELINE AS A WHOLE across all phases, with the
view a pipeline operator has at 08:00 every morning:
    - Did all phases run?
    - Were row counts in the expected range?
    - Is churn rate drifting — which might mean the model needs
      retraining, or something broke in upstream data generation?
These are cross-phase, operational questions — separate from per-phase
structural correctness.


"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import Optional


# ==============================================================================
# CONFIG
# ==============================================================================

MONITORING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(MONITORING_DIR)
CATALOG_DIR    = os.path.join(BASE_DIR, "data", "catalog")
REPORTS_DIR    = MONITORING_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# All thresholds in one place — changing a value here changes it everywhere.
# Same principle as the unified QUALITY_THRESHOLD=95 across the orchestrator
# and Airflow DAG: one source of truth, no silent drift between files.
ALERT_THRESHOLDS = {
    # Phase 2 raw layer
    "raw_quality_score_min_pct":      95.0,
    "raw_customers_min_rows":          1_000,
    "raw_usage_events_min_rows":      10_000,
    "raw_tickets_min_rows":              500,
    # Phase 4 classification
    "classified_customers_min_rows":   1_000,
    "churn_rate_min_pct":               5.0,
    "churn_rate_max_pct":              50.0,
    "churn_rate_drift_threshold_pct":  10.0,
    # Phase 5 warehouse
    "p5_dq_checks_min":                    7,
    # Phase 6 mart
    "mart_customer_360_min_rows":      1_000,
    "mart_root_cause_summary_rows":        4,
}

CRITICAL = "CRITICAL"
WARNING  = "WARNING"
INFO     = "INFO"


# ==============================================================================
# MANIFEST LOADERS
# ==============================================================================

def _latest(pattern: str) -> Optional[dict]:
    """Return parsed JSON of the most recent file matching the glob pattern.
    Alphabetically last = chronologically last because filenames embed
    yyyymmdd_hhmmss timestamps."""
    files = sorted(glob.glob(os.path.join(CATALOG_DIR, pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _all_matching(pattern: str) -> list:
    files = sorted(glob.glob(os.path.join(CATALOG_DIR, pattern)))
    results = []
    for fp in files:
        with open(fp) as f:
            results.append(json.load(f))
    return results


def load_raw_manifest()           -> Optional[dict]: return _latest("raw_manifest_*.json")
def load_root_cause_manifest()    -> Optional[dict]: return _latest("root_cause_manifest_*.json")
def load_iceberg_manifest()       -> Optional[dict]: return _latest("iceberg_write_manifest_*.json")
def load_phase5_manifest()        -> Optional[dict]: return _latest("phase5_manifest_*.json")
def load_phase6_manifest()        -> Optional[dict]: return _latest("phase6_manifest_*.json")
def load_all_root_cause_manifests() -> list:         return _all_matching("root_cause_manifest_*.json")


# ==============================================================================
# ALERT COLLECTOR
# ==============================================================================

class AlertCollector:
    """
    Accumulates all alerts during the monitoring run. Keeps CRITICAL
    and WARNING alerts separate so the final summary can count them and
    exit(1) if any CRITICAL fired — giving Airflow a concrete failure signal.
    """
    def __init__(self):
        self.alerts = []

    def add(self, level: str, phase: str, check: str, message: str, value=None):
        self.alerts.append({
            "level": level, "phase": phase,
            "check": check, "message": message, "value": value,
        })
        icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(level, "⚪")
        print(f"  {icon} [{level:<8}] {phase} | {check}: {message}")

    def ok(self, phase: str, check: str, message: str):
        self.add(INFO, phase, check, message)

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.alerts if a["level"] == CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.alerts if a["level"] == WARNING)


# ==============================================================================
# PHASE MONITORS
# ==============================================================================

def monitor_phase2_raw(alerts: AlertCollector) -> dict:
    """
    Checks Phase 2 raw ingestion manifest.
    A raw layer failure is CRITICAL — every downstream phase depends on it.
    Note: Phase 2 stores clean_rows as formatted strings ("5,000") not ints —
    the _clean_int() helper strips commas before numeric comparisons.
    """
    print("\n─── Phase 2: Raw Layer Ingestion ───")
    manifest = load_raw_manifest()

    if not manifest:
        alerts.add(CRITICAL, "Phase 2", "manifest_exists",
                   "No raw_manifest_*.json found — Phase 2 may not have run")
        return {"status": "MISSING"}

    if not manifest.get("all_passed"):
        alerts.add(CRITICAL, "Phase 2", "all_passed",
                   "all_passed=False — one or more datasets failed the quality gate")
    else:
        alerts.ok("Phase 2", "all_passed",
                  f"All {manifest['datasets_processed']} datasets passed quality gate ✓")

    def _clean_int(s) -> int:
        return int(str(s).replace(",", ""))

    datasets = manifest.get("datasets", {})

    for ds_name, stats in datasets.items():
        score = float(str(stats.get("quality_score", 0)).replace("%", ""))
        if score < ALERT_THRESHOLDS["raw_quality_score_min_pct"]:
            alerts.add(CRITICAL, "Phase 2", f"{ds_name}_quality",
                       f"Quality {score}% below {ALERT_THRESHOLDS['raw_quality_score_min_pct']}% threshold",
                       value=score)
        else:
            alerts.ok("Phase 2", f"{ds_name}_quality", f"{ds_name}: {score}% ✓")

    for ds_name, threshold_key in [
        ("customers",       "raw_customers_min_rows"),
        ("usage_events",    "raw_usage_events_min_rows"),
        ("network_tickets", "raw_tickets_min_rows"),
    ]:
        if ds_name in datasets:
            rows = _clean_int(datasets[ds_name].get("clean_rows", 0))
            floor = ALERT_THRESHOLDS[threshold_key]
            if rows < floor:
                alerts.add(CRITICAL, "Phase 2", f"{ds_name}_row_count",
                           f"{rows:,} rows < minimum {floor:,}", value=rows)
            else:
                alerts.ok("Phase 2", f"{ds_name}_row_count",
                          f"{ds_name}: {rows:,} rows ✓")

    return {
        "status":             "OK" if manifest.get("all_passed") else "FAILED",
        "datasets_processed": manifest.get("datasets_processed"),
        "total_clean_rows":   manifest.get("total_clean_rows"),
        "total_quarantined":  manifest.get("total_quarantined_rows"),
        "run_date":           manifest.get("run_date"),
    }


def monitor_phase4_classification(alerts: AlertCollector) -> dict:
    """
    Checks Phase 4 root_cause and Iceberg manifests.

    DRIFT DETECTION: compares the current run's churn rate against the
    previous run's. If it shifted more than churn_rate_drift_threshold_pct
    percentage points, a WARNING fires. This matters because a sudden drift
    is almost always a pipeline problem (bad upstream data, broken feature
    computation), not a genuine business change — and it should be caught
    BEFORE the Superset dashboard gets refreshed and a director sees it.
    """
    print("\n─── Phase 4: Spark Classification ───")
    rc_manifest      = load_root_cause_manifest()
    iceberg_manifest = load_iceberg_manifest()
    all_rc           = load_all_root_cause_manifests()
    summary          = {"status": "MISSING"}

    if not rc_manifest:
        alerts.add(CRITICAL, "Phase 4", "root_cause_manifest",
                   "No root_cause_manifest_*.json found")
    else:
        total   = rc_manifest.get("total_customers", 0)
        churned = rc_manifest.get("total_churned", 0)
        dist    = rc_manifest.get("root_cause_distribution", {})

        if total < ALERT_THRESHOLDS["classified_customers_min_rows"]:
            alerts.add(CRITICAL, "Phase 4", "customer_count",
                       f"Only {total:,} customers classified — below minimum "
                       f"{ALERT_THRESHOLDS['classified_customers_min_rows']:,}", value=total)
        else:
            alerts.ok("Phase 4", "customer_count", f"{total:,} customers classified ✓")

        churn_rate = 100 * churned / total if total else 0

        if churn_rate < ALERT_THRESHOLDS["churn_rate_min_pct"]:
            alerts.add(WARNING, "Phase 4", "churn_rate_low",
                       f"Churn rate {churn_rate:.1f}% is suspiciously low", value=churn_rate)
        elif churn_rate > ALERT_THRESHOLDS["churn_rate_max_pct"]:
            alerts.add(WARNING, "Phase 4", "churn_rate_high",
                       f"Churn rate {churn_rate:.1f}% is suspiciously high", value=churn_rate)
        else:
            alerts.ok("Phase 4", "churn_rate",
                      f"Churn rate {churn_rate:.1f}% within expected range "
                      f"({ALERT_THRESHOLDS['churn_rate_min_pct']}%–"
                      f"{ALERT_THRESHOLDS['churn_rate_max_pct']}%) ✓")

        # Drift detection — needs at least 2 historical runs
        if len(all_rc) >= 2:
            prev         = all_rc[-2]
            prev_total   = prev.get("total_customers", 1)
            prev_churned = prev.get("total_churned", 0)
            prev_rate    = 100 * prev_churned / prev_total
            drift        = abs(churn_rate - prev_rate)
            threshold    = ALERT_THRESHOLDS["churn_rate_drift_threshold_pct"]
            if drift > threshold:
                alerts.add(WARNING, "Phase 4", "churn_rate_drift",
                           f"Churn rate shifted {drift:.1f}pp vs previous run "
                           f"({prev_rate:.1f}% → {churn_rate:.1f}%) — "
                           f"investigate before dashboard refresh", value=round(drift, 2))
            else:
                alerts.ok("Phase 4", "churn_rate_drift",
                          f"Drift {drift:.1f}pp within {threshold}pp threshold ✓")
        else:
            alerts.add(INFO, "Phase 4", "churn_rate_drift",
                       "Only one classification run exists — drift baseline not yet established")

        summary = {
            "status":                    "OK",
            "total_customers":           total,
            "total_churned":             churned,
            "churn_rate_pct":            round(churn_rate, 2),
            "root_cause_distribution":   dist,
        }

    if not iceberg_manifest:
        alerts.add(CRITICAL, "Phase 4", "iceberg_manifest",
                   "No iceberg_write_manifest_*.json found — Iceberg write may have failed")
    else:
        written    = iceberg_manifest.get("rows_written", 0)
        snapshots  = iceberg_manifest.get("total_snapshots", 0)
        write_mode = iceberg_manifest.get("write_mode", "unknown")
        ref_date   = iceberg_manifest.get("reference_date", "unknown")

        if written < ALERT_THRESHOLDS["classified_customers_min_rows"]:
            alerts.add(CRITICAL, "Phase 4", "iceberg_rows_written",
                       f"Only {written:,} rows in Iceberg table — below minimum", value=written)
        else:
            alerts.ok("Phase 4", "iceberg_write",
                      f"{written:,} rows | mode={write_mode} | "
                      f"snapshots={snapshots} | reference_date={ref_date} ✓")

        summary["iceberg_snapshots"] = snapshots
        summary["reference_date"]    = ref_date
        summary["write_mode"]        = write_mode

    return summary


def monitor_phase5_warehouse(alerts: AlertCollector) -> dict:
    """
    Checks Phase 5 star schema manifest.
    Key concerns: all 7 tables written? All 7 DQ checks passed?
    """
    print("\n─── Phase 5: Star Schema Warehouse ───")
    manifest = load_phase5_manifest()

    if not manifest:
        alerts.add(CRITICAL, "Phase 5", "manifest_exists",
                   "No phase5_manifest_*.json found — Phase 5 may not have run")
        return {"status": "MISSING"}

    if manifest.get("status") != "SUCCESS":
        alerts.add(CRITICAL, "Phase 5", "status",
                   f"Phase 5 status={manifest.get('status')}")
    else:
        alerts.ok("Phase 5", "status", "Star schema build: SUCCESS ✓")

    tables   = manifest.get("tables_written", [])
    dq_count = manifest.get("dq_checks_passed", 0)
    expected = [
        "dim_date", "dim_region", "dim_plan", "dim_root_cause",
        "dim_customer", "fact_customer_churn_snapshot", "fact_network_tickets"
    ]
    missing = [t for t in expected if t not in tables]
    if missing:
        alerts.add(CRITICAL, "Phase 5", "tables_written",
                   f"Missing tables: {missing}", value=missing)
    else:
        alerts.ok("Phase 5", "tables_written", f"All {len(tables)} warehouse tables present ✓")

    if dq_count < ALERT_THRESHOLDS["p5_dq_checks_min"]:
        alerts.add(CRITICAL, "Phase 5", "dq_checks",
                   f"Only {dq_count} DQ checks passed — expected at least "
                   f"{ALERT_THRESHOLDS['p5_dq_checks_min']}", value=dq_count)
    else:
        alerts.ok("Phase 5", "dq_checks", f"{dq_count} DQ checks passed ✓")

    return {
        "status":           manifest.get("status"),
        "tables_written":   tables,
        "dq_checks_passed": dq_count,
        "source_customers": manifest.get("sources", {}).get("classified_customers_rows"),
        "source_tickets":   manifest.get("sources", {}).get("raw_network_ticket_rows"),
    }


def monitor_phase6_mart(alerts: AlertCollector) -> dict:
    """
    Checks Phase 6 analytics mart manifest.

    CROSS-PHASE RECONCILIATION: mart_customer_360 row count must equal
    Phase 4's total_customers. If they don't match, a row was lost somewhere
    between classification and the mart — a lineage break that would make
    any customer-level analysis wrong. Catching this here, reading two
    manifests from different phases, is only possible from a cross-phase
    monitoring script — it can't be detected by either phase on its own.
    """
    print("\n─── Phase 6: Analytics Mart ───")
    manifest    = load_phase6_manifest()
    rc_manifest = load_root_cause_manifest()

    if not manifest:
        alerts.add(CRITICAL, "Phase 6", "manifest_exists",
                   "No phase6_manifest_*.json found — Phase 6 may not have run")
        return {"status": "MISSING"}

    if manifest.get("status") != "SUCCESS":
        alerts.add(CRITICAL, "Phase 6", "status",
                   f"Phase 6 status={manifest.get('status')}")
    else:
        alerts.ok("Phase 6", "status", "Mart build: SUCCESS ✓")

    mart_tables = manifest.get("mart_tables", {})

    m360 = mart_tables.get("mart_customer_360", 0)
    if m360 < ALERT_THRESHOLDS["mart_customer_360_min_rows"]:
        alerts.add(CRITICAL, "Phase 6", "mart_customer_360_rows",
                   f"mart_customer_360 has {m360:,} rows — below minimum", value=m360)
    else:
        alerts.ok("Phase 6", "mart_customer_360_rows",
                  f"mart_customer_360: {m360:,} rows ✓")

    # Cross-phase lineage check
    if rc_manifest:
        p4_total = rc_manifest.get("total_customers", 0)
        if m360 != p4_total:
            alerts.add(WARNING, "Phase 6", "cross_phase_reconciliation",
                       f"mart_customer_360 ({m360:,}) ≠ Phase 4 customers ({p4_total:,}) — "
                       "lineage gap between classification and mart",
                       value={"mart": m360, "phase4": p4_total})
        else:
            alerts.ok("Phase 6", "cross_phase_reconciliation",
                      f"mart_customer_360 ({m360:,}) = Phase 4 customers ({p4_total:,}) ✓")

    # Root cause category count — fixed vocabulary, must always be 4
    rc_rows   = mart_tables.get("mart_root_cause_summary", 0)
    expected_rc = ALERT_THRESHOLDS["mart_root_cause_summary_rows"]
    if rc_rows != expected_rc:
        alerts.add(WARNING, "Phase 6", "root_cause_category_count",
                   f"mart_root_cause_summary has {rc_rows} rows — expected {expected_rc}. "
                   "A new root cause category may have appeared upstream.",
                   value=rc_rows)
    else:
        alerts.ok("Phase 6", "root_cause_category_count",
                  f"mart_root_cause_summary: {rc_rows} categories ✓")

    return {
        "status":      manifest.get("status"),
        "mart_tables": mart_tables,
        "dq_checks":   manifest.get("dq_checks_passed"),
    }


# ==============================================================================
# REPORT WRITER
# ==============================================================================

def write_health_report(phase_summaries: dict, alerts: AlertCollector) -> str:
    report = {
        "report_timestamp": datetime.now().isoformat(),
        "pipeline_status":  "HEALTHY" if alerts.critical_count == 0 else "DEGRADED",
        "alert_summary": {
            "critical": alerts.critical_count,
            "warning":  alerts.warning_count,
            "info":     sum(1 for a in alerts.alerts if a["level"] == INFO),
        },
        "phase_summaries":  phase_summaries,
        "alerts":           alerts.alerts,
        "thresholds_used":  ALERT_THRESHOLDS,
    }

    report_path = os.path.join(
        REPORTS_DIR, f"pipeline_health_report_{RUN_TIMESTAMP}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if alerts.critical_count > 0:
        alert_path = os.path.join(REPORTS_DIR, f"alerts_{RUN_TIMESTAMP}.json")
        with open(alert_path, "w") as f:
            json.dump(
                {"critical_alerts": [a for a in alerts.alerts if a["level"] == CRITICAL]},
                f, indent=2
            )
        print(f"\n[ALERT FILE] {alert_path}")

    return report_path


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    print("=" * 80)
    print("  PHASE 7 — PIPELINE MONITORING & ALERTING")
    print(f"  Run timestamp: {RUN_TIMESTAMP}")
    print("=" * 80)

    alerts = AlertCollector()

    summaries = {
        "phase_2_raw":            monitor_phase2_raw(alerts),
        "phase_4_classification": monitor_phase4_classification(alerts),
        "phase_5_warehouse":      monitor_phase5_warehouse(alerts),
        "phase_6_mart":           monitor_phase6_mart(alerts),
    }

    report_path = write_health_report(summaries, alerts)

    status = "✅  HEALTHY" if alerts.critical_count == 0 else "🔴 DEGRADED"
    print("\n" + "=" * 80)
    print(f"  PIPELINE HEALTH: {status}")
    print("=" * 80)
    print(f"  🔴 Critical : {alerts.critical_count}")
    print(f"  🟡 Warnings : {alerts.warning_count}")
    print(f"  🟢 Passed   : {sum(1 for a in alerts.alerts if a['level'] == INFO)}")
    print(f"  Report      : {report_path}")
    print("=" * 80)

    return 0 if alerts.critical_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
