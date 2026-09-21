"""
Pipeline output tests — validate the artefacts produced by each phase.

These tests do NOT re-run the pipeline.  They read the Parquet files and
JSON manifests written by a prior run and assert the invariants that matter:
quality thresholds, grain uniqueness, orphan-key absence, cross-phase row
counts, and overall monitoring health.

Run after .\run_pipeline.ps1:
    pytest tests/ -v
"""

import pandas as pd

QUALITY_THRESHOLD_PCT = 95.0
EXPECTED_ROOT_CAUSES  = {"active", "behavioral", "network", "mixed"}
CHURN_RATE_MIN_PCT    = 5.0
CHURN_RATE_MAX_PCT    = 50.0
WAREHOUSE_TABLE_COUNT = 7


# ===========================================================================
# Phase 2 — Raw layer quality
# ===========================================================================

def test_phase2_all_datasets_pass_quality_gate(phase2_manifest):
    assert phase2_manifest["all_passed"] is True, \
        "Phase 2 reported at least one dataset below the quality threshold"


def test_phase2_every_quality_score_above_threshold(phase2_manifest):
    for name, stats in phase2_manifest["datasets"].items():
        score = float(stats["quality_score"])
        assert score >= QUALITY_THRESHOLD_PCT, \
            f"{name} quality score {score}% is below the {QUALITY_THRESHOLD_PCT}% threshold"


def test_phase2_zero_quarantined_rows(phase2_manifest):
    assert phase2_manifest["total_quarantined_rows"] == 0, \
        f"Expected 0 quarantined rows, got {phase2_manifest['total_quarantined_rows']}"


# ===========================================================================
# Phase 4 — Spark classification
# ===========================================================================

def test_phase4_exactly_four_root_cause_categories(phase4_manifest):
    actual = set(phase4_manifest["root_cause_distribution"].keys())
    assert actual == EXPECTED_ROOT_CAUSES, \
        f"Unexpected root cause categories: {actual}"


def test_phase4_classification_counts_sum_to_total(phase4_manifest):
    total = phase4_manifest["total_customers"]
    summed = sum(phase4_manifest["root_cause_distribution"].values())
    assert summed == total, \
        f"root_cause_distribution sums to {summed}, expected {total}"


def test_phase4_churn_rate_in_expected_range(phase4_manifest):
    total    = phase4_manifest["total_customers"]
    churned  = phase4_manifest["total_churned"]
    rate_pct = churned / total * 100
    assert CHURN_RATE_MIN_PCT <= rate_pct <= CHURN_RATE_MAX_PCT, \
        f"Churn rate {rate_pct:.1f}% outside expected {CHURN_RATE_MIN_PCT}–{CHURN_RATE_MAX_PCT}% window"


# ===========================================================================
# Phase 5 — Star schema (Parquet checks)
# ===========================================================================

def test_phase5_seven_warehouse_tables_written(phase5_manifest):
    n = len(phase5_manifest["tables_written"])
    assert n == WAREHOUSE_TABLE_COUNT, \
        f"Expected {WAREHOUSE_TABLE_COUNT} warehouse tables, manifest shows {n}"


def test_phase5_all_dq_checks_passed(phase5_manifest):
    assert phase5_manifest["dq_checks_passed"] == 7


def test_phase5_fact_churn_grain_unique(fact_churn):
    dupes = fact_churn.duplicated(subset=["date_key", "customer_key"]).sum()
    assert dupes == 0, \
        f"{dupes} duplicate (date_key, customer_key) rows — fact grain violated"


def test_phase5_no_orphan_customer_keys_in_fact_churn(fact_churn):
    nulls = fact_churn["customer_key"].isna().sum()
    assert nulls == 0, \
        f"{nulls} NULL customer_key rows — dimension lookup failed for these facts"


def test_phase5_fact_ticket_grain_unique(fact_tickets):
    dupes = fact_tickets.duplicated(subset=["ticket_id"]).sum()
    assert dupes == 0, f"{dupes} duplicate ticket_id rows"


def test_phase5_fact_churn_row_count_matches_phase4(fact_churn, phase4_manifest):
    expected = phase4_manifest["total_customers"]
    actual   = len(fact_churn)
    assert actual == expected, \
        f"fact_churn has {actual} rows but Phase 4 classified {expected} customers"


def test_phase5_dim_customer_no_duplicate_current_rows(dim_customer):
    current = dim_customer[dim_customer["is_current"] == True]  # noqa: E712
    dupes   = current.duplicated(subset=["customer_id"]).sum()
    assert dupes == 0, \
        f"{dupes} customers have more than one is_current=True row in dim_customer"


# ===========================================================================
# Phase 6 — Analytics mart
# ===========================================================================

def test_phase6_mart_customer_360_reconciles_to_phase4(mart_customer_360, phase4_manifest):
    assert len(mart_customer_360) == phase4_manifest["total_customers"], \
        "mart_customer_360 row count does not match Phase 4 classified customers"


def test_phase6_root_cause_summary_has_four_rows(mart_root_cause):
    assert len(mart_root_cause) == 4, \
        f"mart_root_cause_summary has {len(mart_root_cause)} rows, expected 4"


def test_phase6_root_cause_customer_counts_sum_to_total(mart_root_cause, phase4_manifest):
    total_in_mart = mart_root_cause["customer_count"].sum()
    assert total_in_mart == phase4_manifest["total_customers"], \
        f"mart_root_cause_summary sums to {total_in_mart}, expected {phase4_manifest['total_customers']}"


# ===========================================================================
# Phase 7 — End-to-end monitoring health
# ===========================================================================

def test_phase7_pipeline_status_healthy(phase7_report):
    status = phase7_report["pipeline_status"]
    assert status == "HEALTHY", f"Pipeline status is {status!r}, expected 'HEALTHY'"


def test_phase7_zero_critical_alerts(phase7_report):
    criticals = phase7_report["alert_summary"]["critical"]
    assert criticals == 0, f"{criticals} critical alert(s) in the health report"
