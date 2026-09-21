"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
spark/root_cause_engine.py — Root Cause Classification Engine
================================================================================

PURPOSE:
    Classifies every churned customer into exactly one of three root causes —
    behavioral, network, or mixed — using a transparent, rule-based decision
    tree. Active (non-churned) customers are labeled separately.

WHY RULE-BASED AND NOT A TRAINED MODEL:
    A rule engine is explainable. When the panel asks "why was this customer
    classified as network churn?" the answer is a specific, auditable
    condition — not a black-box probability. Telecom root cause analysis is
    a domain where interpretability matters more than marginal accuracy:
    a churn-save team needs to know WHY to act, not just THAT a customer
    is at risk. This mirrors how the operator's analysts would actually use it —
    open the rule, see exactly which threshold fired.

THE DECISION TREE (agreed and tested before this file was written):

    1. Not churned (is_churned == 0)               -> "active"
    2. Churned AND 3+ tickets in 30 days before
       silence (hard failures, dropped/no-signal)   -> check behavioral too:
           - if ALSO 2+ unpaid months              -> "mixed"
           - else                                   -> "network"
    3. Churned AND silent 60+ days AND
       2+ unpaid months in last 3 cycles            -> "behavioral"
    4. Churned, no clean signal in either direction  -> "behavioral" (default)

WHY THIS ORDER (network checked before behavioral):
    A network-caused churn ALWAYS produces behavioral symptoms too — a
    customer who suffers repeated dropped calls also goes silent and stops
    paying. If we checked behavioral signals first, every network-caused
    churn would be misclassified as behavioral before its ticket history
    was ever examined. Checking the more specific, harder-to-fake signal
    (a ticket record tied to a hard failure) before the generic symptom
    (silence, non-payment) is what keeps the classification honest.

THRESHOLDS (confirmed, not placeholders):
    days_silent >= 60                — two missed prepaid billing cycles
                                        (Sudan's cycle is ~30 days); one
                                        missed cycle could be a trip or
                                        illness, two signals intentional exit
    unpaid_months_recent >= 2        — same two-cycle reasoning, applied
                                        to payment behavior
    tickets_last_30d_before_silence
        >= 3                         — a single ticket can be a one-off;
                                        three in 30 days immediately before
                                        going quiet is a pattern, not noise
    arpu_drop_pct                    — computed in process_layer.py,
                                        available for Phase 6 deep-dive but
                                        not gating in this version's tree
                                        (see Section 4 notes)

PRODUCTION ARCHITECTURE MAPPING:
    Input:  feature_df from process_layer.py (in-memory, cached)
    Output: classified Spark DataFrame, handed to iceberg_writer.py
    This file IS the "root cause rule engine" referenced in the job
    requirements — explainable, documented, versionable.

AUTHOR: Omnia Abdelrhman
PROJECT: Telecom Churn Platform — Data Engineering & Data Science Portfolio
================================================================================
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from pyspark.sql import functions as F

CATALOG_PATH = Path("../data/catalog")
CATALOG_PATH.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("root_cause_engine")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Rule Thresholds (single source of truth)
# ════════════════════════════════════════════════════════════════════════════
# Defined as named constants, not magic numbers buried in the F.when() chain.
# Anyone reading this file — or asked about it in an interview — can find
# every threshold in one place, with the reasoning documented above.

DAYS_SILENT_THRESHOLD       = 60   # 2 missed prepaid billing cycles
UNPAID_MONTHS_THRESHOLD     = 2    # 2 of last 3 billing cycles unpaid
TICKETS_THRESHOLD           = 3    # 3+ tickets in the pre-silence window


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Classification Rules (as standalone, testable functions)
# ════════════════════════════════════════════════════════════════════════════
# Each rule is its own function so it can be unit tested independently of
# the full F.when() chain — and so the interview answer for "show me the
# network rule" is a single function, not a search through one giant block.

def is_network_signal(df):
    """
    Network churn signal: 3+ tickets in the 30 days before going silent.

    Returns a boolean Column — True where the network signal fires.
    """
    return df["tickets_last_30d_before_silence"] >= TICKETS_THRESHOLD


def is_behavioral_signal(df):
    """
    Behavioral churn signal: silent 60+ days AND 2+ unpaid months recently.

    Both conditions required (AND, not OR) — silence alone could mean a
    customer is traveling; unpaid alone could mean a billing dispute.
    Together, they indicate genuine disengagement.
    """
    return (
        (df["days_silent"] >= DAYS_SILENT_THRESHOLD) &
        (df["unpaid_months_recent"] >= UNPAID_MONTHS_THRESHOLD)
    )


def classify_root_cause(feature_df):
    """
    Apply the full decision tree to a feature DataFrame.

    This is a direct implementation of the tree documented at the top of
    this file — each F.when() branch corresponds to one box in that tree.

    Args:
        feature_df: Spark DataFrame from process_layer.build_feature_table(),
                     must contain: is_churned, days_silent,
                     unpaid_months_recent, tickets_last_30d_before_silence

    Returns:
        Spark DataFrame with an added "root_cause" column:
        one of "active", "network", "mixed", "behavioral"
    """
    network_signal    = is_network_signal(feature_df)
    behavioral_signal = is_behavioral_signal(feature_df)

    classified = feature_df.withColumn(
        "root_cause",

        # Rule 1 — not churned at all
        F.when(feature_df["is_churned"] == 0, "active")

        # Rule 2 — network signal present (checked FIRST — see module docstring
        # for why specific evidence must be checked before generic symptoms)
        .when(
            (feature_df["is_churned"] == 1) & network_signal,
            # Within the network branch: also check behavioral —
            # if BOTH fire, the cause is genuinely ambiguous -> "mixed"
            F.when(behavioral_signal, "mixed").otherwise("network")
        )

        # Rule 3 — behavioral signal present, no network signal
        .when(
            (feature_df["is_churned"] == 1) & behavioral_signal,
            "behavioral"
        )

        # Rule 4 — churned, but neither clean signal fired.
        # Default to "behavioral" rather than leaving it unclassified —
        # a churned customer with no ticket history and partial signals
        # is still more likely behavioral than network in the absence of
        # ticket evidence. This default is itself a modeling decision and
        # is called out explicitly here rather than left implicit.
        .when(feature_df["is_churned"] == 1, "behavioral")

        .otherwise("active")
    )

    return classified


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Classification Explainability
# ════════════════════════════════════════════════════════════════════════════

def add_explanation_column(classified_df):
    """
    Add a human-readable explanation for each classification.

    This is what separates a rule engine from a black box: every row can
    answer "why was I classified this way" in plain language, suitable
    for a churn-save team to act on directly — no model interpretation
    layer needed.
    """
    return classified_df.withColumn(
        "classification_reason",
        F.when(F.col("root_cause") == "active",
               "Active - usage within normal range")
         .when(F.col("root_cause") == "network",
               F.concat(
                   F.lit("Network - "),
                   F.col("tickets_last_30d_before_silence").cast("string"),
                   F.lit(" tickets in 30 days before going silent")
               ))
         .when(F.col("root_cause") == "mixed",
               F.concat(
                   F.lit("Mixed - "),
                   F.col("tickets_last_30d_before_silence").cast("string"),
                   F.lit(" network tickets AND "),
                   F.col("unpaid_months_recent").cast("string"),
                   F.lit(" unpaid months")
               ))
         .when(F.col("root_cause") == "behavioral",
               F.concat(
                   F.lit("Behavioral - silent "),
                   F.col("days_silent").cast("string"),
                   F.lit(" days, "),
                   F.col("unpaid_months_recent").cast("string"),
                   F.lit(" unpaid months")
               ))
         .otherwise("Unclassified")
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Classification Summary & Manifest
# ════════════════════════════════════════════════════════════════════════════

def build_classification_manifest(classified_df) -> dict:
    """
    Summarize classification results and document the rule thresholds used.

    Note on arpu_drop_pct: this feature is computed in process_layer.py and
    is available in the output table for Phase 6's deeper analysis, but is
    intentionally NOT a gating condition in this version of the tree — the
    four features used here (is_churned, days_silent, unpaid_months_recent,
    tickets_last_30d_before_silence) were sufficient to cleanly separate the
    three causes in testing. Adding arpu_drop_pct as a fifth gate risked
    over-fitting the rule set to the synthetic data's specific generation
    logic rather than the general telecom pattern. It remains in the output
    schema as a documented, available signal for future rule refinement.
    """
    counts = classified_df.groupBy("root_cause").count().collect()
    distribution = {row["root_cause"]: row["count"] for row in counts}

    total_churned = classified_df.filter(F.col("is_churned") == 1).count()

    manifest = {
        "classification_run":  datetime.now().strftime("%Y%m%d_%H%M%S"),
        "rule_thresholds": {
            "days_silent_threshold":   DAYS_SILENT_THRESHOLD,
            "unpaid_months_threshold": UNPAID_MONTHS_THRESHOLD,
            "tickets_threshold":       TICKETS_THRESHOLD
        },
        "rule_order": [
            "1. not churned -> active",
            "2. churned AND tickets>=3 in pre-silence window -> network (or mixed if also behavioral)",
            "3. churned AND silent>=60d AND unpaid>=2mo -> behavioral",
            "4. churned, no clean signal -> behavioral (default)"
        ],
        "total_customers":     classified_df.count(),
        "total_churned":       total_churned,
        "root_cause_distribution": distribution,
        "arpu_drop_pct_note":  (
            "Computed but not used as a gating condition in this rule "
            "version - available for Phase 6 deep-dive analysis"
        ),

        # Lineage continues the chain — this manifest's source is the
        # feature table from process_layer.py, and its output is consumed
        # by iceberg_writer.py for persistence
        "lineage": {
            "source":      "spark/process_layer.py (feature_df)",
            "produced_by": "spark/root_cause_engine.py",
            "consumed_by": "spark/iceberg_writer.py"
        }
    }

    manifest_path = CATALOG_PATH / f"root_cause_manifest_{manifest['classification_run']}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Classification manifest saved -> {manifest_path}")
    return manifest


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

def run_root_cause_classification(feature_df):
    """
    Full root cause classification pipeline.

    Args:
        feature_df: output of process_layer.run_feature_engineering()

    Returns:
        (classified_df, manifest): Spark DataFrame with root_cause and
        classification_reason columns added, and the summary manifest dict
    """
    print("=" * 65)
    print("  PHASE 4 — ROOT CAUSE CLASSIFICATION")
    print("=" * 65)

    print("\n🎯 Applying decision tree...")
    classified_df = classify_root_cause(feature_df)

    print("📝 Adding explanations...")
    classified_df = add_explanation_column(classified_df)
    classified_df = classified_df.cache()
    classified_df.count()  # Materialize

    print("📋 Building classification manifest...")
    manifest = build_classification_manifest(classified_df)

    print(f"\n✅ Classification complete")
    print(f"   Total customers: {manifest['total_customers']:,}")
    print(f"   Total churned:   {manifest['total_churned']:,}")
    print(f"   Distribution:")
    for cause, count in manifest["root_cause_distribution"].items():
        print(f"      {cause:<12} {count:,}")
    print(f"   Lineage -> produced_by: {manifest['lineage']['produced_by']}")
    print(f"   Lineage -> consumed_by: {manifest['lineage']['consumed_by']}")
    print("=" * 65)

    return classified_df, manifest


if __name__ == "__main__":
    # Standalone execution: runs process_layer first, then classifies
    import sys
    sys.path.insert(0, ".")
    from process_layer import run_feature_engineering

    spark, feature_df, _ = run_feature_engineering()
    classified_df, manifest = run_root_cause_classification(feature_df)
    classified_df.select(
        "customer_id", "is_churned", "days_silent", "unpaid_months_recent",
        "tickets_last_30d_before_silence", "root_cause", "classification_reason"
    ).show(10, truncate=False)
    spark.stop()
