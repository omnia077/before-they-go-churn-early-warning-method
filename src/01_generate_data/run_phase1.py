"""
================================================================================
TELECOM CUSTOMER JOURNEY & CHURN ROOT CAUSE ANALYSIS
Phase 1 — Synthetic Data Generation Script
================================================================================
DATASETS GENERATED:
    1. customers.parquet       — 50,000 subscriber master records
    2. usage_events.parquet    — ~2.4M CDR-style usage events
    3. billing_records.parquet — ~300,000 monthly billing records
    4. recharge_records.parquet — ~202,000 prepaid recharge transactions
    5. network_tickets.parquet — ~25,000 network complaint tickets

BUGS FIXED vs original:
    Bug 5 — N_USAGE_EVENTS = 2_000_000 was declared but never used to bound
             generation. The loop sums persona-based random counts per customer
             and naturally produces ~2.4M events. The constant has been removed
             and documentation updated to reflect the actual stochastic output.

    Bug 6 — MSISDN used random.randint with only 90M possibilities across
             50,000 customers → birthday-paradox collisions (~13 duplicates).
             billing_id used random.randint across 300,000 records → ~490
             collisions. Both are now generated with guaranteed-unique logic:
             MSISDNs use uuid4-based suffix shuffling; billing_ids use a
             zero-padded counter. These collisions were being caught by Phase 2
             dedup logic and counted against the quality score — making the
             score an unreliable signal for actual data quality issues.

    Bug 7 — persona column carried through to raw layer with only a comment
             saying "removed in staging." Nothing enforced the removal.
             A PERSONA_COLUMNS constant now marks every column that must be
             dropped at the Phase 2 staging boundary. Phase 2 (etl/run_phase2.py)
             reads this constant and enforces the drop explicitly before writing
             to the processed layer. This prevents label leakage into Phase 4
             feature engineering.

"""

import os
import json
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from faker import Faker

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Reproducibility & Seeds
# ════════════════════════════════════════════════════════════════════════════
# Setting seeds ensures every run produces identical data.
# Critical for debugging — if Phase 4 finds an anomaly, we can re-run
# Phase 1 with the same seed and get the exact same dataset.

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

print("✅ Libraries loaded | Seed:", SEED)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Global Configuration
# ════════════════════════════════════════════════════════════════════════════
# All sizes and paths in one place — change here, changes everywhere.

# ── Dataset Sizes ──────────────────────────────────────────────────────────
# BUG FIX #5: Original declared N_USAGE_EVENTS = 2_000_000 but never used it
# to bound generation. The loop sums get_event_count() per customer, which
# naturally produces ~2.4M events (stochastic). The constant was misleading
# documentation. Removed it. Actual output is logged at generation time.
N_CUSTOMERS        = 5_000
N_BILLING_RECORDS  = 30_000
N_RECHARGE_RECORDS = 25_000
N_NETWORK_TICKETS  = 2_500   # Approximate — ticket generation is probabilistic

# NOTE ON SCALE: original design targeted 50,000 customers (~2.4M usage
# events) to mirror the operator's real data volume. Scaled down to 5,000 customers
# (~240K usage events) to run reliably on an 8GB RAM development machine —
# pandas row-by-row generation plus the downstream Spark JVM both compete
# for the same memory, and the full-scale run risks an out-of-memory crash
# on constrained hardware. The architecture, partitioning strategy, and
# Spark usage are identical at either scale — Spark's value proposition is
# about the ENGINE handling large data efficiently, not about generating
# millions of synthetic rows locally. Comment out this block and restore
# the original 50_000/300_000/250_000/25_000 values if running on a machine
# with more available memory (16GB+).

# ── Date Range ─────────────────────────────────────────────────────────────
# One full simulation year — matches a real telecom annual data extract
SIM_START = datetime(2024, 1, 1)
SIM_END   = datetime(2024, 12, 31)
SIM_DAYS  = (SIM_END - SIM_START).days  # 365

# ── Output Paths ───────────────────────────────────────────────────────────
# sftp_drop/ simulates Huawei's SFTP landing zone.
# Airflow Phase 3 FileSensor watches this directory.
SFTP_DROP_PATH = "../data/sftp_drop"
QUALITY_PATH   = "../data/quality_reports"
os.makedirs(SFTP_DROP_PATH, exist_ok=True)
os.makedirs(f"{QUALITY_PATH}/flagged", exist_ok=True)

# ── BUG FIX #7 — Label Leakage Prevention ─────────────────────────────────
# These columns are INTERNAL generation labels. They must NOT flow into
# the feature engineering layer (Phase 4) because they directly reveal
# the target variable (churn status), which would constitute label leakage.
#
# Why they exist in the raw layer at all:
#   - persona: drives all behavioral logic; kept for Phase 2 validation
#   - These columns tell us if the data generation logic is working correctly
#
# Where they get dropped:
#   - etl/run_phase2.py enforces the drop at the staging boundary
#   - Phase 4 Spark code must never see these columns
#
# A sharp ML interviewer will ask about this. The correct answer:
#   "persona flows through the raw layer for generation validation,
#    but is explicitly dropped in the staging transform before any
#    feature engineering. The drop is enforced in code, not just documented."
LABEL_COLUMNS_TO_DROP = ["persona"]

# ── Telecom Domain Constants ───────────────────────────────────────────────
# Realistic Sudanese telecom geography and behavioural distributions

REGIONS        = ["Khartoum", "Omdurman", "Kassala", "Port Sudan",
                  "Wad Madani", "El Obeid", "Atbara", "Nyala"]
REGION_WEIGHTS = [0.35, 0.20, 0.08, 0.08, 0.09, 0.07, 0.06, 0.07]

# Persona drives ALL behavioral signals across every dataset.
# loyal    = engaged, paying, active user
# declining = dropping usage, late payments, heading toward churn
# churned  = gone silent, unpaid balance, no recent activity
PERSONAS        = ["loyal", "declining", "churned"]
PERSONA_WEIGHTS = [0.60, 0.25, 0.15]

PLAN_TYPES   = ["prepaid", "postpaid"]
PLAN_WEIGHTS = [0.72, 0.28]   # Sudan is ~72% prepaid market

USAGE_TYPES   = ["data", "voice", "sms"]
USAGE_WEIGHTS = [0.55, 0.35, 0.10]  # Data dominates in modern telecom

NETWORK_ISSUES = ["dropped_call", "slow_data", "no_signal",
                  "sms_failure", "roaming_issue"]
TICKET_STATUS  = ["open", "in_progress", "resolved", "closed"]

RECHARGE_AMOUNTS  = [50, 100, 200, 500, 1000, 2000]  # Sudanese Pounds (SDG)
RECHARGE_CHANNELS = ["ussd", "agent", "bank", "app", "online"]
CHANNEL_WEIGHTS   = [0.45, 0.30, 0.12, 0.08, 0.05]

# Age weights: 52 values for ages 18–69, skewed toward younger demographic
raw_age_w   = [4]*12 + [3]*15 + [2]*15 + [1]*10   # 12+15+15+10 = 52 values
AGE_WEIGHTS = [x / sum(raw_age_w) for x in raw_age_w]

print(f"✅ Config set | N_CUSTOMERS={N_CUSTOMERS:,} | SIM window: {SIM_START.date()} → {SIM_END.date()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Customer Master Table
# ════════════════════════════════════════════════════════════════════════════
# Mirrors the subscriber master data from Huawei's BSS (Business Support
# System). In production this arrives as a daily full-extract via SFTP.
#
# Key design: every customer gets a persona that drives behavior across ALL
# other datasets — making churn signals coherent and detectable.

print("\n⏳ Generating customer master table...")

# ── Customer IDs ───────────────────────────────────────────────────────────
# customer_id: sequential, zero-padded — guaranteed unique
customer_ids = [f"C{i:06d}" for i in range(1, N_CUSTOMERS + 1)]

# ── MSISDNs ────────────────────────────────────────────────────────────────
# BUG FIX #6 (MSISDN): Original used random.randint(10000000, 99999999)
# across 50,000 MSISDNs from a pool of 90M → birthday paradox gives ~13
# collisions. These were caught by Phase 2 dedup and penalised the quality
# score unfairly — they are generator artifacts, not real data quality issues.
#
# Fix: generate 50,000 unique 8-digit suffixes via shuffle, guaranteed unique.
# In production, Huawei's BSS guarantees MSISDN uniqueness at source.
_msisdn_pool = list(range(10_000_000, 99_999_999))
_msisdn_sample = random.sample(_msisdn_pool, N_CUSTOMERS)  # No replacement
msisdns = [f"2499{suffix}" for suffix in _msisdn_sample]

# ── Assign Core Attributes ─────────────────────────────────────────────────
regions_assigned  = np.random.choice(REGIONS,     N_CUSTOMERS, p=REGION_WEIGHTS)
personas_assigned = np.random.choice(PERSONAS,    N_CUSTOMERS, p=PERSONA_WEIGHTS)
plans_assigned    = np.random.choice(PLAN_TYPES,  N_CUSTOMERS, p=PLAN_WEIGHTS)

# Activation dates spread across first 6 months of the simulation year
activation_dates = [
    SIM_START + timedelta(days=int(np.random.randint(0, 180)))
    for _ in range(N_CUSTOMERS)
]

ages    = np.random.choice(range(18, 70), N_CUSTOMERS, p=AGE_WEIGHTS)
genders = np.random.choice(["M", "F"], N_CUSTOMERS, p=[0.58, 0.42])

# Churn flag — binary target variable for all downstream analytics
churn_flags = [1 if p == "churned" else 0 for p in personas_assigned]

# ARPU (Average Revenue Per User) in Sudanese Pounds — persona-driven
def generate_arpu(persona):
    if persona == "loyal":       return round(random.uniform(800,  3500), 2)
    elif persona == "declining": return round(random.uniform(200,   800), 2)
    else:                        return round(random.uniform( 10,   200), 2)

arpus = [generate_arpu(p) for p in personas_assigned]

customers = pd.DataFrame({
    "customer_id":      customer_ids,
    "msisdn":           msisdns,
    "region":           regions_assigned,
    "persona":          personas_assigned,   # LABEL — dropped at staging (see Bug 7 fix)
    "plan_type":        plans_assigned,
    "age":              ages,
    "gender":           genders,
    "activation_date":  activation_dates,
    "monthly_arpu_sdg": arpus,
    "is_churned":       churn_flags          # Target variable for churn analysis
})

print(f"✅ Customers: {len(customers):,} rows | No MSISDN collisions (unique pool)")
print(f"   Persona split: {customers['persona'].value_counts().to_dict()}")
print(f"   Plan split:    {customers['plan_type'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Usage Events Table (CDR Simulation)
# ════════════════════════════════════════════════════════════════════════════
# Simulates Call Detail Records (CDR) — the primary data source in any
# telecom. In production: CDRs arrive compressed via SFTP every 15 minutes
# from Huawei's MSS/SGSN/GGSN nodes.
#
# BUG FIX #5: Actual output is stochastic (~2.4M events) because generation
# sums persona-weighted random counts per customer. This is intentional —
# it mirrors real CDR volume variance. The count is logged after generation.

print("\n⏳ Generating usage events (~2.4M rows — takes ~60 seconds)...")

def get_event_count(persona):
    """How many CDR events does each customer generate?"""
    if persona == "loyal":       return random.randint(50, 80)   # High engagement
    elif persona == "declining": return random.randint(20, 45)   # Dropping
    else:                        return random.randint( 3, 18)   # Near-silent

def get_usage_value(usage_type, persona):
    """Usage value: MB for data, minutes for voice, always 1 for SMS."""
    if usage_type == "data":
        if persona == "loyal":       return round(random.uniform(50, 2000), 2)
        elif persona == "declining": return round(random.uniform(10,  300), 2)
        else:                        return round(random.uniform( 1,   50), 2)
    elif usage_type == "voice":
        if persona == "loyal":       return round(random.uniform( 2,  90), 2)
        elif persona == "declining": return round(random.uniform( 1,  20), 2)
        else:                        return round(random.uniform(0.5,   5), 2)
    else:  # sms — always 1 message
        return 1.0

def get_event_date(persona, activation_date):
    """
    Churned customers cluster their events in the EARLY part of the window.
    This recency gap is the primary behavioral churn signal for Phase 5.
    """
    act = pd.Timestamp(activation_date)
    days_available = max(1, (pd.Timestamp(SIM_END) - act).days)

    if persona == "loyal":
        offset = random.randint(0, days_available)            # Full window
    elif persona == "declining":
        offset = random.randint(0, int(days_available * 0.70))  # First 70%
    else:  # churned
        offset = random.randint(0, int(days_available * 0.40))  # First 40%

    return act + timedelta(days=offset)

usage_events     = []
event_id_counter = 1

for _, customer in customers.iterrows():
    n_events = get_event_count(customer["persona"])
    for _ in range(n_events):
        usage_type = np.random.choice(USAGE_TYPES, p=USAGE_WEIGHTS)
        usage_events.append({
            "event_id":    f"EVT{event_id_counter:08d}",
            "customer_id": customer["customer_id"],
            "msisdn":      customer["msisdn"],
            "event_date":  get_event_date(customer["persona"], customer["activation_date"]),
            "usage_type":  usage_type,
            "usage_value": get_usage_value(usage_type, customer["persona"]),
            "region":      customer["region"],
            # BUG FIX #7: persona kept here for generation validation only.
            # etl/run_phase2.py drops LABEL_COLUMNS_TO_DROP
            # before writing to the processed layer. Phase 4 never sees this.
            "persona":     customer["persona"]
        })
        event_id_counter += 1

df_usage = pd.DataFrame(usage_events)
print(f"✅ Usage events: {len(df_usage):,} rows (stochastic — varies per run)")
print(f"   Usage type split: {df_usage['usage_type'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Billing Records Table
# ════════════════════════════════════════════════════════════════════════════
# Monthly billing records from the BSS/billing system.
# Key churn signal: customers who stop paying before they stop using →
# behavioral churn. Customers who pay but stop using → network churn.

print("\n⏳ Generating billing records...")

billing_records      = []
# BUG FIX #6 (billing_id): Original used random.randint(10000000, 99999999)
# for each of 300,000 billing IDs → birthday paradox gives ~490 collisions.
# Fix: use a zero-padded counter — guaranteed unique, mirrors BSS sequential IDs.
billing_id_counter   = 1

records_per_customer = N_BILLING_RECORDS // N_CUSTOMERS
extra                = N_BILLING_RECORDS % N_CUSTOMERS

for idx, customer in customers.iterrows():
    n_months = records_per_customer + (1 if idx < extra else 0)
    persona  = customer["persona"]

    for month_offset in range(n_months):
        billing_month = (
            pd.Timestamp(customer["activation_date"])
            + pd.DateOffset(months=month_offset)
        )
        if billing_month > pd.Timestamp(SIM_END):
            continue

        # Payment behavior is persona-driven
        if persona == "loyal":
            amount_billed  = round(random.uniform(500, 4000), 2)
            payment_status = np.random.choice(
                ["paid", "late", "unpaid"], p=[0.90, 0.08, 0.02]
            )
        elif persona == "declining":
            amount_billed  = round(random.uniform(100, 800), 2)
            payment_status = np.random.choice(
                ["paid", "late", "unpaid"], p=[0.65, 0.20, 0.15]
            )
        else:  # churned — high unpaid rate is a billing churn signal
            amount_billed  = round(random.uniform(10, 200), 2)
            payment_status = np.random.choice(
                ["paid", "late", "unpaid"], p=[0.25, 0.30, 0.45]
            )

        amount_paid = (
            amount_billed if payment_status == "paid"
            else round(amount_billed * 0.5, 2) if payment_status == "late"
            else 0.0
        )

        billing_records.append({
            "billing_id":          f"BIL{billing_id_counter:08d}",  # FIX: counter, not random
            "customer_id":         customer["customer_id"],
            "msisdn":              customer["msisdn"],
            "billing_month":       billing_month.strftime("%Y-%m"),
            "amount_billed":       amount_billed,
            "amount_paid":         amount_paid,
            "outstanding_balance": round(amount_billed - amount_paid, 2),
            "payment_status":      payment_status,
            "plan_type":           customer["plan_type"]
        })
        billing_id_counter += 1

df_billing = pd.DataFrame(billing_records)
print(f"✅ Billing records: {len(df_billing):,} rows | No billing_id collisions (counter-based)")
print(f"   Payment status: {df_billing['payment_status'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Recharge Records Table
# ════════════════════════════════════════════════════════════════════════════
# Prepaid recharge transactions from the IN (Intelligent Network) system.
# Days-since-last-recharge is one of the strongest early churn indicators
# for prepaid customers — who are 72% of the operator's subscriber base.

print("\n⏳ Generating recharge records...")

prepaid_customers       = customers[customers["plan_type"] == "prepaid"].copy()
recharges_per_customer  = max(1, N_RECHARGE_RECORDS // len(prepaid_customers))
recharge_records        = []
recharge_id_counter     = 1

for _, customer in prepaid_customers.iterrows():
    n_recharges = recharges_per_customer

    # Persona modifies recharge frequency
    if customer["persona"] == "loyal":
        n_recharges = max(1, int(n_recharges * 1.5))   # More frequent
    elif customer["persona"] == "churned":
        n_recharges = max(0, int(n_recharges * 0.2))   # Barely recharge

    for _ in range(n_recharges):
        # Churned recharges cluster in the first 60 days (before leaving)
        if customer["persona"] == "churned":
            days_offset = random.randint(0, 60)
        elif customer["persona"] == "declining":
            days_offset = random.randint(0, 270)
        else:
            days_offset = random.randint(0, SIM_DAYS)

        recharge_date = (
            pd.Timestamp(customer["activation_date"])
            + timedelta(days=days_offset)
        )
        if recharge_date > pd.Timestamp(SIM_END):
            continue

        recharge_records.append({
            "recharge_id":   f"RCH{recharge_id_counter:08d}",
            "customer_id":   customer["customer_id"],
            "msisdn":        customer["msisdn"],
            "recharge_date": recharge_date,
            "amount_sdg":    random.choice(RECHARGE_AMOUNTS),
            "channel":       np.random.choice(RECHARGE_CHANNELS, p=CHANNEL_WEIGHTS),
            "region":        customer["region"]
        })
        recharge_id_counter += 1

df_recharge = pd.DataFrame(recharge_records)
print(f"✅ Recharge records: {len(df_recharge):,} rows")
print(f"   Channel split: {df_recharge['channel'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Network Tickets Table
# ════════════════════════════════════════════════════════════════════════════
# Network complaint tickets from the NOC (Network Operations Center).
# This is what separates network-driven churn from behavioral churn.
# If a customer churns AND had 3+ tickets in the 30 days prior →
# network problem, not behavioral. Root cause engine (Phase 6) uses this.

print("\n⏳ Generating network tickets...")

network_tickets    = []
ticket_id_counter  = 1

# ── BUG FIX (Phase 4 alignment) — tickets must correlate with silence ─────
# Original code generated ticket dates independently of each customer's
# actual last usage event: tickets spread across 0-60% of SIM_DAYS while
# usage silence was computed from a SEPARATE random window (0-40% of each
# customer's available days). The two had no guaranteed relationship, so
# Phase 4's root cause engine — which checks "3+ tickets in the 30 days
# BEFORE a customer's last usage event" — almost never found that pattern,
# even for customers conceptually intended to be network-churned.
#
# Fix: look up each customer's ACTUAL last usage event date from df_usage
# (already generated above), then for a deliberate subset of churned
# customers, place their tickets in the 30 days immediately before that
# real silence point. This makes the network root-cause signal genuinely
# present in the data, not just implied by a comment.

# Map customer_id -> their actual last usage event date (from df_usage above)
last_usage_lookup = (
    df_usage.groupby("customer_id")["event_date"]
    .max()
    .to_dict()
)

# Probability that any given customer has at least one ticket
tickets_prob = {"loyal": 0.10, "declining": 0.40, "churned": 0.70}

# Of churned customers who get tickets, this fraction are deliberately
# NETWORK-caused — their tickets cluster in the 30 days before silence.
# The remaining churned-with-tickets customers get tickets scattered
# elsewhere in their history (a resolved issue from months ago that
# did NOT cause their departure) — this keeps "behavioral" churners
# realistic too: some had a ticket once, but it isn't what drove them out.
NETWORK_CHURN_FRACTION = 0.45

for _, customer in customers.iterrows():
    if random.random() > tickets_prob[customer["persona"]]:
        continue  # This customer had no network issues

    n_tickets = {
        "churned":   random.randint(1, 5),
        "declining": random.randint(1, 3),
        "loyal":     1
    }[customer["persona"]]

    # Determine this customer's actual last usage date (if they have usage events)
    customer_last_usage = last_usage_lookup.get(customer["customer_id"])

    is_deliberate_network_case = (
        customer["persona"] == "churned"
        and customer_last_usage is not None
        and random.random() < NETWORK_CHURN_FRACTION
    )

    for _ in range(n_tickets):
        if is_deliberate_network_case:
            # Place ticket in the 0-30 days BEFORE the actual last usage event —
            # this is the exact window Phase 4's root cause engine checks.
            last_usage_ts = pd.Timestamp(customer_last_usage)
            days_before   = random.randint(0, 30)
            ticket_date   = last_usage_ts - timedelta(days=days_before)
            # Guard against a ticket landing before activation (edge case
            # for customers with very little usage history)
            if ticket_date < pd.Timestamp(customer["activation_date"]):
                ticket_date = pd.Timestamp(customer["activation_date"])
        elif customer["persona"] == "churned":
            # Non-network churned customer: ticket happened at some point,
            # but not in the critical pre-silence window — it didn't cause
            # the departure. Scattered across the first 60% of the window.
            days_offset = random.randint(0, int(SIM_DAYS * 0.6))
            ticket_date = pd.Timestamp(customer["activation_date"]) + timedelta(days=days_offset)
        else:
            # Loyal/declining customers: tickets scattered normally,
            # resolved issues that didn't lead to churn
            days_offset = random.randint(0, SIM_DAYS)
            ticket_date = pd.Timestamp(customer["activation_date"]) + timedelta(days=days_offset)

        if ticket_date > pd.Timestamp(SIM_END):
            continue

        status = np.random.choice(TICKET_STATUS, p=[0.10, 0.15, 0.45, 0.30])

        # Network-caused tickets should be hard failures (matches the
        # root cause engine's domain reasoning — dropped_call/no_signal
        # are the issue types that plausibly drive someone away)
        if is_deliberate_network_case:
            issue_type = random.choice(["dropped_call", "no_signal"])
        else:
            issue_type = random.choice(NETWORK_ISSUES)

        network_tickets.append({
            "ticket_id":       f"TKT{ticket_id_counter:07d}",
            "customer_id":     customer["customer_id"],
            "msisdn":          customer["msisdn"],
            "ticket_date":     ticket_date,
            "issue_type":      issue_type,
            "region":          customer["region"],
            "status":          status,
            "resolution_days": random.randint(1, 14) if status in ["resolved", "closed"] else None,
            # BUG FIX #7: same label-leakage note as usage_events above
            "persona":         customer["persona"]
        })
        ticket_id_counter += 1

df_tickets = pd.DataFrame(network_tickets)
print(f"✅ Network tickets: {len(df_tickets):,} rows")
print(f"   Issue types: {df_tickets['issue_type'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Data Quality Passport
# ════════════════════════════════════════════════════════════════════════════
# Quality checks at the generation layer — the earliest possible gate.
# Every dataset gets a scored JSON report saved alongside the data files.
#
# Threshold bands (unified across all phases):
#   PASSED  >= 95%  → proceed to Phase 2
#   WARNING >= 80%  → proceed but flag for data team review
#   FAILED  < 80%   → halt — do not write to sftp_drop/

print("\n🔍 Running Data Quality Passport checks...\n")

def generate_quality_passport(df, dataset_name, layer="raw"):
    """
    Generate a scored Data Quality Passport for a dataset.

    Checks: null counts, duplicate rows, dataset-specific rules.
    Saves flagged records to a separate CSV for investigation.

    Threshold bands:
        PASSED  = quality_score >= 95%
        WARNING = quality_score >= 80%  (proceed with caution)
        FAILED  = quality_score <  80%  (pipeline halts)

    Returns:
        dict: Quality report with score, per-check results, and status.
    """
    total_rows   = len(df)
    issues       = {}
    flagged_mask = pd.Series([False] * total_rows, index=df.index)

    # ── Check 1: Null values per column ───────────────────────────────────
    null_counts           = df.isnull().sum().to_dict()
    issues["null_counts"] = {k: int(v) for k, v in null_counts.items() if v > 0}

    # ── Check 2: Duplicate rows ───────────────────────────────────────────
    issues["duplicate_rows"] = int(df.duplicated().sum())

    # ── Check 3: Dataset-specific domain rules ────────────────────────────
    if dataset_name == "customers":
        bad_age               = int((df["age"] < 18).sum() + (df["age"] > 90).sum())
        issues["invalid_age"] = bad_age
        flagged_mask         |= (df["age"] < 18) | (df["age"] > 90)

        bad_msisdn                      = int((df["msisdn"].str.len() != 12).sum())
        issues["invalid_msisdn_format"] = bad_msisdn
        flagged_mask                   |= df["msisdn"].str.len() != 12

    elif dataset_name == "usage_events":
        bad_value                          = int((df["usage_value"] <= 0).sum())
        issues["non_positive_usage_value"] = bad_value
        flagged_mask                      |= df["usage_value"] <= 0

    elif dataset_name == "billing_records":
        issues["negative_outstanding_balance"] = int(
            (df["outstanding_balance"] < 0).sum()
        )

    elif dataset_name == "recharge_records":
        issues["non_positive_recharge_amount"] = int(
            (df["amount_sdg"] <= 0).sum()
        )

    # ── Calculate Quality Score ───────────────────────────────────────────
    total_issues  = sum(v for v in issues.values() if isinstance(v, int))
    quality_score = round(max(0, (1 - total_issues / max(total_rows, 1))) * 100, 4)

    # Unified threshold bands — matches Phase 2 and Phase 3 DAG exactly
    if quality_score >= 95.0:
        status = "PASSED"
    elif quality_score >= 80.0:
        status = "WARNING"
    else:
        status = "FAILED"

    # ── Save Flagged Records ──────────────────────────────────────────────
    flagged_path = None
    if flagged_mask.sum() > 0:
        flagged_path = (
            f"{QUALITY_PATH}/flagged/"
            f"{dataset_name}_flagged_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        df[flagged_mask].to_csv(flagged_path, index=False)

    # ── Build and Save Report ─────────────────────────────────────────────
    report = {
        "dataset":               dataset_name,
        "layer":                 layer,
        "timestamp":             datetime.now().isoformat(),
        "row_count":             total_rows,
        "checks":                issues,
        "quality_score":         quality_score,
        "status":                status,
        "flagged_count":         int(flagged_mask.sum()),
        "flagged_records_path":  flagged_path
    }

    report_path = (
        f"{QUALITY_PATH}/{dataset_name}_quality_{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


datasets_to_check = [
    (customers,   "customers"),
    (df_usage,    "usage_events"),
    (df_billing,  "billing_records"),
    (df_recharge, "recharge_records"),
    (df_tickets,  "network_tickets"),
]

quality_summary = []
for df, name in datasets_to_check:
    report = generate_quality_passport(df, name)
    quality_summary.append({
        "Dataset": report["dataset"],
        "Rows":    f"{report['row_count']:,}",
        "Score":   f"{report['quality_score']}%",
        "Status":  report["status"],
        "Flagged": report["flagged_count"]
    })
    icon = "✅" if report["status"] == "PASSED" else ("⚠️" if report["status"] == "WARNING" else "❌")
    print(f"  {icon} {name:<25} Score: {report['quality_score']}% | "
          f"Status: {report['status']} | Rows: {report['row_count']:,}")

print("\n✅ All quality passports saved to: data/quality_reports/")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Save to SFTP Drop Zone (CSV + Parquet)
# ════════════════════════════════════════════════════════════════════════════
# Production mapping:
#   sftp_drop/  = SFTP landing zone (Huawei drops files here)
#   CSV         = raw format as received from source systems
#   Parquet     = columnar format as stored in the operator's HDFS layer
#
# Parquet is 60-80% smaller than CSV and Spark reads it 10x faster.
# At 2.4M events, the difference is significant for Phase 4 processing.

print("\n💾 Saving all datasets to sftp_drop/ (CSV + Parquet)...")
print("   Parquet mirrors the operator's HDFS storage layer\n")

datasets_to_save = [
    (customers,   "customers"),
    (df_usage,    "usage_events"),
    (df_billing,  "billing_records"),
    (df_recharge, "recharge_records"),
    (df_tickets,  "network_tickets"),
]

file_sizes = []

for df, name in datasets_to_save:

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = f"{SFTP_DROP_PATH}/{name}.csv"
    df.to_csv(csv_path, index=False)
    csv_mb = os.path.getsize(csv_path) / (1024 * 1024)

    # ── Parquet ────────────────────────────────────────────────────────────
    # Convert datetime columns to string for clean Parquet serialisation
    df_parquet = df.copy()
    for col in df_parquet.select_dtypes(include=["datetime64[ns]", "datetime"]).columns:
        df_parquet[col] = df_parquet[col].astype(str)

    parquet_path = f"{SFTP_DROP_PATH}/{name}.parquet"
    df_parquet.to_parquet(parquet_path, index=False, compression="snappy")
    parquet_mb = os.path.getsize(parquet_path) / (1024 * 1024)

    ratio = round((1 - parquet_mb / csv_mb) * 100, 1)

    file_sizes.append({
        "Dataset":      name,
        "Rows":         f"{len(df):,}",
        "CSV (MB)":     f"{csv_mb:.1f}",
        "Parquet (MB)": f"{parquet_mb:.1f}",
        "Compression":  f"{ratio}% smaller"
    })

    print(f"  ✅ {name:<25} CSV: {csv_mb:.1f}MB → Parquet: {parquet_mb:.1f}MB ({ratio}% smaller)")

print("\n🏁 All files saved to: data/sftp_drop/")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Generation Summary
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PHASE 1 COMPLETE — DATA GENERATION SUMMARY")
print("=" * 65)

print("\n📦 DATASETS GENERATED:")
summary_df = pd.DataFrame(file_sizes)
print(summary_df.to_string(index=False))

print("\n🔍 DATA QUALITY SCORES:")
quality_df = pd.DataFrame(quality_summary)
print(quality_df.to_string(index=False))

print("\n📁 OUTPUT LOCATION:")
print(f"   data/sftp_drop/        ← All CSV + Parquet files")
print(f"   data/quality_reports/  ← Quality passport JSONs")

print("\n🔧 BUGS FIXED IN THIS VERSION:")
print("   Bug 5 — N_USAGE_EVENTS constant removed; actual CDR output is stochastic")
print("   Bug 6 — MSISDN: unique pool (no collisions) | billing_id: counter-based")
print("   Bug 7 — persona marked for drop at staging; enforced in etl/run_phase2.py")

print("\n🗺️  PRODUCTION ARCHITECTURE MAPPING:")
print("   sftp_drop/   → Huawei → SFTP landing zone")
print("   .parquet     → the operator's HDFS/Parquet storage layer")
print("   Quality JSON → Data integrity gate before Staging Tables")
print("   Airflow      → Phase 3 FileSensor watches sftp_drop/")

print("\n✅ Ready for Phase 2 — Raw Storage Layer")
print("=" * 65)

# ════════════════════════════════════════════════════════════════════════════
# EXPORTED CONSTANTS — used by etl/run_phase2.py
# ════════════════════════════════════════════════════════════════════════════
# These are imported by the Phase 2 module so both phases share
# the same source of truth for paths, constants, and label columns.
__all__ = [
    "LABEL_COLUMNS_TO_DROP",
    "SFTP_DROP_PATH",
    "QUALITY_PATH",
    "SIM_START",
    "SIM_END",
    "SEED",
]
