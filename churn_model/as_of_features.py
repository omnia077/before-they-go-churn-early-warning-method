"""
Point-in-time (as-of) churn features in PySpark.

Every feature for cutoff C uses only rows dated on or before C. Windows are
(C - days_from, C - days_to], so 30 days = (30, 0] and the prior 60 days
= (90, 30].

This module is self-contained on purpose: 04_classify_spark/process_layer.py runs
side effects at import (creates ../data dirs, opens a log file), so importing it
from here would depend on the working directory. Its four root-cause features
also accept `as_of` now, but the ML feature set below is built here.

Leakage rules honoured here:
  * Only the columns listed in INPUT_COLUMNS are read. `customers.is_churned`
    exists in raw/ and is deliberately never selected.
  * `network_tickets.status` / `resolution_days` are end-of-simulation values,
    not as-of values, so no "open tickets" feature is built from them.
  * `msisdn`, `gender`, `age` are not used as features.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

INPUT_COLUMNS = {
    "customers": ["customer_id", "region", "plan_type", "activation_date", "monthly_arpu_sdg"],
    "usage_events": ["customer_id", "event_date", "usage_type", "usage_value"],
    "recharge_records": ["customer_id", "recharge_date", "amount_sdg"],
    "network_tickets": ["customer_id", "ticket_date", "issue_type", "region"],
    "billing_records": ["customer_id", "billing_month", "amount_billed", "payment_status"],
}
HARD_ISSUES = ["dropped_call", "no_signal", "slow_data"]

FEATURE_COLUMNS = [
    "use_30d", "use_prev60d", "use_trend", "data_mb_30d", "voice_min_30d", "days_since_use",
    "days_since_rech", "rech_30d", "rech_90d", "rech_amt_90d", "rech_gap_ratio",
    "tix_30d", "tix_90d", "hard_tix_30d", "region_tix_rate_30d",
    "unpaid_last3", "late_last3", "bill_ratio",
    "tenure_days", "tenure_lt90", "is_prepaid", "monthly_arpu_sdg",
]


def get_spark(app="churn_as_of_features"):
    return (SparkSession.builder.appName(app).master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8").getOrCreate())


def load_raw(spark, raw_path):
    return {name: spark.read.parquet(f"{raw_path}/{name}").select(*cols)
            for name, cols in INPUT_COLUMNS.items()}


def as_of_window(df, date_col, as_of, days_from, days_to):
    """Rows with as_of - days_from < date <= as_of - days_to."""
    d = F.to_date(F.col(date_col))
    hi = F.date_sub(F.lit(as_of).cast("date"), days_to)
    lo = F.date_sub(F.lit(as_of).cast("date"), days_from)
    return df.filter((d > lo) & (d <= hi))


def _upto(df, date_col, as_of):
    return df.filter(F.to_date(F.col(date_col)) <= F.lit(as_of).cast("date"))


def _count(df, name):
    return df.groupBy("customer_id").agg(F.count("*").alias(name))


def build_as_of_features(tables, as_of, cfg):
    """One row per customer activated on or before `as_of`, with FEATURE_COLUMNS."""
    fc = cfg["features"]
    C = F.lit(as_of).cast("date")
    cust = (tables["customers"]
            .withColumn("activation_date", F.to_date("activation_date"))
            .filter(F.col("activation_date") <= C))

    usage = _upto(tables["usage_events"], "event_date", as_of)
    rech = _upto(tables["recharge_records"], "recharge_date", as_of)
    tix = _upto(tables["network_tickets"], "ticket_date", as_of)
    bill = tables["billing_records"].filter(F.col("billing_month") <= as_of[:7])

    # ---- usage
    u30 = as_of_window(usage, "event_date", as_of, 30, 0)
    u60 = as_of_window(usage, "event_date", as_of, 90, 30)
    use_30d = _count(u30, "use_30d")
    use_prev60d = _count(u60, "use_prev60d")
    data_mb = (u30.filter(F.col("usage_type") == "data").groupBy("customer_id")
               .agg(F.sum("usage_value").alias("data_mb_30d")))
    voice_min = (u30.filter(F.col("usage_type") == "voice").groupBy("customer_id")
                 .agg(F.sum("usage_value").alias("voice_min_30d")))
    last_use = (usage.groupBy("customer_id")
                .agg(F.datediff(C, F.max(F.to_date("event_date"))).alias("days_since_use")))

    # ---- recharge
    r90 = as_of_window(rech, "recharge_date", as_of, 90, 0)
    rech_30d = _count(as_of_window(rech, "recharge_date", as_of, 30, 0), "rech_30d")
    rech_90d = _count(r90, "rech_90d")
    rech_amt = r90.groupBy("customer_id").agg(F.mean("amount_sdg").alias("rech_amt_90d"))
    r180 = as_of_window(rech, "recharge_date", as_of, 180, 0)
    gaps = (r180.groupBy("customer_id")
            .agg(F.count("*").alias("_n"),
                 F.datediff(F.max(F.to_date("recharge_date")),
                            F.min(F.to_date("recharge_date"))).alias("_span"))
            .withColumn("_mean_gap", F.when(F.col("_n") >= 2, F.col("_span") / (F.col("_n") - 1)))
            .select("customer_id", "_mean_gap"))
    last_rech = (rech.groupBy("customer_id")
                 .agg(F.datediff(C, F.max(F.to_date("recharge_date"))).alias("_since_rech")))

    # ---- network tickets
    t30 = as_of_window(tix, "ticket_date", as_of, 30, 0)
    tix_30d = _count(t30, "tix_30d")
    tix_90d = _count(as_of_window(tix, "ticket_date", as_of, 90, 0), "tix_90d")
    hard = _count(t30.filter(F.col("issue_type").isin(HARD_ISSUES)), "hard_tix_30d")

    # regional ticket rate: tickets per 1,000 subscribers active in the region
    lb = fc["region_active_lookback_days"]
    active = (as_of_window(usage, "event_date", as_of, lb, 0)
              .select("customer_id").distinct()
              .join(cust.select("customer_id", "region"), "customer_id")
              .groupBy("region").agg(F.count("*").alias("_active")))
    reg_t = t30.groupBy("region").agg(F.count("*").alias("_reg_tix"))
    reg_rate = (active.join(reg_t, "region", "left").fillna({"_reg_tix": 0})
                .withColumn("region_tix_rate_30d", F.col("_reg_tix") / F.col("_active") * 1000)
                .select("region", "region_tix_rate_30d"))

    # ---- billing (last 3 bills up to the cutoff month)
    w = Window.partitionBy("customer_id").orderBy(F.col("billing_month").desc())
    b3 = bill.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= 3)
    unpaid = b3.groupBy("customer_id").agg(
        F.sum(F.when(F.col("payment_status") == "unpaid", 1).otherwise(0)).alias("unpaid_last3"),
        F.sum(F.when(F.col("payment_status") == "late", 1).otherwise(0)).alias("late_last3"),
        F.avg("amount_billed").alias("_avg3"))
    latest = b3.filter(F.col("rn") == 1).select("customer_id", F.col("amount_billed").alias("_latest"))
    billf = unpaid.join(latest, "customer_id", "left")

    X = cust.select("customer_id", "region", "plan_type", "activation_date", "monthly_arpu_sdg")
    for part in [use_30d, use_prev60d, data_mb, voice_min, last_use, rech_30d, rech_90d,
                 rech_amt, gaps, last_rech, tix_30d, tix_90d, hard, billf]:
        X = X.join(part, "customer_id", "left")
    X = X.join(reg_rate, "region", "left")

    prepaid = F.col("plan_type") == "prepaid"
    X = (X.fillna(0, ["use_30d", "use_prev60d", "data_mb_30d", "voice_min_30d", "rech_30d",
                      "rech_90d", "rech_amt_90d", "tix_30d", "tix_90d", "hard_tix_30d",
                      "unpaid_last3", "late_last3", "region_tix_rate_30d"])
         .withColumn("use_trend", F.col("use_30d") / (F.col("use_prev60d") / 2 + 1))
         .withColumn("days_since_use", F.coalesce("days_since_use", F.lit(fc["fill_days_since_use"])))
         .withColumn("days_since_rech",
                     F.when(prepaid, F.coalesce("_since_rech", F.lit(fc["fill_days_since_recharge"])))
                      .otherwise(F.lit(-1)))
         .withColumn("rech_gap_ratio",
                     F.when(prepaid, F.col("days_since_rech") / (F.coalesce(
                         "_mean_gap", F.lit(float(fc["default_mean_recharge_gap_days"]))) + 1))
                      .otherwise(F.lit(0.0)))
         .withColumn("bill_ratio",
                     F.when(F.col("_avg3") > 0, F.col("_latest") / F.col("_avg3")).otherwise(F.lit(1.0)))
         .withColumn("tenure_days", F.datediff(C, F.col("activation_date")))
         .withColumn("tenure_lt90", (F.col("tenure_days") < 90).cast("int"))
         .withColumn("is_prepaid", prepaid.cast("int")))
    return X.select("customer_id", "region", *FEATURE_COLUMNS)
