"""
================================================================================
TELECOM CHURN PLATFORM - Phase 1b: Hidden-Process Data Generator
================================================================================
WHY THIS FILE EXISTS
    Phase 1 draws a `persona` first (loyal / declining / churned) and then
    generates every table FROM that persona, with non-overlapping ranges
    (churned customers get 3-18 events, loyal get 50-80). Any model trained
    on it relearns the generator and scores ~1.0 AUC. That proves nothing.

    Phase 1b keeps the SAME five tables and column names (so Phases 2-7 can
    run unchanged) but generates behaviour from a hidden, evolving state:

        latent dissatisfaction d[i, m]  (AR(1): persists, gets shocked)
              |-- usage, recharge gaps, tickets, late payments are NOISY
              |   consequences of d
              +-- churn hazard next month = f(d, tenure, plan, hidden trait)

    The label is a random draw from a hazard, never a rule on the features.
    The true drivers are saved in ground_truth/answer_key.json so an ML model
    (and SHAP) can be checked against them.

OUTPUT (default ../data_v2, so your Phase 1 data and README numbers are safe)
    data_v2/sftp_drop/*.csv|parquet      same 5 tables as Phase 1
    data_v2/ground_truth/                NOT for the pipeline or features:
        churn_labels.parquet             customer_id, churn_date (label source)
        latent_state.parquet             d and true hazard per customer-month
        answer_key.json                  planted drivers and coefficients

RUN
    python phase1b_generate_data.py                 # 20,000 customers
    python phase1b_generate_data.py --n 10000       # if memory is tight
================================================================================
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config ----
SEED = 42
SIM_START = pd.Timestamp("2024-01-01")
SIM_END = pd.Timestamp("2024-12-31")
N_MONTHS = 12
TARGET_MONTHLY_CHURN = 0.018     # ~20% of the base churns over the year

REGIONS = ["Khartoum", "Omdurman", "Kassala", "Port Sudan",
           "Wad Madani", "El Obeid", "Atbara", "Nyala"]
REGION_W = np.array([0.35, 0.20, 0.08, 0.08, 0.09, 0.07, 0.06, 0.07])
USAGE_TYPES = np.array(["data", "voice", "sms"])
USAGE_W = [0.55, 0.35, 0.10]
NETWORK_ISSUES = np.array(["dropped_call", "slow_data", "no_signal",
                           "sms_failure", "roaming_issue"])
HARD_ISSUES = np.array(["dropped_call", "no_signal", "slow_data"])
TICKET_STATUS = np.array(["open", "in_progress", "resolved", "closed"])
TICKET_STATUS_W = [0.10, 0.15, 0.45, 0.30]
RECHARGE_AMOUNTS = np.array([50, 100, 200, 500, 1000, 2000])
CHANNELS = np.array(["ussd", "agent", "bank", "app", "online"])
CHANNEL_W = [0.45, 0.30, 0.12, 0.08, 0.05]
_raw_age_w = [4] * 12 + [3] * 15 + [2] * 15 + [1] * 10
AGE_W = np.array(_raw_age_w) / sum(_raw_age_w)

# Planted network stress by region (baseline) and outage episodes
# (region_index, first_month, last_month, extra_stress); months are 0-based
REGION_BASE_STRESS = np.array([0.00, 0.05, 0.20, 0.10, 0.10, 0.25, 0.15, 0.30])
OUTAGE_EPISODES = [(7, 5, 7, 1.6),    # Nyala, Jun-Aug
                   (2, 8, 9, 1.2)]    # Kassala, Sep-Oct

# Latent dynamics: d[m] = AR*d[m-1] + W_NET*net + W_BUD*budget + W_NOISE*eps
AR, W_NET, W_BUD, W_NOISE = 0.7, 0.7, 0.5, 0.3
# How d shows up in observable behaviour (all noisy)
USAGE_SLOPE, RECHARGE_SLOPE, TICKET_SLOPE = 0.7, 0.8, 0.25
# Churn hazard (logit scale); intercept `a` is solved to hit the target rate
B_D, B_SHORT, B_LONG, B_PREPAID, TRAIT_SD = 1.6, 0.7, -0.4, 0.35, 0.5


def month_bounds(m):
    ms = SIM_START + pd.DateOffset(months=m)
    me = SIM_START + pd.DateOffset(months=m + 1) - pd.Timedelta(days=1)
    return ms, me


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_dates(counts, w_start, w_end, rng):
    """For each row k, draw counts[k] dates uniformly in [w_start[k], w_end[k]]."""
    idx = np.repeat(np.arange(len(counts)), counts)
    span = (w_end - w_start).astype("timedelta64[D]").astype(int) + 1
    offs = np.floor(rng.random(len(idx)) * span[idx]).astype("int64")
    return idx, w_start[idx] + offs.astype("timedelta64[D]")


def fmt_ids(prefix, n, width, start=1):
    return np.array([f"{prefix}{i:0{width}d}" for i in range(start, start + n)])


def main(n, seed, out_root):
    rng = np.random.default_rng(seed)
    sftp = f"{out_root}/sftp_drop"
    truth = f"{out_root}/ground_truth"
    os.makedirs(sftp, exist_ok=True)
    os.makedirs(truth, exist_ok=True)
    print(f"Phase 1b | customers={n:,} | seed={seed} | out={out_root}")

    # ------------------------------------------------ customers (static) ----
    cust_ids = fmt_ids("C", n, 6)
    msisdn = np.array([f"2499{s}" for s in
                       rng.choice(90_000_000, n, replace=False) + 10_000_000])
    region_idx = rng.choice(len(REGIONS), n, p=REGION_W)
    prepaid = rng.random(n) < 0.72
    age = rng.choice(np.arange(18, 70), n, p=AGE_W)
    gender = rng.choice(["M", "F"], n, p=[0.58, 0.42])

    pre_window = rng.random(n) < 0.65          # 65% were already customers
    act_days = np.where(pre_window, -rng.integers(1, 1500, n),
                        rng.integers(0, 150, n))
    activation = (SIM_START + pd.to_timedelta(act_days, unit="D")
                  ).values.astype("datetime64[D]")

    income = rng.normal(size=n)
    price_sens = -0.5 * income + np.sqrt(0.75) * rng.normal(size=n)
    trait = rng.normal(0, TRAIT_SD, n)         # hidden: no feature sees this
    base_usage = np.exp(rng.normal(np.log(6.0), 0.55, n))   # sessions / month
    base_rech = np.where(prepaid, np.exp(rng.normal(np.log(1.4), 0.4, n)), 0.0)
    usage_scale = np.exp(rng.normal(0, 0.6, n))
    # heavy-tailed complaint behaviour: a few customers file many tickets, most file few
    complain = np.exp(rng.normal(0, 0.9, n) - 0.9 ** 2 / 2)      # mean = 1
    arpu = np.clip(np.exp(np.log(900) + 0.5 * income
                          + rng.normal(0, 0.3, n)), 50, 6000).round(2)
    amt_pref = np.clip(np.round(2.0 + 0.9 * income + rng.normal(0, 0.8, n)),
                       0, 5).astype(int)

    # ---------------------------------- latent dissatisfaction, all months ----
    stress = np.tile(REGION_BASE_STRESS[:, None], (1, N_MONTHS))
    for r, m0, m1, extra in OUTAGE_EPISODES:
        stress[r, m0:m1 + 1] += extra
    personal = ((rng.random((n, N_MONTHS)) < 0.05)
                * rng.exponential(1.0, (n, N_MONTHS)))
    net_shock = stress[region_idx] + personal                # (n, 12)
    bud_shock = price_sens[:, None] * rng.normal(size=(n, N_MONTHS))

    D = np.zeros((n, N_MONTHS))
    d_prev = rng.normal(0, 0.5, n)
    d_before = np.zeros((n, N_MONTHS))                       # d at month start
    for m in range(N_MONTHS):
        d_before[:, m] = d_prev
        d_prev = (AR * d_prev + W_NET * net_shock[:, m]
                  + W_BUD * bud_shock[:, m] + W_NOISE * rng.normal(size=n))
        D[:, m] = d_prev

    # ------------------------------- churn: hazard depends on LAGGED d -------
    # Hazard in month m uses d at the START of m, so the drivers are already
    # visible in behaviour before the churn happens (a genuine early warning).
    month_start = np.array([month_bounds(m)[0] for m in range(N_MONTHS)],
                           dtype="datetime64[D]")
    month_end = np.array([month_bounds(m)[1] for m in range(N_MONTHS)],
                         dtype="datetime64[D]")
    tenure = (month_start[None, :] - activation[:, None]).astype(int)  # days
    existing = activation[:, None] <= month_end[None, :]
    u_churn = rng.random((n, N_MONTHS))

    def hazard_matrix(a):
        logit = (a + B_D * d_before + B_SHORT * (tenure < 90)
                 + B_LONG * (tenure > 730) + B_PREPAID * prepaid[:, None]
                 + trait[:, None])
        return sigmoid(logit)

    def simulate(a):
        h = hazard_matrix(a)
        churn_month = np.full(n, -1)
        alive = np.ones(n, bool)
        exposed = 0
        for m in range(N_MONTHS):
            elig = alive & existing[:, m]
            hit = elig & (u_churn[:, m] < h[:, m])
            churn_month[hit] = m
            exposed += elig.sum()
            alive &= ~hit
        return churn_month, h, (churn_month >= 0).sum() / max(exposed, 1)

    lo, hi = -9.0, -1.0                       # bisection on the intercept
    for _ in range(30):
        a = (lo + hi) / 2
        _, _, rate = simulate(a)
        lo, hi = (a, hi) if rate < TARGET_MONTHLY_CHURN else (lo, a)
    churn_month, H, realised = simulate(a)
    print(f"intercept a={a:.3f} | monthly churn {realised:.2%} | "
          f"churned in year {(churn_month >= 0).mean():.1%}")

    churn_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    for m in range(N_MONTHS):
        sel = np.where(churn_month == m)[0]
        w0 = np.maximum(month_start[m], activation[sel])
        span = (month_end[m] - w0).astype(int) + 1
        churn_date[sel] = w0 + np.floor(rng.random(len(sel)) * span
                                        ).astype("int64").astype("timedelta64[D]")

    # ------------------------------------------- behaviour, month by month --
    usage_parts, rech_parts, tkt_parts, bill_parts = [], [], [], []
    for m in range(N_MONTHS):
        ms, me = month_start[m], month_end[m]
        ndays = (me - ms).astype(int) + 1
        in_month_alive = (activation <= me) & ((churn_month < 0)
                                               | (churn_month >= m))
        k = np.where(in_month_alive)[0]
        w0 = np.maximum(ms, activation[k])
        w1 = np.where(churn_month[k] == m, churn_date[k], me).astype("datetime64[D]")
        frac = ((w1 - w0).astype(int) + 1) / ndays
        d_m = D[k, m]

        # usage sessions
        lam = base_usage[k] * np.clip(np.exp(-USAGE_SLOPE * d_m), 0.1, 3.0) * frac
        cnt = rng.poisson(lam)
        ri, dt = sample_dates(cnt, w0, w1, rng)
        cid = k[ri]
        utype = rng.choice(3, len(cid), p=USAGE_W)
        fade = np.exp(-0.5 * D[cid, m])
        med = np.array([200.0, 8.0, 1.0])[utype]
        sig = np.array([0.9, 0.8, 0.0])[utype]
        val = np.where(utype == 2, 1.0,
                       usage_scale[cid] * med * np.exp(sig * rng.normal(size=len(cid))) * fade)
        usage_parts.append(pd.DataFrame({
            "ci": cid, "event_date": dt, "usage_type": USAGE_TYPES[utype],
            "usage_value": np.maximum(np.round(val, 2), 0.5).astype(float)}))

        # recharges (prepaid only)
        lam_r = base_rech[k] * np.clip(np.exp(-RECHARGE_SLOPE * d_m), 0.05, 3.0) * frac
        cnt_r = rng.poisson(lam_r)
        ri, dt = sample_dates(cnt_r, w0, w1, rng)
        cid = k[ri]
        pick = np.clip(amt_pref[cid] + rng.choice([-1, 0, 0, 1], len(cid)), 0, 5)
        rech_parts.append(pd.DataFrame({
            "ci": cid, "recharge_date": dt,
            "amount_sdg": RECHARGE_AMOUNTS[pick],
            "channel": rng.choice(CHANNELS, len(cid), p=CHANNEL_W)}))

        # network tickets: noisy consequence of network stress
        ns = net_shock[k, m]
        lam_t = (0.02 + TICKET_SLOPE * np.maximum(ns, 0)) * complain[k] * frac
        cnt_t = rng.poisson(lam_t)
        ri, dt = sample_dates(cnt_t, w0, w1, rng)
        cid = k[ri]
        hard = rng.random(len(cid)) < np.where(net_shock[cid, m] > 0.6, 0.65, 0.3)
        issue = np.where(hard, rng.choice(HARD_ISSUES, len(cid)),
                         rng.choice(NETWORK_ISSUES, len(cid)))
        status = rng.choice(TICKET_STATUS, len(cid), p=TICKET_STATUS_W)
        resol = np.where(np.isin(status, ["resolved", "closed"]),
                         rng.integers(1, 15, len(cid)).astype(float), np.nan)
        tkt_parts.append(pd.DataFrame({
            "ci": cid, "ticket_date": dt, "issue_type": issue,
            "status": status, "resolution_days": resol}))

        # monthly bill (only for months the customer stays through)
        full = churn_month[k] != m
        kb, d_b = k[full], D[k[full], m]
        z_late = -2.0 + 0.5 * d_b + 0.3 * bud_shock[kb, m]
        z_unp = -3.2 + 0.8 * d_b + 0.4 * bud_shock[kb, m]
        den = 1.0 + np.exp(z_late) + np.exp(z_unp)
        u = rng.random(len(kb))
        p_late, p_unp = np.exp(z_late) / den, np.exp(z_unp) / den
        pay = np.where(u < p_unp, "unpaid",
                       np.where(u < p_unp + p_late, "late", "paid"))
        billed = np.round(arpu[kb] * rng.uniform(0.8, 1.2, len(kb))
                          * np.clip(np.exp(-0.2 * d_b), 0.4, 1.5), 2)
        paid = np.where(pay == "paid", billed,
                        np.where(pay == "late", np.round(billed * 0.5, 2), 0.0))
        bill_parts.append(pd.DataFrame({
            "ci": kb, "billing_month": ms.astype(str)[:7],
            "amount_billed": billed, "amount_paid": paid,
            "outstanding_balance": np.round(billed - paid, 2),
            "payment_status": pay}))
        print(f"  month {m + 1:2d}/12 | usage {len(usage_parts[-1]):>7,} | "
              f"recharge {len(rech_parts[-1]):>6,} | tickets {len(tkt_parts[-1]):>5,}")

    # -------------------------------------------------------- assemble ------
    def attach(df, prefix, width):
        df = df.reset_index(drop=True)
        df.insert(0, prefix[0], fmt_ids(prefix[1], len(df), width))
        df["customer_id"] = cust_ids[df["ci"].values]
        df["msisdn"] = msisdn[df["ci"].values]
        df["region"] = np.array(REGIONS)[region_idx[df["ci"].values]]
        return df

    usage = attach(pd.concat(usage_parts), ("event_id", "EVT"), 8)
    recharge = attach(pd.concat(rech_parts), ("recharge_id", "RCH"), 8)
    tickets = attach(pd.concat(tkt_parts), ("ticket_id", "TKT"), 7)
    billing = attach(pd.concat(bill_parts), ("billing_id", "BIL"), 8)
    billing["plan_type"] = np.where(prepaid[billing["ci"].values], "prepaid", "postpaid")

    # `persona` is a constant placeholder. It carries NO information; it exists
    # only so an unchanged Phase 2 drop step (LABEL_COLUMNS_TO_DROP) still finds it.
    usage["persona"] = "n/a"
    tickets["persona"] = "n/a"

    usage = usage[["event_id", "customer_id", "msisdn", "event_date", "usage_type",
                   "usage_value", "region", "persona"]]
    recharge = recharge[["recharge_id", "customer_id", "msisdn", "recharge_date",
                         "amount_sdg", "channel", "region"]]
    tickets = tickets[["ticket_id", "customer_id", "msisdn", "ticket_date",
                       "issue_type", "region", "status", "resolution_days", "persona"]]
    billing = billing[["billing_id", "customer_id", "msisdn", "billing_month",
                       "amount_billed", "amount_paid", "outstanding_balance",
                       "payment_status", "plan_type"]]
    is_churned = (churn_month >= 0).astype(int)
    customers = pd.DataFrame({
        "customer_id": cust_ids, "msisdn": msisdn,
        "region": np.array(REGIONS)[region_idx], "persona": "n/a",
        "plan_type": np.where(prepaid, "prepaid", "postpaid"),
        "age": age, "gender": gender,
        "activation_date": pd.to_datetime(activation),
        "monthly_arpu_sdg": arpu, "is_churned": is_churned})

    # ---------------------------------------------------- integrity checks --
    assert usage["event_id"].is_unique and customers["msisdn"].is_unique
    assert (usage["usage_value"] > 0).all()
    last = usage.groupby("customer_id")["event_date"].max()
    cd = pd.Series(churn_date, index=cust_ids).dropna()
    assert (last.reindex(cd.index).dropna() <= cd.reindex(last.reindex(cd.index).dropna().index)
            ).all(), "activity found after churn date"

    # ---------------------------------------------------------- write out ---
    tables = {"customers": customers, "usage_events": usage,
              "billing_records": billing, "recharge_records": recharge,
              "network_tickets": tickets}
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
        df.to_csv(f"{sftp}/{name}.csv", index=False)
        df.to_parquet(f"{sftp}/{name}.parquet", index=False, compression="snappy")
        print(f"  wrote {name:<18} {len(df):>9,} rows")

    pd.DataFrame({"customer_id": cust_ids, "is_churned": is_churned,
                  "churn_date": pd.to_datetime(churn_date).strftime("%Y-%m-%d"),
                  "churn_month": churn_month + 1}
                 ).to_parquet(f"{truth}/churn_labels.parquet", index=False)
    lat = pd.DataFrame({"customer_id": np.repeat(cust_ids, N_MONTHS),
                        "month": np.tile(np.arange(1, N_MONTHS + 1), n),
                        "d_start": d_before.ravel(), "true_hazard": H.ravel(),
                        "exists": existing.ravel()})
    lat.to_parquet(f"{truth}/latent_state.parquet", index=False)
    answer_key = {
        "generated": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "n_customers": n, "monthly_churn_rate": round(float(realised), 4),
        "churn_definition": "churn_date = last day of any activity; label for "
                            "cutoff C is churn_date in (C, C + 1 month]",
        "hazard_logit": {"intercept": round(float(a), 4), "dissatisfaction_start_of_month": B_D,
                         "tenure_lt_90d": B_SHORT, "tenure_gt_730d": B_LONG,
                         "prepaid": B_PREPAID, "hidden_trait_sd": TRAIT_SD},
        "dissatisfaction_dynamics": {"ar": AR, "network_weight": W_NET,
                                     "budget_weight": W_BUD, "noise": W_NOISE},
        "behaviour_links": {"usage_rate_slope": USAGE_SLOPE,
                            "recharge_rate_slope": RECHARGE_SLOPE,
                            "ticket_rate_slope": TICKET_SLOPE},
        "planted_regional_outages": [
            {"region": REGIONS[r], "months": f"{m0 + 1}-{m1 + 1}", "extra_stress": e}
            for r, m0, m1, e in OUTAGE_EPISODES],
        "expected_top_drivers": [
            "usage trend (recent vs earlier)", "recharge gap growth / days since recharge",
            "tickets in last 30-90 days", "unpaid bills", "tenure < 90 days",
            "prepaid plan", "region during outage episodes"],
        "unobservable": ["hidden_trait", "latent dissatisfaction itself"]}
    with open(f"{truth}/answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)
    print("Done. Ground truth is in", truth, "- keep it out of features.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="../data_v2")
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
