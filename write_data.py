# write_data.py
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple



try:
    from runners import a, b, c, d  # your existing functions
except Exception as e:
    pass


# write_data.py
"""
Write the four dataframes to ./data/<ns>/ for the dashboard to read.

- TEST = True  -> use mock generators defined here
- TEST = False -> call real producers a(), b(), c(), d() from runners.py

Usage:
  python write_data.py --ns team-alpha --n-large 10000
Env override:
  WRITE_TEST=true|false
"""



# -----------------------------
# Toggle real vs. mock producers
# -----------------------------
TEST = True  # set False to call real a/b/c/d
# Optional env override
_env = os.getenv("WRITE_TEST")
if _env is not None:
    TEST = _env.lower() in ("1", "true", "yes", "y")

DATA_ROOT = os.path.abspath("./data")


# =============================
# MOCK GENERATORS (TEST = True)
# =============================
def mock_df1_summary_15d(seed: int = 101) -> pd.DataFrame:
    """df1: per-day totals and issues for last 15 days."""
    rng = np.random.default_rng(seed)
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(14, -1, -1)]
    total = rng.integers(500, 900, size=len(dates))
    issues = (total * rng.uniform(0.06, 0.18, size=len(dates))).astype(int)
    return pd.DataFrame({"date": dates, "total": total, "issues": issues})


def mock_df2_category_daily(seed: int = 102) -> pd.DataFrame:
    """df2: per-day, per-category issue counts for last 15 days."""
    rng = np.random.default_rng(seed)
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(14, -1, -1)]
    categories = ["Data Quality", "Latency", "Mapping", "Compliance", "Duplicates"]
    rows = []
    for d in dates:
        base = rng.integers(20, 80, size=len(categories))
        for c, v in zip(categories, base):
            rows.append({"date": d, "category": c, "issues": int(v)})
    return pd.DataFrame(rows)


def mock_df3_aux(seed: int = 103) -> pd.DataFrame:
    """df3: small auxiliary metrics for top cards/side info."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "metric": ["throughput", "success_rate", "avg_latency_ms"],
        "value": [int(rng.integers(6000, 9000)), round(rng.uniform(0.95, 0.995), 3), int(rng.integers(120, 260))]
    })


def mock_df4_details_large(n: int = 10_000, seed: int = 104) -> pd.DataFrame:
    """
    df4: large ragged table (some optional columns missing per-row).
    Designed to exercise:
      - uneven fields
      - CSV download
      - server-side pagination on UI
    """
    rng = np.random.default_rng(seed)
    merchants = [f"Merchant-{i:04d}" for i in range(1, 1501)]
    issues = ["Mapping", "Latency", "Compliance", "Data Quality", "Duplicates"]
    owners = ["Sam", "Lee", "Mia", "Ari", "Jon", "N/A"]
    rows = []
    for i in range(1, n + 1):
        rec = {
            "id": i,
            "merchant": rng.choice(merchants),
            "issue": rng.choice(issues),
        }
        # optional fields
        if rng.random() < 0.6:
            rec["owner"] = rng.choice(owners)
        if rng.random() < 0.4:
            rec["priority"] = rng.choice(["Low", "Medium", "High", "Critical"])
        if rng.random() < 0.3:
            rec["status"] = rng.choice(["Open", "In Review", "Closed"])
        if rng.random() < 0.2:
            rec["region"] = rng.choice(["NA", "EU", "APAC"])
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================
# REAL PRODUCERS (TEST = False)
# =============================
try:
    def real_producers() -> Tuple:
        """
        Import your real functions only when needed to avoid import-time side effects.
        They must be defined in runners.py as: a() -> df1, b() -> df2, c() -> df3, d(n:int) -> df4
        """
        from runners import a, b, c, d  # noqa: WPS347 (local import on purpose)
        return a, b, c, d

except Exception as e:
    pass


# =============================
# WRITE LOGIC
# =============================
def write_all(ns: str, n_large: int = 10_000) -> None:
    """
    Writes:
      df1 -> df1.parquet
      df2 -> df2.parquet
      df3 -> df3.parquet
      df4 -> df4.csv
    under ./data/<ns>/
    """
    ns_dir = os.path.join(DATA_ROOT, ns)
    os.makedirs(ns_dir, exist_ok=True)

    if TEST:
        df1 = mock_df1_summary_15d()
        df2 = mock_df2_category_daily()
        df3 = mock_df3_aux()
        df4 = mock_df4_details_large(n=n_large)
    else:
        A, B, C, D = real_producers()
        df1 = A()
        df2 = B()
        df3 = C()
        df4 = D(n_large)

    for df in (df1, df2):
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col])

    # Write small frames in Parquet (compact, typed)
    df1.to_parquet(os.path.join(ns_dir, "df1.parquet"), index=False)
    df2.to_parquet(os.path.join(ns_dir, "df2.parquet"), index=False)
    df3.to_parquet(os.path.join(ns_dir, "df3.parquet"), index=False)

    # Write large in CSV for easy browser download
    df4.to_csv(os.path.join(ns_dir, "df4.csv"), index=False)

    print(f"[write_data] Wrote data for ns='{ns}' (TEST={TEST}) to {ns_dir}")


# =============================
# CLI
# =============================
def _parse_args():
    parser = argparse.ArgumentParser(description="Write dashboard dataframes to ./data/<ns>/")
    parser.add_argument("--ns", type=str, default="default", help="Namespace (tenant/user) folder")
    parser.add_argument("--n-large", type=int, default=10_000, help="Rows for df4 (large table)")
    parser.add_argument("--test", type=str, default=None,
                        help="Override TEST flag: true/false (or use env WRITE_TEST)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Optional CLI override for TEST
    if args.test is not None:
        TEST = args.test.lower() in ("1", "true", "yes", "y")
    os.makedirs(DATA_ROOT, exist_ok=True)
    write_all(ns=args.ns, n_large=args.n_large)
