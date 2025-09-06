#!/usr/bin/env python3
"""
Yield Curve Comparison: US vs Canada

This version reads a tenor list from `Tenor.csv` (single column named
`tenor`) and fetches the latest available yields from:
  - US: FRED (Daily Treasury Yield Curve series)
  - Canada: Bank of Canada Valet API (Treasury Bills + Benchmark Bonds)

Notes on coverage:
  - Canada benchmark series are available for 2Y, 3Y, 5Y, 7Y, 10Y, and
    a "Long-term" benchmark. Treasury Bills cover 1M, 3M, 6M, 1Y.
  - There is no explicit 20Y benchmark series in BoC Valet. For 30Y,
    we approximate with the BoC "Long-term" benchmark. 20Y is skipped.

Environment:
  - Set FRED API key in `FRED_API_KEY` (https://fred.stlouisfed.org/docs/api/api_key.html)
  - Requires `requests` and `matplotlib` packages.
"""

from __future__ import annotations

import csv
import os
import math
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta

import requests
import matplotlib.pyplot as plt


TENOR_ORDER: List[str] = [
    "1M", "3M", "6M",
    "1Y", "2Y", "3Y", "5Y", "7Y",
    "10Y", "20Y", "30Y",
]


def normalize_tenor(t: str) -> str:
    """Normalize tenor labels (uppercase, strip spaces)."""
    return t.strip().upper()


def load_tenors_from_csv(path: str) -> List[str]:
    """Load tenor labels from a CSV with a single column header `tenor`.
    Returns the intersection with supported TENOR_ORDER, preserving order.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "tenor" not in headers:
            raise ValueError("Tenor CSV must include a 'tenor' column header")
        tenors_raw: List[str] = []
        for row in reader:
            tenor_val = (row.get("tenor") or "").strip()
            if not tenor_val:
                continue
            tenors_raw.append(normalize_tenor(tenor_val))

    supported = {normalize_tenor(t) for t in TENOR_ORDER}
    tenors = [t for t in TENOR_ORDER if normalize_tenor(t) in supported and normalize_tenor(t) in tenors_raw]
    if not tenors:
        raise ValueError("No supported tenors found in Tenor.csv")
    return tenors


def get_default_data() -> Tuple[List[str], List[float], List[float]]:
    """Default sample data for demonstration. Replace with current values as needed."""
    tenors = TENOR_ORDER.copy()
    # Example spot yields (%). Update to your latest figures.
    us =    [5.30, 5.40, 5.50, 5.30, 4.90, 4.70, 4.30, 4.20, 4.10, 4.30, 4.30]
    canada = [5.00, 5.10, 5.20, 5.00, 4.40, 4.20, 3.80, 3.70, 3.60, math.nan, math.nan]
    return tenors, us, canada


# FRED series mapping for US Treasury yields
FRED_SERIES: Dict[str, str] = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# BoC Valet series mapping for Government of Canada yields
# T-bills (1M,3M,6M,1Y): use V80691342/44/45/46 ("Treasury bills")
# Benchmarks (2Y,3Y,5Y,7Y,10Y): BD.CDN.*YR.DQ.YLD
# Long-term proxy for 30Y: BD.CDN.LONG.DQ.YLD
BOC_SERIES: Dict[str, Optional[str]] = {
    "1M": "V80691342",
    "3M": "V80691344",
    "6M": "V80691345",
    "1Y": "V80691346",
    "2Y": "BD.CDN.2YR.DQ.YLD",
    "3Y": "BD.CDN.3YR.DQ.YLD",
    "5Y": "BD.CDN.5YR.DQ.YLD",
    "7Y": "BD.CDN.7YR.DQ.YLD",
    "10Y": "BD.CDN.10YR.DQ.YLD",
    "20Y": None,  # Not available in Valet benchmark set
    "30Y": "BD.CDN.LONG.DQ.YLD",  # Approximate as long-term
}
def fred_latest_values(series_ids: List[str], api_key: str) -> Dict[str, Optional[float]]:
    """Fetch latest numeric values for FRED series within recent history to avoid '.' values."""
    out: Dict[str, Optional[float]] = {}
    for sid in series_ids:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            start = (date.today() - timedelta(days=120)).isoformat()
            params = {
                "series_id": sid,
                "api_key": api_key,
                "file_type": "json",
                # Fetch last ~4 months to find the most recent numeric observation
                "observation_start": start,
            }
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            val: Optional[float] = None
            # FRED returns observations in ascending date order by default.
            # Iterate from the end (most recent) and pick the first numeric value.
            for obs in reversed(data.get("observations", [])):
                v = obs.get("value")
                try:
                    val = float(v)
                    break
                except Exception:
                    continue
            out[sid] = val
        except Exception:
            out[sid] = None
    return out


def boc_group_latest(group_name: str, recent: int = 30) -> Tuple[Dict[str, float], Optional[str]]:
    """Fetch the most recent numeric value per series in a BoC group over a recent window.
    Returns a map series->value and the most recent date encountered.
    """
    url = f"https://www.bankofcanada.ca/valet/observations/group/{group_name}/json"
    params = {"recent": recent}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        obs_list = data.get("observations", [])
        latest_date: Optional[str] = None
        values: Dict[str, float] = {}
        # Walk from newest to oldest to pick the most recent numeric value
        for obs in reversed(obs_list):
            d = obs.get("d")
            if latest_date is None:
                latest_date = d
            for k, vobj in obs.items():
                if k == "d" or not isinstance(vobj, dict):
                    continue
                v = vobj.get("v")
                try:
                    fv = float(v)
                except Exception:
                    continue
                # Keep the first (newest) numeric value only
                if k not in values:
                    values[k] = fv
        return values, latest_date
    except Exception:
        return {}, None


def boc_values_for_tenors(tenors: List[str]) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    """Build Canada yields for requested tenors using BoC groups.
    - TBILL_ALL provides 1M/3M/6M/1Y (two possible codes each; pick whichever available)
    - bond_yields_benchmark provides 2Y/3Y/5Y/7Y/10Y and long-term (proxy for 30Y)
    Returns tenor->value map and a reference date (most recent seen).
    """
    tbill_map, d1 = boc_group_latest("TBILL_ALL", recent=30)
    bench_map, d2 = boc_group_latest("bond_yields_benchmark", recent=60)
    ref_date = d1 or d2

    # Alternatives per tenor for TBILL
    tbill_choices: Dict[str, List[str]] = {
        "1M": ["V80691342", "V1592248173"],
        "3M": ["V80691344", "V80691303"],
        "6M": ["V80691345", "V80691304"],
        "1Y": ["V80691346", "V80691305"],
    }

    bench_codes: Dict[str, str] = {
        "2Y": "BD.CDN.2YR.DQ.YLD",
        "3Y": "BD.CDN.3YR.DQ.YLD",
        "5Y": "BD.CDN.5YR.DQ.YLD",
        "7Y": "BD.CDN.7YR.DQ.YLD",
        "10Y": "BD.CDN.10YR.DQ.YLD",
        # 20Y: not available
        "30Y": "BD.CDN.LONG.DQ.YLD",
    }

    out: Dict[str, Optional[float]] = {t: None for t in tenors}
    for t in tenors:
        if t in tbill_choices:
            for code in tbill_choices[t]:
                if code in tbill_map:
                    out[t] = tbill_map[code]
                    break
        elif t in bench_codes:
            code = bench_codes[t]
            out[t] = bench_map.get(code)
        else:
            # 20Y (None) or unsupported
            out[t] = None
    return out, ref_date


# US Treasury (no API key) via FiscalData API
TREASURY_FIELD_BY_TENOR: Dict[str, str] = {
    "1M": "bc_1month",
    "3M": "bc_3month",
    "6M": "bc_6month",
    "1Y": "bc_1year",
    "2Y": "bc_2year",
    "3Y": "bc_3year",
    "5Y": "bc_5year",
    "7Y": "bc_7year",
    "10Y": "bc_10year",
    "20Y": "bc_20year",
    "30Y": "bc_30year",
}


def us_treasury_latest_values_no_key(tenors: List[str]) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    """Fetch the most recent US Treasury par yield curve (no API key) from Treasury FiscalData.
    Returns a mapping tenor->value and the record_date string if available.
    """
    endpoints = [
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_par_yield_curve",
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_yield_curve",
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_yield_curve_rates",
    ]

    fields = ["record_date"] + [TREASURY_FIELD_BY_TENOR[t] for t in tenors if t in TREASURY_FIELD_BY_TENOR]
    params = {
        "sort": "-record_date",
        "page[number]": 1,
        "page[size]": 1,
        "format": "json",
        "fields": ",".join(fields),
    }

    last_error: Optional[Exception] = None
    for url in endpoints:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            rows = js.get("data") or js.get("Data") or []
            if not rows:
                continue
            row = rows[0]
            record_date = row.get("record_date")
            out: Dict[str, Optional[float]] = {}
            for t in tenors:
                fld = TREASURY_FIELD_BY_TENOR.get(t)
                v = row.get(fld) if fld else None
                try:
                    out[t] = float(v) if v not in (None, "", "N/A") else None
                except Exception:
                    out[t] = None
            return out, record_date
        except Exception as e:
            last_error = e
            continue

    return {t: None for t in tenors}, None


def fred_csv_latest_values_no_key(series_ids: List[str]) -> Dict[str, Optional[float]]:
    """Fallback path: fetch latest values using FRED's fredgraph.csv (no API key).
    Downloads one CSV per series and parses the last non-empty value.
    """
    out: Dict[str, Optional[float]] = {}
    for sid in series_ids:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            # CSV header is typically DATE,<SID>
            # Find last non-empty numeric value from bottom
            val: Optional[float] = None
            for line in reversed(lines[1:]):
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                v = parts[1].strip()
                if v in ('', '.', 'NaN'):
                    continue
                try:
                    val = float(v)
                    break
                except Exception:
                    continue
            out[sid] = val
        except Exception:
            out[sid] = None
    return out


def plot_yield_curves(tenors: List[str], us: List[Optional[float]], canada: List[Optional[float]]) -> None:
    """Plot exactly two curves: US and Canada."""
    # Filter to indices where both are present (avoid broken lines)
    idx = [i for i, (u, c) in enumerate(zip(us, canada)) if u is not None and c is not None and not (isinstance(u, float) and math.isnan(u)) and not (isinstance(c, float) and math.isnan(c))]
    if not idx:
        raise RuntimeError("No overlapping tenors with values for both US and Canada to plot.")

    tenors_plot = [tenors[i] for i in idx]
    us_plot = [float(us[i]) for i in idx]
    ca_plot = [float(canada[i]) for i in idx]

    x = list(range(len(tenors_plot)))

    plt.figure(figsize=(10, 5))

    # Two curves only (no markers). Canada in red.
    plt.plot(x, us_plot, label="US Treasury", color="#1f77b4", linewidth=2)
    plt.plot(x, ca_plot, label="Government of Canada", color="red", linewidth=2)

    plt.title("Yield Curve: US vs Canada")
    plt.xlabel("Maturity")
    plt.ylabel("Yield (%)")
    plt.xticks(x, tenors_plot)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save and show
    out = "yield_curve_us_canada.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.show()


def main() -> None:
    # Load requested tenors from Tenor.csv (case-insensitive name)
    base_dir = os.path.dirname(__file__)
    tenor_candidates = [
        os.path.join(base_dir, "Tenor.csv"),
        os.path.join(base_dir, "tenor.csv"),
        os.path.join(base_dir, "TENOR.csv"),
    ]
    tenor_path = next((p for p in tenor_candidates if os.path.exists(p)), None)
    if not tenor_path:
        print("Tenor.csv not found; using full default tenor set.")
        tenors = TENOR_ORDER.copy()
    else:
        tenors = load_tenors_from_csv(tenor_path)

    # Fetch US yields: prefer FRED if API key is set; otherwise use Treasury FiscalData API (no key)
    fred_key = os.environ.get("FRED_API_KEY", "").strip()
    us_values_by_tenor: Dict[str, Optional[float]] = {}
    if fred_key:
        fred_ids = [FRED_SERIES[t] for t in tenors if t in FRED_SERIES]
        fred_map = fred_latest_values(fred_ids, fred_key)
        for t in tenors:
            sid = FRED_SERIES.get(t)
            us_values_by_tenor[t] = fred_map.get(sid) if sid else None
    else:
        # Try Treasury FiscalData first
        us_map, record_date = us_treasury_latest_values_no_key(tenors)
        if record_date:
            print(f"US Treasury (FiscalData) record_date: {record_date}")
        # If that failed for all, fall back to FRED fredgraph.csv (no key)
        if all(us_map.get(t) is None for t in tenors):
            fred_ids = [FRED_SERIES[t] for t in tenors if t in FRED_SERIES]
            fred_no_key = fred_csv_latest_values_no_key(fred_ids)
            for t in tenors:
                sid = FRED_SERIES.get(t)
                us_map[t] = fred_no_key.get(sid) if sid else None
            print("US Treasury values fetched via FRED CSV fallback (no API key).")
        us_values_by_tenor = us_map

    # Fetch Canada from BoC Valet via groups
    ca_values_by_tenor, ca_date = boc_values_for_tenors(tenors)
    if ca_date:
        print(f"BoC (Canada) latest date: {ca_date}")

    # Prepare aligned lists in input order
    us_vals = [us_values_by_tenor.get(t) for t in tenors]
    ca_vals = [ca_values_by_tenor.get(t) for t in tenors]

    # Lightweight debug: show which tenors have values
    have = [t for t, u, c in zip(tenors, us_vals, ca_vals) if u is not None and c is not None and not (isinstance(u, float) and math.isnan(u)) and not (isinstance(c, float) and math.isnan(c))]
    missing = [t for t, u, c in zip(tenors, us_vals, ca_vals) if t not in have]
    print(f"Tenors with values: {have}")
    if missing:
        print(f"Tenors missing on one side: {missing}")

    # Plot common tenors where both values are present
    try:
        plot_yield_curves(tenors, us_vals, ca_vals)
    except RuntimeError as e:
        print(str(e))
        print("US values:", {t: us_values_by_tenor.get(t) for t in tenors})
        print("CA values:", {t: ca_values_by_tenor.get(t) for t in tenors})
        return


if __name__ == "__main__":
    main()
