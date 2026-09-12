"""
Consolidated DATA COLLECTION script for the GPR/WUI/EPU/TPU/GSCPI AI-
mitigation panel project. Collects EVERY raw data series this project
uses -- GDP, ICT investment, AI patents, AI investment, national and
semiconductor equity indices, all five global shock series (WUI, GPR,
EPU, TPU, GSCPI), and the AI/ICT-related HS export data -- and writes
them all into a single local workbook, ai_data.xlsx, with exactly the
sheet structure the rest of this project's scripts (the local-data
modeling scripts) expect: gdp, ict_inv, ai_patent, ai_inv, index_nat,
index_sox, hs_export, wui, gpr, epu, tpu, gscpi, com,
trade_openness, population_growth, productivity_growth, gdp_per_capita,
infl.

THIS SCRIPT DOES NOT MODEL ANYTHING: no panel construction, no local
projections, no IRFs, no regressions, no charts. It is purely a data-
collection/refresh step -- run this whenever the underlying source data
needs updating, then run one of the modeling scripts (which read
ai_data.xlsx as their single local data source) separately.

CONSOLIDATES (fetch logic extracted essentially verbatim, not
rewritten, from each source script -- see each function's docstring
for its original provenance):
    ai_model_WUI_pat_inv.py    -> WUI, GDP, AI patents/investment,
                                   national + semiconductor indices
    ai_model_GPR_pat_inv.py    -> GPR
    ..._TPU_pure_cum.py        -> TPU, ICT investment share
    ..._EPU_pure_cum.py        -> EPU
    ..._GSCPI_pure_cum.py      -> GSCPI
    hs_export.py               -> AI/ICT-related HS export data (BIMTS)

LOCAL FILES REQUIRED next to this script (not fetched over the
network): eto_patent.csv, eto_inv.csv (ETO/CSET Country Activity
Tracker direct exports -- see fetch_ai_patents()/fetch_ai_investment()
docstrings for exactly where to get them).
"""

import sys
import subprocess
import io
import os
import itertools
import time
from pathlib import Path


def install_if_needed(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_if_needed("requests")
install_if_needed("pandas")
install_if_needed("numpy")
install_if_needed("openpyxl")
install_if_needed("xlsxwriter")
install_if_needed("yfinance")

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------
# 0. CONFIG (identical across all six source scripts -- verified --
#    EXTENDED here with Ireland, Finland, and Portugal)
# ----------------------------------------------------------------------

COUNTRIES = ["NL", "DE", "FR", "IT", "ES", "BE", "AT", "IE", "FI", "PT"]
                                    # euro-area panel + Ireland, Finland,
                                    # Portugal
SAMPLE_START = "2000-01-01"
SAMPLE_END = "2026-06-30"
EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

OUTPUT_FILE = Path(__file__).resolve().parent / "ai_data.xlsx"

LOCAL_AI_PATENTS_FILE = "eto_patent.csv"
LOCAL_AI_INVESTMENT_FILE = "eto_inv.csv"
# ^ ETO/CSET Country Activity Tracker's own direct CSV exports
# (cat.eto.tech, "Patent" and "Investment" datasets respectively), saved
# locally next to this script. IMPORTANT: these local files must
# themselves be RE-EXPORTED from cat.eto.tech with Ireland, Finland,
# and Portugal ADDED to the country selection in the tool's own URL --
# this script cannot add countries to data that was already exported
# without them; see fetch_ai_patents()/fetch_ai_investment()'s
# docstrings for the exact URL to re-export from.

COUNTRY_NAME_MAP = {
    "AT": "Austria", "BE": "Belgium", "DE": "Germany", "ES": "Spain",
    "FR": "France", "IT": "Italy", "NL": "Netherlands",
    "IE": "Ireland", "FI": "Finland", "PT": "Portugal",
}  # this project's ISO2 codes -> the full English country names used in
   # the ETO CAT exports' "Country" column

NATIONAL_INDEX_TICKERS = {
    "NL": "^AEX",        # AEX, Amsterdam
    "DE": "^GDAXI",       # DAX, Frankfurt
    "FR": "^FCHI",        # CAC 40, Paris
    "IT": "FTSEMIB.MI",   # FTSE MIB, Milan
    "ES": "^IBEX",        # IBEX 35, Madrid
    "BE": "^BFX",         # BEL 20, Brussels
    "AT": "^ATX",         # ATX, Vienna
    "IE": "^ISEQ",        # ISEQ All Share, Dublin
    "FI": "^OMXH25",      # OMX Helsinki 25
    "PT": "PSI20.LS",      # PSI, Lisbon -- NOT "^PSI20": that symbol is
                           # stale/discontinued on Yahoo Finance since the
                           # index's 2021 rename from "PSI-20" to "PSI";
                           # "PSI20.LS" is the currently actively-quoted
                           # ticker (confirmed via live 2025/2026 data)
}

HS_CODES = [
    "847150", "847180", "847330",  # computer processing units, other
                                    # automatic data-processing units,
                                    # parts/accessories of heading 8471
    "848610", "848620", "848630", "848640", "848690",  # HS 8486: machines
                                    # and apparatus for the manufacture of
                                    # semiconductor boules/wafers,
                                    # semiconductor devices, electronic
                                    # integrated circuits, or flat panel
                                    # displays (and their parts/accessories)
]
ISO3_TO_ISO2 = {
    "NLD": "NL", "DEU": "DE", "FRA": "FR", "ITA": "IT",
    "ESP": "ES", "BEL": "BE", "AUT": "AT",
    "IRL": "IE", "FIN": "FI", "PRT": "PT",
}
COUNTRIES_ISO3 = list(ISO3_TO_ISO2.keys())


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def _requests_get_with_retry(url, params=None, timeout=60, max_retries=3, backoff=2.0):
    """
    Thin wrapper around requests.get() that retries on TRANSIENT network
    errors (connection reset, connection aborted, timeout) with
    exponential backoff, instead of letting the whole run crash on the
    first hiccup. From ai_model_WUI_pat_inv.py -- motivated there by
    Eurostat's API intermittently resetting the connection mid-response
    on an otherwise valid request. Does NOT retry on HTTP error status
    codes beyond transient-connection-level.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"  [!] Network error on attempt {attempt}/{max_retries} "
                      f"({e.__class__.__name__}: {e}) -- retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  [!] Network error on final attempt {attempt}/{max_retries} "
                      f"({e.__class__.__name__}) -- giving up. This is usually "
                      f"transient server-side flakiness -- simply re-running the "
                      f"script often succeeds.")
    raise last_exc


def eurostat_json_to_df(dataset, params):
    """Fetch and flatten a Eurostat JSON-stat response into a long DataFrame."""
    url = f"{EUROSTAT_BASE}/{dataset}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    required_keys = {"dimension", "id", "size", "value"}
    missing = required_keys - js.keys()
    if missing:
        raise ValueError(
            f"Eurostat response for dataset '{dataset}' is missing "
            f"{missing} -- this dataset/param combination likely matched "
            f"zero observations, or the dataset code is wrong.\n"
            f"Requested URL: {r.url}\n"
            f"Raw response keys: {list(js.keys())}\n"
            f"Raw response (first 1000 chars): {str(js)[:1000]}"
        )

    dims = js["dimension"]
    dim_ids = js["id"]
    values = js["value"]

    idx_lists = []
    for d in dim_ids:
        cat = dims[d]["category"]
        order = sorted(cat["index"].items(), key=lambda kv: kv[1])
        idx_lists.append([k for k, _ in order])

    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}

    rows = []
    for flat_i, combo in enumerate(itertools.product(*idx_lists)):
        v = values.get(str(flat_i))
        if v is None:
            continue
        rows.append(dict(zip(dim_ids, combo), value=v))

    if not rows:
        available = {d: list(dims[d]["category"]["index"].keys()) for d in dim_ids}
        raise ValueError(
            f"Eurostat returned zero non-null observations for dataset "
            f"'{dataset}' with params={params}.\n"
            f"Requested URL: {r.url}\n"
            f"Valid categories per dimension in this dataset (check your "
            f"filter values against these): {available}"
        )

    return pd.DataFrame(rows)


def _yf_close_series(ticker):
    """
    yfinance (recent versions) returns a MultiIndex-column DataFrame even
    for a single ticker unless told otherwise, so
    yf.download(ticker, ...)["Close"] can come back as a one-column
    DataFrame rather than a Series, which silently breaks a later
    .rename("some_name") call. Force a genuine Series here.
    """
    import yfinance as yf
    px = yf.download(ticker, start=SAMPLE_START, end=SAMPLE_END, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.squeeze("columns")
    return px


# ----------------------------------------------------------------------
# 1. Global shock series -- WUI, GPR, EPU, TPU, GSCPI
# ----------------------------------------------------------------------

def fetch_wui_global():
    """
    Quarterly Global World Uncertainty Index (WUI), GDP-weighted average
    across countries. Ahir, Bloom & Furceri, NBER WP 29763.
    Source: https://worlduncertaintyindex.com/wp-content/uploads/2026/07/WUI_Data.xlsx
    Sheet "F1", header row 2, columns "year" (e.g. "1990q1") and "WUI".
    From ai_model_WUI_pat_inv.py.
    """
    url = ("https://worlduncertaintyindex.com/wp-content/uploads/2026/07/"
           "WUI_Data.xlsx")
    r = _requests_get_with_retry(url, timeout=60)

    candidate = pd.read_excel(io.BytesIO(r.content), sheet_name="F1", header=2)
    cols_upper = [str(c).strip().upper() for c in candidate.columns]

    date_col = next((c for c, cu in zip(candidate.columns, cols_upper) if cu == "YEAR"), None)
    value_col = next((c for c, cu in zip(candidate.columns, cols_upper) if cu == "WUI"), None)

    if date_col is None or value_col is None:
        sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None, nrows=5)
        raise ValueError(
            "fetch_wui_global: expected columns 'year' and 'WUI' not found "
            f"at header row 2 of sheet 'F1' (got columns: {list(candidate.columns)}). "
            "First 5 raw rows of every sheet: "
            f"{ {name: d.values.tolist() for name, d in sheets.items()} }."
        )

    df = candidate[[date_col, value_col]].rename(
        columns={date_col: "quarter_raw", value_col: "wui_global"}
    )
    df["wui_global"] = pd.to_numeric(df["wui_global"], errors="coerce")

    def _parse_quarter(v):
        if pd.isna(v):
            return pd.NaT
        if isinstance(v, pd.Timestamp):
            return v.to_period("Q")
        s = str(v).strip().upper().replace(" ", "")
        try:
            return pd.Period(s, freq="Q")
        except Exception:
            try:
                return pd.Timestamp(v).to_period("Q")
            except Exception:
                return pd.NaT

    df["quarter"] = df["quarter_raw"].apply(_parse_quarter)
    df = df.dropna(subset=["quarter", "wui_global"]).sort_values("quarter")

    if df.empty:
        raise ValueError("fetch_wui_global: zero valid (quarter, value) rows after parsing.")

    print(f"  [diagnostic] WUI: {len(df)} quarters, "
          f"{df['quarter'].min()}-{df['quarter'].max()}.")
    return df[["quarter", "wui_global"]]


def fetch_gpr_global():
    """
    Monthly global GPR index, Caldara & Iacoviello, resampled to
    quarterly (mean). Source: matteoiacoviello.com/gpr_files/data_gpr_export.xls
    From ai_model_GPR_pat_inv.py.
    """
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content))
    df["month"] = pd.to_datetime(df["month"])
    df = df[["month", "GPR"]].rename(columns={"GPR": "gpr_global"})
    df = df.set_index("month")
    q = df.resample("QE").mean()
    q.index = q.index.to_period("Q")
    q = q.reset_index().rename(columns={"month": "quarter"})
    print(f"  [diagnostic] GPR: {len(q)} quarters, "
          f"{q['quarter'].min()}-{q['quarter'].max()}.")
    return q[["quarter", "gpr_global"]]


def fetch_epu_global():
    """
    Monthly Global Economic Policy Uncertainty (GEPU_current) index,
    Baker, Bloom & Davis, resampled to quarterly (mean).
    Source: policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx
    From ..._EPU_pure_cum.py.
    """
    url = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = None
    for header_guess in (0, 1, 2, 3):
        try:
            candidate = pd.read_excel(io.BytesIO(r.content), header=header_guess)
        except Exception:
            continue
        cols = list(candidate.columns)
        value_col = next((c for c in cols if str(c).strip() == "GEPU_current"), None)
        if value_col is None:
            continue

        cols_upper = [str(c).strip().upper() for c in cols]
        year_col = next((c for c, cu in zip(cols, cols_upper) if cu == "YEAR"), None)
        month_col = next((c for c, cu in zip(cols, cols_upper) if cu == "MONTH"), None)
        date_col = next((c for c, cu in zip(cols, cols_upper) if cu == "DATE"), None)

        if year_col is not None and month_col is not None:
            yr = pd.to_numeric(candidate[year_col], errors="coerce")
            mo = pd.to_numeric(candidate[month_col], errors="coerce")
            valid = yr.notna() & mo.notna()
            if valid.sum() == 0:
                continue
            quarter = pd.PeriodIndex(
                pd.to_datetime(dict(year=yr[valid].astype(int), month=mo[valid].astype(int), day=1)),
                freq="M",
            ).asfreq("Q")
            df = pd.DataFrame({
                "quarter": quarter,
                "epu_global": candidate.loc[valid, value_col].values,
            })
            break
        elif date_col is not None:
            parsed_date = pd.to_datetime(candidate[date_col], errors="coerce")
            valid = parsed_date.notna()
            if valid.sum() == 0:
                continue
            quarter = parsed_date[valid].dt.to_period("Q")
            df = pd.DataFrame({
                "quarter": quarter.values,
                "epu_global": candidate.loc[valid, value_col].values,
            })
            break

    if df is None:
        sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None, nrows=5)
        raise ValueError(
            "fetch_epu_global: could not locate 'GEPU_current' alongside a "
            "usable date column across header rows 0-3. First 5 raw rows "
            f"of every sheet: { {name: d.values.tolist() for name, d in sheets.items()} }."
        )

    df["epu_global"] = pd.to_numeric(df["epu_global"], errors="coerce")
    df = df.dropna(subset=["quarter", "epu_global"]).sort_values("quarter")

    if df.empty:
        raise ValueError("fetch_epu_global: zero valid (quarter, value) rows after parsing.")

    q = df.groupby("quarter")["epu_global"].mean().reset_index()
    print(f"  [diagnostic] EPU: {len(q)} quarters, "
          f"{q['quarter'].min()}-{q['quarter'].max()}.")
    return q[["quarter", "epu_global"]]


def fetch_tpu_global():
    """
    Quarterly Trade Policy Uncertainty (TPU) index, Caldara, Iacoviello,
    Molligo, Prestipino & Raffo (JME 2020). Already quarterly at source.
    Source: matteoiacoviello.com/tpu_files/tpu_web_latest.xlsx,
    sheet "TPU_QUARTERLY", column "TPUQ". From ..._TPU_pure_cum.py.
    """
    url = "https://www.matteoiacoviello.com/tpu_files/tpu_web_latest.xlsx"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = None
    for header_guess in (0, 1, 2, 3):
        try:
            candidate = pd.read_excel(io.BytesIO(r.content), sheet_name="TPU_QUARTERLY",
                                       header=header_guess)
        except Exception:
            continue
        cols = list(candidate.columns)
        cols_upper = [str(c).strip().upper() for c in cols]
        value_col = next((c for c, cu in zip(cols, cols_upper) if cu == "TPUQ"), None)
        if value_col is None or len(cols) < 2:
            continue
        date_col = cols[0] if cols[0] != value_col else cols[1]

        def _parse_quarter(v):
            if pd.isna(v):
                return pd.NaT
            if isinstance(v, pd.Timestamp):
                return v.to_period("Q")
            s = str(v).strip().upper().replace(" ", "")
            try:
                return pd.Period(s, freq="Q")
            except Exception:
                try:
                    return pd.Timestamp(v).to_period("Q")
                except Exception:
                    return pd.NaT

        parsed_quarter = candidate[date_col].apply(_parse_quarter)
        value_numeric = pd.to_numeric(candidate[value_col], errors="coerce")
        valid = parsed_quarter.notna() & value_numeric.notna()
        if valid.sum() == 0:
            continue

        df = pd.DataFrame({
            "quarter": parsed_quarter[valid].values,
            "tpu_global": value_numeric[valid].values,
        })
        break

    if df is None:
        sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None, nrows=5)
        raise ValueError(
            "fetch_tpu_global: could not locate a 'TPUQ' column with a "
            "parseable date column in sheet 'TPU_QUARTERLY' across header "
            f"rows 0-3. First 5 raw rows: { {name: d.values.tolist() for name, d in sheets.items()} }."
        )

    df = df.dropna(subset=["quarter", "tpu_global"]).sort_values("quarter")
    if df.empty:
        raise ValueError("fetch_tpu_global: zero valid (quarter, value) rows after parsing.")

    print(f"  [diagnostic] TPU: {len(df)} quarters, "
          f"{df['quarter'].min()}-{df['quarter'].max()}.")
    return df[["quarter", "tpu_global"]]


def fetch_gscpi_global():
    """
    Monthly Global Supply Chain Pressure Index (GSCPI), NY Fed (Benigno,
    di Giovanni, Groen & Noble, 2022), resampled to quarterly (mean).
    Source: newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx
    From ..._GSCPI_pure_cum.py.
    """
    url = ("https://www.newyorkfed.org/medialibrary/research/interactives/"
           "gscpi/downloads/gscpi_data.xlsx")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None)

    df = None
    for sheet_name, raw in sheets.items():
        for header_guess in (0, 1, 2, 3, 4):
            try:
                candidate = pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name,
                                           header=header_guess)
            except Exception:
                continue
            cols_upper = [str(c).strip().upper() for c in candidate.columns]
            date_col = next((c for c, cu in zip(candidate.columns, cols_upper)
                              if cu in ("DATE", "MONTH") or "DATE" in cu), None)
            value_col = next((c for c, cu in zip(candidate.columns, cols_upper)
                               if cu == "GSCPI" or "GSCPI" in cu), None)
            if date_col is not None and value_col is not None:
                df = candidate[[date_col, value_col]].rename(
                    columns={date_col: "month", value_col: "gscpi_global"}
                )
                break
        if df is not None:
            break

    if df is None:
        raise ValueError(
            "fetch_gscpi_global: could not locate a Date/GSCPI-style column "
            "pair. Sheets and columns found: "
            f"{ {name: list(d.columns) for name, d in sheets.items()} }."
        )

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["gscpi_global"] = pd.to_numeric(df["gscpi_global"], errors="coerce")
    df = df.dropna(subset=["month", "gscpi_global"]).sort_values("month")
    if df.empty:
        raise ValueError("fetch_gscpi_global: zero valid (date, value) rows after parsing.")

    df = df.set_index("month")
    q = df.resample("QE").mean()
    q.index = q.index.to_period("Q")
    q = q.reset_index().rename(columns={"month": "quarter"})
    print(f"  [diagnostic] GSCPI: {len(q)} quarters, "
          f"{q['quarter'].min()}-{q['quarter'].max()}.")
    return q[["quarter", "gscpi_global"]]


def fetch_commodity_price_index():
    """
    Global Price Index of All Commodities (IMF, series PALLFNFINDEXQ) --
    already quarterly at source, index points, not seasonally adjusted.
    Covers the full commodity basket INCLUDING oil and natural gas
    (alongside metals, food, agricultural raw materials, etc.), unlike
    e.g. WWOPI (oil-only) or WWFPI (food-only) -- this is the broad,
    all-commodities headline index specifically.
    Source: https://fred.stlouisfed.org/series/PALLFNFINDEXQ

    Downloaded via FRED's public CSV export endpoint
    (fredgraph.csv?id=<series>) -- no API key required, a well-
    established, widely-used FRED access pattern (confirmed via
    multiple independent public examples using this exact URL format).
    FRED marks missing observations with "." in this export, which
    pandas' default read_csv would keep as the literal string "." --
    coerced to NaN explicitly below.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": "PALLFNFINDEXQ"}

    max_attempts = 4
    r = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_attempts:
                raise SystemExit(
                    f"\nfetch_commodity_price_index: network request failed "
                    f"after {max_attempts} attempts: {e.__class__.__name__}: {e}"
                )
            wait_s = 5 * (3 ** (attempt - 1))  # 5s, 15s, 45s
            print(f"  [!] Attempt {attempt}/{max_attempts} failed "
                  f"({e.__class__.__name__}: {e}) -- retrying in {wait_s}s...")
            time.sleep(wait_s)

    df = pd.read_csv(io.StringIO(r.text))
    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    value_col = cols_upper.get("PALLFNFINDEXQ")
    date_col = cols_upper.get("DATE") or cols_upper.get("OBSERVATION_DATE")
    if date_col is None and value_col is not None and len(df.columns) == 2:
        date_col = [c for c in df.columns if c != value_col][0]
    if date_col is None or value_col is None:
        raise ValueError(
            f"fetch_commodity_price_index: could not identify the date "
            f"and/or 'PALLFNFINDEXQ' value column. Actual columns: "
            f"{list(df.columns)}."
        )

    df = df[[date_col, value_col]].rename(
        columns={date_col: "date", value_col: "commodity_price_index"}
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["commodity_price_index"] = pd.to_numeric(df["commodity_price_index"], errors="coerce")
    df = df.dropna(subset=["date", "commodity_price_index"])
    if df.empty:
        raise ValueError(
            "fetch_commodity_price_index: zero valid rows after parsing "
            "the FRED CSV export."
        )

    df["quarter"] = df["date"].dt.to_period("Q")
    df = df.sort_values("quarter")
    print(f"  [diagnostic] Global commodity price index (PALLFNFINDEXQ): "
          f"{len(df)} quarters, {df['quarter'].min()}-{df['quarter'].max()}.")
    return df[["quarter", "commodity_price_index"]]


# ----------------------------------------------------------------------
# 2. GDP (raw level only -- growth-rate derivation is modeling logic,
#    out of scope for this data-collection-only script)
# ----------------------------------------------------------------------

def fetch_gdp_level():
    """
    Quarterly real GDP, chain-linked volumes, seasonally & calendar
    adjusted. Eurostat dataset namq_10_gdp. From
    ai_model_WUI_pat_inv.py's fetch_gdp_growth_yoy() -- growth-rate
    (dgdp) computation removed here since that's a modeling-time
    transformation, not raw data; only gdp_level is collected.
    """
    params = {
        "format": "JSON",
        "na_item": "B1GQ",
        "unit": "CLV10_MEUR",
        "s_adj": "SCA",
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000-Q1",
    }
    df = eurostat_json_to_df("namq_10_gdp", params)
    df = df.rename(columns={"geo": "country", "time": "quarter"})
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["gdp_level"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "quarter", "gdp_level"]].dropna().sort_values(["country", "quarter"])
    print(f"  [diagnostic] GDP level: {len(df)} country-quarter rows across "
          f"{df['country'].nunique()} countries.")
    return df


# ----------------------------------------------------------------------
# 3. ICT investment share of GFCF
# ----------------------------------------------------------------------

def fetch_ict_investment_share():
    """
    (ICT equipment + computer software/databases) GFCF as % of total
    GFCF, annual, per country. Eurostat dataset nama_10_a64_p5, filtered
    to nace_r2="TOTAL". From ..._TPU_pure_cum.py.

    ALSO fetches three additional asset-type breakdowns, reported
    alongside (NOT folded into the ict_share numerator/denominator --
    that computation is unchanged): N112G_secJ, "Other buildings and
    structures" gross fixed capital formation WITHIN NACE Rev.2 Section
    J (Information and communication) specifically; N117G_secJ,
    "Intellectual property products" gross fixed capital formation,
    same Section J restriction; and N1173G_secJ, "Computer software and
    databases" gross fixed capital formation, same Section J
    restriction (the Section-J-specific counterpart to the economy-wide
    N1173G already used in the ict_share numerator above -- a DIFFERENT
    number from N1173G, not a duplicate of it).

    SECTOR FILTER, stated clearly (a correction from an earlier
    version): these three series use nace_r2="J" (NOT "TOTAL" like the
    ICT-share numerator/denominator above) -- i.e. investment in these
    asset types made BY THE INFORMATION-AND-COMMUNICATION SECTOR
    ITSELF specifically (e.g. data centres, broadcasting studios,
    telecom network infrastructure, and software/IP developed within
    that sector), not economy-wide investment in these asset types.
    This needs its OWN separate Eurostat call (params_sector_j below),
    since it uses a different nace_r2 filter than the params_num/
    params_den calls that feed ict_share.

    CODE CORRECTION: the codes "N112" and "N117" (without the "G"
    suffix) were tried first and came back completely empty -- Eurostat's
    asset10 vocabulary (confirmed directly against
    https://dd.eionet.europa.eu/vocabulary/eurostat/asset10/) uses a
    G(ross)/N(et) suffix on EVERY code in this classification, matching
    the convention already used by N1132G/N1173G/N11G elsewhere in this
    same function -- "N112"/"N117" simply aren't valid codes in this
    vocabulary at all, which is why Eurostat silently returned nothing
    for them rather than erroring. N112G and N117G (confirmed via
    Eurostat's own vocabulary listing AND independently via published
    academic usage, e.g. the EU KLEMS/InteVoi methodology and a
    ScienceDirect infrastructure-investment paper both using
    "N112G Other buildings and structures" by that exact name) are the
    correct codes.
    """
    params_num = {
        "format": "JSON", "unit": "CP_MEUR", "nace_r2": "TOTAL",
        "asset10": ["N1132G", "N1173G"], "geo": COUNTRIES, "sinceTimePeriod": "2000",
    }
    params_den = {
        "format": "JSON", "unit": "CP_MEUR", "nace_r2": "TOTAL",
        "asset10": "N11G", "geo": COUNTRIES, "sinceTimePeriod": "2000",
    }
    params_sector_j = {
        "format": "JSON", "unit": "CP_MEUR", "nace_r2": "J",
        "asset10": ["N112G", "N117G", "N1173G"], "geo": COUNTRIES, "sinceTimePeriod": "2000",
    }
    raw_num = eurostat_json_to_df("nama_10_a64_p5", params_num)
    raw_den = eurostat_json_to_df("nama_10_a64_p5", params_den)
    raw_sector_j = eurostat_json_to_df("nama_10_a64_p5", params_sector_j)

    for name, d in (("N1132G+N1173G", raw_num), ("N11G", raw_den),
                     ("N112G+N117G+N1173G, nace_r2=J", raw_sector_j)):
        d.rename(columns={"geo": "country", "time": "year"}, inplace=True)
        missing = {"country", "year"} - set(d.columns)
        if missing:
            raise ValueError(
                f"fetch_ict_investment_share ({name} call): expected columns "
                f"{missing} not found. Actual columns: {list(d.columns)}."
            )
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["year"] = d["year"].astype(int)

    num_wide = raw_num.pivot_table(
        index=["country", "year"], columns="asset10", values="value"
    ).reset_index()
    num_wide.columns.name = None
    for col in ("N1132G", "N1173G"):
        if col not in num_wide.columns:
            num_wide[col] = np.nan

    n1173g_coverage = num_wide["N1173G"].notna().sum()
    if n1173g_coverage == 0:
        print("  [!] N1173G has ZERO non-null values for this country/period "
              "set -- known Eurostat coverage gap. Treating it as 0 in the "
              "numerator throughout.")
    else:
        n_total = len(num_wide)
        print(f"  [diagnostic] N1173G coverage: {n1173g_coverage}/{n_total} "
              f"country-year rows have a non-null value.")

    sector_j_wide = raw_sector_j.pivot_table(
        index=["country", "year"], columns="asset10", values="value"
    ).reset_index()
    sector_j_wide.columns.name = None
    for col in ("N112G", "N117G", "N1173G"):
        if col not in sector_j_wide.columns:
            sector_j_wide[col] = np.nan
    sector_j_wide = sector_j_wide.rename(
        columns={"N112G": "N112G_secJ", "N117G": "N117G_secJ",
                 "N1173G": "N1173G_secJ"})

    for col, label in (("N112G_secJ", "Other buildings and structures, Section J"),
                        ("N117G_secJ", "Intellectual property products, Section J"),
                        ("N1173G_secJ", "Computer software and databases, Section J")):
        coverage = sector_j_wide[col].notna().sum()
        print(f"  [diagnostic] {col} ({label}) coverage: {coverage}/{len(sector_j_wide)} "
              f"country-year rows have a non-null value.")

    den_agg = raw_den.groupby(["country", "year"])["value"].sum().rename("N11G").reset_index()

    wide = num_wide.merge(den_agg, on=["country", "year"], how="inner")
    wide = wide.merge(sector_j_wide, on=["country", "year"], how="left")
    wide["ict_numerator"] = wide["N1132G"] + wide["N1173G"].fillna(0)
    wide["ict_share"] = wide["ict_numerator"] / wide["N11G"]
    wide = wide.dropna(subset=["N1132G", "N11G", "ict_share"])

    if wide.empty:
        raise ValueError("fetch_ict_investment_share: zero rows remain after computing ict_share.")

    return wide[["country", "year", "N1132G", "N1173G", "N112G_secJ", "N117G_secJ",
                  "N1173G_secJ", "N11G", "ict_share"]]


# ----------------------------------------------------------------------
# 4. ETO/CSET AI patents and AI investment (local files)
# ----------------------------------------------------------------------

def fetch_ai_patents():
    """
    Annual AI-related PATENT APPLICATIONS per country. Source: ETO/CSET
    Country Activity Tracker (cat.eto.tech, dataset=Patent), exported
    locally as LOCAL_AI_PATENTS_FILE. From ai_model_WUI_pat_inv.py.
    """
    if not os.path.exists(LOCAL_AI_PATENTS_FILE):
        raise FileNotFoundError(
            f"fetch_ai_patents: local file '{LOCAL_AI_PATENTS_FILE}' not "
            "found next to this script. Export it from "
            "https://cat.eto.tech/?countries=Belgium%2CFrance%2CGermany%2CItaly%2CNetherlands%2CAustria%2CSpain%2CIreland%2CFinland%2CPortugal"
            "&dataset=Patent and save it under this filename."
        )

    df = pd.read_csv(LOCAL_AI_PATENTS_FILE, encoding="utf-8-sig")

    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    country_col = cols_upper.get("COUNTRY")
    metric_col = cols_upper.get("METRIC")
    if country_col is None or metric_col is None:
        raise ValueError(
            "fetch_ai_patents: expected columns 'Country' and 'Metric' "
            f"not found. Actual columns: {list(df.columns)}."
        )

    year_cols = [c for c in df.columns
                 if c not in (country_col, metric_col) and str(c).strip().isdigit()]
    if not year_cols:
        raise ValueError(
            "fetch_ai_patents: no year-like columns found. "
            f"Actual columns: {list(df.columns)}."
        )

    metric_values = set(df[metric_col].astype(str).str.strip())
    target_metric = next(
        (m for m in metric_values if m.strip().lower() == "patent applications"), None
    )
    if target_metric is None:
        raise ValueError(
            "fetch_ai_patents: 'Patent applications' not found in the "
            f"'{metric_col}' column. Actual values present: {sorted(metric_values)}."
        )

    df_apps = df[df[metric_col].astype(str).str.strip() == target_metric].copy()

    long = df_apps.melt(id_vars=[country_col], value_vars=year_cols,
                         var_name="year", value_name="ai_patents")
    long = long.rename(columns={country_col: "entity_name"})
    long["ai_patents"] = pd.to_numeric(long["ai_patents"], errors="coerce")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")

    inv_map = {v: k for k, v in COUNTRY_NAME_MAP.items()}
    raw_entity_names = set(long["entity_name"].unique())
    long_filtered = long[long["entity_name"].isin(inv_map.keys())].copy()
    long_filtered["country"] = long_filtered["entity_name"].map(inv_map)

    found = set(long_filtered["country"].unique())
    missing = set(COUNTRIES) - found
    if missing:
        raise ValueError(
            f"fetch_ai_patents: no rows found for {missing} (ISO2 codes) -- "
            f"expected entity names {[COUNTRY_NAME_MAP[m] for m in missing]} "
            f"were not present. Actual entity names in the file: {sorted(raw_entity_names)}."
        )

    long_filtered = long_filtered.dropna(subset=["ai_patents", "year"]).sort_values(
        ["country", "year"]
    )
    if long_filtered.empty:
        raise ValueError("fetch_ai_patents: zero valid rows remain after filtering.")

    by_year_total = long_filtered.groupby("year")["ai_patents"].sum()
    near_zero_years = by_year_total[by_year_total <= 300].index.tolist()
    if near_zero_years:
        print(f"  [!] fetch_ai_patents: years {near_zero_years} have near-zero "
              f"total patent applications -- likely INCOMPLETE due to reporting "
              f"lag, not genuine zero activity.")

    print(f"  [diagnostic] AI patent applications: {len(long_filtered)} "
          f"country-year rows across {long_filtered['country'].nunique()} "
          f"countries, years {int(long_filtered['year'].min())}-"
          f"{int(long_filtered['year'].max())}.")

    return long_filtered[["country", "year", "ai_patents"]]


def fetch_ai_investment():
    """
    Annual AI-related INCOMING INVESTMENT COUNTS per country. Source:
    ETO/CSET Country Activity Tracker (cat.eto.tech, dataset=Investment),
    exported locally as LOCAL_AI_INVESTMENT_FILE. From
    ai_model_WUI_pat_inv.py.
    """
    if not os.path.exists(LOCAL_AI_INVESTMENT_FILE):
        raise FileNotFoundError(
            f"fetch_ai_investment: local file '{LOCAL_AI_INVESTMENT_FILE}' "
            "not found next to this script. Export it from "
            "https://cat.eto.tech/?countries=Belgium%2CFrance%2CGermany%2CItaly%2CNetherlands%2CAustria%2CSpain%2CIreland%2CFinland%2CPortugal"
            "&dataset=Investment and save it under this filename."
        )

    df = pd.read_csv(LOCAL_AI_INVESTMENT_FILE, encoding="utf-8-sig")

    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    country_col = cols_upper.get("COUNTRY")
    metric_col = cols_upper.get("METRIC")
    if country_col is None or metric_col is None:
        raise ValueError(
            "fetch_ai_investment: expected columns 'Country' and 'Metric' "
            f"not found. Actual columns: {list(df.columns)}."
        )

    year_cols = [c for c in df.columns
                 if c not in (country_col, metric_col) and str(c).strip().isdigit()]
    if not year_cols:
        raise ValueError(
            "fetch_ai_investment: no year-like columns found. "
            f"Actual columns: {list(df.columns)}."
        )

    metric_values = set(df[metric_col].astype(str).str.strip())
    target_metric = next(
        (m for m in metric_values if m.strip().lower() == "incoming investment counts"), None
    )
    if target_metric is None:
        raise ValueError(
            "fetch_ai_investment: 'Incoming investment counts' not found "
            f"in the '{metric_col}' column. Actual values present: {sorted(metric_values)}."
        )

    df_inv = df[df[metric_col].astype(str).str.strip() == target_metric].copy()

    long = df_inv.melt(id_vars=[country_col], value_vars=year_cols,
                        var_name="year", value_name="ai_investment")
    long = long.rename(columns={country_col: "entity_name"})
    long["ai_investment"] = pd.to_numeric(long["ai_investment"], errors="coerce")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")

    inv_map = {v: k for k, v in COUNTRY_NAME_MAP.items()}
    raw_entity_names = set(long["entity_name"].unique())
    long_filtered = long[long["entity_name"].isin(inv_map.keys())].copy()
    long_filtered["country"] = long_filtered["entity_name"].map(inv_map)

    found = set(long_filtered["country"].unique())
    missing = set(COUNTRIES) - found
    if missing:
        raise ValueError(
            f"fetch_ai_investment: no rows found for {missing} (ISO2 codes) -- "
            f"expected entity names {[COUNTRY_NAME_MAP[m] for m in missing]} "
            f"were not present. Actual entity names in the file: {sorted(raw_entity_names)}."
        )

    long_filtered = long_filtered.dropna(subset=["ai_investment", "year"]).sort_values(
        ["country", "year"]
    )
    if long_filtered.empty:
        raise ValueError("fetch_ai_investment: zero valid rows remain after filtering.")

    by_year_total = long_filtered.groupby("year")["ai_investment"].sum()
    near_zero_years = by_year_total[by_year_total <= 300].index.tolist()
    if near_zero_years:
        print(f"  [!] fetch_ai_investment: years {near_zero_years} have "
              f"near-zero total incoming investment counts -- check whether "
              f"this reflects genuine activity or an incomplete/lagged year.")

    print(f"  [diagnostic] AI incoming investment counts: {len(long_filtered)} "
          f"country-year rows across {long_filtered['country'].nunique()} "
          f"countries, years {int(long_filtered['year'].min())}-"
          f"{int(long_filtered['year'].max())}.")

    return long_filtered[["country", "year", "ai_investment"]]


# ----------------------------------------------------------------------
# 5. National and semiconductor equity indices (Yahoo Finance)
# ----------------------------------------------------------------------

def fetch_national_indices_raw(countries=COUNTRIES):
    """
    Raw quarterly close price + log return for each country's national
    headline equity index (see NATIONAL_INDEX_TICKERS). From
    ai_model_WUI_pat_inv.py.
    """
    out = []
    for c in countries:
        ticker = NATIONAL_INDEX_TICKERS.get(c)
        if ticker is None:
            print(f"  [!] WARNING: no ticker configured for {c} in "
                  f"NATIONAL_INDEX_TICKERS -- skipping this country entirely.")
            continue
        px = _yf_close_series(ticker)
        if px.empty:
            print(f"  [!] WARNING: yfinance returned ZERO rows for {c} "
                  f"(ticker '{ticker}') -- skipping this country. This "
                  f"usually means the ticker symbol is stale/wrong (e.g. "
                  f"an index was renamed or moved exchange suffix) rather "
                  f"than a genuine data gap -- verify the symbol still "
                  f"resolves to live data on Yahoo Finance and update "
                  f"NATIONAL_INDEX_TICKERS if not. Without a fix, this "
                  f"country will be SILENTLY missing from the index_nat "
                  f"sheet and from any spec that depends on it (e.g. the "
                  f"Corr/hs_export exposure specs) -- this exact issue "
                  f"previously went unnoticed until the modeling script's "
                  f"own downstream warning caught it much later.")
            continue
        q = px.resample("QE").last()
        ret = np.log(q / q.shift(1))
        df = pd.DataFrame({"close": q, "log_ret": ret})
        df.index = df.index.to_period("Q")
        df = df.reset_index().rename(columns={"index": "quarter"})
        if "quarter" not in df.columns:
            df = df.rename(columns={df.columns[0]: "quarter"})
        df["country"] = c
        df["ticker"] = ticker
        out.append(df)
    result = pd.concat(out, ignore_index=True)
    result = result[["country", "ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])
    print(f"  [diagnostic] National indices: {len(result)} rows across "
          f"{result['country'].nunique()} countries.")
    missing_countries = set(countries) - set(result["country"].unique())
    if missing_countries:
        print(f"  [!] WARNING: {missing_countries} are MISSING from the final "
              f"national-indices result -- see the per-country warnings above "
              f"for why.")
    return result


def fetch_semiconductor_raw():
    """
    Raw quarterly close price + log return for the global semiconductor
    index (^SOX). From ai_model_WUI_pat_inv.py.
    """
    px = _yf_close_series("^SOX")
    q = px.resample("QE").last()
    ret = np.log(q / q.shift(1))
    df = pd.DataFrame({"close": q, "log_ret": ret})
    df.index = df.index.to_period("Q")
    df = df.reset_index().rename(columns={"index": "quarter"})
    if "quarter" not in df.columns:
        df = df.rename(columns={df.columns[0]: "quarter"})
    df["ticker"] = "^SOX"
    result = df[["ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])
    print(f"  [diagnostic] Semiconductor index (^SOX): {len(result)} rows.")
    return result


# ----------------------------------------------------------------------
# 6. AI/ICT-related HS export data (OECD BIMTS)
# ----------------------------------------------------------------------

def fetch_hs_export_data():
    """
    HS 847150+847180+847330+848610+848620+848630+848640+848690 exports
    AND total merchandise exports ("_T") -- all from the SAME BIMTS
    source, in one request -- for the 7 target countries to the World,
    2000-onward. From hs_export.py (adapted here to RETURN a DataFrame
    rather than writing directly to the workbook itself, so this
    collection script can write every sheet in one unified step at the
    end).
    """
    ref_area = "+".join(COUNTRIES_ISO3)
    product_hs = "+".join(f"HS17_{code}" for code in HS_CODES) + "+_T"
    key = f"{ref_area}.W..C..{product_hs}.A.USD_EXC."

    url = (
        "https://sdmx.oecd.org/sti-public/rest/data/"
        "OECD.SDD.TPS,DSD_BIMTS_6D@DF_BIMTS_HS2017_6D,1.0/"
        f"{key}"
    )
    params = {
        "startPeriod": "2000",
        "dimensionAtObservation": "AllDimensions",
        "format": "csvfilewithlabels",
    }

    print("  Requesting BIMTS HS export data:")
    print(f"    {url}")

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/csv,application/csv,text/plain,*/*",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=120)
    except requests.exceptions.RequestException as e:
        raise SystemExit(
            f"\nfetch_hs_export_data: network request failed: "
            f"{e.__class__.__name__}: {e}"
        )

    if r.status_code != 200:
        raise SystemExit(
            f"\nfetch_hs_export_data: request failed with HTTP {r.status_code}.\n"
            f"Response body (first 2000 chars):\n{r.text[:2000]}"
        )

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise SystemExit(
            "\nfetch_hs_export_data: request succeeded (HTTP 200) but "
            "returned ZERO rows."
        )

    cols_upper = {str(c).strip().upper(): c for c in df.columns}

    def find_col(*candidates):
        for cand in candidates:
            if cand in cols_upper:
                return cols_upper[cand]
        return None

    col_ref_area = find_col("REF_AREA", "REFERENCE AREA", "REFERENCE_AREA")
    col_trade_flow = find_col("TRADE_FLOW", "TRADE FLOW")
    col_product_hs = find_col("PRODUCT_HS", "PRODUCT HS", "HS PRODUCT")
    col_adjustment = find_col("ADJUSTMENT")
    col_time = find_col("TIME_PERIOD", "TIME PERIOD")
    col_value = find_col("OBS_VALUE", "OBSERVATION VALUE")

    missing = [name for name, col in [
        ("REF_AREA", col_ref_area), ("TRADE_FLOW", col_trade_flow),
        ("PRODUCT_HS", col_product_hs), ("TIME_PERIOD", col_time),
        ("OBS_VALUE", col_value),
    ] if col is None]
    if missing:
        raise SystemExit(
            f"\nfetch_hs_export_data: could not find expected column(s) "
            f"{missing} in the response. Actual columns: {list(df.columns)}"
        )

    df[col_trade_flow] = df[col_trade_flow].astype(str).str.strip().str.upper()
    mask = df[col_trade_flow] == "X"

    if col_adjustment is not None:
        df[col_adjustment] = df[col_adjustment].astype(str).str.strip().str.upper()
        adj_mask = df[col_adjustment] == "B_ADJ_RX"
        if adj_mask.any():
            mask &= adj_mask
        else:
            print("  [!] WARNING: 'B_ADJ_RX' not found in ADJUSTMENT -- "
                  "proceeding WITHOUT that filter.")

    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise SystemExit("\nfetch_hs_export_data: zero rows remain after filtering.")

    filtered["country"] = filtered[col_ref_area].astype(str).str.strip().str.upper().map(ISO3_TO_ISO2)
    filtered["hs_code_raw"] = filtered[col_product_hs].astype(str).str.strip().str.upper()
    filtered["year"] = pd.to_numeric(filtered[col_time], errors="coerce").astype("Int64")
    filtered["value_usd"] = pd.to_numeric(filtered[col_value], errors="coerce")
    filtered = filtered.dropna(subset=["country", "hs_code_raw", "year", "value_usd"])

    wide = filtered.pivot_table(
        index=["country", "year"], columns="hs_code_raw", values="value_usd", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    rename_map = {f"HS17_{code}": f"HS_{code}" for code in HS_CODES}
    rename_map["_T"] = "total_export"
    wide = wide.rename(columns=rename_map)

    hs_cols = [f"HS_{c}" for c in HS_CODES]
    expected_cols = ["country", "year"] + hs_cols + ["total_export"]
    for col in expected_cols:
        if col not in wide.columns:
            print(f"  [!] WARNING: expected column '{col}' not present -- "
                  f"added as all-missing.")
            wide[col] = pd.NA

    wide = wide[expected_cols].sort_values(["country", "year"]).reset_index(drop=True)
    wide["year"] = wide["year"].astype(int)

    # share: combined HS export value as a share of total merchandise exports
    wide["share"] = wide[hs_cols].sum(axis=1) / wide["total_export"]

    print(f"  [diagnostic] HS export data: {wide.shape[0]} country-year rows.")
    return wide


# ----------------------------------------------------------------------
# 7. Growth-literature control variables (added for a referee-requested
#    robustness check: trade openness, labor productivity level, GDP
#    per capita, population growth -- time-varying controls not
#    captured by entity fixed effects alone).
# ----------------------------------------------------------------------

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country"


def _fetch_world_bank_indicator(indicator_code, value_col_name, countries=COUNTRIES_ISO3):
    """
    Shared logic for a single World Bank Open Data indicator, annual,
    for this project's countries. World Bank's v2 Indicators API
    (well-documented, stable, long-standing -- see
    datahelpdesk.worldbank.org) returns JSON as a 2-element array:
    [metadata_dict, list_of_observations]. Each observation has
    countryiso3code, date (year, as a string), and value (float or
    null for that country-year).

    HONESTY NOTE: this exact query was not independently live-tested
    against the World Bank API before delivery (unlike this project's
    OECD BIMTS queries, which went through extensive live trial-and-
    error earlier) -- the request/response format below follows the
    World Bank's own official API documentation closely and should
    work, but if it doesn't on first run, paste the printed diagnostic
    output (HTTP status, first part of the response) back to Claude.
    """
    country_codes = ";".join(c.lower() for c in countries)
    url = f"{WORLD_BANK_BASE}/{country_codes}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": "20000",
              "date": f"{SAMPLE_START[:4]}:{SAMPLE_END[:4]}"}

    print(f"  Requesting World Bank indicator {indicator_code}:")
    print(f"    {url}")

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(
            f"\n_fetch_world_bank_indicator({indicator_code}): network "
            f"request failed: {e.__class__.__name__}: {e}"
        )

    try:
        payload = r.json()
    except ValueError:
        raise SystemExit(
            f"\n_fetch_world_bank_indicator({indicator_code}): response "
            f"was not valid JSON. First 1000 chars:\n{r.text[:1000]}"
        )

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise SystemExit(
            f"\n_fetch_world_bank_indicator({indicator_code}): unexpected "
            f"response shape (expected a 2-element [metadata, observations] "
            f"list). Raw response (first 1500 chars):\n{str(payload)[:1500]}"
        )

    observations = payload[1]
    if not observations:
        raise SystemExit(
            f"\n_fetch_world_bank_indicator({indicator_code}): zero "
            f"observations returned. Metadata: {payload[0]}"
        )

    df = pd.DataFrame(observations)
    missing = {"countryiso3code", "date", "value"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"\n_fetch_world_bank_indicator({indicator_code}): expected "
            f"columns {missing} not found. Actual columns: {list(df.columns)}."
        )

    df["country"] = df["countryiso3code"].map(ISO3_TO_ISO2)
    df["year"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")
    df[value_col_name] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["country", "year", value_col_name])

    found = set(df["country"].unique())
    missing_countries = set(COUNTRIES) - found
    if missing_countries:
        print(f"  [!] WARNING: no data returned for {missing_countries} -- "
              f"check these countries' ISO3 codes are valid World Bank "
              f"country codes.")

    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"  [diagnostic] {indicator_code}: {len(df)} country-year rows "
          f"across {df['country'].nunique()} countries, years "
          f"{int(df['year'].min())}-{int(df['year'].max())}.")
    return df[["country", "year", value_col_name]]


def fetch_trade_openness():
    """
    Trade (% of GDP) -- exports plus imports of goods and services,
    as a share of GDP. World Bank indicator NE.TRD.GNFS.ZS.
    https://data.worldbank.org/indicator/NE.TRD.GNFS.ZS
    """
    return _fetch_world_bank_indicator("NE.TRD.GNFS.ZS", "trade_openness")


def fetch_population_growth():
    """
    Population growth (annual %). World Bank indicator SP.POP.GROW.
    https://data.worldbank.org/indicator/SP.POP.GROW
    """
    return _fetch_world_bank_indicator("SP.POP.GROW", "population_growth")


def fetch_productivity_growth():
    """
    Real labour productivity per hour worked, YEAR-ON-YEAR growth
    (log-difference x100), computed from the underlying index
    (2015=100). Eurostat dataset nama_10_lp_ulc, na_item=RLPR_HW.
    https://ec.europa.eu/eurostat/databrowser/view/tipsna70

    SWITCHED FROM OECD PDB (MEASURE=GDPHRS) to Eurostat after the OECD
    PDB query consistently failed with HTTP 500 ("Object reference not
    set to an instance of an object") for this project's specific
    10-country combination -- even after the query was corrected to
    exactly match an independently confirmed-working reference
    template. The likely cause was never fully isolated (a data gap
    for one specific country was suspected but not confirmed), and
    Eurostat is both already proven reliable elsewhere in this script
    (fetch_gdp_level(), fetch_ict_investment_share()) and the more
    natural source for a euro-area-only panel like this one.

    DELIBERATE EXCEPTION to this script's usual "collect levels only,
    compute growth at modeling time" convention (see fetch_gdp_level()'s
    docstring for that general rule): this series is explicitly wanted
    as a GROWTH rate (it is a referee-requested growth-literature
    control variable -- see the original request this whole batch of
    four new series was added for), so the transformation is done here
    rather than left to the modeling scripts. Growth is computed as
    100*(ln(index_t) - ln(index_{t-1})), matching this project's
    existing log-difference growth convention used elsewhere (e.g.
    dgdp in the modeling scripts) rather than a simple percentage-
    change formula, for consistency. Annual data, so period-over-period
    here is already year-on-year (no separate multi-period lag needed,
    unlike GDP's quarterly YoY, which needs a 4-quarter shift).
    """
    params = {
        "format": "JSON",
        "na_item": "RLPR_HW",
        "unit": "I15",
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000",
    }
    df = eurostat_json_to_df("nama_10_lp_ulc", params)
    df = df.rename(columns={"geo": "country", "time": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["index_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "year", "index_value"]].dropna().sort_values(["country", "year"])

    df["productivity_growth"] = df.groupby("country")["index_value"].transform(
        lambda s: 100 * (np.log(s) - np.log(s.shift(1)))
    )
    df = df.dropna(subset=["productivity_growth"])

    print(f"  [diagnostic] Real labour productivity growth (YoY, log-diff): "
          f"{len(df)} country-year rows across {df['country'].nunique()} "
          f"countries.")
    return df[["country", "year", "productivity_growth"]]


def fetch_gdp_per_capita():
    """
    GDP per capita, chain-linked volumes (2010), euro per capita.
    Eurostat dataset nama_10_pc, na_item=B1GQ (the same GDP code
    fetch_gdp_level() already uses, just per-capita instead of
    aggregate), unit=CLV10_EUR_HAB.
    https://ec.europa.eu/eurostat/web/products-datasets/-/nama_10_pc

    SWITCHED FROM OECD PDB (MEASURE=GDPPOP) to Eurostat for the same
    reason as fetch_productivity_growth() above: the OECD PDB query
    consistently failed with HTTP 500 for this project's specific
    10-country combination, even using an independently confirmed-
    valid unit/price-base combination for this exact measure -- see
    fetch_productivity_growth()'s docstring for the fuller explanation.
    """
    params = {
        "format": "JSON",
        "na_item": "B1GQ",
        "unit": "CLV10_EUR_HAB",
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000",
    }
    df = eurostat_json_to_df("nama_10_pc", params)
    df = df.rename(columns={"geo": "country", "time": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gdp_per_capita"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "year", "gdp_per_capita"]].dropna().sort_values(["country", "year"])
    print(f"  [diagnostic] GDP per capita: {len(df)} country-year rows "
          f"across {df['country'].nunique()} countries.")
    return df


def fetch_hicp_index():
    """
    HICP (Harmonised Index of Consumer Prices), INDEX LEVEL (2015=100),
    per country -- NOT a year-on-year rate. Eurostat dataset
    prc_hicp_midx ("HICP - monthly data, index"), unit=I15 (index,
    2015=100), coicop=CP00 (All-items HICP -- the headline measure, not
    a COICOP sub-category).
    https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_midx

    Confirmed dataset/parameter combination (unit=I15, coicop=CP00) via
    multiple independent, published query examples for this exact
    dataset.

    Monthly at source (unlike this project's other Eurostat control
    series, which are all annual) -- resampled to quarterly (mean), the
    same convention this project already uses for other monthly
    sources (e.g. fetch_gpr_global()'s monthly-to-quarterly resample).
    This is a plain INDEX LEVEL -- any growth-rate transformation
    (YoY, QoQ, etc.) is left to the modeling scripts, matching this
    project's usual "collect levels only" convention (see
    fetch_gdp_level()'s docstring) -- unlike fetch_productivity_growth(),
    which is a deliberate, referee-requested exception to that
    convention.
    """
    params = {
        "format": "JSON",
        "unit": "I15",
        "coicop": "CP00",
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000-01",
    }
    df = eurostat_json_to_df("prc_hicp_midx", params)
    df = df.rename(columns={"geo": "country", "time": "month"})
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df["hicp_index"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "month", "hicp_index"]].dropna().sort_values(["country", "month"])
    if df.empty:
        raise ValueError(
            "fetch_hicp_index: zero valid (country, month, value) rows "
            "after parsing the Eurostat response -- check unit/coicop codes "
            "are still current."
        )

    out = []
    for country, g in df.groupby("country"):
        g = g.set_index("month")
        q = g["hicp_index"].resample("QE").mean()
        q.index = q.index.to_period("Q")
        q = q.reset_index().rename(columns={"month": "quarter"})
        q["country"] = country
        out.append(q)
    result = pd.concat(out, ignore_index=True)
    result = result[["country", "quarter", "hicp_index"]].dropna()

    print(f"  [diagnostic] HICP index (2015=100): {len(result)} country-quarter "
          f"rows across {result['country'].nunique()} countries.")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("DATA COLLECTION -- fetching every raw series this project uses")
    print("=" * 70)

    sheets = {}

    print("\n[1/18] GDP level (Eurostat)...")
    sheets["gdp"] = fetch_gdp_level()

    print("\n[2/18] ICT investment share (Eurostat)...")
    sheets["ict_inv"] = fetch_ict_investment_share()

    print("\n[3/18] AI patent applications (ETO/CSET, local file)...")
    sheets["ai_patent"] = fetch_ai_patents()

    print("\n[4/18] AI incoming investment counts (ETO/CSET, local file)...")
    sheets["ai_inv"] = fetch_ai_investment()

    print("\n[5/18] National equity indices (Yahoo Finance)...")
    sheets["index_nat"] = fetch_national_indices_raw()

    print("\n[6/18] Semiconductor index ^SOX (Yahoo Finance)...")
    sheets["index_sox"] = fetch_semiconductor_raw()

    print("\n[7/18] AI/ICT-related HS export data (OECD BIMTS)...")
    sheets["hs_export"] = fetch_hs_export_data()

    print("\n[8/18] World Uncertainty Index (WUI)...")
    sheets["wui"] = fetch_wui_global()

    print("\n[9/18] Geopolitical Risk Index (GPR)...")
    sheets["gpr"] = fetch_gpr_global()

    print("\n[10/18] Economic Policy Uncertainty Index (EPU)...")
    sheets["epu"] = fetch_epu_global()

    print("\n[11/18] Trade Policy Uncertainty Index (TPU)...")
    sheets["tpu"] = fetch_tpu_global()

    print("\n[12/18] Global Supply Chain Pressure Index (GSCPI)...")
    sheets["gscpi"] = fetch_gscpi_global()

    print("\n[13/18] Global commodity price index, incl. oil and gas (FRED)...")
    sheets["com"] = fetch_commodity_price_index()

    print("\n[14/18] Trade openness (World Bank)...")
    sheets["trade_openness"] = fetch_trade_openness()

    print("\n[15/18] Population growth (World Bank)...")
    sheets["population_growth"] = fetch_population_growth()

    print("\n[16/18] Labor productivity growth: real productivity per hour worked, YoY (Eurostat)...")
    sheets["productivity_growth"] = fetch_productivity_growth()

    print("\n[17/18] GDP per capita (Eurostat)...")
    sheets["gdp_per_capita"] = fetch_gdp_per_capita()

    print("\n[18/18] HICP index level, 2015=100 (Eurostat)...")
    sheets["infl"] = fetch_hicp_index()

    # Convert any Period-typed "quarter" columns to plain strings
    # (e.g. "2000Q1") for Excel -- matches the existing ai_data.xlsx
    # convention exactly; Excel/openpyxl cannot write pandas Period
    # objects directly.
    for name, df in sheets.items():
        if "quarter" in df.columns:
            df["quarter"] = df["quarter"].astype(str)

    print(f"\n{'=' * 70}")
    print(f"Writing {len(sheets)} sheets to {OUTPUT_FILE}...")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # FORMULA PRESERVATION: ict_inv column F (ict_share) and hs_export
    # column L (share) are written as GENUINE, LIVE EXCEL FORMULAS
    # (visible/editable in Excel's formula bar), matching exactly what
    # the attached reference ai_data.xlsx has in those two columns --
    # not pre-computed static values, per the explicit request.
    #
    # xlsxwriter (not openpyxl) is used for the final write specifically
    # because it supports write_formula(row, col, formula, value=...):
    # a formula string PLUS a manually-supplied cached result in the
    # same call. This matters because openpyxl-written formulas have NO
    # cached value at all until the file is opened and saved by real
    # Excel (or LibreOffice, etc.) at least once -- confirmed by testing
    # that pandas.read_excel() sees NaN for an openpyxl-only formula
    # cell. Since the rest of this project's modeling scripts read
    # ai_data.xlsx via plain pandas.read_excel() and expect a NUMERIC
    # ict_share/share value, an openpyxl-only formula write would
    # silently break them until someone manually opens+resaves the
    # file. xlsxwriter's cached-value parameter avoids that entirely:
    # the formula is genuinely live in Excel, AND pandas immediately
    # sees the correct number, with no manual Excel round-trip needed.
    #
    # IMPORTANT DISCREPANCY, kept exactly as found (not "fixed"): the
    # hs_export formula in the attached reference file,
    # "=SUM(F{row}:J{row})/K{row}", sums ONLY the 5 newer HS_8486xx
    # columns (F:J) -- it does NOT include the original 3 HS_8471xx
    # columns (C:E) in the numerator, unlike this script's own
    # fetch_hs_export_data(), which computes "share" as the sum of ALL
    # 8 HS columns. This was verified against the reference file's own
    # CACHED formula result (0.002045 for AT/2000), which matches
    # "sum of only the 5 columns" and NOT "sum of all 8" (0.008831).
    # This is very likely an unintentional oversight from when the
    # sheet was extended from 3 to 8 HS codes (the SUM range wasn't
    # widened to also cover the original 3 columns) -- but the request
    # was to keep the formula exactly as it is in the attached file, so
    # that is what this script does. If this was NOT intentional, the
    # fix is a one-word change to the formula string below
    # ("=SUM(C{row}:J{row})/K{row}" instead of "=SUM(F{row}:J{row})/K{row}").
    # ------------------------------------------------------------------

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            if sheet_name == "ict_inv":
                # columns: A=country B=year C=N1132G D=N1173G E=N112G_secJ
                #          F=N117G_secJ G=N1173G_secJ H=N11G I=ict_share
                col_f = list(df.columns).index("ict_share")
                # N11G's column letter is derived dynamically (not hardcoded
                # as "E") -- N112G_secJ, N117G_secJ, and N1173G_secJ were
                # inserted before it, shifting N11G from column E to H. A
                # hardcoded "/E{row}" reference here would silently divide
                # by the wrong column (N112G_secJ) instead of N11G after
                # that insertion -- this keeps the
                # formula correct regardless of how many columns end up
                # between N1173G and N11G in the future.
                n11g_col_idx = list(df.columns).index("N11G")
                n11g_letter = chr(ord("A") + n11g_col_idx)
                for i, row in df.iterrows():
                    excel_row = i + 2  # 1-based, +1 for header row
                    formula = f"=SUM(C{excel_row}:D{excel_row})/{n11g_letter}{excel_row}"
                    # Defensive fallback (fetch_ict_investment_share() already
                    # guarantees a non-NaN ict_share for every row it returns,
                    # so this branch shouldn't normally trigger -- kept as a
                    # safety net matching the hs_export fix below, since a
                    # NaN passed to write_formula(value=...) corrupts the
                    # cached result in a way that crashes pandas.read_excel()
                    # later (see the hs_export comment for the full story).
                    ict_share_val = row["ict_share"]
                    cached_ict_share = 0.0 if pd.isna(ict_share_val) else ict_share_val
                    worksheet.write_formula(i + 1, col_f, formula, value=cached_ict_share)

            elif sheet_name == "hs_export":
                # columns: A=country B=year C..J=8 HS codes K=total_export L=share
                col_l = list(df.columns).index("share")
                for i, row in df.iterrows():
                    excel_row = i + 2
                    formula = f"=SUM(F{excel_row}:J{excel_row})/K{excel_row}"
                    # BUGFIX: a plain Python "+" between the 5 HS_848xxx
                    # values propagates NaN if ANY single one is missing
                    # (e.g. BIMTS genuinely has no recorded IE/2019/HS_848630
                    # observation) -- unlike Excel's own SUM() function (which
                    # treats a blank cell as 0), and unlike pandas' .sum()
                    # (skipna=True by default, used when "share" was first
                    # computed in fetch_hs_export_data()). That mismatch is
                    # exactly what produced a NaN cached_value here, which
                    # xlsxwriter's write_formula(value=...) then stored as
                    # literal text "nan" in the cell's cached-result XML --
                    # invisible under a normal openpyxl read, but fatal for
                    # pandas.read_excel()'s default (read_only + data_only)
                    # reading path, which tries int("nan") and crashes.
                    # Fixed by skipping NaN terms explicitly (matching both
                    # Excel's own SUM() semantics and the original "share"
                    # column's pandas .sum() -- so the cached value now
                    # matches what Excel will compute on next recalculation,
                    # not just "some non-crashing number").
                    hs_cols_l = ["HS_848610", "HS_848620", "HS_848630", "HS_848640", "HS_848690"]
                    hs_sum = sum(0.0 if pd.isna(row[c]) else row[c] for c in hs_cols_l)
                    total_val = row["total_export"]
                    if pd.isna(total_val) or total_val == 0:
                        cached_value = 0.0
                    else:
                        cached_value = hs_sum / total_val
                    worksheet.write_formula(i + 1, col_l, formula, value=cached_value)

    print(f"\nDONE. Sheets written: {list(sheets.keys())}")
    for name, df in sheets.items():
        print(f"  {name}: {df.shape[0]} rows, columns={list(df.columns)}")
    print(f"\n{OUTPUT_FILE.name} is ready -- upload it back to Claude, or run one of "
          f"the modeling scripts locally, which read this file as their single "
          f"local data source.")
