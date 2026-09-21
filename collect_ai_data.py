"""
Consolidated DATA COLLECTION script for the GPR/WUI/EPU/TPU/GSCPI AI-
mitigation panel project. Collects EVERY raw data series this project
uses -- GDP, ICT investment, AI patents, AI investment, national,
semiconductor, MSCI World, and STOXX Europe 600 equity indices, all
five global shock series (WUI, GPR, EPU, TPU, GSCPI), and the AI/ICT-
related HS export data -- and writes them all into a single local
workbook, ai_data.xlsx, with exactly the sheet structure the rest of
this project's scripts (the local-data modeling scripts) expect: gdp,
ict_inv, ai_patent, ai_inv, index_nat, index_sox, index_msci,
index_stoxx600, hs_export, eur_export, wui, gpr, epu, tpu, gscpi, com,
trade_openness, population_growth, productivity_growth, gdp_per_capita,
infl.

THIS SCRIPT DOES NOT MODEL ANYTHING: no panel construction, no local
projections, no IRFs, no regressions, no charts. It is purely a data-
collection/refresh step -- run this whenever the underlying source data
needs updating, then run one of the modeling scripts (which read
ai_data.xlsx as their single local data source) separately.

FUTURE-PROOFING, stated honestly (what is and isn't automatic here):
  - SAMPLE_END is now DYNAMIC (today + a 32-day forward buffer), not a
    hardcoded date -- Eurostat/World Bank/Yahoo Finance sources below
    will pick up newly published data automatically on every run,
    with no manual date bump ever needed again.
  - The WUI download URL CANNOT be made fully automatic -- each new
    WUI release is published at an unpredictable new URL (not a
    stable/versioned endpoint), so the hardcoded URL in
    fetch_wui_global() will eventually go stale. What IS automated:
    a staleness check that prints a loud, actionable warning (with
    the exact URL to check) if the most recently fetched WUI quarter
    is more than ~9 months behind today.
  - The local AI-patents/AI-investment files (from cat.eto.tech, no
    scriptable API exists) have the same limitation and the same kind
    of staleness check -- a warning fires if the local file's most
    recent year is more than 2 years behind today.
  - AMECO EC-forecast vintages are handled by the SEPARATE
    collect_ec_forecast_data.py script, not this one -- see that
    script's own docstring for its (also only partially automatable)
    situation.

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
import datetime
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
# SAMPLE_END is now DYNAMIC (today + a small forward buffer), not a
# hardcoded date -- confirmed both usages of SAMPLE_END below (Yahoo
# Finance equity prices, World Bank indicators) are purely HISTORICAL
# data sources with no forward-looking/forecast component, so setting
# this to "today" carries no risk of silently excluding data. The
# +32-day buffer absorbs timezone differences between this machine and
# the data providers, and lets a same-day run still pick up whatever
# has already been published for "today" in a provider's own timezone,
# without needing this constant to be manually bumped ever again.
SAMPLE_END = (datetime.date.today() + datetime.timedelta(days=32)).isoformat()
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


def eurostat_json_to_df(dataset, params, base_url=None):
    """Fetch and flatten a Eurostat JSON-stat response into a long DataFrame.
    base_url: override for the standard Eurostat dissemination endpoint --
    used for DS-prefixed Comext/Prodcom datasets, which are served from a
    DIFFERENT API base (.../api/comext/dissemination/... instead of
    .../api/dissemination/...); see fetch_eur_export_data() below."""
    url = f"{base_url or EUROSTAT_BASE}/{dataset}"
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


def _comext_dimension_categories(dataset, params, base_url):
    """
    Runs a Comext/Eurostat query and returns {dimension_id: {code: label}}
    for EVERY dimension in the response, regardless of whether any actual
    observations matched -- used as a DISCOVERY step (see
    _discover_cpa_codes() below) to find the real, valid codes
    for a dimension in a dataset whose exact code notation isn't known
    in advance, rather than guessing at a code and hoping it's right.
    Unlike eurostat_json_to_df(), this does NOT raise when zero rows
    match; a dimension's code list is still present and usable in that
    case (which is exactly the situation this function is FOR).
    """
    url = f"{base_url}/{dataset}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if "dimension" not in js or "id" not in js:
        raise ValueError(
            f"_comext_dimension_categories: response for '{dataset}' has no "
            f"'dimension'/'id' keys at all (not even a structure-only "
            f"response) -- params={params}. Raw response (first 1000 chars): "
            f"{str(js)[:1000]}"
        )
    dims = js["dimension"]
    result = {}
    for d in js["id"]:
        cat = dims[d]["category"]
        labels = cat.get("label", {})
        codes = list(cat["index"].keys())
        result[d] = {c: labels.get(c, c) for c in codes}
    return result


def _discover_cpa_codes(reporter_probe, dataset, base_url, probe_year, all_reporters):
    """
    Discovers the REAL product codes for CPA groups 26.1, 26.2, 28.99,
    "total", and the REAL intra-EU partner code, in this specific Comext
    dataset -- by running one query with NO product/partner filter (a
    single reporter/flow/year, so the request stays small), then VALIDATING
    each candidate product code with its own small query that actually
    includes it as a filter value (rather than trusting the structure
    listing alone). ALSO probes whether "+"-joining multiple values into
    one request actually works for the reporter and product dimensions in
    this dataset, so the caller can use far fewer, larger requests when it
    does, instead of always falling back to one request per value.

    Live-run problems that drove this design:
    1. This project's original guesses at the product code notation
       (e.g. "CPA22_26.1") were confirmed WRONG. A later attempt found
       candidate codes ("261" etc.) via the STRUCTURE listing (the set
       of categories Eurostat declares for the product dimension), but
       those candidates STILL matched zero observations when actually
       submitted as filter values -- meaning appearing in the structure
       listing does not guarantee a code works as a filter value here.
       This function now VALIDATES each candidate with its own query.
    2. The assumed intra-EU partner code "EU27_2020" was NEVER
       recognized in any live run, while "EXT_EU27_2020" (extra-EU) was
       -- following that same "EXT_"-prefix pattern, this function also
       discovers the intra-EU code by label-matching ("intra" in the
       partner label) rather than assuming "EU27_2020".
    3. "+"-joining multiple reporter OR product values into one request
       was separately confirmed, in a live run, to silently match zero
       observations for BOTH dimensions -- forcing one request per
       individual value, which is safe but multiplies the request count
       by (countries x products) and made a full 2000-2025 fetch far too
       slow. This function now tests each dimension's joinability
       directly, so the fetch only pays that cost where it's genuinely
       required.

    Returns (product_code_map, intra_partner_code, reporter_can_join,
    product_can_join):
      product_code_map: {"26.1": code_or_None, "26.2": ..., "28.99": ...,
                          "total": ...} -- a None value means no VALIDATED
                          code was found for that target.
      intra_partner_code: the discovered code, or None if not found.
      reporter_can_join / product_can_join: True if "+"-joining multiple
        values for that dimension returned data for more than one of
        them in a live probe -- False if not (or if the probe itself
        failed), meaning the caller should request one value at a time.
    """
    structure_params = {
        "format": "JSON", "freq": "A", "reporter": reporter_probe,
        "flow": "2", "indicators": "VALUE_EUR", "time": str(probe_year),
    }
    cats = _comext_dimension_categories(dataset, structure_params, base_url)
    product_cats = cats.get("product", {})
    partner_cats = cats.get("partner", {})
    if not product_cats or not partner_cats:
        print(f"  [!] WARNING: structure-discovery query returned no "
              f"product and/or partner categories at all -- the reporter "
              f"code ('{reporter_probe}') or another fixed parameter may "
              f"ALSO be wrong. Raw categories returned: {cats}")
        return {}, None, False, False

    # Discover the intra-EU partner code by label match (we already know
    # "EXT_EU27_2020" -> extra-EU from a prior live run; find its
    # sibling here rather than assuming "EU27_2020").
    intra_partner = None
    for code, label in partner_cats.items():
        if "intra" in str(label).lower() or "intra" in str(code).lower():
            intra_partner = code
            break
    if intra_partner is None:
        print(f"  [!] WARNING: could not find an intra-EU partner code "
              f"among {len(partner_cats)} partner categories: {partner_cats}")

    # Candidate product codes by label/code pattern match.
    targets = {"26.1": ["26.1", "26_1", "261"], "26.2": ["26.2", "26_2", "262"],
               "28.99": ["28.99", "28_99", "2899"], "total": ["total"]}
    candidates = {}
    for target, patterns in targets.items():
        for code, label in product_cats.items():
            code_l, label_l = str(code).lower(), str(label).lower()
            if any(p in code_l or p in label_l for p in patterns):
                candidates[target] = code
                break
    print(f"  [diagnostic] CPA product-code candidates from structure "
          f"listing: {candidates} (validating each against a real query "
          f"next -- appearing in the structure listing does not guarantee "
          f"a code resolves data)")

    # VALIDATE each candidate with its own small query (single reporter,
    # single product, the discovered extra-EU partner as a known-good
    # partner value) -- confirms the code actually returns data, not just
    # that Eurostat's dimension listing mentions it.
    resolved = {}
    for target, code in candidates.items():
        probe_params = {
            "format": "JSON", "freq": "A", "reporter": reporter_probe,
            "partner": "EXT_EU27_2020", "product": code, "flow": "2",
            "indicators": "VALUE_EUR", "time": str(probe_year),
        }
        try:
            eurostat_json_to_df(dataset, probe_params, base_url=base_url)
            resolved[target] = code
        except ValueError:
            print(f"  [!] WARNING: candidate product code '{code}' for "
                  f"'{target}' appeared in the structure listing but "
                  f"returned zero observations when actually queried -- "
                  f"treating it as unresolved.")
            resolved[target] = None
        except requests.exceptions.HTTPError as e:
            print(f"  [!] WARNING: validating candidate product code "
                  f"'{code}' for '{target}' failed with an HTTP error "
                  f"({e}) rather than a clean zero-match -- treating it "
                  f"as unresolved.")
            resolved[target] = None
    for target in targets:
        resolved.setdefault(target, None)

    print(f"  [diagnostic] CPA product-code discovery result (validated): "
          f"{resolved}; intra-EU partner code: {intra_partner!r}")

    if intra_partner is None or any(v is None for v in resolved.values()):
        # Can't usefully probe joinability without a full, valid set of
        # codes to test with -- the caller will raise on the unresolved
        # codes anyway, so just report no-join-support conservatively.
        return resolved, intra_partner, False, False

    # PROBE JOINABILITY: does "+"-joining multiple reporters, or multiple
    # products, into one request actually return data for more than one
    # of them? Tested against the already-known-good EXT_EU27_2020
    # partner and the validated product codes, at the same probe_year.
    all_product_codes = "+".join(resolved.values())
    product_can_join = _probe_multi_value_support(
        dataset, base_url, "product", all_product_codes,
        {"format": "JSON", "freq": "A", "reporter": reporter_probe,
         "partner": "EXT_EU27_2020", "flow": "2", "indicators": "VALUE_EUR",
         "time": str(probe_year)},
        expected_count=2)

    joined_reporters = "+".join(all_reporters)
    reporter_can_join = len(all_reporters) < 2 or _probe_multi_value_support(
        dataset, base_url, "reporter", joined_reporters,
        {"format": "JSON", "freq": "A", "partner": "EXT_EU27_2020",
         "product": resolved["total"], "flow": "2", "indicators": "VALUE_EUR",
         "time": str(probe_year)},
        expected_count=2)

    print(f"  [diagnostic] Multi-value join support: reporter="
          f"{reporter_can_join}, product={product_can_join} -- requests "
          f"will be batched accordingly to minimise their number.")
    return resolved, intra_partner, reporter_can_join, product_can_join


def _probe_multi_value_support(dataset, base_url, dimension, joined_value,
                                other_params, expected_count):
    """
    Tests whether "+"-joining several values into ONE request for a given
    dimension actually works in this dataset, by running a single probe
    query and checking it returns data for MORE THAN ONE distinct value
    of that dimension (not just that the query succeeds at all, since a
    query can return non-zero rows for only the first/one value even when
    the "+"-join itself is silently ignored or partially honoured).

    This exists because live testing repeatedly showed that "+"-joining
    multiple reporter or product values into one Comext DS-059366 request
    silently matched zero observations, forcing this project to fall back
    to one request per individual value -- which is safe but drastically
    increases the number of HTTP requests (and therefore run time) for a
    26-year x 10-country x 4-product x 2-partner fetch. Rather than
    assuming multi-value joins never work and always paying that cost,
    this function checks ONCE whether they happen to work for THIS
    dimension in THIS dataset (they might, depending on the specific
    combination of other filters) and lets the caller use the cheaper
    joined-request approach when they do.

    Returns True only if the probe response actually contains more than
    one distinct value for `dimension` -- not just a non-empty response.
    """
    params = dict(other_params)
    params[dimension] = joined_value
    try:
        df = eurostat_json_to_df(dataset, params, base_url=base_url)
    except (ValueError, requests.exceptions.HTTPError):
        return False
    col = next((c for c in df.columns if c.lower() == dimension), None)
    if col is None:
        return False
    return df[col].nunique() >= expected_count


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
    # STALENESS CHECK: the URL above is hardcoded with a specific
    # publication month/year, since WUI publishes each new release at
    # an unpredictable new URL (not a stable/versioned endpoint this
    # script could reconstruct automatically) -- this cannot be made
    # fully future-proof the way SAMPLE_END above was. What CAN be
    # automated is detecting when this hardcoded URL has gone stale:
    # WUI is published roughly quarterly, so if the most recent
    # quarter in the data we just parsed is more than ~9 months behind
    # today, the hardcoded URL is very likely pointing at an outdated
    # release and needs updating by hand.
    latest_quarter_end = df["quarter"].max().end_time.date()
    staleness_days = (datetime.date.today() - latest_quarter_end).days
    if staleness_days > 270:
        print(f"  [!] WARNING: WUI's most recent quarter ({df['quarter'].max()}) is "
              f"{staleness_days} days behind today -- the hardcoded URL above "
              f"(.../wp-content/uploads/2026/07/WUI_Data.xlsx) is very likely pointing "
              f"at an outdated release. Visit https://worlduncertaintyindex.com "
              f"directly, find the current download link for WUI_Data.xlsx, and update "
              f"the url variable in fetch_wui_global() to match.")
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
    # CRITICAL: reset_index(drop=True) after sort_values() -- without
    # this, the DataFrame's index keeps its PRE-SORT values, which no
    # longer match each row's actual POSITION after sorting. Any code
    # elsewhere in this script that computes an Excel row number from
    # gdp_df.index[mask] (e.g. the ai_inv "share" formula-writing logic
    # in main(), which looks up a specific country's Q4 row) would then
    # silently point at the WRONG row -- confirmed live: cell D2 in
    # ai_inv pointed at gdp!C807 (Italy's 2015Q4 row) instead of the
    # correct gdp!C65 (Austria's own 2015Q4 row), because match
    # position 63 (AT's own pre-sort index) no longer corresponded to
    # Excel row 65 once the DataFrame had been re-sorted. Fixed here at
    # the source, since gdp_df is reused by multiple downstream
    # functions, not just patched at each individual call site.
    df = df.reset_index(drop=True)
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
    # ict_share formula, per explicit instruction, matching the
    # reference workbook's own ict_inv!I2 cell exactly:
    # "=(C2+D2+E2+F2-G2)/H2" -- i.e.
    # (N1132G + N1173G + N112G_secJ + N117G_secJ - N1173G_secJ) / N11G.
    # This now DOES fold the three Section-J asset breakdowns into the
    # numerator (N112G_secJ and N117G_secJ added, N1173G_secJ
    # subtracted) -- a genuine change from an earlier version of this
    # function, which reported those three columns alongside ict_share
    # without including them in its calculation. Missing Section-J
    # values are treated as 0 (fillna), matching how N1173G's own
    # occasional gaps are already handled below, so a missing
    # Section-J observation doesn't turn the whole row's ict_share NaN.
    wide["ict_numerator"] = (wide["N1132G"] + wide["N1173G"].fillna(0)
                              + wide["N112G_secJ"].fillna(0) + wide["N117G_secJ"].fillna(0)
                              - wide["N1173G_secJ"].fillna(0))
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
    # STALENESS CHECK: this is a LOCAL file the user exports by hand
    # from cat.eto.tech -- no automatic download exists (ETO/CSET does
    # not offer a stable, scriptable API for this), so this cannot be
    # made fully future-proof either. What CAN be automated is flagging
    # when the local export itself looks stale relative to today.
    latest_year = int(long_filtered["year"].max())
    if datetime.date.today().year - latest_year > 2:
        print(f"  [!] WARNING: the local file's most recent year ({latest_year}) is "
              f"more than 2 years behind today ({datetime.date.today().year}) -- this "
              f"local export from cat.eto.tech may be stale. Consider re-exporting a "
              f"current version and replacing LOCAL_AI_PATENTS_FILE.")

    return long_filtered[["country", "year", "ai_patents"]]


def fetch_ai_investment(gdp_df):
    """
    Annual AI-related INCOMING INVESTMENT COUNTS per country. Source:
    ETO/CSET Country Activity Tracker (cat.eto.tech, dataset=Investment),
    exported locally as LOCAL_AI_INVESTMENT_FILE. From
    ai_model_WUI_pat_inv.py.

    ALSO computes a "share" column: ai_investment divided by that
    country-year's Q4 (fourth-quarter) GDP level specifically (from
    gdp_df, the same DataFrame fetch_gdp_level() returns), times 1000
    -- following the CURRENT reference workbook's own D2/D3 cells
    exactly: "=(C2/gdp!C65)*1000" and "=(C3/gdp!C69)*1000", i.e. a
    direct reference to that one Q4 cell, NOT an average across all 4
    quarters (an earlier version of this function used AVERAGE() over
    the full year -- confirmed superseded by checking the reference
    workbook directly: C65 is AT/2015Q4, matching D2's own AT/2015
    row). This function computes the same VALUE in Python (used as the
    cached result behind a live formula written in main() below, which
    is what actually reproduces the "gdp!C65"-style reference as a
    real cross-sheet Excel formula rather than a fixed number).
    gdp_df must have columns country, quarter (e.g. "2015Q1"),
    gdp_level -- exactly fetch_gdp_level()'s own return shape.
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

    gdp = gdp_df.copy()
    gdp["gdp_year"] = gdp["quarter"].astype(str).str[:4].astype(int)

    # --- "share" column: ai_investment / that country-year's Q4
    # (fourth-quarter) GDP level, x1000 -- matching the reference
    # workbook's own D2/D3 cells exactly: "=(C2/gdp!C65)*1000" and
    # "=(C3/gdp!C69)*1000" (a direct reference to ONE specific cell,
    # NOT an AVERAGE() over all 4 quarters as an earlier version of
    # this function used -- confirmed by checking the gdp sheet
    # directly: C65 is AT/2015Q4, matching D2's own AT/2015 row, and
    # C69 is AT/2016Q4, matching D3's own AT/2016 row -- i.e. this
    # country-year's own Q4 level specifically, not an annual average).
    gdp = gdp_df.copy()
    gdp["gdp_year"] = gdp["quarter"].astype(str).str[:4].astype(int)
    gdp_q4 = gdp[gdp["quarter"].astype(str).str.endswith("Q4")].copy()
    gdp_q4_level = gdp_q4[["country", "gdp_year", "gdp_level"]].rename(
        columns={"gdp_level": "q4_gdp_level"})

    long_filtered = long_filtered.merge(
        gdp_q4_level, left_on=["country", "year"], right_on=["country", "gdp_year"],
        how="left",
    )
    missing_gdp = long_filtered["q4_gdp_level"].isna().sum()
    if missing_gdp:
        print(f"  [!] fetch_ai_investment: {missing_gdp} country-year rows have no "
              f"matching Q4 GDP data (share will be blank for these) -- check gdp_df's "
              f"year coverage against ai_inv's own {int(long_filtered['year'].min())}-"
              f"{int(long_filtered['year'].max())} range.")
    long_filtered["share"] = (long_filtered["ai_investment"]
                               / long_filtered["q4_gdp_level"]) * 1000

    # STALENESS CHECK: same reasoning as fetch_ai_patents() above --
    # this is a local, hand-exported file with no automatic-download
    # path, so flag it if the export itself looks stale.
    latest_year = int(long_filtered["year"].max())
    if datetime.date.today().year - latest_year > 2:
        print(f"  [!] WARNING: the local file's most recent year ({latest_year}) is "
              f"more than 2 years behind today ({datetime.date.today().year}) -- this "
              f"local export from cat.eto.tech may be stale. Consider re-exporting a "
              f"current version and replacing LOCAL_AI_INVESTMENT_FILE.")

    return long_filtered[["country", "year", "ai_investment", "share"]]


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


def _fetch_quarterly_index(ticker, index_label, min_start_quarter="2000Q2"):
    """
    Shared logic for a single-ticker global/regional equity index,
    fetched via Yahoo Finance since SAMPLE_START (2000): quarterly
    close price + log return, with loud diagnostics if the ticker
    returns nothing or doesn't reach back far enough. Used by
    fetch_msci_world_raw() and fetch_stoxx600_raw() (and originally
    matches fetch_semiconductor_raw()'s own inline pattern).
    """
    px = _yf_close_series(ticker)
    if px.empty:
        print(f"  [!] WARNING: yfinance returned ZERO rows for {index_label} "
              f"(ticker '{ticker}'). This ticker was identified via web "
              f"research, not verified against a live call -- if this "
              f"warning fires, the symbol may have changed on Yahoo "
              f"Finance. Try searching \"{index_label}\" on "
              f"https://finance.yahoo.com directly and update this "
              f"ticker to whatever it currently resolves to.")
        return pd.DataFrame(columns=["ticker", "quarter", "close", "log_ret"])
    q = px.resample("QE").last()
    ret = np.log(q / q.shift(1))
    df = pd.DataFrame({"close": q, "log_ret": ret})
    df.index = df.index.to_period("Q")
    df = df.reset_index().rename(columns={"index": "quarter"})
    if "quarter" not in df.columns:
        df = df.rename(columns={df.columns[0]: "quarter"})
    df["ticker"] = ticker
    result = df[["ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])
    earliest = result["quarter"].min() if not result.empty else None
    print(f"  [diagnostic] {index_label} ({ticker}): {len(result)} rows, "
          f"earliest quarter: {earliest}.")
    if earliest is not None and str(earliest) > min_start_quarter:
        print(f"  [!] WARNING: earliest available quarter is {earliest}, "
              f"later than the requested 2000 start -- yfinance/Yahoo "
              f"Finance may not have full history for this ticker back to "
              f"2000. Check https://finance.yahoo.com/quote/{ticker.replace('^', '%5E')}"
              f"/history/ directly to confirm how far back data actually goes.")
    return result


def fetch_msci_world_raw():
    """
    Raw quarterly close price + log return for the MSCI World Index
    (global developed-market equities), since SAMPLE_START (2000).

    Ticker: "^990100-USD-STRD" -- the actual MSCI World Index itself
    (USD, gross/price return), NOT an ETF proxy. This distinction
    matters for the "since 2000" requirement specifically: the most
    obvious alternative, the iShares MSCI World ETF (URTH), only
    launched in 2012 and would have NO data at all for 2000-2011,
    silently truncating the requested history. The direct index
    ticker does not have that limitation, since the underlying MSCI
    World Index itself predates 2000 by decades.

    HONESTY NOTE: this ticker was identified via web research (Yahoo
    Finance's own listing for "MSCI WORLD (^990100-USD-STRD)"), not
    independently verified against a live yfinance call in this
    sandbox (no network access here). The diagnostic warning inside
    _fetch_quarterly_index() fires loudly and specifically if this
    assumption turns out wrong on a real run, so a stale/incorrect
    ticker cannot fail silently.
    """
    return _fetch_quarterly_index("^990100-USD-STRD", "MSCI World")


def fetch_stoxx600_raw():
    """
    Raw quarterly close price + log return for the STOXX Europe 600
    Index (pan-European large/mid/small-cap equities), since
    SAMPLE_START (2000) -- SPLICED from two sources, since neither
    alone covers the full requested period:

    - 2004(Q2)-onward: the STOXX Europe 600 index itself, ticker
      "^STOXX" (Yahoo Finance displays it as "STXE 600 I").
    - 2000-2004(Q1): PROXIED with the German DAX index, ticker
      "^GDAXI" -- confirmed via a live run of this script's own
      earlier version that Yahoo's own "^STOXX" data series does NOT
      go back before 2004, despite the underlying STOXX Europe 600
      index itself having existed (with backfilled history) since the
      late 1990s. The FIRST proxy attempted was EURO STOXX 50
      (^STOXX50E) -- also confirmed, via a second live run, to have
      the SAME ~2004 Yahoo Finance data-availability limit (evidently
      a limitation shared across Yahoo's whole STOXX-family index
      feed, not specific to ^STOXX alone) -- so DAX was chosen
      specifically because it is NOT part of that family and is one
      of the longest-running European index series on Yahoo Finance
      (data available back into the early 1990s).

    HONESTY NOTE on the proxy itself: DAX (30-40 German blue-chip
    large caps only) is compositionally much NARROWER than STOXX
    Europe 600 (pan-European, all caps, ~17 countries) -- it is a
    reasonable, commonly-used directional proxy for broad European
    equity movements (Germany is STOXX 600's largest single-country
    weight), but NOT a precise substitute. Every proxied row is
    explicitly flagged via the "source" column ("STOXX600" vs
    "DAX_proxy") so this substitution is never silently
    indistinguishable from genuine STOXX 600 data -- check that
    column before treating the whole series as uniform.

    The proxy period's price LEVEL is reconstructed by chaining DAX's
    own quarterly log returns backward from the first genuine STOXX
    600 close -- this keeps the "close" column a single continuous,
    usable price index across the full sample, rather than leaving
    2000-2004Q1 as a gap in "close" while only "log_ret" is populated.
    """
    stoxx600 = _fetch_quarterly_index("^STOXX", "STOXX Europe 600")
    if stoxx600.empty:
        print("  [!] WARNING: no STOXX 600 data at all -- cannot splice in the "
              "DAX proxy either (nothing to chain it to). Returning "
              "empty.")
        return stoxx600

    first_real_quarter = stoxx600["quarter"].min()
    first_start_quarter = pd.Period(SAMPLE_START, freq="Q")
    if first_real_quarter <= first_start_quarter:
        # STOXX 600 itself already covers the full requested range --
        # no proxy needed at all.
        stoxx600["source"] = "STOXX600"
        return stoxx600[["ticker", "quarter", "close", "log_ret", "source"]]

    proxy = _fetch_quarterly_index("^GDAXI", "DAX (STOXX 600 proxy)")
    gap = proxy[proxy["quarter"] < first_real_quarter].copy()
    if gap.empty:
        print("  [!] WARNING: DAX proxy has no data before "
              f"{first_real_quarter} either -- the 2000-{first_real_quarter} "
              "gap in STOXX 600 could not be filled. Returning STOXX 600 "
              "alone, still starting at its own real first quarter.")
        stoxx600["source"] = "STOXX600"
        return stoxx600[["ticker", "quarter", "close", "log_ret", "source"]]

    gap = gap.sort_values("quarter")
    first_real_close = stoxx600.loc[stoxx600["quarter"] == first_real_quarter, "close"].iloc[0]
    # Chain BACKWARD from the first genuine STOXX 600 close. To get
    # quarter q's synthetic price, we need the return for "q -> q+1"
    # (NOT q's own log_ret, which is "q-1 -> q") -- that return lives
    # on the NEXT quarter's row in the FULL proxy series (which
    # includes first_real_quarter itself, one quarter past the gap,
    # specifically to supply this). Confirmed via a dedicated
    # mathematical-continuity test that an earlier version of this
    # function got this off-by-one wrong (it used q's own log_ret
    # instead of q+1's), producing a synthetic series that did NOT
    # connect smoothly to the real STOXX 600 series at the splice
    # point -- fixed here.
    proxy_indexed = proxy.set_index("quarter")["log_ret"]
    gap_quarters_desc = sorted(gap["quarter"].tolist(), reverse=True)
    synthetic_closes = {}
    running_close = first_real_close
    for q in gap_quarters_desc:
        next_q = q + 1
        next_log_ret = proxy_indexed.get(next_q, np.nan)
        if pd.isna(next_log_ret) or pd.isna(running_close):
            synthetic_closes[q] = np.nan
            running_close = np.nan
            continue
        running_close = running_close / np.exp(next_log_ret)
        synthetic_closes[q] = running_close
    gap["close"] = gap["quarter"].map(synthetic_closes)
    gap["ticker"] = "^STOXX (proxied via ^GDAXI)"
    gap["source"] = "DAX_proxy"

    stoxx600["source"] = "STOXX600"
    result = pd.concat(
        [gap[["ticker", "quarter", "close", "log_ret", "source"]],
         stoxx600[["ticker", "quarter", "close", "log_ret", "source"]]],
        ignore_index=True,
    ).sort_values("quarter").reset_index(drop=True)
    print(f"  [diagnostic] STOXX 600 (spliced): {len(result)} total rows -- "
          f"{len(gap)} proxied via DAX ({gap['quarter'].min()}-"
          f"{gap['quarter'].max()}), {len(stoxx600)} genuine STOXX 600 "
          f"({first_real_quarter} onward).")
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
# 6b. CPA 2.2-classified EU trade data (Comext DS-059366) -- a SEPARATE
#     source from fetch_hs_export_data() above (which uses HS/CN codes
#     via OECD BIMTS). This section fetches export values classified by
#     CPA (Statistical Classification of Products by Activity) instead,
#     for three specific CPA 2.2 product groups plus the total export
#     value, split by intra-EU vs. extra-EU trade flow.
# ----------------------------------------------------------------------

COMEXT_BASE = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data"
CPA_EUR_EXPORT_DATASET = "DS-059366"


def fetch_eur_export_data():
    """
    Export values (EUR) for CPA 2.2 product groups 26.1, 26.2, and 28.99,
    plus the total export value, BOTH intra-EU and extra-EU, for the 10
    target euro area countries, 2000-2025 -- from Comext dataset
    DS-059366 ("International trade of EU and non-EU countries since
    2002 by CPA 2.2"), per explicit instruction.

    FULL CODE DISCOVERY, not hardcoded guesses -- multiple rounds of live
    testing showed that guessing at this dataset's code notation (product
    codes like "CPA22_26.1", the measure code "VALUE_IN_EUR", the intra-EU
    partner code "EU27_2020") repeatedly matched zero observations, and
    that even codes found via the dataset's own STRUCTURE listing (the
    declared category list for a dimension) were not guaranteed to work
    as actual filter values. This function now:
    1. Runs _discover_cpa_codes() to read the real product-code candidates
       AND the real intra-EU partner code from the dataset's structure,
       then VALIDATES each candidate product code with its own live query
       before trusting it (see that function's docstring for why the
       structure listing alone was not sufficient).
    2. Requests EVERY dimension that has more than one target value
       (reporter, product) as a SEPARATE request per value, not "+"-joined
       -- confirmed via a live run that "+"-joining 10 reporters into one
       string matched zero observations even after every other parameter
       was corrected, while single-reporter requests worked; product is
       treated the same way on the (unconfirmed but consistent) assumption
       that the same limitation applies, since "+"-joined products also
       matched zero observations throughout testing.
    3. Uses "VALUE_EUR" (not "VALUE_IN_EUR") as the measure code --
       confirmed correct both by inspecting the user's own manually
       downloaded export from this exact dataset (its "INDICATORS" filter
       row reads "VALUE_EUR") and by it being recognized as a valid
       category in live error responses after the fix.

    Note also (separate from the above): CPA Version 2.2 itself only
    became the OFFICIAL classification for European statistics from
    2025 onward (Commission Delegated Regulation (EU) 2024/3103) -- this
    dataset's own title ("since 2002 by CPA 2.2") indicates Eurostat has
    back-cast the full historical series onto CPA 2.2, so requesting
    2000-2025 here is consistent with how the dataset is meant to be
    used, not a mismatch with when CPA 2.2 itself was adopted.

    Batches by ONE (year, partner, country, product) combination per
    request -- confirmed across several live runs that this dataset
    rejects "+"-joined multi-value requests for at least the reporter
    dimension (and, per testing, apparently product too) with silent
    zero-match responses rather than a clear "not supported" error, and
    separately that combining too many dimension values in one call can
    also trigger HTTP 413 (request too large). This means substantially
    more, smaller HTTP requests than earlier versions of this function
    (roughly years x 2 partners x 10 countries x 4 products), which is
    slower but is the only approach confirmed not to silently return
    nothing.
    """
    # NOTE: this dataset's own title says "since 2002" -- confirmed in a
    # live run that requesting years before that (2000, 2001) matches
    # zero observations for the TIME dimension specifically, with every
    # other dimension (reporter, partner, product, flow, indicators) all
    # resolving correctly. This is genuinely missing data, not a code
    # error, so those years are skipped here rather than requested and
    # guaranteed to fail.
    start_year = max(2002, int(SAMPLE_START[:4]))
    end_year = min(2025, int(SAMPLE_END[:4]))

    # DISCOVERY STEP: find the dataset's REAL, VALIDATED product codes
    # for CPA 26.1, 26.2, 28.99, and "total", the REAL intra-EU partner
    # code, and whether "+"-joining multiple reporter/product values
    # into one request actually works -- see _discover_cpa_codes()'s
    # docstring for why each of these needed live validation rather than
    # being assumed. Probed on the most recent year in range (more
    # likely to have complete, non-camouflaged data than the earliest
    # years) and the FIRST country in COUNTRIES (arbitrary -- product/
    # partner codes don't vary by reporter).
    discovered, intra_partner_code, reporter_can_join, product_can_join = _discover_cpa_codes(
        reporter_probe=COUNTRIES[0], dataset=CPA_EUR_EXPORT_DATASET,
        base_url=COMEXT_BASE, probe_year=end_year, all_reporters=COUNTRIES)
    code_map = {  # target label -> real dataset code (or None if unresolved)
        "CPA_261": discovered.get("26.1"),
        "CPA_262": discovered.get("26.2"),
        "CPA_2899": discovered.get("28.99"),
        "total_export": discovered.get("total"),
    }
    unresolved = [label for label, code in code_map.items() if code is None]
    if unresolved or intra_partner_code is None:
        problems = []
        if unresolved:
            problems.append(f"product code(s) for {unresolved}")
        if intra_partner_code is None:
            problems.append("the intra-EU partner code")
        raise SystemExit(
            f"\nfetch_eur_export_data: could not discover/validate "
            f"{' and '.join(problems)} -- see the [!] WARNING/[diagnostic] "
            f"lines above for what the discovery query actually returned, "
            f"and either widen the pattern-matching in "
            f"_discover_cpa_codes() or hardcode the correct code(s) "
            f"directly if you can identify them from that output."
        )
    partner_codes = {intra_partner_code: "intra", "EXT_EU27_2020": "extra"}

    # Batch reporter and/or product into single "+"-joined requests where
    # the discovery step confirmed that actually works in this dataset --
    # cutting the request count by up to (countries x products) = 40x
    # relative to always requesting one value at a time, while still
    # falling back safely to per-value requests wherever joining was NOT
    # confirmed to work.
    if reporter_can_join:
        reporter_batches = ["+".join(COUNTRIES)]
    else:
        reporter_batches = list(COUNTRIES)
    if product_can_join:
        product_batches = {"ALL": "+".join(code_map.values())}
    else:
        product_batches = {label: code for label, code in code_map.items()}

    total_requests = len(range(start_year, end_year + 1)) * len(partner_codes) * \
        len(reporter_batches) * len(product_batches)
    print(f"  [diagnostic] Fetching DS-059366: {total_requests} request(s) "
          f"planned ({end_year - start_year + 1} years x {len(partner_codes)} "
          f"partners x {len(reporter_batches)} reporter batch(es) x "
          f"{len(product_batches)} product batch(es)).")

    all_rows = []
    failed_requests = []
    request_num = 0
    for year in range(start_year, end_year + 1):
        for partner_code in partner_codes:
            for reporter_batch in reporter_batches:
                for target_label, product_code in product_batches.items():
                    request_num += 1
                    params = {
                        "format": "JSON",
                        "freq": "A",  # REQUIRED -- confirmed via a live run
                                       # that omitting this made the API
                                       # return an entirely empty result,
                                       # since without it the API cannot
                                       # resolve time="2000" as an ANNUAL
                                       # period.
                        "reporter": reporter_batch,
                        "partner": partner_code,
                        "product": product_code,
                        "flow": "2",  # export (Comext convention: 1=import, 2=export)
                        "indicators": "VALUE_EUR",
                        "time": str(year),
                    }
                    try:
                        df = eurostat_json_to_df(CPA_EUR_EXPORT_DATASET, params, base_url=COMEXT_BASE)
                    except ValueError as e:
                        # eurostat_json_to_df() already prints the full set of valid
                        # categories per dimension in its own error message when a
                        # query matches zero observations -- surface that here
                        # rather than aborting the whole collection run, since a
                        # single bad combination shouldn't necessarily block every
                        # other one from being collected.
                        print(f"  [!] WARNING: DS-059366 request for "
                              f"{year}/{partner_code}/{reporter_batch}/{target_label} failed: {e}")
                        failed_requests.append((year, partner_code, reporter_batch, target_label))
                        continue
                    except requests.exceptions.HTTPError as e:
                        print(f"  [!] WARNING: DS-059366 request for "
                              f"{year}/{partner_code}/{reporter_batch}/{target_label} failed: {e}")
                        failed_requests.append((year, partner_code, reporter_batch, target_label))
                        continue
                    df["year"] = year
                    all_rows.append(df)
                    time.sleep(0.05)  # be polite to the API between requests
        # Progress marker after each YEAR (not each request) -- with
        # potentially hundreds of requests total, printing only once per
        # year keeps output readable while still making clear the fetch
        # is actively progressing rather than hanging.
        print(f"  [diagnostic] ...year {year} done "
              f"({request_num}/{total_requests} requests so far).")

    if not all_rows:
        raise SystemExit(
            "\nfetch_eur_export_data: EVERY request failed -- see the [!] "
            "WARNING lines above for Eurostat's own reported valid "
            "categories, and re-check the discovered product/partner "
            "codes or the flow code in fetch_eur_export_data() accordingly."
        )
    if failed_requests:
        print(f"  [!] WARNING: {len(failed_requests)} combination(s) "
              f"returned no data and were skipped "
              f"(showing up to 20): {failed_requests[:20]}")

    raw = pd.concat(all_rows, ignore_index=True)

    col_reporter = next((c for c in raw.columns if c.lower() == "reporter"), None)
    col_partner = next((c for c in raw.columns if c.lower() == "partner"), None)
    col_product = next((c for c in raw.columns if c.lower() == "product"), None)
    col_value = next((c for c in raw.columns if c.lower() == "value"), None)
    missing = [n for n, c in [("reporter", col_reporter), ("partner", col_partner),
                               ("product", col_product), ("value", col_value)] if c is None]
    if missing:
        raise SystemExit(
            f"\nfetch_eur_export_data: expected column(s) {missing} not found "
            f"in the flattened response. Actual columns: {list(raw.columns)}"
        )

    raw["country"] = raw[col_reporter].astype(str).str.strip().str.upper()
    raw["value_eur"] = pd.to_numeric(raw[col_value], errors="coerce")
    raw = raw.dropna(subset=["value_eur"])

    partner_label = dict(partner_codes)
    raw["flow_label"] = raw[col_partner].map(partner_label)
    unmapped = raw.loc[raw["flow_label"].isna(), col_partner].unique()
    if len(unmapped):
        print(f"  [!] WARNING: unrecognised partner code(s) in the response, "
              f"dropped: {list(unmapped)} -- if this is intra-/extra-EU under "
              f"a different code, update the partner_label mapping above.")
        raw = raw.dropna(subset=["flow_label"])

    product_label = {code: label for label, code in code_map.items()}
    raw["product_label"] = raw[col_product].map(product_label)
    unmapped_p = raw.loc[raw["product_label"].isna(), col_product].unique()
    if len(unmapped_p):
        print(f"  [!] WARNING: unrecognised product code(s) in the response, "
              f"dropped: {list(unmapped_p)}.")
        raw = raw.dropna(subset=["product_label"])

    raw["col_name"] = raw["product_label"] + "_" + raw["flow_label"]

    wide = raw.pivot_table(
        index=["country", "year"], columns="col_name", values="value_eur", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    expected_cols = ["country", "year"]
    for base in ["CPA_261", "CPA_262", "CPA_2899"]:
        expected_cols += [f"{base}_intra", f"{base}_extra"]
    expected_cols += ["total_export_intra", "total_export_extra"]
    for col in expected_cols:
        if col not in wide.columns:
            print(f"  [!] WARNING: expected column '{col}' not present in the "
                  f"result -- added as all-missing.")
            wide[col] = pd.NA

    wide = wide[expected_cols].sort_values(["country", "year"]).reset_index(drop=True)
    wide["year"] = wide["year"].astype(int)

    # "share": combined CPA export value (all groups, intra+extra) as a
    # share of total intra+extra exports -- computed here as a plain
    # pandas column (matching fetch_hs_export_data()'s own "share"
    # column) so this function's return value is self-consistent even
    # when used outside the Excel-writing path (e.g. in tests). The
    # Excel output itself REPLACES this column's cell values with a live
    # formula in the write loop below ("eur_export" branch), matching
    # the reference workbook's own eur_export!K2 formula -- this
    # pandas-computed value becomes that formula's cached result, so
    # the two are guaranteed to agree rather than silently drifting
    # apart.
    cpa_cols = [c for c in expected_cols if c.startswith("CPA_")]
    cpa_sum = wide[cpa_cols].sum(axis=1, skipna=True)
    total_all = wide["total_export_intra"].fillna(0) + wide["total_export_extra"].fillna(0)
    wide["share"] = np.where(total_all == 0, 0.0, cpa_sum / total_all)

    print(f"  [diagnostic] CPA 2.2 EU export data (DS-059366): {wide.shape[0]} "
          f"country-year rows, {wide['country'].nunique()} countries, years "
          f"{wide['year'].min()}-{wide['year'].max()}.")
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

    print("\n[1/21] GDP level (Eurostat)...")
    sheets["gdp"] = fetch_gdp_level()

    print("\n[2/21] ICT investment share (Eurostat)...")
    sheets["ict_inv"] = fetch_ict_investment_share()

    print("\n[3/21] AI patent applications (ETO/CSET, local file)...")
    sheets["ai_patent"] = fetch_ai_patents()

    print("\n[4/21] AI incoming investment counts (ETO/CSET, local file)...")
    sheets["ai_inv"] = fetch_ai_investment(sheets["gdp"])

    print("\n[5/21] National equity indices (Yahoo Finance)...")
    sheets["index_nat"] = fetch_national_indices_raw()

    print("\n[6/21] Semiconductor index ^SOX (Yahoo Finance)...")
    sheets["index_sox"] = fetch_semiconductor_raw()

    print("\n[7/21] MSCI World index (Yahoo Finance)...")
    sheets["index_msci"] = fetch_msci_world_raw()

    print("\n[8/21] STOXX Europe 600 index (Yahoo Finance)...")
    sheets["index_stoxx600"] = fetch_stoxx600_raw()

    print("\n[9/21] AI/ICT-related HS export data (OECD BIMTS)...")
    sheets["hs_export"] = fetch_hs_export_data()

    print("\n[10/21] CPA 2.2 EU export data, intra-/extra-EU (Comext DS-059366)...")
    sheets["eur_export"] = fetch_eur_export_data()

    print("\n[11/21] World Uncertainty Index (WUI)...")
    sheets["wui"] = fetch_wui_global()

    print("\n[12/21] Geopolitical Risk Index (GPR)...")
    sheets["gpr"] = fetch_gpr_global()

    print("\n[13/21] Economic Policy Uncertainty Index (EPU)...")
    sheets["epu"] = fetch_epu_global()

    print("\n[14/21] Trade Policy Uncertainty Index (TPU)...")
    sheets["tpu"] = fetch_tpu_global()

    print("\n[15/21] Global Supply Chain Pressure Index (GSCPI)...")
    sheets["gscpi"] = fetch_gscpi_global()

    print("\n[16/21] Global commodity price index, incl. oil and gas (FRED)...")
    sheets["com"] = fetch_commodity_price_index()

    print("\n[17/21] Trade openness (World Bank)...")
    sheets["trade_openness"] = fetch_trade_openness()

    print("\n[18/21] Population growth (World Bank)...")
    sheets["population_growth"] = fetch_population_growth()

    print("\n[19/21] Labor productivity growth: real productivity per hour worked, YoY (Eurostat)...")
    sheets["productivity_growth"] = fetch_productivity_growth()

    print("\n[20/21] GDP per capita (Eurostat)...")
    sheets["gdp_per_capita"] = fetch_gdp_per_capita()

    print("\n[21/21] HICP index level, 2015=100 (Eurostat)...")
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
    # HISTORY NOTE on hs_export's formula (now resolved): an earlier
    # version of this script's reference file used
    # "=SUM(F{row}:J{row})/K{row}" -- summing ONLY the 5 newer
    # HS_8486xx columns (F:J), excluding the original 3 HS_8471xx
    # columns (C:E) -- while fetch_hs_export_data() itself always
    # computed "share" as the sum of ALL 8 HS columns. At the time,
    # the instruction was to match that narrower reference formula
    # exactly rather than "fix" what looked like an oversight. This
    # has since been explicitly superseded: the formula below now sums
    # all 8 columns (C:J), matching both the reference workbook's
    # current cell and fetch_hs_export_data()'s own calculation.
    # ------------------------------------------------------------------

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            if sheet_name == "ict_inv":
                # columns: A=country B=year C=N1132G D=N1173G E=N112G_secJ
                #          F=N117G_secJ G=N1173G_secJ H=N11G I=ict_share
                col_f = list(df.columns).index("ict_share")
                # Every source column's letter is derived dynamically (not
                # hardcoded) from its actual position in df.columns, so the
                # formula stays correct even if the column order ever
                # changes -- matching the reference workbook's own
                # ict_inv!I2 cell exactly: "=(C2+D2+E2+F2-G2)/H2", i.e.
                # (N1132G + N1173G + N112G_secJ + N117G_secJ - N1173G_secJ)
                # / N11G.
                def _col_letter(name):
                    return chr(ord("A") + list(df.columns).index(name))
                l_n1132g = _col_letter("N1132G")
                l_n1173g = _col_letter("N1173G")
                l_n112g_j = _col_letter("N112G_secJ")
                l_n117g_j = _col_letter("N117G_secJ")
                l_n1173g_j = _col_letter("N1173G_secJ")
                l_n11g = _col_letter("N11G")
                for i, row in df.iterrows():
                    excel_row = i + 2  # 1-based, +1 for header row
                    formula = (f"=({l_n1132g}{excel_row}+{l_n1173g}{excel_row}"
                               f"+{l_n112g_j}{excel_row}+{l_n117g_j}{excel_row}"
                               f"-{l_n1173g_j}{excel_row})/{l_n11g}{excel_row}")
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
                    # Formula now sums ALL 8 HS columns (C:J), per explicit
                    # instruction, matching the reference workbook's own
                    # hs_export!L2 cell exactly: "=SUM(C2:J2)/K2". An
                    # EARLIER version of this script deliberately kept a
                    # narrower "=SUM(F{row}:J{row})/K{row})" (only the 5
                    # newer HS_8486xx columns, excluding the original 3
                    # HS_8471xx columns) because that mismatch was found in
                    # a then-current reference file and the instruction at
                    # the time was to match it exactly rather than "fix"
                    # what looked like an oversight -- this has now been
                    # superseded by an explicit request for the full-sum
                    # version, so that is what this script produces.
                    formula = f"=SUM(C{excel_row}:J{excel_row})/K{excel_row}"
                    # BUGFIX, unchanged from before: a plain Python "+"
                    # between the HS_84xxxx values propagates NaN if ANY
                    # single one is missing (e.g. BIMTS genuinely has no
                    # recorded IE/2019/HS_848630 observation) -- unlike
                    # Excel's own SUM() function (which treats a blank cell
                    # as 0), and unlike pandas' .sum() (skipna=True by
                    # default, used when "share" was first computed in
                    # fetch_hs_export_data()). That mismatch is exactly what
                    # produced a NaN cached_value here in an earlier version,
                    # which xlsxwriter's write_formula(value=...) then
                    # stored as literal text "nan" in the cell's cached-
                    # result XML -- invisible under a normal openpyxl read,
                    # but fatal for pandas.read_excel()'s default (read_only
                    # + data_only) reading path, which tries int("nan") and
                    # crashes. Fixed by skipping NaN terms explicitly
                    # (matching both Excel's own SUM() semantics and the
                    # original "share" column's pandas .sum()) -- now
                    # applied across ALL 8 HS columns, matching the widened
                    # formula above.
                    hs_cols_l = [c for c in df.columns if c.startswith("HS_")]
                    hs_sum = sum(0.0 if pd.isna(row[c]) else row[c] for c in hs_cols_l)
                    total_val = row["total_export"]
                    if pd.isna(total_val) or total_val == 0:
                        cached_value = 0.0
                    else:
                        cached_value = hs_sum / total_val
                    worksheet.write_formula(i + 1, col_l, formula, value=cached_value)

            elif sheet_name == "eur_export":
                # columns: A=country B=year C..H=6 CPA intra/extra value
                # columns I=total_export_intra J=total_export_extra K=share.
                # Formula matches the reference workbook's own eur_export!K2
                # cell exactly: "=SUM(C2:H2)/(I2+J2)" -- sum of every CPA
                # value column (both intra- and extra-EU, across all CPA
                # groups present) divided by total intra-EU plus total
                # extra-EU exports. Column letters are derived dynamically
                # from df.columns rather than hardcoded, so this stays
                # correct if a CPA group is ever added, removed, or
                # reordered (e.g. the CPA_2899 -> CPA_289920/CPA_289951
                # split some earlier sessions of this project explored).
                col_k = list(df.columns).index("share")
                cpa_cols_l = [c for c in df.columns if c.startswith("CPA_")]

                def _col_letter_eur(name):
                    return chr(ord("A") + list(df.columns).index(name))
                l_first_cpa = _col_letter_eur(cpa_cols_l[0])
                l_last_cpa = _col_letter_eur(cpa_cols_l[-1])
                l_total_intra = _col_letter_eur("total_export_intra")
                l_total_extra = _col_letter_eur("total_export_extra")

                for i, row in df.iterrows():
                    excel_row = i + 2
                    formula = (f"=SUM({l_first_cpa}{excel_row}:{l_last_cpa}{excel_row})"
                               f"/({l_total_intra}{excel_row}+{l_total_extra}{excel_row})")
                    # Same NaN-safety reasoning as hs_export above: sum the
                    # CPA columns skipping any missing value (matching
                    # Excel's own SUM(), which treats a blank cell as 0),
                    # rather than a plain Python "+" that would propagate a
                    # single NaN into the whole cached result.
                    cpa_sum = sum(0.0 if pd.isna(row[c]) else row[c] for c in cpa_cols_l)
                    total_intra_val = row["total_export_intra"]
                    total_extra_val = row["total_export_extra"]
                    denom = (0.0 if pd.isna(total_intra_val) else total_intra_val) + \
                            (0.0 if pd.isna(total_extra_val) else total_extra_val)
                    cached_value = 0.0 if denom == 0 else cpa_sum / denom
                    worksheet.write_formula(i + 1, col_k, formula, value=cached_value)

            elif sheet_name == "ai_inv":
                # columns: A=country B=year C=ai_investment D=share
                # Formula matches the reference workbook's own D2/D3
                # cells exactly: "=(C2/gdp!C65)*1000" and
                # "=(C3/gdp!C69)*1000" -- a DIRECT reference to that
                # country-year's own Q4 cell in the "gdp" sheet (NOT an
                # AVERAGE() over all 4 quarters, which an earlier
                # version of this script used). The Q4 row is located
                # dynamically (not hardcoded) via the in-memory gdp
                # DataFrame (sheets["gdp"]) that fed that sheet -- since
                # both were written in the SAME row order, a DataFrame
                # row position + 2 (1-based, +1 for the header row) is
                # exactly that quarter's Excel row.
                col_d = list(df.columns).index("share")
                gdp_df = sheets["gdp"]
                gdp_quarter_str = gdp_df["quarter"].astype(str)
                for i, row in df.iterrows():
                    excel_row = i + 2
                    q4_label = f"{int(row['year'])}Q4"
                    match_positions = gdp_df.index[
                        (gdp_df["country"] == row["country"])
                        & (gdp_quarter_str == q4_label)
                    ].tolist()
                    if not match_positions:
                        # No matching Q4 GDP row for this country-year --
                        # write a plain 0 (not a formula referencing an
                        # empty/invalid cell), matching fetch_ai_investment()'s
                        # own diagnostic warning for this same condition.
                        worksheet.write(i + 1, col_d, 0.0)
                        continue
                    gdp_q4_row = match_positions[0] + 2
                    formula = f"=(C{excel_row}/gdp!C{gdp_q4_row})*1000"
                    share_val = row["share"]
                    cached_share = 0.0 if pd.isna(share_val) else share_val
                    worksheet.write_formula(i + 1, col_d, formula, value=cached_share)

    print(f"\nDONE. Sheets written: {list(sheets.keys())}")
    for name, df in sheets.items():
        print(f"  {name}: {df.shape[0]} rows, columns={list(df.columns)}")
    print(f"\n{OUTPUT_FILE.name} is ready -- upload it back to Claude, or run one of "
          f"the modeling scripts locally, which read this file as their single "
          f"local data source.")
