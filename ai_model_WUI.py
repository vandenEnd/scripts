"""
WUI shock propagation and AI-exposure mitigation: NL + euro area panel
========================================================================

Estimates a MODERATED-REGRESSION panel local projection (main effects
AND interaction terms, not a "difference" spec) with ENTITY fixed
effects only (no time effects):

  ICT spec:
    Delta_gdp[i,t+h] = a[i,h] + b1_h*WUI_shock_t + b2_h*F[i,t]
                       + b3_h*(WUI_shock_t * F[i,t]) + Gamma_h*X[i,t]
                       + e[i,t+h]

  Corr spec (Dum = is_boom_t; Dum=0/"bust" is the reference level):
    Delta_gdp[i,t+h] = a[i,h] + b1_h*WUI_shock_t + b2_h*F[i,t]
                       + b3_h*Dum_t + b4_h*(WUI_shock_t*F[i,t])
                       + b5_h*(WUI_shock_t*Dum_t)
                       + b6_h*(WUI_shock_t*F[i,t]*Dum_t)
                       + Gamma_h*X[i,t] + e[i,t+h]

for h = 0..8 quarters, where F[i,t] in (0,1) is a logistic transform of a
country-level AI/ICT-exposure state variable z[i,t], and WUI_shock_t is
an AR(2)-purified version of the global, GDP-weighted World Uncertainty
Index (Ahir, Bloom & Furceri; see purify_wui_shock()) -- the residual
after regressing WUI on its own lags, an approximate "identified shock"
rather than the raw level.

IDENTIFICATION: WUI_shock_t, Dum_t, and WUI_shock_t*Dum_t are
entity-invariant (identical across every country at a given quarter --
WUI is a global shock, Dum is derived from the global ^SOX index). Time
fixed effects would absorb all three completely, so estimating b1, b3
(Corr spec), and b5 as genuinely identified coefficients requires
ENTITY effects only -- see build_panel()/run_local_projections() for the
full derivation. Cost: common-across-countries time-varying confounders
(ECB policy stance, euro-area-wide demand shocks) are no longer swept
out of the residual the way two-way FE would have done.

INTERPRETATION: b3 (ICT spec) and b4 (Corr spec) are the mitigating
effect of AI/ICT exposure on the WUI shock's impact, net of the WUI and
F main effects (which are now separately estimated, not folded into an
implicit baseline). For the Corr spec specifically, the mitigating
effect of WUI*F is b4 alone in the bust regime (Dum=0) and b4+b6 in the
boom regime (Dum=1) -- see summarize_irf() for where that sum is
computed with a correctly derived combined standard error.

Two versions of z (AI/ICT exposure) are estimated:
  (A) ICT investment share of GFCF  (primary; Eurostat nama_10_a64_p5)
  (B) Rolling correlation between each country's national equity index
      and the global semiconductor index (robustness; contaminated by
      the same uncertainty shocks used as the RHS variable -- see
      caveats printed at the end of the script)

All results (IRF tables, all series used -- including both the raw WUI
level and the purified shock, for transparency -- variable definitions,
and IRF plots) are saved to a single Excel workbook: model_results.xlsx.

Run:  pip install pandas numpy requests statsmodels linearmodels yfinance openpyxl matplotlib Pillow
      python gpr_ai_mitigation_pipeline.py

Network access required (Eurostat API, worlduncertaintyindex.com, Yahoo Finance).
"""

SCRIPT_VERSION = "2025-08-11-v37-WUI-time-trend-control"  # bump this whenever the file changes;
                                    # print it at runtime to confirm you're
                                    # not running a stale cached copy

import io
import numpy as np
import pandas as pd
import requests

print(f"[gpr_ai_mitigation_pipeline.py version {SCRIPT_VERSION}]")


def _requests_get_with_retry(url, params=None, timeout=60, max_retries=3, backoff=2.0):
    """
    Thin wrapper around requests.get() that retries on TRANSIENT network
    errors (connection reset, connection aborted, timeout) with
    exponential backoff, instead of letting the whole run crash on the
    first hiccup.

    Motivating case: Eurostat's API has been observed to intermittently
    reset the connection mid-response (WinError 10054 /
    ConnectionResetError: "An existing connection was forcibly closed by
    the remote host") on an otherwise perfectly valid, well-formed
    request -- this is server-side flakiness (likely rate-limiting or
    load-shedding on Eurostat's end for a burst of calls, since this
    script issues several Eurostat requests in quick succession), not a
    bug in the request itself. A short retry-with-backoff resolves most
    such failures without any change to what's actually being requested.

    Does NOT retry on HTTP error status codes (4xx/5xx, via
    raise_for_status()) beyond what's already transient-connection-level
    -- a genuine "this endpoint/dataset doesn't exist" 404 or a
    malformed-request 400 should fail immediately and clearly, not
    after 3 pointless retries of the exact same bad request.
    """
    import time

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
                      f"({e.__class__.__name__}) -- giving up. This is usually transient "
                      f"server-side flakiness (e.g. Eurostat rate-limiting a burst of "
                      f"requests) -- simply re-running the script often succeeds.")
    raise last_exc


# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------

COUNTRIES = ["NL", "DE", "FR", "IT", "ES", "BE", "AT"]   # euro-area panel
SAMPLE_START = "2000-01-01"
SAMPLE_END   = "2026-06-30"
HORIZONS = range(0, 9)          # h = 0..8 quarters
THETA = 2.0                     # logistic transition steepness (standardized z)
FOCUS_COUNTRY = "NL"            # country singled out for its own interaction term
STANDARDIZE_MODE = "pooled"     # 'pooled' (default, recommended) or 'within_country'
                                 # -- see the long comment in build_panel() for why
                                 # 'within_country' can cause a two-way-FE absorption
                                 # error for the exposure interaction term.

# ----------------------------------------------------------------------
# 1. WUI (global, GDP-weighted) -- Ahir, Bloom & Furceri
# ----------------------------------------------------------------------

def fetch_wui_global():
    """
    Quarterly Global World Uncertainty Index (WUI), GDP-weighted average
    across countries. Ahir, H., N. Bloom, and D. Furceri, "The World
    Uncertainty Index," NBER Working Paper 29763. Constructed by counting
    the frequency of the word "uncertain" (or variants) in the quarterly
    Economist Intelligence Unit country reports for 143 countries,
    GDP-weighted into a single global series.
    Source page: https://worlduncertaintyindex.com/data/
    Direct data file: https://worlduncertaintyindex.com/wp-content/uploads/2026/07/WUI_Data.xlsx
    Sheet: "F1" (the workbook tab backing the paper's Figure 1, "Global
    WUI -- GDP-Weighted Average").

    Confirmed layout (as of the 2026Q2 data vintage): two title rows,
    then a header row (2 title rows above it, i.e. header=2 in
    pandas' 0-indexed row numbering) with columns "year" (quarter string
    like "1990q1"), "year2" (plain calendar year, unused here), "WUI"
    (the GDP-weighted global index value -- 12496.47 in 1990Q1, rising to
    77923 by 2026Q2). Already quarterly at the source -- no monthly-to-
    quarterly resampling needed.

    Column presence is still checked explicitly (not blindly trusted)
    before use -- if "year" or "WUI" aren't found at header row 2, this
    raises a clear error showing the actual columns and the first few
    raw rows of every sheet, instead of failing several calls downstream
    with a bare KeyError.
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
            "at header row 2 of sheet 'F1' (got columns: "
            f"{list(candidate.columns)}). The file layout may have changed. "
            "First 5 raw rows of every sheet in the workbook: "
            f"{ {name: d.values.tolist() for name, d in sheets.items()} }. "
            "Inspect the layout above and update the sheet name / header "
            "row / column names accordingly."
        )

    df = candidate[[date_col, value_col]].rename(
        columns={date_col: "quarter_raw", value_col: "wui_global"}
    )
    df["wui_global"] = pd.to_numeric(df["wui_global"], errors="coerce")

    def _parse_quarter(v):
        # expected format: "1990q1" (case-insensitive) -> Period('1990Q1')
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
        raise ValueError(
            "fetch_wui_global: zero valid (quarter, value) rows after "
            "parsing sheet 'F1' -- check the date-column format manually "
            "(quarter_raw values before parsing: "
            f"{df['quarter_raw'].head(10).tolist() if 'quarter_raw' in df else 'N/A'})."
        )

    return df.set_index("quarter")[["wui_global"]]


AR_PURIFICATION_LAGS = 2  # number of WUI own-lags used to purify the shock


def purify_wui_shock(wui_global_df, lags=AR_PURIFICATION_LAGS):
    """
    AR(lags) purification of the WUI index: regress WUI_t on its own lags
    and take the residual as an approximate "identified shock" -- the
    part of WUI not predictable from its own recent history.

    WUI is itself already a quarterly, relatively persistent uncertainty
    index (unlike a daily/monthly news-count index), so its own lags
    typically explain a meaningful share of its variance -- the residual
    strips out that predictable component, leaving something closer to a
    "surprise" in uncertainty. This is a lightweight identification
    choice, not a structural one: it does NOT control for other macro/
    financial variables the way a full VAR would (e.g. the WUI paper's
    own VAR uses log average stock return, WUI, and GDP growth with a
    Cholesky ordering -- see Ahir, Bloom & Furceri, NBER WP 29763,
    Section on the global WUI). Treat this as a supplementary robustness
    check on the baseline spec, not a replacement for a fully identified
    structural shock.

    Returns (df, model): df has columns quarter, wui_global, wui_shock;
    model is the fitted statsmodels OLS result (printed diagnostics
    include R^2 and the AR coefficients, worth checking before trusting
    the residual -- a very low R^2 means WUI has little own-persistence
    to purify out in the first place, in which case wui_shock will look
    a lot like wui_global anyway).
    """
    import statsmodels.api as sm

    df = wui_global_df[["quarter", "wui_global"]].copy()
    df = df.sort_values("quarter").reset_index(drop=True)

    lag_cols = []
    for l in range(1, lags + 1):
        col = f"wui_lag{l}"
        df[col] = df["wui_global"].shift(l)
        lag_cols.append(col)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    X = sm.add_constant(df[lag_cols])
    y = df["wui_global"]
    model = sm.OLS(y, X).fit()
    df["wui_shock"] = model.resid.values

    print(f"  [AR({lags}) WUI purification] R^2 = {model.rsquared:.3f}")
    print(f"  {model.params.to_string()}")

    return df[["quarter", "wui_global", "wui_shock"]], model


def compute_wui_shock_std():
    """
    Fetches and AR-purifies the global WUI series (same steps as
    build_panel() does internally) and returns the standard deviation of
    the resulting wui_shock -- the "1 standard deviation shock" size
    used to rescale the IRF charts (see plot_baseline_irf(), plot_irf(),
    plot_regime_irf()) from "response per 1 raw unit of wui_used" (an
    arbitrary scale -- wui_used is an AR(2) residual, not a naturally
    interpretable unit) into "response per 1-stdev WUI shock" (the
    standard, directly interpretable convention in the shock-IRF
    literature).

    Cheap and idempotent to call again here independently of
    build_panel() -- just an Excel download and a simple AR(2)
    regression, not a network-heavy operation, and calling it twice
    (once per exposure spec inside build_panel(), once here) always
    returns the same value since the underlying WUI series and AR
    specification don't change between calls.
    """
    wui_global = fetch_wui_global().reset_index()
    shock_df, _ar_model = purify_wui_shock(wui_global)
    std = shock_df["wui_shock"].std()
    print(f"  [diagnostic] WUI shock (wui_used) standard deviation = {std:.4f} "
          f"-- this is the '1-stdev shock' used to rescale IRF charts.")
    return std


# Note: build_baseline_shock_panel() (a separate lightweight panel just for
# the shock baseline) has been removed -- now that build_panel() itself
# uses the AR-purified shock as wui_used, the baseline spec can reuse
# panel_ict/panel_corr directly (they already carry wui_used = wui_shock),
# the same way it did before the shock/level split was introduced.


# ----------------------------------------------------------------------
# 2. Eurostat pulls: quarterly GDP growth, ICT investment share
# ----------------------------------------------------------------------

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def eurostat_json_to_df(dataset, params):
    url = f"{EUROSTAT_BASE}/{dataset}"
    r = _requests_get_with_retry(url, params=params, timeout=60)
    js = r.json()

    # If the query matches nothing, Eurostat's JSON-stat response can omit
    # "value" (and sometimes "dimension"/"id") ENTIRELY rather than
    # returning them empty. Check for a well-formed response up front and
    # fail with the actual server payload shown, rather than a bare
    # KeyError three functions downstream.
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
    values = js["value"]  # dict (sparse) or list (dense): index -> value

    idx_lists = []
    for d in dim_ids:
        cat = dims[d]["category"]
        order = sorted(cat["index"].items(), key=lambda kv: kv[1])
        idx_lists.append([k for k, _ in order])

    # "value" is a dict of {flat_index_str: value} when sparse (the common
    # case), but Eurostat can return it as a plain list when dense.
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}

    import itertools
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


def fetch_gdp_growth_yoy():
    """
    Quarterly real GDP, chain-linked volumes, seasonally & calendar
    adjusted, dataset namq_10_gdp -- returns both the raw level
    (gdp_level, for the Raw_GDP export sheet) and the derived YEAR-ON-
    YEAR log growth (dgdp, used as the lag control throughout this
    script -- see the single dgdp_lag1 control in build_panel(),
    run_local_projections(), and run_baseline_wui_projections()).

    YoY (4-quarter log difference) rather than QoQ (1-quarter): this is
    a deliberate choice for the CONTROL variable specifically, not for
    the dependent variable -- the actual regression outcome
    (dgdp_cum_lead, built directly from gdp_level in
    run_local_projections()/run_baseline_wui_projections()) is already a
    cumulative h-period log difference by construction and is
    unaffected by this change; only the single lagged growth-rate
    control (dgdp_lag1) is redefined here, from QoQ to YoY, and only one
    lag of it is used (dgdp_lag2 has been removed throughout).
    """
    params = {
        "format": "JSON",
        "na_item": "B1GQ",
        "unit": "CLV10_MEUR",       # chain-linked volumes, 2010 ref, mEUR
        "s_adj": "SCA",             # seasonally + calendar adjusted
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000-Q1",
    }
    df = eurostat_json_to_df("namq_10_gdp", params)
    df = df.rename(columns={"geo": "country", "time": "quarter"})
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["gdp_level"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "quarter", "gdp_level"]].dropna().sort_values(["country", "quarter"])
    # YoY: 4-quarter (i.e. one full year) log difference, not 1-quarter.
    df["dgdp"] = df.groupby("country")["gdp_level"].transform(lambda s: 100 * np.log(s / s.shift(4)))
    return df[["country", "quarter", "gdp_level", "dgdp"]]


def fetch_ict_investment_share():
    """
    (ICT equipment + computer software/databases) GFCF as % of total
    GFCF, annual, per country. Dataset nama_10_a64_p5 ("Capital formation
    by industry (NACE Rev.2) and detailed asset type" --
    ec.europa.eu/eurostat/databrowser/view/nama_10_a64_p5), filtered to
    nace_r2="TOTAL" (the whole-economy aggregate across all NACE
    industries, rather than summing 64 individual industry categories
    ourselves) -- confirmed as the correct way to get an economy-wide
    total from this dataset via an independently found working query
    (fgeerolf.com/data/eurostat/nama_10_a64_p5.html: `nace_r2 == "TOTAL",
    asset10 == "N1132G"`). ESA2010 asset10 vocabulary
    (dd.eionet.europa.eu/vocabulary/eurostat/asset10):
      N1132G  ICT equipment (gross) -- numerator, component 1
      N1173G  Computer software and databases (gross) -- numerator,
              component 2
      N11G    Total fixed assets (gross) -- denominator

    N1173G has a KNOWN Eurostat coverage gap for some country/period
    combinations (a real data-availability gap, not a wrong ESA2010
    code). Missing N1173G values are treated as a zero contribution when
    summing rather than letting NaN propagate and wipe out the whole row
    -- N1132G alone still counts for a country/year that has equipment
    data but no software breakdown. N1132G and N11G are still required
    (dropped if missing); only N1173G is defaulted to zero.

    A diagnostic is printed reporting N1173G's actual non-null coverage,
    including the all-zero-coverage case (in which case the numerator is
    effectively N1132G alone for the whole sample, but the code still
    works rather than crashing).
    """
    params_num = {
        "format": "JSON",
        "unit": "CP_MEUR",
        "nace_r2": "TOTAL",
        "asset10": ["N1132G", "N1173G"],
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000",
    }
    params_den = {
        "format": "JSON",
        "unit": "CP_MEUR",
        "nace_r2": "TOTAL",
        "asset10": "N11G",
        "geo": COUNTRIES,
        "sinceTimePeriod": "2000",
    }
    raw_num = eurostat_json_to_df("nama_10_a64_p5", params_num)
    raw_den = eurostat_json_to_df("nama_10_a64_p5", params_den)

    for name, d in (("N1132G+N1173G", raw_num), ("N11G", raw_den)):
        d.rename(columns={"geo": "country", "time": "year"}, inplace=True)
        missing = {"country", "year"} - set(d.columns)
        if missing:
            raise ValueError(
                f"fetch_ict_investment_share ({name} call): expected columns "
                f"{missing} not found after renaming. Actual columns returned "
                f"by Eurostat: {list(d.columns)}."
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
        print("  [!] N1173G (computer software and databases) has ZERO "
              "non-null values for this country/period set -- a known "
              "Eurostat coverage gap, not a wrong code. Treating it as 0 "
              "in the numerator throughout (numerator is effectively "
              "N1132G alone for the whole sample); ict_share is still "
              "computed and the code does not crash.")
    else:
        n_total = len(num_wide)
        print(f"  [diagnostic] N1173G coverage: {n1173g_coverage}/{n_total} "
              f"country-year rows have a non-null value; the rest are "
              f"treated as 0 in the numerator.")

    den_agg = raw_den.groupby(["country", "year"])["value"].sum().rename("N11G").reset_index()

    wide = num_wide.merge(den_agg, on=["country", "year"], how="inner")
    wide["ict_numerator"] = wide["N1132G"] + wide["N1173G"].fillna(0)
    wide["ict_share"] = wide["ict_numerator"] / wide["N11G"]
    # N1132G and N11G are still required for a valid row -- only N1173G's
    # absence is tolerated (already zero-filled above).
    wide = wide.dropna(subset=["N1132G", "N11G", "ict_share"])

    if wide.empty:
        raise ValueError(
            "fetch_ict_investment_share: zero rows remain after computing "
            "ict_share. Diagnostics -- non-null counts: "
            f"N1132G={num_wide['N1132G'].notna().sum()}, "
            f"N1173G={num_wide['N1173G'].notna().sum()}, "
            f"N11G={den_agg['N11G'].notna().sum()}, "
            f"num rows={len(num_wide)}, den rows={len(den_agg)}."
        )
    return wide[["country", "year", "N1132G", "N1173G", "N11G", "ict_share"]]


def annual_to_quarterly(df_annual, value_col, quarters_index):
    """Flat-repeat annual value across the 4 quarters of that year (simple
    step interpolation; use PCHIP/cubic if you want smoothing)."""
    if df_annual.empty:
        raise ValueError(
            "annual_to_quarterly: input DataFrame is empty -- the upstream "
            "fetch (e.g. fetch_ict_investment_share) returned zero rows. "
            "Fix that first; an empty input here would otherwise silently "
            "produce a columnless output that fails later with a cryptic "
            "KeyError('country') at the merge step."
        )
    out = []
    for c, g in df_annual.groupby("country"):
        s = g.set_index("year")[value_col]
        for q in quarters_index:
            y = q.year
            if y in s.index:
                out.append({"country": c, "quarter": q, value_col: s.loc[y]})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 3. Semiconductor-correlation exposure proxy
# ----------------------------------------------------------------------

def _yf_close_series(ticker):
    """
    yfinance (recent versions) returns a MultiIndex-column DataFrame even
    for a single ticker unless told otherwise, so
    yf.download(ticker, ...)["Close"] can come back as a one-column
    DataFrame rather than a Series, which silently breaks any later
    .rename("some_name") call (DataFrame.rename() treats a bare string as
    an index-mapper function, not a name). Force a genuine Series here.
    """
    import yfinance as yf
    px = yf.download(ticker, start=SAMPLE_START, end=SAMPLE_END, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.squeeze("columns")
    return px


# National headline equity index tickers (Yahoo Finance). Check these are
# still the right/liquid ticker before running -- exchanges occasionally
# change index compositions or Yahoo's ticker symbols.
NATIONAL_INDEX_TICKERS = {
    "NL": "^AEX",        # AEX, Amsterdam
    "DE": "^GDAXI",       # DAX, Frankfurt
    "FR": "^FCHI",        # CAC 40, Paris
    "IT": "FTSEMIB.MI",   # FTSE MIB, Milan
    "ES": "^IBEX",        # IBEX 35, Madrid
    "BE": "^BFX",         # BEL 20, Brussels
    "AT": "^ATX",         # ATX, Vienna
}

CORR_WINDOW_QUARTERS = 8   # rolling window for national-vs-semiconductor
                           # correlation (the AI-exposure state variable, F)
SOX_REGIME_WINDOW_QUARTERS = 4   # rolling window for the SOX boom/bust regime
                                 # classification (Dum) -- DELIBERATELY separate
                                 # from CORR_WINDOW_QUARTERS above; shortened to
                                 # react faster to regime changes than the
                                 # 8-quarter exposure-correlation measure does.


def fetch_national_indices_raw(countries=COUNTRIES):
    """
    Raw quarterly close price + log return for each country's national
    headline equity index (see NATIONAL_INDEX_TICKERS), for the
    Raw_National_Index export sheet. Kept separate from
    fetch_corr_with_semiconductor_index() (which re-downloads the same
    tickers) so that function's already-checked correlation logic isn't
    touched by this addition -- the cost is a duplicate download per
    ticker, which is cheap relative to correctness risk here.
    """
    out = []
    for c in countries:
        ticker = NATIONAL_INDEX_TICKERS.get(c)
        if ticker is None:
            continue
        px = _yf_close_series(ticker)
        if px.empty:
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
    return result[["country", "ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])


def fetch_semiconductor_raw():
    """
    Raw quarterly close price + log return for the global semiconductor
    index (^SOX), for the Raw_Semiconductor export sheet. No country
    dimension -- this is a single global series.
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
    return df[["ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])


def fetch_corr_with_semiconductor_index(countries=COUNTRIES, window=CORR_WINDOW_QUARTERS):
    """
    AI-exposure state variable: rolling correlation between each
    country's national equity index returns and the global semiconductor
    index (^SOX) returns. A country whose stock market co-moves more
    tightly with the semiconductor cycle is coded as more "AI/chip
    exposed" through its listed corporates.

    Caveat: national-index-vs-^SOX correlation can rise during global
    risk-off episodes for reasons unrelated to AI diffusion (e.g. NL's
    AEX is trade- and semiconductor-supply-chain exposed via ASML
    regardless of AI-specific mechanisms). Treat this as a supplementary
    / robustness specification, not a substitute for the ICT-investment-
    share version.
    """
    semi_px = _yf_close_series("^SOX")
    semi_q = semi_px.resample("QE").last()
    semi_ret = np.log(semi_q / semi_q.shift(1))
    semi_ret.index = semi_ret.index.to_period("Q")

    out = []
    for c in countries:
        ticker = NATIONAL_INDEX_TICKERS.get(c)
        if ticker is None:
            print(f"  [!] No national index ticker configured for {c}, skipping.")
            continue
        px = _yf_close_series(ticker)
        if px.empty:
            print(f"  [!] Empty price series for {c} ({ticker}) -- check ticker.")
            continue
        q = px.resample("QE").last()
        ret = np.log(q / q.shift(1))
        ret.index = ret.index.to_period("Q")

        both = pd.concat([ret.rename("nat_ret"), semi_ret.rename("semi_ret")], axis=1).dropna()
        roll_corr = both["nat_ret"].rolling(window).corr(both["semi_ret"])
        roll_corr = roll_corr.rename("z_raw").reset_index()
        roll_corr["country"] = c
        out.append(roll_corr.rename(columns={"index": "quarter", roll_corr.columns[0]: "quarter"})
                    if "quarter" not in roll_corr.columns else roll_corr)

    df = pd.concat(out, ignore_index=True)
    date_col = [c for c in df.columns if c not in ("z_raw", "country")][0]
    df = df.rename(columns={date_col: "quarter"})
    return df[["country", "quarter", "z_raw"]].dropna()


def compute_sox_regime(window=SOX_REGIME_WINDOW_QUARTERS):
    """
    Boom/bust regime indicator for the AI/chip cycle, based on the global
    semiconductor index (^SOX) alone -- entity-invariant (depends only on
    quarter, not on country). Uses its OWN rolling window
    (SOX_REGIME_WINDOW_QUARTERS, shorter than CORR_WINDOW_QUARTERS used
    for the exposure-correlation measure -- see the config section for
    why they're kept separate): is_boom=1 (Dum=1 in the moderated
    regression -- see build_panel()) if the trailing `window`-quarter
    average log return of ^SOX is ABOVE THE MEDIAN of that same average-
    return series over the sample; 0 ("bust", the regression's reference
    level, Dum=0) otherwise.

    Thresholding at the MEDIAN (rather than at zero) is a deliberate
    choice so boom and bust each cover close to half the sample by
    construction -- ^SOX (like most equity indices) trends upward over
    most multi-year windows, so a literal zero threshold would classify
    considerably more quarters as "boom" than "bust", leaving the bust
    regime's coefficients (b4 alone, the reference level -- see
    summarize_irf()) identified off a much smaller, less balanced
    sub-sample than the boom regime's. Median-thresholding avoids that
    imbalance without changing the underlying economic interpretation
    (still "is the AI-chip cycle currently doing relatively well or
    relatively poorly", just calibrated to the sample's own central
    tendency rather than an arbitrary absolute zero).

    Purpose: F_it (built from the MAGNITUDE of a country's correlation
    with SOX) captures exposure to the AI-capex cycle, but not its
    DIRECTION at a given point in time. A country with high F stays
    high-F whether SOX is booming or busting -- so a single F*shock
    interaction term mixes two economically opposite regimes: during a
    boom, a positive demand/terms-of-trade tailwind from the AI-capex
    cycle should DAMPEN the WUI shock's impact (Channel 1, tailwind
    dominates); during a bust, shared/concentrated exposure to the same
    risk factor should AMPLIFY it (Channel 1, concentration-risk
    dominates). Splitting F*shock by is_boom/is_bust (see build_panel())
    lets these two sub-regimes be estimated separately instead of
    averaging over both.
    """
    px = _yf_close_series("^SOX")
    q = px.resample("QE").last()
    ret = np.log(q / q.shift(1))
    roll_mean_ret = ret.rolling(window).mean()

    df = pd.DataFrame({"sox_roll_mean_ret": roll_mean_ret})
    df.index = df.index.to_period("Q")
    df = df.reset_index().rename(columns={"index": "quarter"})
    if "quarter" not in df.columns:
        df = df.rename(columns={df.columns[0]: "quarter"})

    median_ret = df["sox_roll_mean_ret"].median()
    df["is_boom"] = (df["sox_roll_mean_ret"] > median_ret).astype(int)

    n_valid = df["sox_roll_mean_ret"].notna().sum()
    n_boom = int(df["is_boom"].sum())
    n_bust = n_valid - n_boom
    print(f"  [diagnostic] SOX regime split (median threshold = {median_ret:.6f}): "
          f"boom={n_boom} quarters, bust={n_bust} quarters, "
          f"out of {n_valid} valid quarters ({n_boom/n_valid:.1%} boom).")

    return df[["quarter", "sox_roll_mean_ret", "is_boom"]].dropna(subset=["sox_roll_mean_ret"])


# ----------------------------------------------------------------------
# 4. Build panel
# ----------------------------------------------------------------------

def build_panel(exposure="ict"):
    """
    exposure: 'ict'  -> ICT investment share of GFCF (primary)
              'corr' -> rolling correlation of national index with the
                        semiconductor index (robustness)

    WUI is always the global Ahir-Bloom-Furceri World Uncertainty Index
    (GDP-weighted), AR-purified (see purify_wui_shock()): wui_used is the
    AR(2) residual of the WUI level, not the raw level itself. This
    replaces the raw-level version used in earlier iterations of this
    script throughout -- the exposure interaction (wui_x_exposure = F *
    wui_used) and every downstream coefficient/IRF/chart are now based on
    the purified shock.
    """
    wui_global = fetch_wui_global().reset_index()
    wui_shock_df, _ar_model = purify_wui_shock(wui_global)
    gdp = fetch_gdp_growth_yoy()

    quarters = pd.PeriodIndex(pd.period_range(SAMPLE_START, SAMPLE_END, freq="Q"))

    if exposure == "ict":
        ict_annual = fetch_ict_investment_share()
        exp_q = annual_to_quarterly(ict_annual, "ict_share", quarters)
        exp_q = exp_q.rename(columns={"ict_share": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
    elif exposure == "corr":
        corr = fetch_corr_with_semiconductor_index(countries=COUNTRIES)
        panel = gdp.merge(corr, on=["country", "quarter"], how="left")
    else:
        raise ValueError("exposure must be 'ict' or 'corr'")

    panel = panel.merge(wui_shock_df, on="quarter", how="left")
    panel["wui_used"] = panel["wui_shock"]  # AR-purified shock, not the raw level

    panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # STANDARDIZATION OF z -- pooled across the panel, NOT within-country.
    #
    # Standardizing within each country makes F "relative to that
    # country's own history" -- but if several countries share a similar
    # underlying trend (e.g. ICT investment share rising steadily across
    # most euro-area members), each country's own-history z-score ends up
    # nearly identical to every other country's at the same date, even
    # though absolute exposure levels differ. Pooled standardization
    # preserves genuine between-country LEVEL differences in exposure
    # instead. (Historical note: this choice was originally motivated by
    # avoiding two-way-FE absorption of F_it -- see run_local_projections
    # for why this script now uses entity-only FE throughout instead,
    # which changes F's identification requirement to within-entity time
    # variation rather than cross-sectional variation at a given quarter.
    # Pooled standardization is kept regardless, since preserving
    # absolute cross-country exposure differences is a reasonable design
    # choice on its own merits, not just a fix for the old FE structure.)
    # ------------------------------------------------------------------
    if STANDARDIZE_MODE == "pooled":
        panel["z"] = (panel["z_raw"] - panel["z_raw"].mean()) / panel["z_raw"].std()
    elif STANDARDIZE_MODE == "within_country":
        panel["z"] = panel.groupby("country")["z_raw"].transform(
            lambda s: (s - s.mean()) / s.std()
        )
    else:
        raise ValueError("STANDARDIZE_MODE must be 'pooled' or 'within_country'")
    panel["F"] = 1 / (1 + np.exp(-THETA * panel["z"]))

    # GDP-growth lag control, year-on-year (see fetch_gdp_growth_yoy()).
    panel["dgdp_lag1"] = panel.groupby("country")["dgdp"].shift(1)

    # ------------------------------------------------------------------
    # EXPLICIT LINEAR TIME TREND -- an additional control, entity-
    # invariant (same value for every country at a given quarter, just a
    # 0,1,2,... counter over the panel's own quarters). Added because F
    # (the exposure state variable) trends upward over most of the
    # sample for nearly every country at once -- a shared, roughly
    # global AI/ICT-adoption trend -- which risks the WUI*F interaction
    # coefficient (b3/b4) partly picking up "how the economy's growth-
    # shock relationship changed over calendar time in general" rather
    # than "how it changed specifically WITH exposure." A plain additive
    # trend does NOT fully resolve this (it controls for a generic drift
    # in the LEVEL of growth, not specifically for drift in the shock's
    # own sensitivity -- that would require WUI*trend as a further
    # interaction, not implemented here), but it is a first, standard
    # control against confounding by a shared secular trend.
    #
    # IDENTIFICATION CAVEAT: time_trend is entity-invariant, exactly
    # like wui_used and is_boom -- estimable here only because this
    # script uses entity-only FE (no time FE, which would absorb it
    # completely, same reasoning as for the shock itself). It is NOT
    # perfectly collinear with wui_used by construction (the AR(2)
    # purification in purify_wui_shock() already strips out most smooth,
    # trend-like predictability from the shock series), but some
    # residual collinearity is possible -- see the printed correlation
    # diagnostic below, and treat a high correlation there as a signal
    # that b1 (wui_used's own coefficient) and the time_trend
    # coefficient may be poorly separately identified in that run.
    # ------------------------------------------------------------------
    quarter_order = sorted(panel["quarter"].unique())
    quarter_to_trend = {q: i for i, q in enumerate(quarter_order)}
    panel["time_trend"] = panel["quarter"].map(quarter_to_trend)

    _trend_shock_corr = panel[["time_trend", "wui_used"]].dropna().corr().iloc[0, 1]
    print(f"  [diagnostic] corr(time_trend, wui_used) = {_trend_shock_corr:.3f} "
          f"-- a magnitude above ~0.5-0.6 suggests b1 and the time_trend "
          f"coefficient may be weakly separately identified in this run.")

    # ------------------------------------------------------------------
    # MODERATED-REGRESSION SPECIFICATION (main effects AND interaction
    # terms, not a "difference" spec).
    #
    # ICT spec:  GDPgrowth = b0 + b1*WUI + b2*F + b3*(WUI*F) + controls
    # Corr spec: GDPgrowth = b0 + b1*WUI + b2*F + b3*Dum + b4*(WUI*F)
    #                        + b5*(WUI*Dum) + b6*(WUI*F*Dum) + controls
    #            (Dum = is_boom; the reference level Dum=0 is "bust", so
    #            the mitigating effect of WUI*F is b4 in bust and b4+b6
    #            in boom -- see run_local_projections()/summarize_irf()
    #            for where that sum is computed, with a correctly derived
    #            standard error, not just b4 read off on its own.)
    #
    # WUI (wui_used) and F are included as separate regressors (b1, b2),
    # not folded into an implicit baseline/intercept -- so b3/b4 are
    # genuine interaction-effect coefficients net of both main effects,
    # not a "F=1 minus F=0" difference.
    #
    # IDENTIFICATION: wui_used, is_boom, and their product (wui_x_dum)
    # are entity-invariant (identical across every country at a given
    # quarter -- WUI is a global shock, is_boom is derived from the
    # global ^SOX index). Time fixed effects would absorb all three
    # completely. Estimating b1, b3 (Corr spec's Dum term), and b5 as
    # genuinely identified coefficients therefore requires ENTITY fixed
    # effects only, no time effects -- see run_local_projections(), which
    # uses entity-only FE for both specs. Cost: common-across-countries
    # time-varying confounders (ECB policy, euro-area-wide demand shocks)
    # are not swept out of either spec's residual the way two-way FE
    # would.
    # ------------------------------------------------------------------
    panel["wui_x_exposure"] = panel["F"] * panel["wui_used"]

    # focus-country (NL) interaction terms: let the LP recover NL-specific
    # deviations from the panel-average coefficients, rather than only
    # reading off the 7-country average. Applied to every regressor in
    # the moderated spec (main effects and interactions alike).
    panel["is_focus"] = (panel["country"] == FOCUS_COUNTRY).astype(int)
    panel["wui_used_focus"] = panel["is_focus"] * panel["wui_used"]
    panel["F_focus"] = panel["is_focus"] * panel["F"]
    panel["wui_x_exposure_focus"] = panel["is_focus"] * panel["wui_x_exposure"]

    if exposure == "corr":
        regime = compute_sox_regime()
        panel = panel.merge(regime[["quarter", "is_boom"]], on="quarter", how="left")
        # is_bust is kept for reference/export only -- NOT used as a
        # regressor (a second, complementary dummy would be perfectly
        # collinear with is_boom + the intercept; standard k-1-dummies
        # practice, same principle discussed earlier for a 3-regime
        # low/normal/high design).
        panel["is_bust"] = 1 - panel["is_boom"]

        panel["wui_x_dum"] = panel["wui_used"] * panel["is_boom"]                    # b5: WUI x Dum
        panel["wui_x_exposure_x_dum"] = panel["wui_x_exposure"] * panel["is_boom"]   # b6: WUI x F x Dum

        panel["is_boom_focus"] = panel["is_focus"] * panel["is_boom"]
        panel["wui_x_dum_focus"] = panel["is_focus"] * panel["wui_x_dum"]
        panel["wui_x_exposure_x_dum_focus"] = panel["is_focus"] * panel["wui_x_exposure_x_dum"]

    return panel


# ----------------------------------------------------------------------
# 5. Local projections, panel FE, Driscoll-Kraay SE
# ----------------------------------------------------------------------

def check_time_variation(panel, col):
    """
    Pre-flight diagnostic for the entity-only-FE design used throughout
    this script: identification needs `col` to vary OVER TIME within
    each country (entity FE only remove each country's own time-mean;
    they don't require cross-sectional variation the way two-way FE
    would). Reports the average and minimum within-country time-series
    std -- near zero for some country means that country contributes
    ~nothing to identifying the coefficient on `col`.
    """
    by_i = panel.groupby("country")[col].std()
    avg_std, min_std = by_i.mean(), by_i.min()
    print(f"  [diagnostic] {col}: mean within-country time-series std "
          f"= {avg_std:.6g}, min = {min_std:.6g}")
    if min_std < 1e-8 or avg_std < 1e-6:
        print(f"  [!] WARNING: {col} has near-zero within-country time "
              f"variation for at least one country -- check that "
              f"country's underlying data for this regressor.")


def _dk_bandwidth(h):
    """
    Driscoll-Kraay (Bartlett-kernel) bandwidth for horizon h, GROWING
    with h rather than a small fixed constant.

    The dependent variable at horizon h (dgdp_cum_lead) is a CUMULATIVE
    log difference from t-1 to t+h -- it telescopes to the sum of h+1
    individual-period growth rates. This overlapping-window construction
    induces MA(h)-type serial correlation in the residuals (the same
    reasoning as Hodrick (1992) standard errors for overlapping-return
    regressions): at h=0 the induced correlation is minimal, but it
    grows directly with h. A FIXED bandwidth (this script previously
    used bandwidth=4 at every horizon) is therefore adequate only for
    the shortest horizons and UNDERSTATES the true serial correlation at
    longer ones (roughly h>=4 in this script's HORIZONS=range(0,9)),
    making those horizons' standard errors too narrow/overconfident.
    max(4, h+1) keeps the original floor of 4 for short horizons (a
    reasonable Newey-West-style minimum given the panel's overall
    quarterly sample size) while ensuring the bandwidth is always at
    least as large as the horizon itself.
    """
    return max(4, h + 1)


def run_local_projections(panel, include_focus_interaction=True):
    """
    Moderated-regression specification (main effects AND interaction
    terms, not a "difference" spec) -- see the long comment in
    build_panel() for the full equations. Auto-detects whether `panel`
    has the boom/bust regime split (is_boom, present only for the Corr
    spec) and switches the regressor set accordingly:

    - ICT spec:  wui_used (b1), F (b2), wui_x_exposure (b3, = WUI*F).
    - Corr spec: wui_used (b1), F (b2), is_boom (b3, Dum), wui_x_exposure
      (b4, WUI*F), wui_x_dum (b5, WUI*Dum), wui_x_exposure_x_dum (b6,
      WUI*F*Dum). The mitigating effect of WUI*F is b4 in the reference
      regime (Dum=0, "bust") and b4+b6 in the other regime (Dum=1,
      "boom") -- summarize_irf() computes both, the latter with a
      correctly derived combined standard error (not just b4 and b6's
      individual SEs combined naively).

    ENTITY fixed effects only, NO time effects -- required because
    wui_used, is_boom, and wui_x_dum are entity-invariant (identical
    across every country at a given quarter); time effects would absorb
    all three completely. See build_panel()'s docstring for the full
    identification argument and the resulting trade-off (this spec no
    longer sweeps out common-across-countries time-varying confounders
    the way a two-way-FE design would).

    If include_focus_interaction=True, also adds the FOCUS_COUNTRY
    interaction term(s) for every regressor above, so NL's total
    coefficient(s) can be recovered as beta + beta_focus_deviation.

    drop_absorbed=True (on the PanelOLS constructor) is a safety net: if
    a regressor is fully absorbed despite the diagnostic checks, the fit
    drops it and continues rather than crashing the whole run.

    STANDARD ERRORS: Driscoll-Kraay (cov_type="kernel", Bartlett kernel),
    robust to heteroskedasticity, within-country serial correlation, AND
    cross-sectional dependence across the 7 countries (the latter
    matters specifically because wui_used/is_boom/wui_x_dum are
    entity-invariant common shocks, and this spec uses entity-only FE --
    no time effects to otherwise sweep out unmodeled common factors).
    The Bartlett-kernel BANDWIDTH GROWS WITH HORIZON h (see
    _dk_bandwidth()) rather than using a small fixed constant, since the
    cumulative dependent variable's overlapping-window construction
    induces serial correlation that itself grows with h.
    """
    from linearmodels.panel import PanelOLS

    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()  # PanelOLS needs date-like time index

    has_boom_bust = "is_boom" in panel.columns
    if has_boom_bust:
        base_terms = ["wui_used", "F", "is_boom", "wui_x_exposure", "wui_x_dum",
                      "wui_x_exposure_x_dum"]
        focus_terms = ["wui_used_focus", "F_focus", "is_boom_focus", "wui_x_exposure_focus",
                       "wui_x_dum_focus", "wui_x_exposure_x_dum_focus"]
    else:
        base_terms = ["wui_used", "F", "wui_x_exposure"]
        focus_terms = ["wui_used_focus", "F_focus", "wui_x_exposure_focus"]

    for col in base_terms + (focus_terms if include_focus_interaction else []):
        check_time_variation(panel, col)

    regressors = base_terms + (focus_terms if include_focus_interaction else []) + \
        ["dgdp_lag1", "time_trend"]

    results = {}
    for h in HORIZONS:
        p = panel.copy()
        # CUMULATIVE IRF -- see the identical construction/rationale in
        # run_baseline_wui_projections() below.
        p["dgdp_cum_lead"] = p.groupby("country")["gdp_level"].transform(
            lambda s, h=h: 100 * (np.log(s.shift(-h)) - np.log(s.shift(1)))
        )
        p = p.dropna(subset=["dgdp_cum_lead"] + regressors)
        p = p.set_index(["entity", "time"])

        exog = p[regressors]
        mod = PanelOLS(p["dgdp_cum_lead"], exog, entity_effects=True, time_effects=False,
                        drop_absorbed=True)
        bw = _dk_bandwidth(h)
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
        results[h] = res
        dropped = [r for r in regressors if r not in res.params.index]
        if dropped:
            print(f"  [!] h={h}: {dropped} were ABSORBED and dropped -- "
                  f"not estimated at this horizon.")
        print(f"--- h={h} (Driscoll-Kraay bandwidth={bw}) ---")
        print(res.params)
        print(res.std_errors)
        print()
    return results


def _linear_combination(res, weights: dict):
    """
    Point estimate and correct standard error for an arbitrary weighted
    sum of coefficients, e.g. {"wui_x_exposure": 1, "wui_x_exposure_x_dum": 1}
    for the boom-regime mitigating effect (b4+b6), or including focus-
    country deviation terms too for NL's own total (a 4-term sum). Uses
    the FULL coefficient covariance matrix -- NOT SEs summed in
    quadrature, which ignores covariance between terms and is wrong here
    since they're estimated on overlapping data. Terms missing from
    res.params (absorbed at this horizon) are skipped with a printed
    note; the returned estimate then covers only the terms that survived.
    """
    present = {t: w for t, w in weights.items() if t in res.params.index}
    missing = [t for t in weights if t not in present]
    if missing:
        print(f"  [!] _linear_combination: {missing} absorbed/missing, "
              f"excluded from this combination.")
    if not present:
        return np.nan, np.nan
    b = sum(w * res.params[t] for t, w in present.items())
    cov = res.cov
    var = sum(present[i] * present[j] * cov.loc[i, j] for i in present for j in present)
    se = np.sqrt(var) if var > 0 else np.nan
    return b, se


def _summarize_single_term(rows_by_h, results, term, focus_term, label, focus_country):
    """Shared logic for one regressor (any of b1..b6): fills panel-average
    and NL-total columns into rows_by_h (keyed by h), handling
    per-horizon absorption gracefully."""
    for h, res in results.items():
        row = rows_by_h[h]
        if term in res.params.index:
            row[f"beta_{label}_panelavg"] = res.params[term]
            row[f"se_{label}_panelavg"] = res.std_errors[term]
            row[f"is_absorbed_{label}_panelavg"] = False
        else:
            row[f"beta_{label}_panelavg"] = np.nan
            row[f"se_{label}_panelavg"] = np.nan
            row[f"is_absorbed_{label}_panelavg"] = True

        has_focus_col = focus_term in res.params.index
        if has_focus_col:
            b_nl, se_nl = _linear_combination(res, {term: 1, focus_term: 1})
            row[f"beta_{label}_{focus_country}_deviation"] = res.params[focus_term]
            row[f"beta_{label}_{focus_country}_total"] = b_nl
            row[f"se_{label}_{focus_country}_total"] = se_nl
            row[f"is_absorbed_{label}_{focus_country}"] = False
        else:
            row[f"beta_{label}_{focus_country}_deviation"] = np.nan
            row[f"beta_{label}_{focus_country}_total"] = np.nan
            row[f"se_{label}_{focus_country}_total"] = np.nan
            row[f"is_absorbed_{label}_{focus_country}"] = True


def summarize_irf(results, focus_country=FOCUS_COUNTRY):
    """
    Reports panel-average and NL-total coefficients for EVERY regressor
    in the moderated spec (see run_local_projections() for the equations
    and identification argument):

    ICT spec:  b1_wui, b2_F, b3_wui_x_F
    Corr spec: b1_wui, b2_F, b3_dum, b4_wui_x_F, b5_wui_x_dum,
               b6_wui_x_F_x_dum

    For the Corr spec, also derives two quantities that answer the
    actual question of interest directly (rather than requiring the
    reader to add coefficients by hand): mitigating_effect_bust (= b4
    alone, since Dum=0/bust is the reference level) and
    mitigating_effect_boom (= b4+b6), both with a correctly derived
    combined standard error via _linear_combination() -- NOT b4 and b6's
    individual SEs added naively.

    Checked per-horizon (not once from the first result) since absorption
    can be horizon-specific; absorbed terms get NaN with is_absorbed_*
    flagged True instead of crashing on a missing key.
    """
    has_boom_bust = any("is_boom" in r.params.index for r in results.values())

    if has_boom_bust:
        term_label_triples = [
            ("wui_used", "wui_used_focus", "b1_wui"),
            ("F", "F_focus", "b2_F"),
            ("is_boom", "is_boom_focus", "b3_dum"),
            ("wui_x_exposure", "wui_x_exposure_focus", "b4_wui_x_F"),
            ("wui_x_dum", "wui_x_dum_focus", "b5_wui_x_dum"),
            ("wui_x_exposure_x_dum", "wui_x_exposure_x_dum_focus", "b6_wui_x_F_x_dum"),
        ]
    else:
        term_label_triples = [
            ("wui_used", "wui_used_focus", "b1_wui"),
            ("F", "F_focus", "b2_F"),
            ("wui_x_exposure", "wui_x_exposure_focus", "b3_wui_x_F"),
        ]

    rows_by_h = {h: {"h": h} for h in results}
    for term, focus_term, label in term_label_triples:
        _summarize_single_term(rows_by_h, results, term, focus_term, label, focus_country)

    if has_boom_bust:
        for h, res in results.items():
            row = rows_by_h[h]

            # Mitigating effect in BUST (Dum=0, the reference level) = b4
            # alone -- already computed above under the b4_wui_x_F label.
            row["mitigating_effect_bust_panelavg"] = row.get("beta_b4_wui_x_F_panelavg", np.nan)
            row["se_mitigating_effect_bust_panelavg"] = row.get("se_b4_wui_x_F_panelavg", np.nan)
            row[f"mitigating_effect_bust_{focus_country}_total"] = \
                row.get(f"beta_b4_wui_x_F_{focus_country}_total", np.nan)
            row[f"se_mitigating_effect_bust_{focus_country}_total"] = \
                row.get(f"se_b4_wui_x_F_{focus_country}_total", np.nan)

            # Mitigating effect in BOOM (Dum=1) = b4 + b6.
            b_boom, se_boom = _linear_combination(
                res, {"wui_x_exposure": 1, "wui_x_exposure_x_dum": 1}
            )
            row["mitigating_effect_boom_panelavg"] = b_boom
            row["se_mitigating_effect_boom_panelavg"] = se_boom

            b_boom_nl, se_boom_nl = _linear_combination(res, {
                "wui_x_exposure": 1, "wui_x_exposure_x_dum": 1,
                "wui_x_exposure_focus": 1, "wui_x_exposure_x_dum_focus": 1,
            })
            row[f"mitigating_effect_boom_{focus_country}_total"] = b_boom_nl
            row[f"se_mitigating_effect_boom_{focus_country}_total"] = se_boom_nl

    return pd.DataFrame(list(rows_by_h.values()))


# ----------------------------------------------------------------------
# 5b. Baseline WUI shock IRF -- NO exposure interaction, NO time effects
# ----------------------------------------------------------------------

def run_baseline_wui_projections(panel):
    """
    The "raw" GDP response to a WUI shock, with the exposure/mitigation
    channel switched off entirely (no F, no interaction term) -- this is
    what the mitigation coefficient in run_local_projections() is a
    DEVIATION from, so plotting the two together shows whether AI
    exposure is dampening a response that is itself significant.

    IMPORTANT: this regression can NOT include time fixed effects.
    wui_used = WUI_t is identical across every country at each t (a
    common/global shock), so time effects would absorb it completely --
    exactly the collinearity problem documented in build_panel() for the
    interaction term, but here it hits the shock variable itself, not
    just a split of it. So this spec uses entity (country) fixed effects
    only, which control for average cross-country growth-level
    differences but NOT for other common-across-countries confounders at
    a given quarter (ECB policy, global demand shocks, etc.) -- those are
    absent here by construction, since removing them would remove WUI's
    effect too. Treat this as a plain average effect of the shock across
    the panel, not a fully "clean" IRF in the two-way-FE sense used
    elsewhere in this script.

    STANDARD ERRORS: Driscoll-Kraay, same rationale and growing
    bandwidth (_dk_bandwidth()) as run_local_projections() -- see that
    function's docstring.
    """
    from linearmodels.panel import PanelOLS

    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()

    regressors = ["wui_used", "dgdp_lag1", "time_trend"]

    results = {}
    for h in HORIZONS:
        p = panel.copy()
        # Cumulative IRF -- same construction as run_local_projections();
        # see that function's comment for why this is preferred over
        # summing point estimates from separate per-period regressions.
        p["dgdp_cum_lead"] = p.groupby("country")["gdp_level"].transform(
            lambda s, h=h: 100 * (np.log(s.shift(-h)) - np.log(s.shift(1)))
        )
        p = p.dropna(subset=["dgdp_cum_lead"] + regressors)
        p = p.set_index(["entity", "time"])

        exog = p[regressors]
        mod = PanelOLS(p["dgdp_cum_lead"], exog, entity_effects=True, time_effects=False,
                        drop_absorbed=True)
        bw = _dk_bandwidth(h)
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
        results[h] = res
        if "wui_used" not in res.params.index:
            print(f"  [!] h={h}: wui_used was ABSORBED and dropped -- "
                  f"not estimated at this horizon.")
        print(f"--- baseline h={h} (Driscoll-Kraay bandwidth={bw}) ---")
        print(res.params)
        print(res.std_errors)
        print()
    return results


def summarize_baseline_irf(results):
    """
    beta_wui_baseline / se_wui_baseline: the average GDP response to a
    one-unit WUI increase at horizon h, with exposure/mitigation switched
    off (see run_baseline_wui_projections). Compare its t-stat
    (beta/se) and significance against the mitigation coefficients in
    summarize_irf() -- a significant baseline response with a
    significant, offsetting mitigation coefficient is the "AI buffers
    the shock" story; a baseline that's already insignificant means
    there's little for exposure to mitigate in the first place.
    """
    rows = []
    for h, res in results.items():
        row = {"h": h}
        if "wui_used" in res.params.index:
            b = res.params["wui_used"]
            se = res.std_errors["wui_used"]
            row["beta_wui_baseline"] = b
            row["se_wui_baseline"] = se
            row["t_stat_baseline"] = b / se if se > 0 else np.nan
            row["is_absorbed_baseline"] = False
        else:
            row["beta_wui_baseline"] = np.nan
            row["se_wui_baseline"] = np.nan
            row["t_stat_baseline"] = np.nan
            row["is_absorbed_baseline"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def plot_baseline_irf(irf_df, title="Baseline WUI shock IRF (no exposure interaction)",
                       shock_std=1.0):
    """
    Plots the unconditional, CUMULATIVE WUI-shock IRF with a 90% CI band,
    so you can see directly whether the cumulative response is
    statistically significant at each horizon (the band crossing zero
    means "not significant at that h"). beta_wui_baseline is already a
    cumulative coefficient -- see run_baseline_wui_projections().

    shock_std: rescales the plotted beta/se from "response per 1 RAW
    unit of wui_used" (an arbitrary scale -- wui_used is an AR(2)
    residual, not a naturally interpretable unit) into "response per
    1-STANDARD-DEVIATION WUI shock" -- pass compute_wui_shock_std()'s
    return value here. Defaults to 1.0 (no rescaling, i.e. raw units) if
    not provided, so this function still works standalone. Only the
    CHART is rescaled -- the underlying irf_df/Excel table keeps the
    original per-raw-unit coefficients (t-stat is unaffected either way,
    since scaling beta and se by the same constant leaves their ratio
    unchanged).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = irf_df["h"]
    b = irf_df["beta_wui_baseline"] * shock_std
    se = irf_df["se_wui_baseline"] * shock_std
    ax.plot(h, b, marker="o", color="#2ca02c", label="Baseline WUI effect")
    ax.fill_between(h, b - 1.645 * se, b + 1.645 * se, alpha=0.2, color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("Cumulative dGDP response per 1-stdev WUI shock (0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 6. IRF plots
# ----------------------------------------------------------------------

def plot_irf(irf_df, title, focus_country=FOCUS_COUNTRY, shock_std=1.0):
    """
    Plots the panel-average and NL-specific CUMULATIVE b3 (WUI*F
    interaction) coefficient across horizons, each with a 90% CI band
    (+/- 1.645 SE). This is the ICT spec's mitigating-effect coefficient,
    net of the WUI and F main effects (see run_local_projections()).
    Returns the matplotlib Figure (caller decides whether to
    save/embed/show it).

    shock_std: rescales the plotted beta/se from "per 1 raw unit of
    wui_used" into "per 1-stdev WUI shock" -- see plot_baseline_irf()'s
    docstring for the full rationale. Defaults to 1.0 (no rescaling).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    h = irf_df["h"]
    b_panel = irf_df["beta_b3_wui_x_F_panelavg"] * shock_std
    se_panel = irf_df["se_b3_wui_x_F_panelavg"] * shock_std
    ax.plot(h, b_panel, marker="o", label="Panel average", color="#1f77b4")
    ax.fill_between(h, b_panel - 1.645 * se_panel, b_panel + 1.645 * se_panel,
                     alpha=0.2, color="#1f77b4")

    col_nl = f"beta_b3_wui_x_F_{focus_country}_total"
    se_nl_col = f"se_b3_wui_x_F_{focus_country}_total"
    if col_nl in irf_df.columns:
        b_nl = irf_df[col_nl] * shock_std
        se_nl = irf_df[se_nl_col] * shock_std
        ax.plot(h, b_nl, marker="s", label=f"{focus_country} total", color="#d62728")
        ax.fill_between(h, b_nl - 1.645 * se_nl, b_nl + 1.645 * se_nl,
                         alpha=0.15, color="#d62728")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("Cumulative WUI x F interaction effect per 1-stdev WUI shock (b3, 0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_regime_irf(irf_df, regime, title, focus_country=FOCUS_COUNTRY, color="#2ca02c",
                     color_nl="#98df8a", shock_std=1.0):
    """
    Plots ONE regime's mitigating-effect coefficient (panel-average and
    NL-total) across horizons, each with a 90% CI band. Call this once
    with regime="mitigating_effect_boom" and once with
    regime="mitigating_effect_bust" to get two separate charts -- see
    summarize_irf() for how these are derived: mitigating_effect_bust is
    b4 alone (the Dum=0 reference level), mitigating_effect_boom is
    b4+b6 (with a correctly combined SE).

    shock_std: rescales the plotted beta/se from "per 1 raw unit of
    wui_used" into "per 1-stdev WUI shock" -- see plot_baseline_irf()'s
    docstring for the full rationale. Defaults to 1.0 (no rescaling).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = irf_df["h"]

    col_panel = f"{regime}_panelavg"
    se_panel_col = f"se_{regime}_panelavg"
    b_panel = irf_df[col_panel] * shock_std
    se_panel = irf_df[se_panel_col] * shock_std
    ax.plot(h, b_panel, marker="o", label="Panel average", color=color)
    ax.fill_between(h, b_panel - 1.645 * se_panel, b_panel + 1.645 * se_panel,
                     alpha=0.2, color=color)

    col_nl = f"{regime}_{focus_country}_total"
    se_nl_col = f"se_{regime}_{focus_country}_total"
    if col_nl in irf_df.columns:
        b_nl = irf_df[col_nl] * shock_std
        se_nl = irf_df[se_nl_col] * shock_std
        ax.plot(h, b_nl, marker="s", label=f"{focus_country} total", color=color_nl)
        ax.fill_between(h, b_nl - 1.645 * se_nl, b_nl + 1.645 * se_nl,
                         alpha=0.15, color=color_nl)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("Cumulative mitigating effect per 1-stdev WUI shock (0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 7. Export everything to one Excel workbook (data + IRF tables + plots)
# ----------------------------------------------------------------------

VARIABLE_DEFINITIONS = {
    "country":  "ISO2 country code",
    "quarter":  "Calendar quarter (period)",
    "year":     "Calendar year (annual-frequency raw series)",
    "gdp_level": "Real GDP level, chain-linked volumes (Eurostat namq_10_gdp, CLV10_MEUR, seasonally & calendar adjusted) -- raw source series, not growth",
    "dgdp":     "Real GDP growth, YEAR-ON-YEAR, log-difference x100 (derived from gdp_level, 4-quarter shift -- see fetch_gdp_growth_yoy()). Used only to build the single lag control (dgdp_lag1); the regression's actual dependent variable (dgdp_cum_lead) is a separate, directly-constructed cumulative measure, unaffected by this.",
    "dgdp_lag1": "dgdp (year-on-year GDP growth), lagged 1 quarter -- one of two controls used throughout this script (ICT spec, Corr spec, and the baseline spec alike), alongside time_trend; a second GDP lag (dgdp_lag2) is not used.",
    "time_trend": "Explicit LINEAR TIME TREND control: an integer counter (0, 1, 2, ...) over the panel's own quarters, entity-invariant (same value for every country at a given quarter). Added because F trends upward over most of the sample for nearly every country at once (a shared AI/ICT-adoption trend), which risks the WUI*F interaction coefficient partly reflecting a generic drift in growth over calendar time rather than something specifically tied to exposure. Does NOT fully resolve this (a plain additive trend controls for drift in the LEVEL of growth, not specifically for drift in the shock's own sensitivity -- that would need a further WUI*time_trend interaction, not included here). See build_panel() for the printed corr(time_trend, wui_used) diagnostic and its interpretation.",
    "wui_global": "Global World Uncertainty Index (RAW LEVEL), GDP-weighted average across countries, quarterly (Ahir, Bloom & Furceri, NBER WP 29763) -- kept in the panel for reference, but NOT what the model estimates on (see wui_used)",
    "wui_used": "The WUI series actually used in the model: wui_shock, the AR(2)-purified WUI shock (see purify_wui_shock()) -- NOT the raw level. Every coefficient, IRF, and chart in this workbook is based on this purified shock.",
    "z_raw":    "Raw AI/ICT-exposure state variable before standardization -- ICT investment share of GFCF (Spec A) or rolling correlation of the national equity index with the semiconductor index (Spec B)",
    "z":        "Standardized z_raw (STANDARDIZE_MODE='pooled' across the whole panel by default)",
    "F":        "Logistic transform of z: F = 1 / (1 + exp(-THETA * z)), the smooth 0-1 AI-exposure state weight",
    "wui_x_exposure": "b3 (ICT spec) / b4 (Corr spec) interaction regressor: F * wui_used. Its coefficient is the mitigating effect NET OF the WUI and F main effects (wui_used, F), which are now separately included in the regression -- not a 'difference' spec.",
    "is_focus": f"1 if country == FOCUS_COUNTRY ({FOCUS_COUNTRY}), 0 otherwise",
    "wui_used_focus": "is_focus * wui_used -- NL's own b1 (WUI main effect) interaction term",
    "F_focus": "is_focus * F -- NL's own b2 (F main effect) interaction term",
    "wui_x_exposure_focus": "is_focus * wui_x_exposure -- NL's own interaction-term deviation (b3 for ICT spec, b4 for Corr spec)",
    "is_boom": "1 if the trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) is ABOVE THE MEDIAN of that series over the sample (AI-capex boom, roughly the top half of quarters by this measure), 0 otherwise (bust, roughly the bottom half, the reference level Dum=0) -- median-thresholded so boom and bust each cover close to half the sample by construction, rather than a literal zero threshold (which would skew toward 'boom' given equity indices trend upward over most multi-year windows). Uses its own (shorter) rolling window than the exposure-correlation measure -- see compute_sox_regime() and SOX_REGIME_WINDOW_QUARTERS. Entity-invariant (same for all countries in a given quarter). Also used directly as regressor b3 (Dum) in the Corr spec's moderated regression. Corr spec only.",
    "is_bust": "1 - is_boom. Kept for reference/export only -- NOT used as a regressor (would be perfectly collinear with is_boom + the intercept). Corr spec only.",
    "is_boom_focus": "is_focus * is_boom -- NL's own b3 (Dum main effect) interaction term. Corr spec only.",
    "sox_roll_mean_ret": "Trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) -- the continuous series is_boom is thresholded from",
    "wui_x_dum": "b5 (Corr spec): wui_used * is_boom (WUI x Dum interaction). Entity-invariant like wui_used and is_boom themselves -- estimable only because this spec uses entity-only fixed effects (see run_local_projections()).",
    "wui_x_dum_focus": "is_focus * wui_x_dum -- NL's own b5 interaction term. Corr spec only.",
    "wui_x_exposure_x_dum": "b6 (Corr spec): wui_x_exposure * is_boom (WUI x F x Dum, the full three-way interaction). Corr spec only.",
    "wui_x_exposure_x_dum_focus": "is_focus * wui_x_exposure_x_dum -- NL's own b6 interaction term. Corr spec only.",
    "h": "Local-projection horizon, in quarters",
    "beta_b1_wui_panelavg": "Panel-average CUMULATIVE coefficient on wui_used (b1, the WUI main effect) over horizons 0..h. Both ICT and Corr specs.",
    "se_b1_wui_panelavg": "Standard error of beta_b1_wui_panelavg (Driscoll-Kraay-style)",
    "beta_b2_F_panelavg": "Panel-average CUMULATIVE coefficient on F (b2, the exposure-weight main effect) over horizons 0..h. Both ICT and Corr specs.",
    "se_b2_F_panelavg": "Standard error of beta_b2_F_panelavg",
    "beta_b3_wui_x_F_panelavg": "Panel-average CUMULATIVE coefficient on wui_x_exposure (b3, ICT spec's WUI*F interaction) over horizons 0..h -- the ICT spec's mitigating-effect coefficient, net of the WUI and F main effects. A POSITIVE value means exposure dampens/offsets the (typically negative) WUI shock response; negative means amplification.",
    "se_b3_wui_x_F_panelavg": "Standard error of beta_b3_wui_x_F_panelavg",
    f"beta_b1_wui_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b1 (WUI main effect) coefficient (panel-average + deviation)",
    f"se_b1_wui_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b1 coefficient",
    f"beta_b2_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b2 (F main effect) coefficient (panel-average + deviation)",
    f"se_b2_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b2 coefficient",
    f"beta_b3_wui_x_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b3 (ICT spec's WUI*F interaction) coefficient (panel-average + deviation) -- {FOCUS_COUNTRY}'s own mitigating-effect coefficient",
    f"se_b3_wui_x_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b3 coefficient",
    "beta_b3_dum_panelavg": "Panel-average CUMULATIVE coefficient on is_boom (b3, the Dum/regime main effect) over horizons 0..h. Corr spec only.",
    "se_b3_dum_panelavg": "Standard error of beta_b3_dum_panelavg",
    "beta_b4_wui_x_F_panelavg": "Panel-average CUMULATIVE coefficient on wui_x_exposure (b4, Corr spec's WUI*F interaction) over horizons 0..h -- this is ALSO the mitigating effect in the reference regime (Dum=0, 'bust'); see mitigating_effect_bust_panelavg, which equals this exactly.",
    "se_b4_wui_x_F_panelavg": "Standard error of beta_b4_wui_x_F_panelavg",
    "beta_b5_wui_x_dum_panelavg": "Panel-average CUMULATIVE coefficient on wui_x_dum (b5, WUI*Dum interaction) over horizons 0..h. Corr spec only.",
    "se_b5_wui_x_dum_panelavg": "Standard error of beta_b5_wui_x_dum_panelavg",
    "beta_b6_wui_x_F_x_dum_panelavg": "Panel-average CUMULATIVE coefficient on wui_x_exposure_x_dum (b6, the full three-way WUI*F*Dum interaction) over horizons 0..h -- this is the ADDITIONAL mitigating effect specific to the boom regime, on top of b4. Corr spec only.",
    "se_b6_wui_x_F_x_dum_panelavg": "Standard error of beta_b6_wui_x_F_x_dum_panelavg",
    "mitigating_effect_bust_panelavg": "The mitigating effect of WUI*F in the BUST regime (Dum=0, the reference level) = b4 alone. Expected NEGATIVE if shared/concentrated exposure to the AI-chip cycle amplifies the (typically negative) WUI shock response during downturns.",
    "se_mitigating_effect_bust_panelavg": "Standard error of mitigating_effect_bust_panelavg (= se_b4_wui_x_F_panelavg exactly, since the bust effect IS b4)",
    "mitigating_effect_boom_panelavg": "The mitigating effect of WUI*F in the BOOM regime (Dum=1) = b4 + b6, with a correctly derived COMBINED standard error (via the full coefficient covariance matrix, NOT b4's and b6's individual SEs summed naively). Expected POSITIVE if the demand/terms-of-trade tailwind dampens the WUI shock response during AI-capex booms.",
    "se_mitigating_effect_boom_panelavg": "Correctly derived combined standard error of mitigating_effect_boom_panelavg (b4+b6)",
    f"mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the bust regime (panel-average b4 + {FOCUS_COUNTRY}'s b4 deviation)",
    f"se_mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s bust mitigating effect",
    f"mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the boom regime (panel-average b4+b6, plus {FOCUS_COUNTRY}'s own b4 and b6 deviations, all four terms combined with a correctly derived SE)",
    f"se_mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"Correctly derived combined standard error of {FOCUS_COUNTRY}'s boom mitigating effect (4-term combination)",
    "beta_wui_baseline": "Average CUMULATIVE GDP response to a one-unit WUI shock increase over horizons 0..h, WITHOUT the exposure/mitigation interaction (entity FE only, no time FE -- see run_baseline_wui_projections)",
    "se_wui_baseline": "Standard error of beta_wui_baseline",
    "t_stat_baseline": "t-statistic of beta_wui_baseline (beta/se) -- |t|>~1.96 is significant at the 5% level",
    "is_absorbed_baseline": "True if wui_used was fully absorbed at this horizon",
    "wui_shock": "Residual from an AR(2) regression of wui_global on its own lags -- an approximate 'identified shock' (the unpredictable part of WUI). This IS wui_used throughout the model now. See purify_wui_shock() for why this is a simplified stand-in for a fully identified structural shock, not a reproduction of one.",
    "N1132G": "ICT equipment (data-centre hardware, servers, networking equipment fall under this category), gross fixed capital formation, current prices, MEUR (Eurostat nama_10_a64_p5, nace_r2=TOTAL, asset10=N1132G) -- raw source series, numerator component 1 of ict_share",
    "N1173G": "Computer software and databases, gross fixed capital formation, current prices, MEUR (Eurostat nama_10_a64_p5, nace_r2=TOTAL, asset10=N1173G) -- raw source series, numerator component 2 of ict_share. Has a known Eurostat coverage gap in the related nama_10_an6 dataset; missing values are treated as 0 when computing ict_share rather than dropping the row (see fetch_ict_investment_share())",
    "N11G": "Total fixed assets, gross fixed capital formation, current prices, MEUR (Eurostat nama_10_a64_p5, nace_r2=TOTAL, asset10=N11G) -- raw source series, denominator for ict_share",
    "ict_share": "(N1132G + N1173G, with missing N1173G treated as 0) / N11G -- AI/ICT investment share of total GFCF, used as the exposure proxy",
    "ticker": "Yahoo Finance ticker symbol for the raw price series on this row",
    "close": "Quarter-end closing price (Yahoo Finance) -- raw source series",
    "log_ret": "Quarter-on-quarter log return of `close`",
}


def export_results_to_excel(panels: dict, irfs: dict, figs: dict, raw_data: dict, filepath: str):
    """
    Single workbook with everything: one sheet per spec's panel data, one
    sheet per spec's IRF table, one sheet per raw source series, a
    Variable_definitions sheet, and an IRF_plots sheet with the embedded
    charts (embedded directly from memory -- no PNG files are written to
    disk).

    panels:   {spec_name: panel DataFrame from build_panel()}
    irfs:     {spec_name: IRF DataFrame from summarize_irf()}
    figs:     {spec_name: matplotlib Figure from plot_irf()}
    raw_data: {sheet_name: raw source DataFrame, e.g. "Raw_GDP": ...}
    """
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from PIL import Image as PILImage

    def _safe_sheet_name(name):
        for ch in "[]:*?/\\":
            name = name.replace(ch, "_")
        return name[:31]

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for spec_name, panel in panels.items():
            out = panel.copy()
            if "quarter" in out.columns:
                out["quarter"] = out["quarter"].astype(str)
            out.to_excel(writer, sheet_name=_safe_sheet_name(f"{spec_name}_data"), index=False)

        for spec_name, irf in irfs.items():
            irf.to_excel(writer, sheet_name=_safe_sheet_name(f"{spec_name}_IRF"), index=False)

        for sheet_name, df in raw_data.items():
            out = df.copy()
            for col in ("quarter", "year"):
                if col in out.columns:
                    out[col] = out[col].astype(str)
            out.to_excel(writer, sheet_name=_safe_sheet_name(sheet_name), index=False)

        all_cols = []
        for df in list(panels.values()) + list(irfs.values()) + list(raw_data.values()):
            for c in df.columns:
                if c not in all_cols:
                    all_cols.append(c)
        defs = pd.DataFrame({
            "variable": all_cols,
            "description": [VARIABLE_DEFINITIONS.get(c, "(no description on file)") for c in all_cols],
        })
        defs.to_excel(writer, sheet_name="Variable_definitions", index=False)

    # Add the IRF plots as a new sheet with embedded images (openpyxl can't
    # embed images via pandas' ExcelWriter context, so reopen the workbook).
    # Images are rendered to an in-memory buffer and embedded directly --
    # no PNG files are ever written to disk.
    wb = openpyxl.load_workbook(filepath)
    ws = wb.create_sheet("IRF_plots")
    row_cursor = 1
    for spec_name, fig in figs.items():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = XLImage(PILImage.open(buf))
        ws.add_image(img, f"A{row_cursor}")
        row_cursor += int(img.height / 15) + 2  # rough row spacing so plots don't overlap
    wb.save(filepath)

    print(f"Saved all results (data + IRF tables + raw series + plots) to '{filepath}'.")


# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Computing WUI shock standard deviation (for 1-stdev-shock IRF charts)")
    print("=" * 70)
    wui_shock_std = compute_wui_shock_std()

    print("\n" + "=" * 70)
    print("SPEC A: ICT investment share as exposure (primary), global WUI")
    print("=" * 70)
    panel_ict = build_panel(exposure="ict")
    res_ict = run_local_projections(panel_ict)
    irf_ict = summarize_irf(res_ict)
    print(irf_ict.to_string())
    fig_ict = plot_irf(irf_ict, "Spec A: ICT investment share exposure", shock_std=wui_shock_std)

    print("\n" + "=" * 70)
    print("SPEC B: Rolling correlation(national index, semiconductor index)")
    print("        as exposure (robustness), global WUI")
    print("        -- Channel 1 boom/bust split (see build_panel() docstring)")
    print("=" * 70)
    panel_corr = build_panel(exposure="corr")
    res_corr = run_local_projections(panel_corr)
    irf_corr = summarize_irf(res_corr)
    print(irf_corr.to_string())
    fig_corr_boom = plot_regime_irf(
        irf_corr, "mitigating_effect_boom",
        "Spec B: AI-capex-cycle exposure -- BOOM regime mitigating effect (b4+b6)",
        color="#2ca02c", color_nl="#98df8a", shock_std=wui_shock_std,
    )
    fig_corr_bust = plot_regime_irf(
        irf_corr, "mitigating_effect_bust",
        "Spec B: AI-capex-cycle exposure -- BUST regime mitigating effect (b4)",
        color="#d62728", color_nl="#ff9896", shock_std=wui_shock_std,
    )

    print("\n" + "=" * 70)
    print("BASELINE: WUI shock IRF, exposure/mitigation switched off")
    print("          (entity FE only, no time FE -- see docstring for why;")
    print("          wui_used = AR-purified shock, not the raw WUI level)")
    print("=" * 70)
    # wui_used and dgdp are identical across specs (only the exposure proxy
    # differs), so this only needs to be run once, off either panel.
    res_baseline = run_baseline_wui_projections(panel_ict)
    irf_baseline = summarize_baseline_irf(res_baseline)
    print(irf_baseline.to_string())
    fig_baseline = plot_baseline_irf(irf_baseline, title="Baseline WUI IRF (AR-purified shock, no exposure)",
                                      shock_std=wui_shock_std)

    print("\n" + "=" * 70)
    print("Fetching raw source series for export (GDP, GFCF, indices, WUI)")
    print("=" * 70)
    raw_gdp = fetch_gdp_growth_yoy()[["country", "quarter", "gdp_level"]]
    raw_gfcf = fetch_ict_investment_share()  # country, year, N1132G, N1173G, N11G, ict_share
    raw_national_index = fetch_national_indices_raw()
    raw_semiconductor = fetch_semiconductor_raw()
    raw_wui_level = fetch_wui_global().reset_index()
    raw_wui_shock, _ar_model = purify_wui_shock(raw_wui_level)  # quarter, wui_global, wui_shock
    raw_sox_regime = compute_sox_regime()  # quarter, sox_roll_mean_ret, is_boom

    export_results_to_excel(
        panels={"ICT_exposure": panel_ict, "Corr_exposure": panel_corr},
        irfs={
            "ICT_exposure": irf_ict,
            "Corr_exposure": irf_corr,
            "Baseline_WUI": irf_baseline,
        },
        figs={
            "ICT_exposure": fig_ict,
            "Corr_exposure_Boom": fig_corr_boom,
            "Corr_exposure_Bust": fig_corr_bust,
            "Baseline_WUI": fig_baseline,
        },
        raw_data={
            "Raw_GDP": raw_gdp,
            "Raw_GFCF": raw_gfcf,
            "Raw_National_Index": raw_national_index,
            "Raw_Semiconductor": raw_semiconductor,
            "Raw_WUI_Level": raw_wui_level,
            "Raw_WUI_Shock": raw_wui_shock,  # quarter, wui_global, wui_shock -- what the model actually uses
            "Raw_SOX_Regime": raw_sox_regime,  # quarter, sox_roll_mean_ret, is_boom
        },
        filepath="model_results.xlsx",
    )

    print(f"""
    MODEL SPECIFICATION (moderated regression, not a "difference" spec):
    ICT spec:  GDPgrowth = b0 + b1*WUI + b2*F + b3*(WUI*F) + controls
    Corr spec: GDPgrowth = b0 + b1*WUI + b2*F + b3*Dum + b4*(WUI*F)
                           + b5*(WUI*Dum) + b6*(WUI*F*Dum) + controls
    (Dum = is_boom; Dum=0/"bust" is the reference level.) Both specs use
    ENTITY fixed effects only, no time effects -- required because WUI,
    Dum, and their product are entity-invariant (see build_panel() /
    run_local_projections() for the full identification argument). b3/b4
    here are genuine interaction-effect coefficients net of both main
    effects, not a "F=1 minus F=0" difference.

    NL-SPECIFIC RESULT, SPEC A (ICT / Channel 2): in the ICT_exposure_IRF
    sheet, look at beta_b3_wui_x_F_{FOCUS_COUNTRY}_total and
    se_b3_wui_x_F_{FOCUS_COUNTRY}_total -- NL's total b3 coefficient
    (panel-average + NL-specific deviation), with a correctly computed
    combined SE. A POSITIVE value means higher AI/ICT exposure dampens
    the WUI shock's impact on growth (consistent with firm-level
    resilience); NEGATIVE means exposure amplifies it. beta_b1_wui_* and
    beta_b2_F_* report the WUI and F main effects themselves, now
    separately estimated rather than absorbed.

    NL-SPECIFIC RESULT, SPEC B (Corr / Channel 1, boom vs bust): in the
    Corr_exposure_IRF sheet, look at mitigating_effect_bust_panelavg (=
    b4 alone, the Dum=0/bust reference level) and
    mitigating_effect_boom_panelavg (= b4+b6, the Dum=1/boom regime, with
    a correctly derived combined SE) -- or the {FOCUS_COUNTRY}_total
    versions of each for NL's own coefficients. Economic prior:
    mitigating_effect_boom POSITIVE (demand/terms-of-trade tailwind
    dampens the WUI shock's impact during an AI-capex boom) and
    mitigating_effect_bust NEGATIVE (shared exposure to the same risk
    factor amplifies it during an AI downturn) -- see build_panel() for
    the full reasoning. The individual b1..b6 coefficients (main effects
    and interactions) are also reported in full for anyone who wants to
    verify the derived quantities by hand.

    NOTE ON SPECIFICATION: WUI_shock is the AR(2)-purified shock (see
    purify_wui_shock()), not the raw WUI level -- this applies throughout
    the workbook, including the baseline spec below.

    BASELINE WUI IRF (Baseline_WUI sheet): the unconditional GDP response
    to the AR-purified WUI shock, with the exposure/mitigation channel
    removed entirely -- this answers "is the shock itself significant"
    before asking whether AI exposure changes that response. Check
    t_stat_baseline (|t|>~1.96 ~ 5% significance) at each horizon. This
    spec (like both moderated specs above) uses entity FE only, no time
    FE -- so read it as "does the WUI shock move growth on average
    across the panel", not as a fully "clean" two-way-FE IRF. The
    Raw_WUI_Level and Raw_WUI_Shock sheets let you compare the purified
    series against the original index if you want to sanity-check the
    purification, and Raw_SOX_Regime shows the boom/bust classification
    underlying Spec B.

    CAVEATS TO CHECK BEFORE INTERPRETING:
    1. ICT investment share (Spec A) is annual -> quarterly step
       interpolation is crude; consider spline interpolation or a
       genuinely quarterly ICT proxy (Eurostat STS) if available.
    2. Correlation-with-semiconductor-index (Spec B): national-index-vs-
       ^SOX correlation can rise during global risk-off episodes for
       reasons unrelated to AI diffusion (e.g. NL's AEX is trade- and
       supply-chain exposed via ASML regardless of AI-specific
       mechanisms). The CORR_WINDOW_QUARTERS=8 rolling window is used for
       the correlation measure (F); the boom/bust regime split (Dum) uses
       its own, shorter SOX_REGIME_WINDOW_QUARTERS=4 window, chosen to
       react faster to regime changes than the 8-quarter exposure measure
       -- sensitivity-check both windows independently if results seem
       sensitive to this choice.
    3. NL is one of 7 countries; the focus-country interaction terms let
       you read off its coefficient directly, but with only ~26 years of
       quarterly NL-specific variation -- split further into boom/bust
       sub-samples for Spec B -- NL total-effect standard errors will be
       wide relative to the panel average, and boom or bust periods with
       very few NL observations may show as absorbed (is_absorbed_* flag)
       at some horizons -- expect this.
    4. theta=2.0 for the logistic transition is a starting calibration --
       sensitivity-check theta in {{1, 1.5, 2, 3}}.
    5. ALL THREE specs use entity fixed effects only, no time fixed
       effects (required by including WUI/Dum main effects -- see the
       model specification note above). This means common-across-
       countries time-varying confounders (ECB monetary policy stance,
       euro-area-wide demand shocks at a given quarter) are not swept
       out of any spec's residual. Treat every coefficient in this
       workbook accordingly -- as identified off entity-level variation
       plus the panel's time-series variation combined, not off
       within-quarter cross-country comparisons alone.
    6. Standard errors are Driscoll-Kraay throughout (robust to
       heteroskedasticity, within-country serial correlation, and
       cross-sectional dependence across the 7 countries), with the
       Bartlett-kernel bandwidth growing with horizon h (see
       _dk_bandwidth()) rather than a fixed constant -- the cumulative
       dependent variable's overlapping-window construction induces
       serial correlation that itself grows with the horizon, so a
       fixed bandwidth would understate it at longer horizons (roughly
       h>=4), making those horizons' standard errors too narrow.
    7. ALL THREE IRF CHARTS (baseline, ICT, Corr) are rescaled to show
       the cumulative dGDP response per 1-STANDARD-DEVIATION WUI shock
       (wui_shock_std, printed above -- see compute_wui_shock_std()),
       NOT per 1 raw unit of wui_used -- wui_used is an AR(2) residual,
       an arbitrary scale with no natural economic unit, so "per 1
       standard deviation" is the interpretable convention here.
       IMPORTANT: this rescaling applies ONLY to the CHARTS -- the
       underlying Excel tables (ICT_exposure_IRF, Corr_exposure_IRF,
       Baseline_WUI_IRF) still report the raw per-unit coefficients
       exactly as estimated, so any further hand-calculation from the
       tables needs to multiply by wui_shock_std manually to match what
       the charts show. t-statistics are unaffected by this rescaling
       either way (scaling beta and se by the same constant leaves their
       ratio unchanged).

    WHAT "MITIGATING EFFECT" MEANS IN ECONOMIC TERMS:

    The baseline spec asks a simple question: does a WUI (uncertainty)
    shock move GDP growth at all, on average across the panel? beta_wui_
    baseline is that average marginal response -- typically expected to
    be NEGATIVE (higher uncertainty depresses growth), though whether it
    is actually significant here is itself informative.

    The ICT and Corr specs ask a follow-up question: does a country's
    AI/ICT exposure change HOW SENSITIVE its growth is to that same
    shock? This is captured by an INTERACTION term (WUI shock x exposure
    F), so the coefficients on it (b3 for ICT, b4/b6 for Corr) are not
    themselves "the effect of the shock" -- they are the effect of
    exposure ON that sensitivity. Concretely: the marginal GDP response
    to a WUI shock, as a function of exposure F, is

        d(GDP growth) / d(WUI) = b1 + b3*F        (ICT spec)
        d(GDP growth) / d(WUI) = b1 + b4 + b6*Dum  (Corr spec, at F=1)

    A POSITIVE b3 (or b4/b6) means: as exposure rises, the growth
    response to an uncertainty shock becomes LESS negative -- i.e.
    exposure DAMPENS the shock's bite. This is the "mitigation"
    hypothesis: firms/countries with more AI/ICT capacity (Channel 2:
    firm-level resilience -- automation, data-driven decision-making,
    substituting capital for volatile labour/supply-chain inputs) or
    more exposure to the AI-capex investment cycle (Channel 1: demand/
    terms-of-trade tailwind during a boom) absorb uncertainty shocks
    better than less-exposed peers.

    A NEGATIVE b3 (or b4, in the bust regime specifically) means the
    opposite: exposure AMPLIFIES the shock's impact rather than
    cushioning it -- e.g. because heavy reliance on a concentrated set of
    AI-related suppliers/customers (think: the semiconductor supply
    chain) makes a country's growth MORE fragile to broad uncertainty
    shocks, not less, especially when that same AI-capex cycle is itself
    in a downturn (bust).

    In short: this script is not testing "is uncertainty bad for
    growth" (the baseline spec) -- it is testing "does AI/ICT exposure
    make a country's growth more or less fragile to uncertainty," which
    is a claim about a SECOND DERIVATIVE (how the shock's own effect
    changes with exposure), not the shock's effect itself. A positive
    mitigating coefficient supports the "AI/ICT capacity as an economic
    buffer" story; a negative one supports an "AI/ICT concentration as a
    new source of fragility" story instead. Both are economically
    plausible ex ante -- that is precisely why this needs to be
    estimated rather than assumed.
    """)
