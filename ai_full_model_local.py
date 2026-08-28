"""
Global shock propagation and AI-exposure mitigation: NL + euro area panel
========================================================================

Estimates a MODERATED-REGRESSION panel local projection (main effects
AND interaction terms, not a "difference" spec) with ENTITY fixed
effects only (no time effects):

  ICT spec:
    Delta_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
                       + b3_h*(Shock_t * F[i,t]) + Gamma_h*X[i,t]
                       + e[i,t+h]

  Corr spec (Dum = is_boom_t; Dum=0/"bust" is the reference level):
    Delta_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
                       + b3_h*Dum_t + b4_h*(Shock_t*F[i,t])
                       + b5_h*(Shock_t*Dum_t)
                       + b6_h*(Shock_t*F[i,t]*Dum_t)
                       + Gamma_h*X[i,t] + e[i,t+h]

for h = 0..8 quarters, where F[i,t] in (0,1) is a logistic transform of a
country-level AI/ICT-exposure state variable z[i,t], and Shock_t is an
AR(2)-purified version of a GLOBAL shock series -- either the World
Uncertainty Index (WUI, Ahir, Bloom & Furceri) or the Geopolitical Risk
Index (GPR, Caldara & Iacoviello), whichever SHOCK_VARIABLE selects (see
the CONFIG section) -- the residual after regressing that series on its
own lags, an approximate "identified shock" rather than the raw level.
EVERY coefficient, IRF table, and chart in this script is based on
whichever shock SHOCK_VARIABLE currently selects; change that one
setting and every downstream result follows automatically.

IDENTIFICATION: Shock_t, Dum_t, and Shock_t*Dum_t are entity-invariant
(identical across every country at a given quarter -- the shock is a
global series either way, Dum is derived from the global ^SOX index).
Time fixed effects would absorb all three completely, so estimating b1,
b3 (Corr spec), and b5 as genuinely identified coefficients requires
ENTITY effects only -- see build_panel()/run_local_projections() for the
full derivation. Cost: common-across-countries time-varying confounders
(ECB policy stance, euro-area-wide demand shocks) are no longer swept
out of the residual the way two-way FE would have done.

INTERPRETATION: b3 (ICT spec) and b4 (Corr spec) are the mitigating
effect of AI/ICT exposure on the shock's impact, net of the shock and
F main effects (which are now separately estimated, not folded into an
implicit baseline). For the Corr spec specifically, the mitigating
effect of Shock*F is b4 alone in the bust regime (Dum=0) and b4+b6 in
the boom regime (Dum=1) -- see summarize_irf() for where that sum is
computed with a correctly derived combined standard error.

Four versions of z (AI exposure) are estimated as separate models:
  (A) ICT investment share of GFCF
  (B) Annual AI patent applications per country (see fetch_ai_patents());
      charted together with (C) in a single combined, panel-average-only
      comparison chart (see plot_combined_mitigating_irf())
  (C) Annual AI-related incoming investment counts per country (see
      fetch_ai_investment())
  (D) Rolling correlation between each country's national equity index
      and the global semiconductor index (robustness; contaminated by
      the same uncertainty/risk shocks used as the RHS variable -- see
      caveats printed at the end of the script)

All results (IRF tables, all series used -- including both the raw shock
level and the purified shock, for transparency -- variable definitions,
and IRF plots) are saved to a single Excel workbook: model_results.xlsx.

ALL INPUT DATA is read from a single LOCAL workbook, LOCAL_DATA_FILE
("Data_AI.xlsx"), which must sit next to this script -- see
_read_data_ai_sheet() and the fetch_*() functions below. No network
access, no live API/website calls of any kind are made anywhere in this
script; every series (WUI/GPR, GDP, ICT investment shares, national
equity indices, the semiconductor index) is a plain read from one of
that workbook's sheets (gdp, ict_inv, index_nat, index_sox, wui, gpr).
Only the rolling correlation (Spec B's exposure measure) and the
boom/bust regime classification are still COMPUTED in this script --
from the raw index_nat/index_sox sheets -- since those are derived
quantities, not raw source data.

Run:  pip install pandas numpy statsmodels linearmodels openpyxl matplotlib Pillow
      python gpr_ai_mitigation_pipeline.py
"""

SCRIPT_VERSION = "2025-08-11-v39-patent-investment-added"  # bump this whenever the file changes;
                                    # print it at runtime to confirm you're
                                    # not running a stale cached copy

import io
import os
import numpy as np
import pandas as pd

print(f"[gpr_ai_mitigation_pipeline.py version {SCRIPT_VERSION}]")


LOCAL_DATA_FILE = "ai_data.xlsx"
# ^ Single local workbook, sitting next to this script, that replaces
# every network/website data source this script used to fetch from.
# Expected sheets (all confirmed against the actual workbook this script
# was adapted for): "gdp" (country, quarter, gdp_level), "ict_inv"
# (country, year, N1132G, N1173G, N11G, ict_share -- already computed),
# "ai_patent" (country, year, ai_patents), "ai_inv" (country, year,
# ai_investment), "index_nat" (country, ticker, quarter, close, log_ret),
# "index_sox" (ticker, quarter, close, log_ret), "wui" (quarter,
# wui_global), "gpr" (quarter, gpr_global). Which of the last two sheets
# is actually read
# depends on SHOCK_VARIABLE (see the CONFIG section below) -- only one
# of them is used per run, via fetch_shock_global().


def _read_data_ai_sheet(sheet_name):
    """
    Reads one sheet from the local LOCAL_DATA_FILE workbook. Raises a
    clear, actionable error (naming the missing file or the missing
    sheet, and listing the sheets that DO exist) rather than letting a
    bare FileNotFoundError or a pandas-internal KeyError surface several
    calls downstream.
    """
    if not os.path.exists(LOCAL_DATA_FILE):
        raise FileNotFoundError(
            f"_read_data_ai_sheet: local file '{LOCAL_DATA_FILE}' not found "
            "next to this script. This script reads ALL its input data from "
            "this single workbook -- make sure it's saved in the same "
            "directory you're running the script from."
        )
    try:
        return pd.read_excel(LOCAL_DATA_FILE, sheet_name=sheet_name)
    except ValueError as e:
        try:
            available = pd.ExcelFile(LOCAL_DATA_FILE).sheet_names
        except Exception:
            available = "(could not list sheets either)"
        raise ValueError(
            f"_read_data_ai_sheet: sheet '{sheet_name}' not found in "
            f"'{LOCAL_DATA_FILE}' ({e}). Sheets present in the workbook: "
            f"{available}."
        )


# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------

COUNTRIES = ["NL", "DE", "FR", "IT", "ES", "BE", "AT"]   # euro-area panel
SAMPLE_START = "2000-01-01"
SAMPLE_END   = "2026-06-30"
SHOCK_VARIABLE = "WUI"          # "WUI" or "GPR" -- which global shock series
                                 # drives every spec in this script (baseline,
                                 # ICT, and Corr alike). Change this ONE line
                                 # to switch the whole script (data source,
                                 # AR-purification, panel construction, every
                                 # IRF table and chart) from the World
                                 # Uncertainty Index to the Caldara-Iacoviello
                                 # Geopolitical Risk Index, or back -- nothing
                                 # else needs to change. Both series are read
                                 # from LOCAL_DATA_FILE (sheets "wui" and
                                 # "gpr" respectively); see fetch_shock_global().
if SHOCK_VARIABLE not in ("WUI", "GPR"):
    raise ValueError(f"SHOCK_VARIABLE must be 'WUI' or 'GPR', got {SHOCK_VARIABLE!r}")
HORIZONS = range(0, 9)          # h = 0..8 quarters
THETA = 2.0                     # logistic transition steepness (standardized z)
FOCUS_COUNTRY = "NL"            # country singled out for its own interaction term
STANDARDIZE_MODE = "pooled"     # 'pooled' (default, recommended) or 'within_country'
                                 # -- see the long comment in build_panel() for why
                                 # 'within_country' can cause a two-way-FE absorption
                                 # error for the exposure interaction term.

# ----------------------------------------------------------------------
# 1. Global shock series -- WUI (Ahir, Bloom & Furceri) or GPR
#    (Caldara & Iacoviello), selected via SHOCK_VARIABLE above
# ----------------------------------------------------------------------

_SHOCK_SHEET = {"WUI": "wui", "GPR": "gpr"}
_SHOCK_SOURCE_COL = {"WUI": "wui_global", "GPR": "gpr_global"}
_SHOCK_CITATION = {
    "WUI": 'the World Uncertainty Index (Ahir, Bloom & Furceri, NBER WP 29763)',
    "GPR": 'the Geopolitical Risk Index (Caldara & Iacoviello, AER 2022)',
}


def fetch_shock_global():
    """
    Quarterly GLOBAL shock series -- either the World Uncertainty Index
    (WUI) or the Geopolitical Risk Index (GPR), whichever SHOCK_VARIABLE
    currently selects. See _SHOCK_CITATION for the source paper of
    each.

    Reads the "wui" or "gpr" sheet (columns: quarter, wui_global / quarter,
    gpr_global respectively) of LOCAL_DATA_FILE -- both already quarterly,
    no further resampling needed. Regardless of which is selected, the
    sheet's own value column is renamed here to a single GENERIC name,
    "shock_level", so every downstream function (purify_shock(),
    build_panel(), etc.) is written once and works unchanged for either
    shock variable -- they never need to know or care which one was
    actually chosen.

    Returns a DataFrame indexed by quarter, with the single column
    "shock_level".
    """
    sheet = _SHOCK_SHEET[SHOCK_VARIABLE]
    source_col = _SHOCK_SOURCE_COL[SHOCK_VARIABLE]

    df = _read_data_ai_sheet(sheet)
    missing = {"quarter", source_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_shock_global: expected columns {missing} not found in "
            f"the '{sheet}' sheet of '{LOCAL_DATA_FILE}' (SHOCK_VARIABLE="
            f"{SHOCK_VARIABLE!r}). Actual columns: {list(df.columns)}."
        )
    df = df[["quarter", source_col]].rename(columns={source_col: "shock_level"}).copy()
    df["shock_level"] = pd.to_numeric(df["shock_level"], errors="coerce")
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.dropna(subset=["quarter", "shock_level"]).sort_values("quarter")

    if df.empty:
        raise ValueError(
            f"fetch_shock_global: zero valid (quarter, value) rows after "
            f"reading the '{sheet}' sheet of '{LOCAL_DATA_FILE}'."
        )

    return df.set_index("quarter")[["shock_level"]]


AR_PURIFICATION_LAGS = 2  # number of the chosen shock series' own lags used
                          # to purify it (same specification regardless of
                          # SHOCK_VARIABLE)


def purify_shock(shock_level_df, lags=AR_PURIFICATION_LAGS):
    """
    AR(lags) purification of the chosen global shock series (WUI or GPR,
    per SHOCK_VARIABLE): regress its level on its own lags and take the
    residual as an approximate "identified shock" -- the part not
    predictable from its own recent history.

    Both WUI and GPR are already quarterly, relatively persistent
    indices (unlike a daily/monthly news-count series), so their own
    lags typically explain a meaningful share of variance -- the
    residual strips out that predictable component, leaving something
    closer to a "surprise." This is a lightweight identification
    choice, not a structural one: it does NOT control for other macro/
    financial variables the way a full VAR would (WUI's own paper uses
    log average stock return, WUI, and GDP growth with a Cholesky
    ordering -- Ahir, Bloom & Furceri, NBER WP 29763; GPR's own paper
    uses an 8-variable VAR with GPR ordered first -- Caldara &
    Iacoviello, AER 2022, Section III). Treat this as a supplementary
    robustness check on the baseline spec, not a replacement for either
    paper's own fully identified structural shock.

    Returns (df, model): df has columns quarter, shock_level,
    shock_innov; model is the fitted statsmodels OLS result (printed
    diagnostics include R^2 and the AR coefficients, worth checking
    before trusting the residual -- a very low R^2 means the series has
    little own-persistence to purify out in the first place, in which
    case shock_innov will look a lot like shock_level anyway).
    """
    import statsmodels.api as sm

    df = shock_level_df[["quarter", "shock_level"]].copy()
    df = df.sort_values("quarter").reset_index(drop=True)

    lag_cols = []
    for l in range(1, lags + 1):
        col = f"shock_lag{l}"
        df[col] = df["shock_level"].shift(l)
        lag_cols.append(col)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    X = sm.add_constant(df[lag_cols])
    y = df["shock_level"]
    model = sm.OLS(y, X).fit()
    df["shock_innov"] = model.resid.values

    print(f"  [AR({lags}) {SHOCK_VARIABLE} purification] R^2 = {model.rsquared:.3f}")
    print(f"  {model.params.to_string()}")

    return df[["quarter", "shock_level", "shock_innov"]], model


def compute_shock_std():
    """
    Fetches and AR-purifies the chosen global shock series (same steps
    as build_panel() does internally) and returns the standard
    deviation of the resulting shock_innov -- the "1 standard deviation
    shock" size used to rescale the IRF charts from "response per 1 raw
    unit of shock_used" (an arbitrary scale -- shock_used is an AR(2)
    residual, not a naturally interpretable unit) into "response per
    1-stdev {SHOCK_VARIABLE} shock" (the standard, directly
    interpretable convention in the shock-IRF literature).

    Cheap and idempotent to call again here independently of
    build_panel() -- just a local Excel-sheet read and a simple AR(2)
    regression, and calling it twice (once per exposure spec inside
    build_panel(), once here) always returns the same value since the
    underlying shock series and AR specification don't change between
    calls.
    """
    shock_global = fetch_shock_global().reset_index()
    shock_df, _ar_model = purify_shock(shock_global)
    std = shock_df["shock_innov"].std()
    print(f"  [diagnostic] {SHOCK_VARIABLE} shock (shock_used) standard "
          f"deviation = {std:.4f} -- this is the '1-stdev shock' used to "
          f"rescale IRF charts.")
    return std


# Note: build_baseline_shock_panel() (a separate lightweight panel just for
# the shock baseline) has been removed -- now that build_panel() itself
# uses the AR-purified shock as shock_used, the baseline spec can reuse
# panel_ict/panel_corr directly (they already carry shock_used = shock_innov),
# the same way it did before the shock/level split was introduced.


# ----------------------------------------------------------------------
# 2. Local pulls: quarterly GDP growth, ICT investment share
# ----------------------------------------------------------------------

def fetch_gdp_growth_yoy():
    """
    Quarterly real GDP level per country -- returns both the raw level
    (gdp_level, for the Raw_GDP export sheet) and the derived YEAR-ON-
    YEAR log growth (dgdp, used as the lag control throughout this
    script -- see the single dgdp_lag1 control in build_panel(),
    run_local_projections(), and run_baseline_shock_projections()).

    Reads the "gdp" sheet (columns: country, quarter, gdp_level) of
    LOCAL_DATA_FILE; dgdp itself is not stored in the workbook and is
    computed here exactly as the network-fetch version of this function
    did -- 100*ln(gdp_level_t / gdp_level_{t-4}), a genuine 4-quarter
    (one full year) log difference, not a 1-quarter one. Only this
    lagged growth-rate CONTROL is YoY; the regression's actual
    dependent variable (dgdp_cum_lead, built directly from gdp_level in
    run_local_projections()) is a separate, directly-constructed
    cumulative measure, unaffected by this.
    """
    df = _read_data_ai_sheet("gdp")
    missing = {"country", "quarter", "gdp_level"} - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_gdp_growth_yoy: expected columns {missing} not found "
            f"in the 'gdp' sheet of '{LOCAL_DATA_FILE}'. Actual columns: "
            f"{list(df.columns)}."
        )
    df = df.copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["gdp_level"] = pd.to_numeric(df["gdp_level"], errors="coerce")
    df = df[["country", "quarter", "gdp_level"]].dropna().sort_values(["country", "quarter"])
    df["dgdp"] = df.groupby("country")["gdp_level"].transform(lambda s: 100 * np.log(s / s.shift(4)))
    return df[["country", "quarter", "gdp_level", "dgdp"]]


def fetch_ict_investment_share():
    """
    (ICT equipment + computer software/databases) GFCF as % of total
    GFCF, annual, per country -- ESA2010 asset10 vocabulary
    (dd.eionet.europa.eu/vocabulary/eurostat/asset10):
      N1132G  ICT equipment (gross) -- numerator, component 1
      N1173G  Computer software and databases (gross) -- numerator,
              component 2
      N11G    Total fixed assets (gross) -- denominator
      ict_share = (N1132G + N1173G) / N11G

    Reads the "ict_inv" sheet (columns: country, year, N1132G, N1173G,
    N11G, ict_share) of LOCAL_DATA_FILE directly -- ict_share is already
    computed in the workbook, so no recomputation happens here; this
    function only validates and returns it.

    A diagnostic is printed reporting N1173G's actual non-null coverage
    (N1173G has a known real-world data-availability gap for some
    country/period combinations -- worth knowing about even though the
    workbook's own ict_share values already reflect however that gap
    was handled when the workbook was built).
    """
    df = _read_data_ai_sheet("ict_inv")
    expected_cols = {"country", "year", "N1132G", "N1173G", "N11G", "ict_share"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_ict_investment_share: expected columns {missing} not "
            f"found in the 'ict_inv' sheet of '{LOCAL_DATA_FILE}'. Actual "
            f"columns: {list(df.columns)}."
        )
    df = df.copy()
    for col in ("N1132G", "N1173G", "N11G", "ict_share"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["N1132G", "N11G", "ict_share"])

    n1173g_coverage = df["N1173G"].notna().sum()
    n_total = len(df)
    print(f"  [diagnostic] N1173G coverage: {n1173g_coverage}/{n_total} "
          f"country-year rows have a non-null value in the local workbook.")

    if df.empty:
        raise ValueError(
            f"fetch_ict_investment_share: zero valid rows after reading "
            f"the 'ict_inv' sheet of '{LOCAL_DATA_FILE}'."
        )
    return df[["country", "year", "N1132G", "N1173G", "N11G", "ict_share"]]


def fetch_ai_patents():
    """
    Annual AI-related patent applications per country. Source: Emerging
    Technology Observatory (ETO) / CSET Country Activity Tracker:
    Artificial Intelligence (cat.eto.tech, Patent dataset), pre-extracted
    into this workbook.

    Reads the "ai_patent" sheet (columns: country, year, ai_patents) of
    LOCAL_DATA_FILE directly -- already a clean per-country annual count,
    no filtering/melting needed (unlike the raw multi-metric CSV export
    an earlier version of this script parsed directly from cat.eto.tech).
    """
    df = _read_data_ai_sheet("ai_patent")
    expected_cols = {"country", "year", "ai_patents"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_ai_patents: expected columns {missing} not found in "
            f"the 'ai_patent' sheet of '{LOCAL_DATA_FILE}'. Actual "
            f"columns: {list(df.columns)}."
        )
    df = df.copy()
    df["ai_patents"] = pd.to_numeric(df["ai_patents"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["ai_patents"]).sort_values(["country", "year"])

    if df.empty:
        raise ValueError(
            f"fetch_ai_patents: zero valid rows after reading the "
            f"'ai_patent' sheet of '{LOCAL_DATA_FILE}'."
        )

    print(f"  [diagnostic] AI patent applications: {len(df)} country-year "
          f"rows across {df['country'].nunique()} countries, years "
          f"{int(df['year'].min())}-{int(df['year'].max())}.")

    return df[["country", "year", "ai_patents"]]


def fetch_ai_investment():
    """
    Annual AI-related incoming investment counts per country (the number
    of inbound investment deals into AI-related companies -- NOT their
    dollar value). Source: ETO/CSET Country Activity Tracker: Artificial
    Intelligence (cat.eto.tech, Investment dataset), pre-extracted into
    this workbook.

    Reads the "ai_inv" sheet (columns: country, year, ai_investment) of
    LOCAL_DATA_FILE directly -- already a clean per-country annual
    count, no filtering/melting needed.
    """
    df = _read_data_ai_sheet("ai_inv")
    expected_cols = {"country", "year", "ai_investment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_ai_investment: expected columns {missing} not found in "
            f"the 'ai_inv' sheet of '{LOCAL_DATA_FILE}'. Actual columns: "
            f"{list(df.columns)}."
        )
    df = df.copy()
    df["ai_investment"] = pd.to_numeric(df["ai_investment"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["ai_investment"]).sort_values(["country", "year"])

    if df.empty:
        raise ValueError(
            f"fetch_ai_investment: zero valid rows after reading the "
            f"'ai_inv' sheet of '{LOCAL_DATA_FILE}'."
        )

    print(f"  [diagnostic] AI incoming investment counts: {len(df)} "
          f"country-year rows across {df['country'].nunique()} countries, "
          f"years {int(df['year'].min())}-{int(df['year'].max())}.")

    return df[["country", "year", "ai_investment"]]


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

# ----------------------------------------------------------------------
# 3. Semiconductor-correlation exposure proxy
# ----------------------------------------------------------------------

CORR_WINDOW_QUARTERS = 8   # rolling window for national-vs-semiconductor
                           # correlation (the AI-exposure state variable, F)
SOX_REGIME_WINDOW_QUARTERS = 4   # rolling window for the SOX boom/bust regime
                                 # classification (Dum). Kept as a SEPARATE
                                 # named constant from CORR_WINDOW_QUARTERS
                                 # above even though both are currently 4 --
                                 # they measure conceptually different things
                                 # (exposure-correlation vs. regime direction)
                                 # and may be tuned independently again later.


def fetch_national_indices_raw(countries=COUNTRIES):
    """
    Raw quarterly close price + log return for each country's national
    headline equity index, for the Raw_National_Index export sheet.

    Reads the "index_nat" sheet (columns: country, ticker, quarter,
    close, log_ret) of LOCAL_DATA_FILE, filtered to `countries`.
    """
    df = _read_data_ai_sheet("index_nat")
    expected_cols = {"country", "ticker", "quarter", "close", "log_ret"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_national_indices_raw: expected columns {missing} not "
            f"found in the 'index_nat' sheet of '{LOCAL_DATA_FILE}'. "
            f"Actual columns: {list(df.columns)}."
        )
    df = df.copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["log_ret"] = pd.to_numeric(df["log_ret"], errors="coerce")
    df = df[df["country"].isin(countries)]
    return df[["country", "ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])


def fetch_semiconductor_raw():
    """
    Raw quarterly close price + log return for the global semiconductor
    index (^SOX), for the Raw_Semiconductor export sheet. No country
    dimension -- this is a single global series.

    Reads the "index_sox" sheet (columns: ticker, quarter, close,
    log_ret) of LOCAL_DATA_FILE.
    """
    df = _read_data_ai_sheet("index_sox")
    expected_cols = {"ticker", "quarter", "close", "log_ret"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_semiconductor_raw: expected columns {missing} not "
            f"found in the 'index_sox' sheet of '{LOCAL_DATA_FILE}'. "
            f"Actual columns: {list(df.columns)}."
        )
    df = df.copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["log_ret"] = pd.to_numeric(df["log_ret"], errors="coerce")
    return df[["ticker", "quarter", "close", "log_ret"]].dropna(subset=["close"])


def fetch_corr_with_semiconductor_index(countries=COUNTRIES, window=CORR_WINDOW_QUARTERS):
    """
    AI-exposure state variable: rolling correlation between each
    country's national equity index returns and the global semiconductor
    index (^SOX) returns. A country whose stock market co-moves more
    tightly with the semiconductor cycle is coded as more "AI/chip
    exposed" through its listed corporates.

    This is a DERIVED computation (a rolling correlation), not raw
    source data, so it is computed here from the log_ret columns of
    fetch_national_indices_raw() and fetch_semiconductor_raw() -- both
    of which now read from the local workbook rather than the network --
    exactly as this function did when those two inputs came from a live
    Yahoo Finance download.

    Caveat: national-index-vs-^SOX correlation can rise during global
    risk-off episodes for reasons unrelated to AI diffusion (e.g. NL's
    AEX is trade- and semiconductor-supply-chain exposed via ASML
    regardless of AI-specific mechanisms). Treat this as a supplementary
    / robustness specification, not a substitute for the ICT-investment-
    share version.
    """
    semi = fetch_semiconductor_raw()
    semi_ret = semi.set_index("quarter")["log_ret"]

    nat = fetch_national_indices_raw(countries=countries)

    out = []
    for c in countries:
        g = nat[nat["country"] == c]
        if g.empty:
            print(f"  [!] No national index data for {c} in the local workbook, skipping.")
            continue
        nat_ret = g.set_index("quarter")["log_ret"]

        both = pd.concat([nat_ret.rename("nat_ret"), semi_ret.rename("semi_ret")], axis=1).dropna()
        roll_corr = both["nat_ret"].rolling(window).corr(both["semi_ret"])
        roll_corr = roll_corr.rename("z_raw").reset_index()
        roll_corr["country"] = c
        out.append(roll_corr)

    df = pd.concat(out, ignore_index=True)
    return df[["country", "quarter", "z_raw"]].dropna()


def compute_sox_regime(window=SOX_REGIME_WINDOW_QUARTERS):
    """
    Boom/bust regime indicator for the AI/chip cycle, based on the global
    semiconductor index (^SOX) alone -- entity-invariant (depends only on
    quarter, not on country). Uses its OWN rolling window
    (SOX_REGIME_WINDOW_QUARTERS -- currently the same value as
    CORR_WINDOW_QUARTERS used for the exposure-correlation measure, but
    kept as a separate named constant so the two can be tuned
    independently again if needed): is_boom=1 (Dum=1 in the moderated
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
    cycle should DAMPEN the shock's impact (Channel 1, tailwind
    dominates); during a bust, shared/concentrated exposure to the same
    risk factor should AMPLIFY it (Channel 1, concentration-risk
    dominates). Splitting F*shock by is_boom/is_bust (see build_panel())
    lets these two sub-regimes be estimated separately instead of
    averaging over both.

    This is a DERIVED computation (a rolling mean + median threshold),
    not raw source data, so it is computed here from
    fetch_semiconductor_raw()'s log_ret column, exactly as this function
    did when that input came from a live Yahoo Finance download.
    """
    semi = fetch_semiconductor_raw()
    ret = semi.set_index("quarter")["log_ret"]
    roll_mean_ret = ret.rolling(window).mean()

    df = roll_mean_ret.rename("sox_roll_mean_ret").reset_index()

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
    exposure: 'ict'        -> ICT investment share of GFCF
              'patent'     -> annual AI patent applications per country
                              (see fetch_ai_patents())
              'investment' -> annual AI-related incoming investment
                              counts per country (see fetch_ai_investment())
              'corr'       -> rolling correlation of national index with
                              the semiconductor index (robustness)

    The global shock (WUI or GPR, per SHOCK_VARIABLE -- see the CONFIG
    section) is always AR-purified (see purify_shock()): shock_used is
    the AR(2) residual of the chosen series' level, not the raw level
    itself. The exposure interaction (shock_x_exposure = F * shock_used)
    and every downstream coefficient/IRF/chart are based on this
    purified shock, whichever series it actually is.
    """
    shock_level = fetch_shock_global().reset_index()
    shock_innov_df, _ar_model = purify_shock(shock_level)
    gdp = fetch_gdp_growth_yoy()

    quarters = pd.PeriodIndex(pd.period_range(SAMPLE_START, SAMPLE_END, freq="Q"))

    if exposure == "ict":
        ict_annual = fetch_ict_investment_share()
        exp_q = annual_to_quarterly(ict_annual, "ict_share", quarters)
        exp_q = exp_q.rename(columns={"ict_share": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
    elif exposure == "patent":
        patent_annual = fetch_ai_patents()
        exp_q = annual_to_quarterly(patent_annual, "ai_patents", quarters)
        exp_q = exp_q.rename(columns={"ai_patents": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
    elif exposure == "investment":
        investment_annual = fetch_ai_investment()
        exp_q = annual_to_quarterly(investment_annual, "ai_investment", quarters)
        exp_q = exp_q.rename(columns={"ai_investment": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
    elif exposure == "corr":
        corr = fetch_corr_with_semiconductor_index(countries=COUNTRIES)
        panel = gdp.merge(corr, on=["country", "quarter"], how="left")
    else:
        raise ValueError("exposure must be 'ict', 'patent', 'investment', or 'corr'")

    panel = panel.merge(shock_innov_df, on="quarter", how="left")
    panel["shock_used"] = panel["shock_innov"]  # AR-purified shock, not the raw level

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
    # MODERATED-REGRESSION SPECIFICATION (main effects AND interaction
    # terms, not a "difference" spec).
    #
    # ICT spec:  GDPgrowth = b0 + b1*Shock + b2*F + b3*(Shock*F) + controls
    # Corr spec: GDPgrowth = b0 + b1*Shock + b2*F + b3*Dum + b4*(Shock*F)
    #                        + b5*(Shock*Dum) + b6*(Shock*F*Dum) + controls
    #            (Dum = is_boom; the reference level Dum=0 is "bust", so
    #            the mitigating effect of Shock*F is b4 in bust and b4+b6
    #            in boom -- see run_local_projections()/summarize_irf()
    #            for where that sum is computed, with a correctly derived
    #            standard error, not just b4 read off on its own.)
    #
    # Shock (shock_used, whichever series SHOCK_VARIABLE selects) and F
    # are included as separate regressors (b1, b2), not folded into an
    # implicit baseline/intercept -- so b3/b4 are genuine interaction-
    # effect coefficients net of both main effects, not a "F=1 minus
    # F=0" difference.
    #
    # IDENTIFICATION: shock_used, is_boom, and their product (shock_x_dum)
    # are entity-invariant (identical across every country at a given
    # quarter -- the shock is a global series either way, is_boom is
    # derived from the global ^SOX index). Time fixed effects would
    # absorb all three completely. Estimating b1, b3 (Corr spec's Dum
    # term), and b5 as genuinely identified coefficients therefore
    # requires ENTITY fixed effects only, no time effects -- see
    # run_local_projections(), which uses entity-only FE for both specs.
    # Cost: common-across-countries time-varying confounders (ECB
    # policy, euro-area-wide demand shocks) are not swept out of either
    # spec's residual the way two-way FE would.
    # ------------------------------------------------------------------
    panel["shock_x_exposure"] = panel["F"] * panel["shock_used"]

    # focus-country (NL) interaction terms: let the LP recover NL-specific
    # deviations from the panel-average coefficients, rather than only
    # reading off the 7-country average. Applied to every regressor in
    # the moderated spec (main effects and interactions alike).
    panel["is_focus"] = (panel["country"] == FOCUS_COUNTRY).astype(int)
    panel["shock_used_focus"] = panel["is_focus"] * panel["shock_used"]
    panel["F_focus"] = panel["is_focus"] * panel["F"]
    panel["shock_x_exposure_focus"] = panel["is_focus"] * panel["shock_x_exposure"]

    if exposure == "corr":
        regime = compute_sox_regime()
        panel = panel.merge(regime[["quarter", "is_boom"]], on="quarter", how="left")
        # is_bust is kept for reference/export only -- NOT used as a
        # regressor (a second, complementary dummy would be perfectly
        # collinear with is_boom + the intercept; standard k-1-dummies
        # practice, same principle discussed earlier for a 3-regime
        # low/normal/high design).
        panel["is_bust"] = 1 - panel["is_boom"]

        panel["shock_x_dum"] = panel["shock_used"] * panel["is_boom"]                    # b5: Shock x Dum
        panel["shock_x_exposure_x_dum"] = panel["shock_x_exposure"] * panel["is_boom"]   # b6: Shock x F x Dum

        panel["is_boom_focus"] = panel["is_focus"] * panel["is_boom"]
        panel["shock_x_dum_focus"] = panel["is_focus"] * panel["shock_x_dum"]
        panel["shock_x_exposure_x_dum_focus"] = panel["is_focus"] * panel["shock_x_exposure_x_dum"]

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

    - ICT spec:  shock_used (b1), F (b2), shock_x_exposure (b3, = Shock*F).
    - Corr spec: shock_used (b1), F (b2), is_boom (b3, Dum), shock_x_exposure
      (b4, Shock*F), shock_x_dum (b5, Shock*Dum), shock_x_exposure_x_dum (b6,
      Shock*F*Dum). The mitigating effect of Shock*F is b4 in the reference
      regime (Dum=0, "bust") and b4+b6 in the other regime (Dum=1,
      "boom") -- summarize_irf() computes both, the latter with a
      correctly derived combined standard error (not just b4 and b6's
      individual SEs combined naively).

    ENTITY fixed effects only, NO time effects -- required because
    shock_used, is_boom, and shock_x_dum are entity-invariant (identical
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
    matters specifically because shock_used/is_boom/shock_x_dum are
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
        base_terms = ["shock_used", "F", "is_boom", "shock_x_exposure", "shock_x_dum",
                      "shock_x_exposure_x_dum"]
        focus_terms = ["shock_used_focus", "F_focus", "is_boom_focus", "shock_x_exposure_focus",
                       "shock_x_dum_focus", "shock_x_exposure_x_dum_focus"]
    else:
        base_terms = ["shock_used", "F", "shock_x_exposure"]
        focus_terms = ["shock_used_focus", "F_focus", "shock_x_exposure_focus"]

    for col in base_terms + (focus_terms if include_focus_interaction else []):
        check_time_variation(panel, col)

    regressors = base_terms + (focus_terms if include_focus_interaction else []) + \
        ["dgdp_lag1"]

    results = {}
    for h in HORIZONS:
        p = panel.copy()
        # CUMULATIVE IRF -- see the identical construction/rationale in
        # run_baseline_shock_projections() below.
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
    sum of coefficients, e.g. {"shock_x_exposure": 1, "shock_x_exposure_x_dum": 1}
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

    ICT spec:  b1_shock, b2_F, b3_shock_x_F
    Corr spec: b1_shock, b2_F, b3_dum, b4_shock_x_F, b5_shock_x_dum,
               b6_shock_x_F_x_dum

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
            ("shock_used", "shock_used_focus", "b1_shock"),
            ("F", "F_focus", "b2_F"),
            ("is_boom", "is_boom_focus", "b3_dum"),
            ("shock_x_exposure", "shock_x_exposure_focus", "b4_shock_x_F"),
            ("shock_x_dum", "shock_x_dum_focus", "b5_shock_x_dum"),
            ("shock_x_exposure_x_dum", "shock_x_exposure_x_dum_focus", "b6_shock_x_F_x_dum"),
        ]
    else:
        term_label_triples = [
            ("shock_used", "shock_used_focus", "b1_shock"),
            ("F", "F_focus", "b2_F"),
            ("shock_x_exposure", "shock_x_exposure_focus", "b3_shock_x_F"),
        ]

    rows_by_h = {h: {"h": h} for h in results}
    for term, focus_term, label in term_label_triples:
        _summarize_single_term(rows_by_h, results, term, focus_term, label, focus_country)

    if has_boom_bust:
        for h, res in results.items():
            row = rows_by_h[h]

            # Mitigating effect in BUST (Dum=0, the reference level) = b4
            # alone -- already computed above under the b4_shock_x_F label.
            row["mitigating_effect_bust_panelavg"] = row.get("beta_b4_shock_x_F_panelavg", np.nan)
            row["se_mitigating_effect_bust_panelavg"] = row.get("se_b4_shock_x_F_panelavg", np.nan)
            row[f"mitigating_effect_bust_{focus_country}_total"] = \
                row.get(f"beta_b4_shock_x_F_{focus_country}_total", np.nan)
            row[f"se_mitigating_effect_bust_{focus_country}_total"] = \
                row.get(f"se_b4_shock_x_F_{focus_country}_total", np.nan)

            # Mitigating effect in BOOM (Dum=1) = b4 + b6.
            b_boom, se_boom = _linear_combination(
                res, {"shock_x_exposure": 1, "shock_x_exposure_x_dum": 1}
            )
            row["mitigating_effect_boom_panelavg"] = b_boom
            row["se_mitigating_effect_boom_panelavg"] = se_boom

            b_boom_nl, se_boom_nl = _linear_combination(res, {
                "shock_x_exposure": 1, "shock_x_exposure_x_dum": 1,
                "shock_x_exposure_focus": 1, "shock_x_exposure_x_dum_focus": 1,
            })
            row[f"mitigating_effect_boom_{focus_country}_total"] = b_boom_nl
            row[f"se_mitigating_effect_boom_{focus_country}_total"] = se_boom_nl

    return pd.DataFrame(list(rows_by_h.values()))


# ----------------------------------------------------------------------
# 5b. Baseline shock IRF -- NO exposure interaction, NO time effects
# ----------------------------------------------------------------------

def run_baseline_shock_projections(panel):
    """
    The "raw" GDP response to the chosen global shock (WUI or GPR, per
    SHOCK_VARIABLE), with the exposure/mitigation channel switched off
    entirely (no F, no interaction term) -- this is what the mitigation
    coefficient in run_local_projections() is a DEVIATION from, so
    plotting the two together shows whether AI exposure is dampening a
    response that is itself significant.

    IMPORTANT: this regression can NOT include time fixed effects.
    shock_used is identical across every country at each t (a
    common/global shock, whichever series it is), so time effects would
    absorb it completely -- exactly the collinearity problem documented
    in build_panel() for the interaction term, but here it hits the
    shock variable itself, not just a split of it. So this spec uses
    entity (country) fixed effects only, which control for average
    cross-country growth-level differences but NOT for other common-
    across-countries confounders at a given quarter (ECB policy, global
    demand shocks, etc.) -- those are absent here by construction, since
    removing them would remove the shock's effect too. Treat this as a
    plain average effect of the shock across the panel, not a fully
    "clean" IRF in the two-way-FE sense used elsewhere in this script.

    STANDARD ERRORS: Driscoll-Kraay, same rationale and growing
    bandwidth (_dk_bandwidth()) as run_local_projections() -- see that
    function's docstring.
    """
    from linearmodels.panel import PanelOLS

    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()

    regressors = ["shock_used", "dgdp_lag1"]

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
        if "shock_used" not in res.params.index:
            print(f"  [!] h={h}: shock_used was ABSORBED and dropped -- "
                  f"not estimated at this horizon.")
        print(f"--- baseline h={h} (Driscoll-Kraay bandwidth={bw}) ---")
        print(res.params)
        print(res.std_errors)
        print()
    return results


def summarize_baseline_irf(results):
    """
    beta_shock_baseline / se_shock_baseline: the average GDP response to
    a one-unit increase in the chosen shock (WUI or GPR, per
    SHOCK_VARIABLE) at horizon h, with exposure/mitigation switched off
    (see run_baseline_shock_projections). Compare its t-stat (beta/se)
    and significance against the mitigation coefficients in
    summarize_irf() -- a significant baseline response with a
    significant, offsetting mitigation coefficient is the "AI buffers
    the shock" story; a baseline that's already insignificant means
    there's little for exposure to mitigate in the first place.
    """
    rows = []
    for h, res in results.items():
        row = {"h": h}
        if "shock_used" in res.params.index:
            b = res.params["shock_used"]
            se = res.std_errors["shock_used"]
            row["beta_shock_baseline"] = b
            row["se_shock_baseline"] = se
            row["t_stat_baseline"] = b / se if se > 0 else np.nan
            row["is_absorbed_baseline"] = False
        else:
            row["beta_shock_baseline"] = np.nan
            row["se_shock_baseline"] = np.nan
            row["t_stat_baseline"] = np.nan
            row["is_absorbed_baseline"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def plot_baseline_irf(irf_df, title=None,
                       shock_std=1.0):
    """
    Plots the unconditional, CUMULATIVE shock IRF (WUI or GPR, per
    SHOCK_VARIABLE) with a 90% CI band, so you can see directly whether
    the cumulative response is statistically significant at each
    horizon (the band crossing zero means "not significant at that h").
    beta_shock_baseline is already a cumulative coefficient -- see
    run_baseline_shock_projections().

    shock_std: rescales the plotted beta/se from "response per 1 RAW
    unit of shock_used" (an arbitrary scale -- shock_used is an AR(2)
    residual, not a naturally interpretable unit) into "response per
    1-STANDARD-DEVIATION shock" -- pass compute_shock_std()'s return
    value here. Defaults to 1.0 (no rescaling, i.e. raw units) if not
    provided, so this function still works standalone. Only the CHART is
    rescaled -- the underlying irf_df/Excel table keeps the original
    per-raw-unit coefficients (t-stat is unaffected either way, since
    scaling beta and se by the same constant leaves their ratio
    unchanged).

    title: defaults to a title naming the currently selected
    SHOCK_VARIABLE if not given explicitly.
    """
    import matplotlib.pyplot as plt

    if title is None:
        title = f"Baseline {SHOCK_VARIABLE} shock IRF (no exposure interaction)"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = irf_df["h"]
    b = irf_df["beta_shock_baseline"] * shock_std
    se = irf_df["se_shock_baseline"] * shock_std
    ax.plot(h, b, marker="o", color="#2ca02c", label=f"Baseline {SHOCK_VARIABLE} effect")
    ax.fill_between(h, b - 1.645 * se, b + 1.645 * se, alpha=0.2, color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel(f"Cumulative dGDP response per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 6. IRF plots
# ----------------------------------------------------------------------

def plot_irf(irf_df, title, focus_country=FOCUS_COUNTRY, shock_std=1.0):
    """
    Plots the panel-average and NL-specific CUMULATIVE b3 (Shock*F
    interaction) coefficient across horizons, each with a 90% CI band
    (+/- 1.645 SE). This is the ICT spec's mitigating-effect coefficient,
    net of the shock and F main effects (see run_local_projections()).
    Returns the matplotlib Figure (caller decides whether to
    save/embed/show it).

    shock_std: rescales the plotted beta/se from "per 1 raw unit of
    shock_used" into "per 1-stdev {SHOCK_VARIABLE} shock" -- see
    plot_baseline_irf()'s docstring for the full rationale. Defaults to
    1.0 (no rescaling).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    h = irf_df["h"]
    b_panel = irf_df["beta_b3_shock_x_F_panelavg"] * shock_std
    se_panel = irf_df["se_b3_shock_x_F_panelavg"] * shock_std
    ax.plot(h, b_panel, marker="o", label="Panel average", color="#1f77b4")
    ax.fill_between(h, b_panel - 1.645 * se_panel, b_panel + 1.645 * se_panel,
                     alpha=0.2, color="#1f77b4")

    col_nl = f"beta_b3_shock_x_F_{focus_country}_total"
    se_nl_col = f"se_b3_shock_x_F_{focus_country}_total"
    if col_nl in irf_df.columns:
        b_nl = irf_df[col_nl] * shock_std
        se_nl = irf_df[se_nl_col] * shock_std
        ax.plot(h, b_nl, marker="s", label=f"{focus_country} total", color="#d62728")
        ax.fill_between(h, b_nl - 1.645 * se_nl, b_nl + 1.645 * se_nl,
                         alpha=0.15, color="#d62728")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel(f"Cumulative Shock x F interaction effect per 1-stdev {SHOCK_VARIABLE} shock (b3, 0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_combined_mitigating_irf(irf_patent, irf_investment, shock_std=1.0, title=None):
    """
    Combines the PANEL-AVERAGE-ONLY mitigating-effect coefficient (b3,
    the Shock*F interaction) from TWO separately-estimated Spec A models
    -- one using AI patent applications as the exposure variable
    (irf_patent, from build_panel(exposure="patent")), one using AI
    incoming investment counts (irf_investment, from
    build_panel(exposure="investment")) -- onto a SINGLE chart, so the
    two exposure channels' mitigating effects can be compared directly
    on the same axes, for whichever global shock SHOCK_VARIABLE
    currently selects.

    Deliberately panel-average ONLY (no NL-specific "_total" lines) --
    this chart is meant to give a clean, direct two-line comparison of
    "does the shock's growth impact respond to AI-patenting exposure"
    vs "...to AI-investment exposure", not a per-country breakdown.

    Both irf_patent and irf_investment come from the SAME underlying
    shock construction and the SAME HORIZONS grid, so their "h" columns
    align directly -- no need to merge/reindex before plotting.
    """
    import matplotlib.pyplot as plt

    if title is None:
        title = f"Spec A mitigating effect ({SHOCK_VARIABLE}): AI patents vs AI investment (panel average)"

    fig, ax = plt.subplots(figsize=(7.5, 5))
    h = irf_patent["h"]

    b_pat = irf_patent["beta_b3_shock_x_F_panelavg"] * shock_std
    se_pat = irf_patent["se_b3_shock_x_F_panelavg"] * shock_std
    ax.plot(h, b_pat, marker="o", label="AI patent applications (panel avg)", color="#1f77b4")
    ax.fill_between(h, b_pat - 1.645 * se_pat, b_pat + 1.645 * se_pat, alpha=0.15, color="#1f77b4")

    b_inv = irf_investment["beta_b3_shock_x_F_panelavg"] * shock_std
    se_inv = irf_investment["se_b3_shock_x_F_panelavg"] * shock_std
    ax.plot(h, b_inv, marker="s", label="AI incoming investment counts (panel avg)", color="#ff7f0e")
    ax.fill_between(h, b_inv - 1.645 * se_inv, b_inv + 1.645 * se_inv, alpha=0.15, color="#ff7f0e")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel(f"Cumulative Shock x F interaction effect per 1-stdev {SHOCK_VARIABLE} shock (b3, 0..h)")
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
    shock_used" into "per 1-stdev {SHOCK_VARIABLE} shock" -- see
    plot_baseline_irf()'s docstring for the full rationale. Defaults to
    1.0 (no rescaling).
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
    ax.set_ylabel(f"Cumulative mitigating effect per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
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
    "dgdp_lag1": "dgdp (year-on-year GDP growth), lagged 1 quarter -- the ONLY lag control used throughout this script (ICT spec, Corr spec, and the baseline spec alike); a second lag (dgdp_lag2) is not used.",
    "shock_level": f"Global {SHOCK_VARIABLE} (RAW LEVEL), quarterly -- kept in the panel for reference, but NOT what the model estimates on (see shock_used). See _SHOCK_CITATION for the source paper.",
    "shock_used": f"The {SHOCK_VARIABLE} series actually used in the model: shock_innov, the AR(2)-purified {SHOCK_VARIABLE} shock (see purify_shock()) -- NOT the raw level. Every coefficient, IRF, and chart in this workbook is based on this purified shock.",
    "z_raw":    "Raw AI-exposure state variable before standardization -- ICT investment share of GFCF (Spec A/ict), AI patent applications (Spec A/patent), AI incoming investment counts (Spec A/investment), or rolling correlation of the national equity index with the semiconductor index (Spec B/corr), depending on which build_panel(exposure=...) produced this panel",
    "ai_patents": "Annual AI-related patent applications per country (ETO/CSET Country Activity Tracker, Patent dataset -- see fetch_ai_patents()). Used as the exposure proxy (z_raw) for the 'patent' spec.",
    "ai_investment": "Annual AI-related INCOMING investment counts per country -- the number of inbound investment deals into AI-related companies, NOT their dollar value (ETO/CSET Country Activity Tracker, Investment dataset -- see fetch_ai_investment()). Used as the exposure proxy (z_raw) for the 'investment' spec.",
    "z":        "Standardized z_raw (STANDARDIZE_MODE='pooled' across the whole panel by default)",
    "F":        "Logistic transform of z: F = 1 / (1 + exp(-THETA * z)), the smooth 0-1 AI-exposure state weight",
    "shock_x_exposure": f"b3 (ICT spec) / b4 (Corr spec) interaction regressor: F * shock_used. Its coefficient is the mitigating effect NET OF the {SHOCK_VARIABLE} and F main effects (shock_used, F), which are now separately included in the regression -- not a 'difference' spec.",
    "is_focus": f"1 if country == FOCUS_COUNTRY ({FOCUS_COUNTRY}), 0 otherwise",
    "shock_used_focus": f"is_focus * shock_used -- NL's own b1 ({SHOCK_VARIABLE} main effect) interaction term",
    "F_focus": "is_focus * F -- NL's own b2 (F main effect) interaction term",
    "shock_x_exposure_focus": "is_focus * shock_x_exposure -- NL's own interaction-term deviation (b3 for ICT spec, b4 for Corr spec)",
    "is_boom": "1 if the trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) is ABOVE THE MEDIAN of that series over the sample (AI-capex boom, roughly the top half of quarters by this measure), 0 otherwise (bust, roughly the bottom half, the reference level Dum=0) -- median-thresholded so boom and bust each cover close to half the sample by construction, rather than a literal zero threshold (which would skew toward 'boom' given equity indices trend upward over most multi-year windows). Uses its own (shorter) rolling window than the exposure-correlation measure -- see compute_sox_regime() and SOX_REGIME_WINDOW_QUARTERS. Entity-invariant (same for all countries in a given quarter). Also used directly as regressor b3 (Dum) in the Corr spec's moderated regression. Corr spec only.",
    "is_bust": "1 - is_boom. Kept for reference/export only -- NOT used as a regressor (would be perfectly collinear with is_boom + the intercept). Corr spec only.",
    "is_boom_focus": "is_focus * is_boom -- NL's own b3 (Dum main effect) interaction term. Corr spec only.",
    "sox_roll_mean_ret": "Trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) -- the continuous series is_boom is thresholded from",
    "shock_x_dum": f"b5 (Corr spec): shock_used * is_boom ({SHOCK_VARIABLE} x Dum interaction). Entity-invariant like shock_used and is_boom themselves -- estimable only because this spec uses entity-only fixed effects (see run_local_projections()).",
    "shock_x_dum_focus": "is_focus * shock_x_dum -- NL's own b5 interaction term. Corr spec only.",
    "shock_x_exposure_x_dum": f"b6 (Corr spec): shock_x_exposure * is_boom ({SHOCK_VARIABLE} x F x Dum, the full three-way interaction). Corr spec only.",
    "shock_x_exposure_x_dum_focus": "is_focus * shock_x_exposure_x_dum -- NL's own b6 interaction term. Corr spec only.",
    "h": "Local-projection horizon, in quarters",
    "beta_b1_shock_panelavg": f"Panel-average CUMULATIVE coefficient on shock_used (b1, the {SHOCK_VARIABLE} main effect) over horizons 0..h. Both ICT and Corr specs.",
    "se_b1_shock_panelavg": "Standard error of beta_b1_shock_panelavg (Driscoll-Kraay-style)",
    "beta_b2_F_panelavg": "Panel-average CUMULATIVE coefficient on F (b2, the exposure-weight main effect) over horizons 0..h. Both ICT and Corr specs.",
    "se_b2_F_panelavg": "Standard error of beta_b2_F_panelavg",
    "beta_b3_shock_x_F_panelavg": f"Panel-average CUMULATIVE coefficient on shock_x_exposure (b3, ICT spec's {SHOCK_VARIABLE}*F interaction) over horizons 0..h -- the ICT spec's mitigating-effect coefficient, net of the {SHOCK_VARIABLE} and F main effects. A POSITIVE value means exposure dampens/offsets the (typically negative) {SHOCK_VARIABLE} shock response; negative means amplification.",
    "se_b3_shock_x_F_panelavg": "Standard error of beta_b3_shock_x_F_panelavg",
    f"beta_b1_shock_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b1 ({SHOCK_VARIABLE} main effect) coefficient (panel-average + deviation)",
    f"se_b1_shock_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b1 coefficient",
    f"beta_b2_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b2 (F main effect) coefficient (panel-average + deviation)",
    f"se_b2_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b2 coefficient",
    f"beta_b3_shock_x_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b3 (ICT spec's {SHOCK_VARIABLE}*F interaction) coefficient (panel-average + deviation) -- {FOCUS_COUNTRY}'s own mitigating-effect coefficient",
    f"se_b3_shock_x_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b3 coefficient",
    "beta_b3_dum_panelavg": "Panel-average CUMULATIVE coefficient on is_boom (b3, the Dum/regime main effect) over horizons 0..h. Corr spec only.",
    "se_b3_dum_panelavg": "Standard error of beta_b3_dum_panelavg",
    "beta_b4_shock_x_F_panelavg": f"Panel-average CUMULATIVE coefficient on shock_x_exposure (b4, Corr spec's {SHOCK_VARIABLE}*F interaction) over horizons 0..h -- this is ALSO the mitigating effect in the reference regime (Dum=0, 'bust'); see mitigating_effect_bust_panelavg, which equals this exactly.",
    "se_b4_shock_x_F_panelavg": "Standard error of beta_b4_shock_x_F_panelavg",
    "beta_b5_shock_x_dum_panelavg": f"Panel-average CUMULATIVE coefficient on shock_x_dum (b5, {SHOCK_VARIABLE}*Dum interaction) over horizons 0..h. Corr spec only.",
    "se_b5_shock_x_dum_panelavg": "Standard error of beta_b5_shock_x_dum_panelavg",
    "beta_b6_shock_x_F_x_dum_panelavg": f"Panel-average CUMULATIVE coefficient on shock_x_exposure_x_dum (b6, the full three-way {SHOCK_VARIABLE}*F*Dum interaction) over horizons 0..h -- this is the ADDITIONAL mitigating effect specific to the boom regime, on top of b4. Corr spec only.",
    "se_b6_shock_x_F_x_dum_panelavg": "Standard error of beta_b6_shock_x_F_x_dum_panelavg",
    "mitigating_effect_bust_panelavg": f"The mitigating effect of {SHOCK_VARIABLE}*F in the BUST regime (Dum=0, the reference level) = b4 alone. Expected NEGATIVE if shared/concentrated exposure to the AI-chip cycle amplifies the (typically negative) {SHOCK_VARIABLE} shock response during downturns.",
    "se_mitigating_effect_bust_panelavg": "Standard error of mitigating_effect_bust_panelavg (= se_b4_shock_x_F_panelavg exactly, since the bust effect IS b4)",
    "mitigating_effect_boom_panelavg": f"The mitigating effect of {SHOCK_VARIABLE}*F in the BOOM regime (Dum=1) = b4 + b6, with a correctly derived COMBINED standard error (via the full coefficient covariance matrix, NOT b4's and b6's individual SEs summed naively). Expected POSITIVE if the demand/terms-of-trade tailwind dampens the {SHOCK_VARIABLE} shock response during AI-capex booms.",
    "se_mitigating_effect_boom_panelavg": "Correctly derived combined standard error of mitigating_effect_boom_panelavg (b4+b6)",
    f"mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the bust regime (panel-average b4 + {FOCUS_COUNTRY}'s b4 deviation)",
    f"se_mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s bust mitigating effect",
    f"mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the boom regime (panel-average b4+b6, plus {FOCUS_COUNTRY}'s own b4 and b6 deviations, all four terms combined with a correctly derived SE)",
    f"se_mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"Correctly derived combined standard error of {FOCUS_COUNTRY}'s boom mitigating effect (4-term combination)",
    "beta_shock_baseline": f"Average CUMULATIVE GDP response to a one-unit {SHOCK_VARIABLE} shock increase over horizons 0..h, WITHOUT the exposure/mitigation interaction (entity FE only, no time FE -- see run_baseline_shock_projections)",
    "se_shock_baseline": "Standard error of beta_shock_baseline",
    "t_stat_baseline": "t-statistic of beta_shock_baseline (beta/se) -- |t|>~1.96 is significant at the 5% level",
    "is_absorbed_baseline": "True if shock_used was fully absorbed at this horizon",
    "shock_innov": f"Residual from an AR(2) regression of shock_level on its own lags -- an approximate 'identified shock' (the unpredictable part of {SHOCK_VARIABLE}). This IS shock_used throughout the model now. See purify_shock() for why this is a simplified stand-in for a fully identified structural shock, not a reproduction of one.",
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
    print(f"SHOCK_VARIABLE = {SHOCK_VARIABLE!r} -- every IRF, coefficient, and")
    print(f"chart below is based on this shock series. Change SHOCK_VARIABLE")
    print(f"near the top of the CONFIG section (currently 'WUI' or 'GPR') to")
    print(f"switch the whole script to the other one.")
    print("=" * 70)
    print(f"\nComputing {SHOCK_VARIABLE} shock standard deviation (for 1-stdev-shock IRF charts)")
    print("=" * 70)
    shock_std = compute_shock_std()

    print("\n" + "=" * 70)
    print(f"SPEC A: ICT investment share as exposure (primary), global {SHOCK_VARIABLE}")
    print("=" * 70)
    panel_ict = build_panel(exposure="ict")
    res_ict = run_local_projections(panel_ict)
    irf_ict = summarize_irf(res_ict)
    print(irf_ict.to_string())
    fig_ict = plot_irf(irf_ict, "Spec A: ICT investment share exposure", shock_std=shock_std)

    print("\n" + "=" * 70)
    print(f"SPEC A (patent): AI patent applications as exposure, global {SHOCK_VARIABLE}")
    print("                 (table only -- see the combined chart further below)")
    print("=" * 70)
    panel_patent = build_panel(exposure="patent")
    res_patent = run_local_projections(panel_patent)
    irf_patent = summarize_irf(res_patent)
    print(irf_patent.to_string())

    print("\n" + "=" * 70)
    print(f"SPEC A (investment): AI incoming investment counts as exposure, global {SHOCK_VARIABLE}")
    print("                     (table only -- see the combined chart further below)")
    print("=" * 70)
    panel_investment = build_panel(exposure="investment")
    res_investment = run_local_projections(panel_investment)
    irf_investment = summarize_irf(res_investment)
    print(irf_investment.to_string())

    print("\n" + "=" * 70)
    print("COMBINED CHART: AI patent vs AI investment mitigating effect (panel avg only)")
    print("=" * 70)
    fig_combined = plot_combined_mitigating_irf(irf_patent, irf_investment, shock_std=shock_std)

    print("\n" + "=" * 70)
    print("SPEC B: Rolling correlation(national index, semiconductor index)")
    print(f"        as exposure (robustness), global {SHOCK_VARIABLE}")
    print("        -- Channel 1 boom/bust split (see build_panel() docstring)")
    print("=" * 70)
    panel_corr = build_panel(exposure="corr")
    res_corr = run_local_projections(panel_corr)
    irf_corr = summarize_irf(res_corr)
    print(irf_corr.to_string())
    fig_corr_boom = plot_regime_irf(
        irf_corr, "mitigating_effect_boom",
        "Spec B: AI-capex-cycle exposure -- BOOM regime mitigating effect (b4+b6)",
        color="#2ca02c", color_nl="#98df8a", shock_std=shock_std,
    )
    fig_corr_bust = plot_regime_irf(
        irf_corr, "mitigating_effect_bust",
        "Spec B: AI-capex-cycle exposure -- BUST regime mitigating effect (b4)",
        color="#d62728", color_nl="#ff9896", shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print(f"BASELINE: {SHOCK_VARIABLE} shock IRF, exposure/mitigation switched off")
    print("          (entity FE only, no time FE -- see docstring for why;")
    print(f"          shock_used = AR-purified shock, not the raw {SHOCK_VARIABLE} level)")
    print("=" * 70)
    # shock_used and dgdp are identical across specs (only the exposure proxy
    # differs), so this only needs to be run once, off either panel.
    res_baseline = run_baseline_shock_projections(panel_ict)
    irf_baseline = summarize_baseline_irf(res_baseline)
    print(irf_baseline.to_string())
    fig_baseline = plot_baseline_irf(irf_baseline,
                                      title=f"Baseline {SHOCK_VARIABLE} IRF (AR-purified shock, no exposure)",
                                      shock_std=shock_std)

    print("\n" + "=" * 70)
    print(f"Fetching raw source series for export (GDP, GFCF, AI patents, AI investment, indices, {SHOCK_VARIABLE})")
    print("=" * 70)
    raw_gdp = fetch_gdp_growth_yoy()[["country", "quarter", "gdp_level"]]
    raw_gfcf = fetch_ict_investment_share()  # country, year, N1132G, N1173G, N11G, ict_share
    raw_ai_patents = fetch_ai_patents()  # country, year, ai_patents
    raw_ai_investment = fetch_ai_investment()  # country, year, ai_investment
    raw_national_index = fetch_national_indices_raw()
    raw_semiconductor = fetch_semiconductor_raw()
    raw_shock_level = fetch_shock_global().reset_index()
    raw_shock_innov, _ar_model = purify_shock(raw_shock_level)  # quarter, shock_level, shock_innov
    raw_sox_regime = compute_sox_regime()  # quarter, sox_roll_mean_ret, is_boom

    export_results_to_excel(
        panels={"ICT_exposure": panel_ict, "Patent_exposure": panel_patent,
                "Investment_exposure": panel_investment, "Corr_exposure": panel_corr},
        irfs={
            "ICT_exposure": irf_ict,
            "Patent_exposure": irf_patent,
            "Investment_exposure": irf_investment,
            "Corr_exposure": irf_corr,
            f"Baseline_{SHOCK_VARIABLE}": irf_baseline,
        },
        figs={
            "ICT_exposure": fig_ict,
            "Combined_Patent_vs_Investment": fig_combined,
            "Corr_exposure_Boom": fig_corr_boom,
            "Corr_exposure_Bust": fig_corr_bust,
            f"Baseline_{SHOCK_VARIABLE}": fig_baseline,
        },
        raw_data={
            "Raw_GDP": raw_gdp,
            "Raw_GFCF": raw_gfcf,
            "Raw_AI_Patents": raw_ai_patents,
            "Raw_AI_Investment": raw_ai_investment,
            "Raw_National_Index": raw_national_index,
            "Raw_Semiconductor": raw_semiconductor,
            f"Raw_{SHOCK_VARIABLE}_Level": raw_shock_level,
            f"Raw_{SHOCK_VARIABLE}_Shock": raw_shock_innov,  # quarter, shock_level, shock_innov -- what the model actually uses
            "Raw_SOX_Regime": raw_sox_regime,  # quarter, sox_roll_mean_ret, is_boom
        },
        filepath="model_results.xlsx",
    )

    print(f"""
    MODEL SPECIFICATION (moderated regression, not a "difference" spec):
    Shock variable in this run: {SHOCK_VARIABLE}
    ICT spec:  GDPgrowth = b0 + b1*Shock + b2*F + b3*(Shock*F) + controls
    Corr spec: GDPgrowth = b0 + b1*Shock + b2*F + b3*Dum + b4*(Shock*F)
                           + b5*(Shock*Dum) + b6*(Shock*F*Dum) + controls
    (Dum = is_boom; Dum=0/"bust" is the reference level.) Both specs use
    ENTITY fixed effects only, no time effects -- required because the
    shock, Dum, and their product are entity-invariant (see build_panel()
    / run_local_projections() for the full identification argument).
    b3/b4 here are genuine interaction-effect coefficients net of both
    main effects, not a "F=1 minus F=0" difference.

    NL-SPECIFIC RESULT, SPEC A (ICT / Channel 2): in the ICT_exposure_IRF
    sheet, look at beta_b3_shock_x_F_{FOCUS_COUNTRY}_total and
    se_b3_shock_x_F_{FOCUS_COUNTRY}_total -- NL's total b3 coefficient
    (panel-average + NL-specific deviation), with a correctly computed
    combined SE. A POSITIVE value means higher AI/ICT exposure dampens
    the {SHOCK_VARIABLE} shock's impact on growth (consistent with
    firm-level resilience); NEGATIVE means exposure amplifies it.
    beta_b1_shock_* and beta_b2_F_* report the {SHOCK_VARIABLE} and F
    main effects themselves, now separately estimated rather than
    absorbed.

    NL-SPECIFIC RESULT, SPEC B (Corr / Channel 1, boom vs bust): in the
    Corr_exposure_IRF sheet, look at mitigating_effect_bust_panelavg (=
    b4 alone, the Dum=0/bust reference level) and
    mitigating_effect_boom_panelavg (= b4+b6, the Dum=1/boom regime, with
    a correctly derived combined SE) -- or the {FOCUS_COUNTRY}_total
    versions of each for NL's own coefficients. Economic prior:
    mitigating_effect_boom POSITIVE (demand/terms-of-trade tailwind
    dampens the {SHOCK_VARIABLE} shock's impact during an AI-capex boom)
    and mitigating_effect_bust NEGATIVE (shared exposure to the same
    risk factor amplifies it during an AI downturn) -- see build_panel()
    for the full reasoning. The individual b1..b6 coefficients (main
    effects and interactions) are also reported in full for anyone who
    wants to verify the derived quantities by hand.

    NOTE ON SPECIFICATION: shock_used is the AR(2)-purified
    {SHOCK_VARIABLE} shock (see purify_shock()), not the raw
    {SHOCK_VARIABLE} level -- this applies throughout the workbook,
    including the baseline spec below.

    BASELINE {SHOCK_VARIABLE} IRF (Baseline_{SHOCK_VARIABLE} sheet): the
    unconditional GDP response to the AR-purified {SHOCK_VARIABLE}
    shock, with the exposure/mitigation channel removed entirely -- this
    answers "is the shock itself significant" before asking whether AI
    exposure changes that response. Check t_stat_baseline (|t|>~1.96 ~
    5% significance) at each horizon. This spec (like both moderated
    specs above) uses entity FE only, no time FE -- so read it as "does
    the {SHOCK_VARIABLE} shock move growth on average across the panel",
    not as a fully "clean" two-way-FE IRF. The Raw_{SHOCK_VARIABLE}_Level
    and Raw_{SHOCK_VARIABLE}_Shock sheets let you compare the purified
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
       effects (required by including {SHOCK_VARIABLE}/Dum main effects
       -- see the model specification note above). This means common-
       across-countries time-varying confounders (ECB monetary policy
       stance, euro-area-wide demand shocks at a given quarter) are not
       swept out of any spec's residual. Treat every coefficient in this
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
       the cumulative dGDP response per 1-STANDARD-DEVIATION {SHOCK_VARIABLE} shock
       (shock_std, printed above -- see compute_shock_std()),
       NOT per 1 raw unit of shock_used -- shock_used is an AR(2) residual,
       an arbitrary scale with no natural economic unit, so "per 1
       standard deviation" is the interpretable convention here.
       IMPORTANT: this rescaling applies ONLY to the CHARTS -- the
       underlying Excel tables (ICT_exposure_IRF, Corr_exposure_IRF,
       Baseline_{SHOCK_VARIABLE}_IRF) still report the raw per-unit coefficients
       exactly as estimated, so any further hand-calculation from the
       tables needs to multiply by shock_std manually to match what
       the charts show. t-statistics are unaffected by this rescaling
       either way (scaling beta and se by the same constant leaves their
       ratio unchanged).

    WHAT "MITIGATING EFFECT" MEANS IN ECONOMIC TERMS:

    The baseline spec asks a simple question: does a {SHOCK_VARIABLE}
    shock move GDP growth at all, on average across the panel?
    beta_shock_baseline is that average marginal response -- typically
    expected to be NEGATIVE (higher {SHOCK_VARIABLE} depresses growth),
    though whether it is actually significant here is itself
    informative.

    The ICT and Corr specs ask a follow-up question: does a country's
    AI/ICT exposure change HOW SENSITIVE its growth is to that same
    shock? This is captured by an INTERACTION term ({SHOCK_VARIABLE}
    shock x exposure F), so the coefficients on it (b3 for ICT, b4/b6
    for Corr) are not themselves "the effect of the shock" -- they are
    the effect of exposure ON that sensitivity. Concretely: the
    marginal GDP response to a {SHOCK_VARIABLE} shock, as a function of
    exposure F, is

        d(GDP growth) / d(Shock) = b1 + b3*F        (ICT spec)
        d(GDP growth) / d(Shock) = b1 + b4 + b6*Dum  (Corr spec, at F=1)

    A POSITIVE b3 (or b4/b6) means: as exposure rises, the growth
    response to a {SHOCK_VARIABLE} shock becomes LESS negative -- i.e.
    exposure DAMPENS the shock's bite. This is the "mitigation"
    hypothesis: firms/countries with more AI/ICT capacity (Channel 2:
    firm-level resilience -- automation, data-driven decision-making,
    substituting capital for volatile labour/supply-chain inputs) or
    more exposure to the AI-capex investment cycle (Channel 1: demand/
    terms-of-trade tailwind during a boom) absorb {SHOCK_VARIABLE}
    shocks better than less-exposed peers.

    A NEGATIVE b3 (or b4, in the bust regime specifically) means the
    opposite: exposure AMPLIFIES the shock's impact rather than
    cushioning it -- e.g. because heavy reliance on a concentrated set of
    AI-related suppliers/customers (think: the semiconductor supply
    chain) makes a country's growth MORE fragile to broad {SHOCK_VARIABLE}
    shocks, not less, especially when that same AI-capex cycle is itself
    in a downturn (bust).

    In short: this script is not testing "is {SHOCK_VARIABLE} bad for
    growth" (the baseline spec) -- it is testing "does AI/ICT exposure
    make a country's growth more or less fragile to {SHOCK_VARIABLE}
    shocks," which is a claim about a SECOND DERIVATIVE (how the shock's
    own effect changes with exposure), not the shock's effect itself. A
    positive mitigating coefficient supports the "AI/ICT capacity as an
    economic buffer" story; a negative one supports an "AI/ICT
    concentration as a new source of fragility" story instead. Both are
    economically plausible ex ante -- that is precisely why this needs
    to be estimated rather than assumed.
    """)
