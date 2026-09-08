"""
Global shock propagation and AI-exposure mitigation: NL + euro area panel
========================================================================

Estimates a MODERATED-REGRESSION panel local projection (main effects
AND interaction terms, not a "difference" spec) with ENTITY fixed
effects only (no time effects):

  ICT spec:
    qoq_cum_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
                       + b3_h*(Shock_t * F[i,t]) + Gamma_h*X[i,t]
                       + e[i,t+h]

  Corr spec (Dum = is_boom_t; Dum=0/"bust" is the reference level):
    qoq_cum_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
                       + b3_h*Dum_t + b4_h*(Shock_t*F[i,t])
                       + b5_h*(Shock_t*Dum_t)
                       + b6_h*(Shock_t*F[i,t]*Dum_t)
                       + Gamma_h*X[i,t] + e[i,t+h]

X[i,t] is the CONTROL VECTOR. In the two specs above, X[i,t] =
{qoq_lag1[i,t]} -- GDP's own quarter-on-quarter growth rate, one
quarter lagged (see fetch_gdp_level()/build_panel()) -- the only
control used.

The "*_robustness" variant of EVERY spec above (ict_robustness,
patent_robustness, investment_robustness, corr_robustness,
hs_export_robustness -- referee-requested, testing whether the
exposure-mitigation effect survives once other commonly-cited channels
of cross-country heterogeneity in shock sensitivity are also allowed
to interact with the shock) extends X[i,t] with four additional
growth-literature control variables, each ALSO entering via its own
interaction with Shock_t:

  Robustness spec (any spec above, "_robustness" suffix):
    qoq_cum_gdp[i,t+h] = [that spec's equation above, unchanged]
                       + c1_h*TradeOpen[i,t] + c1x_h*(Shock_t*TradeOpen[i,t])
                       + c2_h*PopGrowth[i,t] + c2x_h*(Shock_t*PopGrowth[i,t])
                       + c3_h*ProdGrowth[i,t] + c3x_h*(Shock_t*ProdGrowth[i,t])
                       + c4_h*GDPPerCapita[i,t] + c4x_h*(Shock_t*GDPPerCapita[i,t])
                       + e[i,t+h]

  i.e. X[i,t] = {qoq_lag1[i,t], TradeOpen[i,t], PopGrowth[i,t],
  ProdGrowth[i,t], GDPPerCapita[i,t]} for the robustness variants --
  see build_panel()'s "*_robustness" branches and
  _build_regressor_lists() for exactly which terms this adds
  (trade_openness, population_growth, productivity_growth,
  gdp_per_capita, and their four shock_x_* interaction terms), and
  fetch_trade_openness()/fetch_population_growth()/
  fetch_productivity_growth()/fetch_gdp_per_capita() for each series'
  source and construction.

for h = 0..8 quarters, where qoq_cum_gdp[i,t+h] is the CUMULATIVE
quarter-on-quarter GDP growth from t-1 through t+h (100*(ln(gdp[t+h])
- ln(gdp[t-1])) -- by telescoping, exactly the sum of the h+1
individual QoQ growth rates from t through t+h, so this single
regression's coefficients ARE the cumulative IRF on QoQ growth
directly). F[i,t] in (0,1)
is a logistic transform of a
country-level AI/ICT-exposure state variable z[i,t], and Shock_t is an
AR(2)-purified version of a GLOBAL shock series -- the World
Uncertainty Index (WUI, Ahir, Bloom & Furceri), the Geopolitical Risk
Index (GPR, Caldara & Iacoviello), the Economic Policy Uncertainty
Index (EPU, Baker, Bloom & Davis), the Trade Policy Uncertainty Index
(TPU, Caldara, Iacoviello, Molligo, Prestipino & Raffo), the Global
Supply Chain Pressure Index (GSCPI, Benigno, di Giovanni, Groen &
Noble), or PC (the first principal component of GPR and EPU, derived
here rather than a published index -- see fetch_pc_gpr_epu()) --
whichever SHOCK_VARIABLE selects (see
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

INTERPRETATION: b3 (ICT spec) and b4 (Corr and hs_export specs) are the
mitigating effect of AI/ICT exposure on the shock's impact, net of the
shock and F main effects (which are now separately estimated, not
folded into an implicit baseline). For the Corr and hs_export specs
specifically, the mitigating effect of Shock*F is b4 alone in the bust
regime (Dum=0) and b4+b6 in the boom regime (Dum=1) -- see
summarize_irf() for where that sum is computed with a correctly
derived combined standard error.

Five versions of z (AI exposure) are estimated as separate models:
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
  (E) AI/ICT-related HS export share -- combined export value of HS
      847150+847180+847330+848610+848620+848630+848640+848690 (HS 8471
      computer processing units/parts and HS 8486 semiconductor/flat-
      panel-display manufacturing machinery) as a share of total
      merchandise exports
      (see fetch_hs_export()); uses the SAME boom/bust regime split as
      (D) (identical SOX-based is_boom classification -- only the
      exposure variable F itself differs between the two)

All results (IRF tables, all series used -- including both the raw shock
level and the purified shock, for transparency -- variable definitions,
and IRF plots) are saved to a single Excel workbook: model_results.xlsx.

ALL INPUT DATA is read from a single LOCAL workbook, LOCAL_DATA_FILE
("Data_AI.xlsx"), which must sit next to this script -- see
_read_data_ai_sheet() and the fetch_*() functions below. No network
access, no live API/website calls of any kind are made anywhere in this
script; every series (the chosen shock, GDP, ICT investment shares,
national equity indices, the semiconductor index) is a plain read from
one of that workbook's sheets (gdp, ict_inv, index_nat, index_sox, wui,
gpr, epu, tpu, gscpi). Only the rolling correlation (Spec B's exposure
measure) and the boom/bust regime classification are still COMPUTED in
this script --
from the raw index_nat/index_sox sheets -- since those are derived
quantities, not raw source data.

Run:  pip install pandas numpy statsmodels linearmodels openpyxl matplotlib Pillow
      python gpr_ai_mitigation_pipeline.py
"""

SCRIPT_VERSION = "2025-08-11-v56-pc-shock-variable"  # bump this whenever the file changes;
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
# "index_sox" (ticker, quarter, close, log_ret), and FIVE global shock
# sheets -- "wui" (quarter, wui_global), "gpr" (quarter, gpr_global),
# "epu" (quarter, epu_global), "tpu" (quarter, tpu_global), "gscpi"
# (quarter, gscpi_global). A SIXTH option, "PC" (SHOCK_VARIABLE == "PC"),
# is not its own sheet -- it is the first principal component of the
# "gpr" and "epu" sheets, computed on the fly (see fetch_pc_gpr_epu()).
# Which ONE of these six is actually used depends on SHOCK_VARIABLE
# (see the CONFIG section below), via fetch_shock_global().


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

COUNTRIES = ["NL", "DE", "FR", "IT", "ES", "BE", "AT", "IE", "FI", "PT"]
SAMPLE_START = "2000-01-01"
SAMPLE_END   = "2026-06-30"
SHOCK_VARIABLE = "WUI"          # "WUI", "GPR", "EPU", "TPU", "GSCPI", or
                                 # "PC" (first principal component of GPR
                                 # and EPU, standardized -- see
                                 # fetch_pc_gpr_epu()) -- which global
                                 # shock series drives every
                                 # spec in this script (baseline, ICT, and
                                 # Corr alike). Change this ONE line to
                                 # switch the whole script (data source,
                                 # AR-purification, panel construction,
                                 # every IRF table and chart) between any
                                 # of the six -- nothing else needs to
                                 # change. The first five series are read
                                 # directly from
                                 # LOCAL_DATA_FILE (sheets "wui", "gpr",
                                 # "epu", "tpu", "gscpi" respectively);
                                 # "PC" is computed on the fly from the
                                 # "gpr" and "epu" sheets -- see
                                 # fetch_shock_global().
_VALID_SHOCK_VARIABLES = ("WUI", "GPR", "EPU", "TPU", "GSCPI", "PC")
if SHOCK_VARIABLE not in _VALID_SHOCK_VARIABLES:
    raise ValueError(
        f"SHOCK_VARIABLE must be one of {_VALID_SHOCK_VARIABLES}, "
        f"got {SHOCK_VARIABLE!r}"
    )
HORIZONS = range(0, 9)          # h = 0..8 quarters
THETA = 2.0                     # logistic transition steepness (standardized z)
FOCUS_COUNTRY = "NL"            # country singled out for its own interaction term
STANDARDIZE_MODE = "pooled"     # 'pooled' (default, recommended) or 'within_country'
                                 # -- see the long comment in build_panel() for why
                                 # 'within_country' can cause a two-way-FE absorption
                                 # error for the exposure interaction term.

# ----------------------------------------------------------------------
# 1. Global shock series -- WUI (Ahir, Bloom & Furceri), GPR (Caldara &
#    Iacoviello), EPU (Baker, Bloom & Davis), TPU (Caldara, Iacoviello,
#    Molligo, Prestipino & Raffo), or GSCPI (Benigno, di Giovanni, Groen
#    & Noble, FRBNY), selected via SHOCK_VARIABLE above
# ----------------------------------------------------------------------

_SHOCK_SHEET = {"WUI": "wui", "GPR": "gpr", "EPU": "epu", "TPU": "tpu", "GSCPI": "gscpi"}
_SHOCK_SOURCE_COL = {
    "WUI": "wui_global", "GPR": "gpr_global", "EPU": "epu_global",
    "TPU": "tpu_global", "GSCPI": "gscpi_global",
}
_SHOCK_CITATION = {
    "WUI": 'the World Uncertainty Index (Ahir, Bloom & Furceri, NBER WP 29763)',
    "GPR": 'the Geopolitical Risk Index (Caldara & Iacoviello, AER 2022)',
    "EPU": 'the Economic Policy Uncertainty Index (Baker, Bloom & Davis, QJE 2016)',
    "TPU": 'the Trade Policy Uncertainty Index (Caldara, Iacoviello, Molligo, Prestipino & Raffo, JME 2020)',
    "GSCPI": 'the Global Supply Chain Pressure Index (Benigno, di Giovanni, Groen & Noble, FRBNY Staff Reports 2022)',
    "PC": 'the first principal component of the (standardized) GPR and EPU series -- a composite uncertainty measure derived here, not itself a published index; see fetch_pc_gpr_epu() for the exact construction',
}


def fetch_pc_gpr_epu():
    """
    First principal component of the (standardized) GPR and EPU series
    -- a composite global-uncertainty measure derived here, used when
    SHOCK_VARIABLE == "PC". NOT itself a published index (unlike WUI,
    GPR, EPU, TPU, GSCPI, each from its own cited paper -- see
    _SHOCK_CITATION); this is a straightforward PCA of two of this
    project's own already-loaded shock series, computed fresh on every
    call.

    Reads the "gpr" and "epu" sheets (quarter, gpr_global / epu_global)
    of LOCAL_DATA_FILE directly, INNER-joins them on quarter (a quarter
    missing from either series is dropped entirely -- PCA needs a
    complete, aligned sample), z-scores each series (mean 0, std 1 --
    essential before PCA, since GPR and EPU are on very different raw
    scales and PCA is scale-sensitive), then extracts PC1 via
    eigendecomposition of the resulting 2x2 covariance matrix (plain
    numpy -- no additional dependency needed for a 2-variable PCA).

    Sign convention: PC1's loadings are flipped if their sum is
    negative, so PC1 is always POSITIVELY associated with both GPR and
    EPU (an eigenvector's sign is otherwise arbitrary -- numpy could
    return either direction), keeping "higher PC1 = higher uncertainty
    on both source measures" as the natural interpretation.

    Returns a DataFrame indexed by quarter, with the single column
    "shock_level" -- matching fetch_shock_global()'s output shape
    exactly, so every downstream function (purify_shock(), build_panel(),
    etc.) works unchanged regardless of whether SHOCK_VARIABLE selects
    this composite or one of the five published indices.
    """
    gpr_df = _read_data_ai_sheet("gpr")
    epu_df = _read_data_ai_sheet("epu")
    for name, df, col in [("gpr", gpr_df, "gpr_global"), ("epu", epu_df, "epu_global")]:
        missing = {"quarter", col} - set(df.columns)
        if missing:
            raise ValueError(
                f"fetch_pc_gpr_epu: expected columns {missing} not found in "
                f"the '{name}' sheet of '{LOCAL_DATA_FILE}'. Actual columns: "
                f"{list(df.columns)}."
            )

    gpr_df = gpr_df[["quarter", "gpr_global"]].copy()
    epu_df = epu_df[["quarter", "epu_global"]].copy()
    gpr_df["quarter"] = pd.PeriodIndex(gpr_df["quarter"], freq="Q")
    epu_df["quarter"] = pd.PeriodIndex(epu_df["quarter"], freq="Q")
    gpr_df["gpr_global"] = pd.to_numeric(gpr_df["gpr_global"], errors="coerce")
    epu_df["epu_global"] = pd.to_numeric(epu_df["epu_global"], errors="coerce")

    merged = gpr_df.merge(epu_df, on="quarter", how="inner").dropna()
    if len(merged) < 10:
        raise ValueError(
            f"fetch_pc_gpr_epu: only {len(merged)} quarters have BOTH GPR "
            f"and EPU available after inner-joining -- too few to compute "
            f"a meaningful principal component."
        )

    gpr_z = (merged["gpr_global"] - merged["gpr_global"].mean()) / merged["gpr_global"].std()
    epu_z = (merged["epu_global"] - merged["epu_global"].mean()) / merged["epu_global"].std()

    X = np.column_stack([gpr_z.to_numpy(), epu_z.to_numpy()])
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    pc1_loadings = eigenvectors[:, -1]  # largest eigenvalue = last column
    if pc1_loadings.sum() < 0:
        pc1_loadings = -pc1_loadings

    pc1_scores = X @ pc1_loadings
    var_explained = eigenvalues[-1] / eigenvalues.sum()

    print(f"  [diagnostic] PC1(GPR, EPU): loadings=[GPR={pc1_loadings[0]:.4f}, "
          f"EPU={pc1_loadings[1]:.4f}], variance explained={var_explained:.1%}, "
          f"n={len(merged)} quarters.")

    df = pd.DataFrame({"quarter": merged["quarter"].values, "shock_level": pc1_scores})
    return df.set_index("quarter")[["shock_level"]]


def fetch_shock_global():
    """
    Quarterly GLOBAL shock series -- whichever of WUI, GPR, EPU, TPU,
    GSCPI, or PC (the first principal component of GPR and EPU -- see
    fetch_pc_gpr_epu()) SHOCK_VARIABLE currently selects. See
    _SHOCK_CITATION for the source paper of each (PC has no formal
    citation -- it is derived here, not a published index).

    For the five published indices, reads the corresponding sheet
    ("wui", "gpr", "epu", "tpu", or "gscpi" -- columns: quarter,
    {name}_global) of LOCAL_DATA_FILE directly -- already quarterly, no
    further resampling needed. For "PC", delegates entirely to
    fetch_pc_gpr_epu() instead (computed from the "gpr" and "epu"
    sheets, not read from a single sheet of its own). Regardless of
    which is selected, the result always has a single GENERIC column
    name, "shock_level", so every downstream function (purify_shock(),
    build_panel(), etc.) is written once and works unchanged for any of
    the six shock variables -- they never need to know or care which
    one was actually chosen.

    Returns a DataFrame indexed by quarter, with the single column
    "shock_level".
    """
    if SHOCK_VARIABLE == "PC":
        return fetch_pc_gpr_epu()

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
    AR(lags) purification of the chosen global shock series (WUI, GPR,
    EPU, TPU, GSCPI, or PC -- per SHOCK_VARIABLE): regress its level on
    its own lags and take the residual as an approximate "identified
    shock" -- the part not predictable from its own recent history.

    All six candidate series are already quarterly, relatively
    persistent indices (unlike a daily/monthly news-count series), so
    their own lags typically explain a meaningful share of variance --
    the residual strips out that predictable component, leaving
    something closer to a "surprise." This is a lightweight
    identification choice, not a structural one: it does NOT control
    for other macro/financial variables the way a full VAR would (each
    series' own paper uses a different, fuller identification scheme --
    see _SHOCK_CITATION for the source paper of whichever one is
    currently selected). Treat this as a supplementary robustness check
    on the baseline spec, not a replacement for any paper's own fully
    identified structural shock.

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

def fetch_gdp_level():
    """
    Quarterly real GDP level per country (gdp_level) -- the raw series
    used both for the Raw_GDP export sheet and, in log-difference form,
    as the model's DEPENDENT VARIABLE and lag CONTROL throughout this
    script (see build_panel(), run_local_projections(), and
    run_baseline_shock_projections()): the dependent variable
    (qoq_cum_lead) is the CUMULATIVE quarter-on-quarter GDP growth from
    t-1 through t+h, and the lag control (qoq_lag1) is the underlying
    one-period QoQ growth rate itself, one quarter before the
    regression's own reference time t -- both are built from log
    differences of the SAME underlying series, at different points in
    time.

    Reads the "gdp" sheet (columns: country, quarter, gdp_level) of
    LOCAL_DATA_FILE and returns it unchanged (level only -- no growth-
    rate transformation of any kind is computed here; qoq_cum_lead and
    qoq_lag1 are built directly from this level elsewhere).
    """
    df = _read_data_ai_sheet("gdp")
    missing = {"country", "quarter", "gdp_level"} - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_gdp_level: expected columns {missing} not found "
            f"in the 'gdp' sheet of '{LOCAL_DATA_FILE}'. Actual columns: "
            f"{list(df.columns)}."
        )
    df = df.copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["gdp_level"] = pd.to_numeric(df["gdp_level"], errors="coerce")
    df = df[["country", "quarter", "gdp_level"]].dropna().sort_values(["country", "quarter"])
    return df


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


def fetch_hs_export():
    """
    Annual AI/ICT-related HS export share per country: the "hs_export"
    sheet's pre-computed share column (the combined export value of HS
    847150+847180+847330+848610+848620+848630+848640+848690 -- HS 8471
    computer processing units/parts and HS 8486 semiconductor/flat-
    panel-display manufacturing machinery -- as a share of that
    country's total merchandise exports to the world -- both sourced
    from the same OECD BIMTS/Data Explorer pull; see
    fetch_bimts_hs_export.py, the local-fetch script used to build
    this sheet).

    Reads the "hs_export" sheet (columns: country, year, per-HS-code
    export values, a total_export column, and a share column -- exact
    HS/total column names not otherwise relied on here) of
    LOCAL_DATA_FILE. Only country, year, and the share column are used;
    the underlying HS-code and total-export values stay in the sheet
    for reference/export but aren't otherwise used in this script. The
    share column is located by name defensively (matched
    case-insensitively on containing "SHARE"), since its exact literal
    name in the workbook wasn't independently confirmed.
    """
    df = _read_data_ai_sheet("hs_export")
    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    share_col = next((cols_upper[c] for c in cols_upper if "SHARE" in c), None)
    if share_col is None or "COUNTRY" not in cols_upper or "YEAR" not in cols_upper:
        raise ValueError(
            f"fetch_hs_export: expected columns 'country', 'year', and a "
            f"column containing 'share' not all found in the 'hs_export' "
            f"sheet of '{LOCAL_DATA_FILE}'. Actual columns: {list(df.columns)}."
        )
    country_col, year_col = cols_upper["COUNTRY"], cols_upper["YEAR"]
    df = df.rename(columns={country_col: "country", year_col: "year", share_col: "share"})
    df["share"] = pd.to_numeric(df["share"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["share"]).sort_values(["country", "year"])

    if df.empty:
        raise ValueError(
            f"fetch_hs_export: zero valid rows after reading the "
            f"'hs_export' sheet of '{LOCAL_DATA_FILE}' (share column "
            f"detected as '{share_col}')."
        )

    print(f"  [diagnostic] HS export share (column '{share_col}'): {len(df)} "
          f"country-year rows across {df['country'].nunique()} countries, "
          f"years {int(df['year'].min())}-{int(df['year'].max())}.")

    return df[["country", "year", "share"]]


def _fetch_growth_control_sheet(sheet_name, value_col):
    """
    Shared logic for the four referee-requested growth-literature
    control variables (trade_openness, population_growth,
    productivity_growth, gdp_per_capita), each its own simple annual
    (country, year, value) sheet in LOCAL_DATA_FILE, built by
    collect_ai_data.py. Used only by the "ict_robustness" spec (see
    build_panel()) -- these are NOT used as controls in any of the
    other specs, only in the dedicated robustness check.
    """
    df = _read_data_ai_sheet(sheet_name)
    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    missing = {"COUNTRY", "YEAR", value_col.upper()} - set(cols_upper.keys())
    if missing:
        raise ValueError(
            f"_fetch_growth_control_sheet({sheet_name!r}): expected columns "
            f"'country', 'year', '{value_col}' not all found. Actual "
            f"columns: {list(df.columns)}."
        )
    df = df.rename(columns={cols_upper["COUNTRY"]: "country", cols_upper["YEAR"]: "year",
                             cols_upper[value_col.upper()]: value_col})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).sort_values(["country", "year"])

    if df.empty:
        raise ValueError(
            f"_fetch_growth_control_sheet({sheet_name!r}): zero valid rows "
            f"after reading."
        )

    print(f"  [diagnostic] {sheet_name}: {len(df)} country-year rows across "
          f"{df['country'].nunique()} countries, years "
          f"{int(df['year'].min())}-{int(df['year'].max())}.")
    return df[["country", "year", value_col]]


def fetch_trade_openness():
    """Trade (% of GDP), annual -- see _fetch_growth_control_sheet()."""
    return _fetch_growth_control_sheet("trade_openness", "trade_openness")


def fetch_population_growth():
    """Population growth (annual %) -- see _fetch_growth_control_sheet()."""
    return _fetch_growth_control_sheet("population_growth", "population_growth")


def fetch_productivity_growth():
    """
    Real labour productivity per hour worked, YoY growth (log-diff) --
    see _fetch_growth_control_sheet(). NOTE: this series starts one
    year later than the others per country (2001, not 2000) since it
    is itself a year-on-year growth rate, computed from an index at
    collection time -- see collect_ai_data.py's fetch_productivity_growth().
    """
    return _fetch_growth_control_sheet("productivity_growth", "productivity_growth")


def fetch_gdp_per_capita():
    """GDP per capita, chain-linked volumes -- see _fetch_growth_control_sheet()."""
    return _fetch_growth_control_sheet("gdp_per_capita", "gdp_per_capita")


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

def _merge_robustness_controls(panel, quarters):
    """
    Shared helper: merges the four referee-requested growth-literature
    control variables (trade_openness, population_growth,
    productivity_growth, gdp_per_capita) onto `panel`, interpolated
    from annual to quarterly the same way every other annual exposure
    series in this script is (see annual_to_quarterly()). Used by every
    "*_robustness" exposure option in build_panel() -- factored out
    here since five separate exposures (ict_robustness,
    patent_robustness, investment_robustness, corr_robustness,
    hs_export_robustness) all need the exact same four merges.
    """
    for control_name, fetch_fn in [
        ("trade_openness", fetch_trade_openness),
        ("population_growth", fetch_population_growth),
        ("productivity_growth", fetch_productivity_growth),
        ("gdp_per_capita", fetch_gdp_per_capita),
    ]:
        control_annual = fetch_fn()
        control_q = annual_to_quarterly(control_annual, control_name, quarters)
        panel = panel.merge(control_q, on=["country", "quarter"], how="left")
    return panel


def build_baseline_robustness_panel():
    """
    Panel for the referee-requested robustness check on the BASELINE
    model specifically: the "raw" AR(1) shock model (see
    run_baseline_shock_projections()), extended with the same four
    growth-literature control variables used in the "*_robustness"
    exposure specs (trade_openness, population_growth,
    productivity_growth, gdp_per_capita) as ADDITIVE controls ONLY --
    NOT their own interactions with the shock (unlike the exposure
    specs' "*_robustness" variants, which DO add both the main effect
    and a Shock interaction for each of the four controls; this
    baseline check adds only the main effect of each, per the explicit
    scope given for this specific chart).

    Tests whether the shock's average, UNCONDITIONAL effect (with no
    exposure/mitigation channel at all) is itself sensitive to simply
    controlling for these four commonly-cited country characteristics,
    not whether the shock's SENSITIVITY varies with them (that would
    require the interaction terms this check deliberately omits).

    Unlike build_panel()'s "*_robustness" branches, this panel has NO
    F/exposure term at all (the baseline model never had one to begin
    with -- see run_baseline_shock_projections()'s docstring), so there
    is no exposure-related merge/construction here, only: GDP, the
    purified shock, qoq_lag1, and the four controls (main effects
    only).
    """
    shock_level = fetch_shock_global().reset_index()
    shock_innov_df, _ar_model = purify_shock(shock_level)
    gdp = fetch_gdp_level()

    quarters = pd.PeriodIndex(pd.period_range(SAMPLE_START, SAMPLE_END, freq="Q"))

    panel = gdp.merge(shock_innov_df, on="quarter", how="left")
    panel["shock_used"] = panel["shock_innov"]

    panel["qoq_growth"] = panel.groupby("country")["gdp_level"].transform(
        lambda s: 100 * np.log(s / s.shift(1))
    )
    panel["qoq_lag1"] = panel.groupby("country")["qoq_growth"].shift(1)

    panel = _merge_robustness_controls(panel, quarters)
    # NOTE: deliberately NOT adding shock_x_* interaction terms here --
    # see this function's docstring for why this check is scoped to
    # additive controls only, unlike the exposure specs' own
    # "*_robustness" variants.

    panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)
    return panel


def build_panel(exposure="ict"):
    """
    exposure: 'ict'        -> ICT investment share of GFCF
              'patent'     -> annual AI patent applications per country
                              (see fetch_ai_patents())
              'investment' -> annual AI-related incoming investment
                              counts per country (see fetch_ai_investment())
              'corr'       -> rolling correlation of national index with
                              the semiconductor index (robustness), with
                              a boom/bust regime split
              'hs_export'  -> AI/ICT-related HS export share (HS 847150+
                              847180+847330+848610+848620+848630+848640+
                              848690 as a share of total
                              merchandise exports; see fetch_hs_export()),
                              with the SAME boom/bust regime split as
                              'corr' (same SOX-based is_boom
                              classification -- only the exposure
                              variable F itself differs between the two
                              specs, not the regime definition)
              'ict_robustness', 'patent_robustness',
              'investment_robustness', 'corr_robustness',
              'hs_export_robustness' -> REFEREE ROBUSTNESS CHECK
                              variants of 'ict'/'patent'/'investment'/
                              'corr'/'hs_export' respectively: same F as
                              the base spec, but adds four growth-
                              literature control variables (trade
                              openness, population growth, productivity
                              growth, GDP per capita) AND their own
                              interactions with the shock, to test
                              whether the exposure's mitigating effect
                              (b3, Shock*F) survives once these
                              alternative channels for cross-country
                              heterogeneity in shock sensitivity are
                              also allowed to interact with the shock --
                              see the long comment further below (after
                              the shared F/shock_x_exposure block) for
                              the full specification. corr_robustness
                              and hs_export_robustness keep the SAME
                              boom/bust regime split as their base
                              specs.

    The global shock (WUI, GPR, EPU, TPU, GSCPI, or PC, per SHOCK_VARIABLE -- see the CONFIG
    section) is always AR-purified (see purify_shock()): shock_used is
    the AR(2) residual of the chosen series' level, not the raw level
    itself. The exposure interaction (shock_x_exposure = F * shock_used)
    and every downstream coefficient/IRF/chart are based on this
    purified shock, whichever series it actually is.
    """
    shock_level = fetch_shock_global().reset_index()
    shock_innov_df, _ar_model = purify_shock(shock_level)
    gdp = fetch_gdp_level()

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
    elif exposure == "hs_export":
        hs_export_annual = fetch_hs_export()
        exp_q = annual_to_quarterly(hs_export_annual, "share", quarters)
        exp_q = exp_q.rename(columns={"share": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
    elif exposure == "ict_robustness":
        # Same F as 'ict' (ICT investment share) -- see the docstring
        # above and the long comment further below for what this spec
        # adds on top.
        ict_annual = fetch_ict_investment_share()
        exp_q = annual_to_quarterly(ict_annual, "ict_share", quarters)
        exp_q = exp_q.rename(columns={"ict_share": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
        panel = _merge_robustness_controls(panel, quarters)
    elif exposure == "patent_robustness":
        # Same F as 'patent' (AI patent applications).
        patent_annual = fetch_ai_patents()
        exp_q = annual_to_quarterly(patent_annual, "ai_patents", quarters)
        exp_q = exp_q.rename(columns={"ai_patents": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
        panel = _merge_robustness_controls(panel, quarters)
    elif exposure == "investment_robustness":
        # Same F as 'investment' (AI incoming investment counts).
        investment_annual = fetch_ai_investment()
        exp_q = annual_to_quarterly(investment_annual, "ai_investment", quarters)
        exp_q = exp_q.rename(columns={"ai_investment": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
        panel = _merge_robustness_controls(panel, quarters)
    elif exposure == "corr_robustness":
        # Same F as 'corr' (rolling correlation with the semiconductor
        # index). Still gets the SAME boom/bust regime split as 'corr'
        # -- see the "if exposure in (...)" block further below, which
        # this exposure name is also included in.
        corr = fetch_corr_with_semiconductor_index(countries=COUNTRIES)
        panel = gdp.merge(corr, on=["country", "quarter"], how="left")
        panel = _merge_robustness_controls(panel, quarters)
    elif exposure == "hs_export_robustness":
        # Same F as 'hs_export' (AI/ICT-related HS export share). Still
        # gets the SAME boom/bust regime split as 'hs_export'/'corr'.
        hs_export_annual = fetch_hs_export()
        exp_q = annual_to_quarterly(hs_export_annual, "share", quarters)
        exp_q = exp_q.rename(columns={"share": "z_raw"})
        panel = gdp.merge(exp_q, on=["country", "quarter"], how="left")
        panel = _merge_robustness_controls(panel, quarters)
    else:
        raise ValueError(
            "exposure must be 'ict', 'patent', 'investment', 'corr', 'hs_export', "
            "'ict_robustness', 'patent_robustness', 'investment_robustness', "
            "'corr_robustness', or 'hs_export_robustness'"
        )

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

    # Quarter-on-quarter GDP growth lag control: qoq_growth[t] =
    # 100*(ln(gdp[t])-ln(gdp[t-1])), then shifted by 1 quarter so
    # qoq_lag1[t] = qoq_growth[t-1] -- QoQ growth as of ONE quarter
    # before the panel's own reference time t (see fetch_gdp_level()).
    panel["qoq_growth"] = panel.groupby("country")["gdp_level"].transform(
        lambda s: 100 * np.log(s / s.shift(1))
    )
    panel["qoq_lag1"] = panel.groupby("country")["qoq_growth"].shift(1)

    # NOTE: an explicit linear time_trend control (guarding against F's
    # shared upward trend confounding the Shock*F interaction -- see the
    # earlier discussion of this exact risk) used to be added here.
    # Removed: the "*_robustness" specs' growth-literature controls
    # (population_growth, gdp_per_capita, etc. -- see
    # _merge_robustness_controls()) already include time-varying
    # information that substitutes for a generic trend there. The
    # PLAIN specs (ict, patent, investment, corr, hs_export -- without
    # robustness controls) have NO such substitute, so removing
    # time_trend leaves them with no explicit protection against that
    # confound at all; that trade-off was accepted deliberately here,
    # not overlooked.

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

    if exposure in ("corr", "hs_export", "corr_robustness", "hs_export_robustness"):
        # SAME boom/bust regime definition for both specs -- SOX-based
        # is_boom classification via compute_sox_regime(), independent
        # of which exposure variable (SOX correlation, or HS export
        # share) actually feeds F. Only F itself differs between the
        # two specs; the regime split does not.
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

    if exposure.endswith("_robustness"):
        # ------------------------------------------------------------
        # REFEREE ROBUSTNESS CHECK: does b3 (Shock*F, whichever exposure
        # variable F this spec's non-"_robustness" base spec uses)
        # survive once alternative, commonly-cited sources of cross-
        # country heterogeneity in shock sensitivity are ALSO allowed
        # to interact with the shock? The baseline model's identifying
        # assumption is that ALL cross-sectional variation in shock
        # impact comes through F_it -- entity FE absorb only LEVEL
        # differences between countries, not differences in
        # SENSITIVITY to the shock (alpha_i is not itself interacted
        # with Shock_t, so it cannot substitute for this). If some
        # omitted channel is correlated with F_it, b3 could be (partly)
        # picking up that other channel instead of, or in addition to,
        # genuine AI/ICT exposure.
        #
        # This spec adds FOUR additional Shock*Control interactions
        # (plus each control's own main effect) alongside the existing
        # b1 (Shock), b2 (F), b3 (Shock*F) -- and, for corr_robustness/
        # hs_export_robustness, alongside the boom/bust terms (b4/b5/b6)
        # built just above this block:
        #   trade_openness    + shock_x_trade_openness
        #   population_growth + shock_x_population_growth
        #   productivity_growth + shock_x_productivity_growth
        #   gdp_per_capita    + shock_x_gdp_per_capita
        # If the base spec's mitigating-effect coefficient(s) remain
        # similar in sign/magnitude/significance once these are
        # included, that is evidence the exposure's mitigating effect
        # is not simply proxying for one of these more standard
        # growth-literature channels.
        #
        # DELIBERATELY NOT focus-country-interacted (unlike the core
        # shock_used/F/shock_x_exposure trio, and the boom/bust terms
        # where applicable): the purpose here is a robustness check on
        # the base spec's mitigating-effect coefficient(s), not a NL-
        # specific breakdown of four additional channels -- keeping
        # these 8 extra terms non-focus-interacted avoids doubling the
        # parameter count for a question this spec isn't trying to
        # answer, which also speaks to the separate referee concern
        # about degrees of
        # freedom/overfitting from adding too many controls.
        # ------------------------------------------------------------
        for control_name in ["trade_openness", "population_growth",
                              "productivity_growth", "gdp_per_capita"]:
            panel[f"shock_x_{control_name}"] = panel["shock_used"] * panel[control_name]

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

    The dependent variable at horizon h (qoq_cum_lead) is a CUMULATIVE
    log difference from t-1 to t+h -- it telescopes to the sum of h+1
    individual-period QoQ growth rates. This overlapping-window
    construction induces MA(h)-type serial correlation in the residuals
    (the same reasoning as Hodrick (1992) standard errors for
    overlapping-return regressions): at h=0 the induced correlation is
    minimal, but it grows directly with h. A FIXED bandwidth would be
    adequate only for the shortest horizons and UNDERSTATE the true
    serial correlation at longer ones, making those horizons' standard
    errors too narrow/overconfident. max(4, h+1) keeps a floor of 4 for
    short horizons (a reasonable Newey-West-style minimum given the
    panel's overall quarterly sample size) while ensuring the bandwidth
    is always at least as large as the horizon itself.
    """
    return max(4, h + 1)


def _build_regressor_lists(panel):
    """
    Shared logic to derive base_terms/focus_terms/regressors from a
    panel's columns -- factored out of run_local_projections() so the
    bias-correction functions below (split_panel_jackknife(),
    bootstrap_bias_correction()) can build the EXACT SAME regressor set
    without duplicating the has_boom_bust/has_robustness_controls
    detection logic a second time.
    """
    has_boom_bust = "is_boom" in panel.columns
    has_robustness_controls = "shock_x_trade_openness" in panel.columns

    if has_boom_bust:
        base_terms = ["shock_used", "F", "is_boom", "shock_x_exposure", "shock_x_dum",
                      "shock_x_exposure_x_dum"]
        focus_terms = ["shock_used_focus", "F_focus", "is_boom_focus", "shock_x_exposure_focus",
                       "shock_x_dum_focus", "shock_x_exposure_x_dum_focus"]
    else:
        base_terms = ["shock_used", "F", "shock_x_exposure"]
        focus_terms = ["shock_used_focus", "F_focus", "shock_x_exposure_focus"]

    if has_robustness_controls:
        base_terms = base_terms + [
            "trade_openness", "shock_x_trade_openness",
            "population_growth", "shock_x_population_growth",
            "productivity_growth", "shock_x_productivity_growth",
            "gdp_per_capita", "shock_x_gdp_per_capita",
        ]

    return base_terms, focus_terms


def _fit_panel_at_horizon(panel, regressors, h, verbose=True):
    """
    Single-horizon PanelOLS fit, factored out of run_local_projections()
    so the exact same fitting logic (dependent-variable construction,
    dropna, entity-FE, Driscoll-Kraay SEs) can be reused by the bias-
    correction functions below without duplicating it. Returns the
    fitted result, or None if the fit fails outright (e.g. a jackknife
    half-sample or bootstrap resample too small/degenerate to identify
    the model at all) -- callers must handle a None return.

    DEPENDENT VARIABLE: qoq_cum_lead is the CUMULATIVE quarter-on-
    quarter GDP growth from t-1 through t+h -- 100*(ln(gdp[t+h]) -
    ln(gdp[t-1])). By telescoping (sum of one-period log differences),
    this is EXACTLY the sum of the h+1 individual QoQ growth rates from
    t through t+h -- so a single regression of this cumulative quantity
    on shock_used, F, etc. IS the cumulative IRF on QoQ growth directly,
    without needing to separately sum h+1 different regressions'
    coefficients (which would also require deriving the covariance
    BETWEEN those separate regressions' coefficients to get a correct
    cumulative standard error -- this construction sidesteps that
    entirely, since it is already a single regression with its own
    correctly-specified Driscoll-Kraay SE).

    qoq_lag1 -- the lag CONTROL -- is the underlying one-period QoQ
    growth rate itself (not cumulative), one quarter before t:
    100*(ln(gdp[t-1])-ln(gdp[t-2])) -- see build_panel().
    """
    from linearmodels.panel import PanelOLS

    p = panel.copy()
    p["qoq_cum_lead"] = p.groupby("country")["gdp_level"].transform(
        lambda s, h=h: 100 * (np.log(s.shift(-h)) - np.log(s.shift(1)))
    )
    p = p.dropna(subset=["qoq_cum_lead"] + regressors)
    if p.empty or p["entity"].nunique() < 2:
        if verbose:
            print(f"  [!] h={h}: sub-sample has too few observations/entities "
                  f"to fit -- skipping.")
        return None
    p = p.set_index(["entity", "time"])

    exog = p[regressors]
    try:
        mod = PanelOLS(p["qoq_cum_lead"], exog, entity_effects=True, time_effects=False,
                        drop_absorbed=True)
        bw = _dk_bandwidth(h)
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    except Exception as e:
        if verbose:
            print(f"  [!] h={h}: fit failed ({e.__class__.__name__}: {e}) -- skipping.")
        return None
    return res


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
    cross-sectional dependence across the 10 countries (the latter
    matters specifically because shock_used/is_boom/shock_x_dum are
    entity-invariant common shocks, and this spec uses entity-only FE --
    no time effects to otherwise sweep out unmodeled common factors).
    The Bartlett-kernel BANDWIDTH GROWS WITH HORIZON h (see
    _dk_bandwidth()) rather than using a small fixed constant, since the
    cumulative dependent variable's overlapping-window construction
    induces serial correlation that itself grows with h.

    FINITE-SAMPLE BIAS: this entity-FE ("within") estimator, combined
    with qoq_lag1 (a lagged-dependent-variable-style regressor, the
    same one-period QoQ growth rate as the dependent variable's own
    building block, just one quarter before t rather than at t+h) --
    is exactly the classic setup where Nickell (1981)-
    type finite-sample bias arises -- an O(1/T) bias that does NOT
    vanish just because N
    (10 countries) is large; it only shrinks as the TIME dimension T
    grows. This function does NOT itself bias-correct -- see
    split_panel_jackknife() and bootstrap_bias_correction() below,
    applied specifically to the "*_robustness" specs per the referee's
    request, for two different ways of estimating and correcting for
    this bias.
    """
    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()  # PanelOLS needs date-like time index

    base_terms, focus_terms = _build_regressor_lists(panel)

    for col in base_terms + (focus_terms if include_focus_interaction else []):
        check_time_variation(panel, col)

    regressors = base_terms + (focus_terms if include_focus_interaction else []) + \
        ["qoq_lag1"]

    results = {}
    for h in HORIZONS:
        res = _fit_panel_at_horizon(panel, regressors, h)
        if res is None:
            raise RuntimeError(f"run_local_projections: fit failed at h={h} on the FULL "
                                f"sample -- this should not happen; check the panel for "
                                f"structural problems (e.g. a regressor that is all-NaN).")
        results[h] = res
        dropped = [r for r in regressors if r not in res.params.index]
        if dropped:
            print(f"  [!] h={h}: {dropped} were ABSORBED and dropped -- "
                  f"not estimated at this horizon.")
        bw = _dk_bandwidth(h)
        print(f"--- h={h} (Driscoll-Kraay bandwidth={bw}) ---")
        print(res.params)
        print(res.std_errors)
        print()
    return results


BOOTSTRAP_BIAS_CORRECTION = False  # off by default -- computationally
                                    # expensive (see bootstrap_bias_correction()'s
                                    # docstring); set True to enable
N_BOOTSTRAP_REPS = 100             # bootstrap replications per horizon, if enabled


def split_panel_jackknife(panel, include_focus_interaction=True, seed=None):
    """
    Split-panel jackknife bias correction (Dhaene & Jochmans, 2015,
    Review of Economic Studies) for the entity-FE ("within") estimator
    used throughout this script -- specifically targets Nickell
    (1981)-type finite-sample bias, which arises here because
    qoq_lag1 (a lagged-dependent-variable-style regressor) is included
    alongside entity fixed effects. This bias is O(1/T) and does NOT
    shrink just because N (10 countries) is comfortably large -- it
    only vanishes as T (the number of quarters) grows, which is why a
    within-estimator with a lagged dependent variable and a "reasonably
    long but not enormous" T panel like this one (~100 quarters) can
    still carry non-trivial bias even with plenty of cross-sectional
    observations overall.

    METHOD: for each horizon h and each regressor, split the panel's
    TIME dimension into two independent halves (first half of quarters
    vs. second half, split at the median quarter across the WHOLE
    panel -- the same split point for every country, so each half-panel
    is still a balanced, complete country-quarter panel in its own
    right, just spanning T/2 quarters instead of T). Fit the SAME
    specification on: (a) the full panel (beta_full), (b) each half
    separately (beta_half1, beta_half2). If the bias of an estimator
    using T periods behaves as beta_hat(T) ~ beta_true + b/T + o(1/T),
    then beta_hat(T/2) ~ beta_true + 2b/T + o(1/T) -- so the linear
    combination

        beta_JK = 2*beta_full - 0.5*(beta_half1 + beta_half2)

    cancels the leading O(1/T) bias term, leaving a smaller-order
    remainder. This requires NO resampling/randomness (hence the
    unused `seed` argument is accepted only for a consistent call
    signature with bootstrap_bias_correction() below, not used here) and
    costs only 3 total fits per horizon (vs. hundreds for a bootstrap),
    making it cheap enough to run by default on every "*_robustness"
    spec.

    LIMITATION, stated plainly: this function reports beta_JK (the
    bias-corrected point estimate) but does NOT derive a separate,
    theoretically-correct jackknife standard error -- it reports the
    ORIGINAL full-sample Driscoll-Kraay SE alongside beta_JK, which is
    a standard, common simplification in applied bias-correction work
    but is not itself bias-corrected. Treat the reported "t-stat" for
    beta_JK (beta_JK / se_full) as an approximation, not a fully
    rigorous inferential statistic -- what this function is designed to
    answer well is "does the POINT ESTIMATE move much once the leading
    finite-sample bias term is removed", not "here is a fully corrected
    confidence interval."

    Returns a dict: {h: {term: {"beta_full":..., "beta_half1":...,
    "beta_half2":..., "beta_jk":..., "se_full":...}}}.
    """
    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()

    base_terms, focus_terms = _build_regressor_lists(panel)
    regressors = base_terms + (focus_terms if include_focus_interaction else []) + \
        ["qoq_lag1"]

    quarters_sorted = sorted(panel["quarter"].unique())
    median_quarter = quarters_sorted[len(quarters_sorted) // 2]
    panel_h1 = panel[panel["quarter"] < median_quarter]
    panel_h2 = panel[panel["quarter"] >= median_quarter]
    print(f"  [jackknife] time split at {median_quarter}: "
          f"half1={panel_h1['quarter'].nunique()} quarters, "
          f"half2={panel_h2['quarter'].nunique()} quarters.")

    out = {}
    for h in HORIZONS:
        res_full = _fit_panel_at_horizon(panel, regressors, h, verbose=False)
        res_h1 = _fit_panel_at_horizon(panel_h1, regressors, h, verbose=False)
        res_h2 = _fit_panel_at_horizon(panel_h2, regressors, h, verbose=False)

        out[h] = {}
        for term in regressors:
            beta_full = res_full.params[term] if (res_full is not None and term in res_full.params.index) else np.nan
            se_full = res_full.std_errors[term] if (res_full is not None and term in res_full.params.index) else np.nan
            beta_h1 = res_h1.params[term] if (res_h1 is not None and term in res_h1.params.index) else np.nan
            beta_h2 = res_h2.params[term] if (res_h2 is not None and term in res_h2.params.index) else np.nan

            if np.isnan(beta_full) or np.isnan(beta_h1) or np.isnan(beta_h2):
                beta_jk = np.nan
            else:
                beta_jk = 2 * beta_full - 0.5 * (beta_h1 + beta_h2)

            out[h][term] = {"beta_full": beta_full, "beta_half1": beta_h1,
                            "beta_half2": beta_h2, "beta_jk": beta_jk, "se_full": se_full}
    return out


def bootstrap_bias_correction(panel, include_focus_interaction=True, n_reps=N_BOOTSTRAP_REPS,
                               seed=42):
    """
    Cluster (pairs) bootstrap bias correction: an alternative to
    split_panel_jackknife() above for the SAME finite-sample-bias
    concern (see that function's docstring for the Nickell/qoq_lag1
    background). Resamples ENTITIES (countries) WITH REPLACEMENT --
    not individual observations -- since resampling observations would
    break each country's own serial-correlation structure, which the
    Driscoll-Kraay SEs already assume is present; resampling whole
    country-panels at a time ("pairs"/"cluster" bootstrap) respects
    that structure, consistent with treating the 10 countries as the
    natural resampling unit.

    Each bootstrap draw gets a NEW, unique entity id per country pulled
    (e.g. "NL_0", "NL_1" if NL happens to be drawn twice in the same
    resample) so PanelOLS never sees a duplicate (entity, time) index,
    which would otherwise raise an error.

    beta_BC = 2*beta_full - mean(beta_b for b in 1..n_reps)
    -- the same "double the full-sample estimate, subtract the
    resampling-based estimate" logic as the jackknife above, just using
    a many-draw bootstrap mean in place of the two-way split average.

    COMPUTATIONAL COST: n_reps refits PER HORIZON (default 100), across
    9 horizons and (if applied to every "*_robustness" spec) 5 specs =
    up to 4,500 total PanelOLS fits for a single run at the default
    n_reps. This is why BOOTSTRAP_BIAS_CORRECTION defaults to False --
    turn it on deliberately, and consider lowering n_reps for a faster,
    less precise run, or raising it for a slower, more precise one.

    Returns a dict in the SAME shape as split_panel_jackknife():
    {h: {term: {"beta_full":..., "beta_bootstrap_mean":..., "beta_bc":...,
    "se_full":..., "n_successful_reps":...}}}.
    """
    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()

    base_terms, focus_terms = _build_regressor_lists(panel)
    regressors = base_terms + (focus_terms if include_focus_interaction else []) + \
        ["qoq_lag1"]

    rng = np.random.default_rng(seed)
    countries = panel["country"].unique()

    print(f"  [bootstrap] {n_reps} reps/horizon x {len(HORIZONS)} horizons "
          f"= {n_reps * len(HORIZONS)} total fits -- this may take a while.")

    out = {}
    for h in HORIZONS:
        res_full = _fit_panel_at_horizon(panel, regressors, h, verbose=False)
        boot_betas = {term: [] for term in regressors}

        for b in range(n_reps):
            sampled_countries = rng.choice(countries, size=len(countries), replace=True)
            pieces = []
            for i, c in enumerate(sampled_countries):
                piece = panel[panel["country"] == c].copy()
                piece["entity"] = f"{c}_{i}"  # unique per draw -- avoids duplicate (entity,time)
                pieces.append(piece)
            panel_b = pd.concat(pieces, ignore_index=True)
            res_b = _fit_panel_at_horizon(panel_b, regressors, h, verbose=False)
            if res_b is not None:
                for term in regressors:
                    if term in res_b.params.index:
                        boot_betas[term].append(res_b.params[term])

        out[h] = {}
        for term in regressors:
            beta_full = res_full.params[term] if (res_full is not None and term in res_full.params.index) else np.nan
            se_full = res_full.std_errors[term] if (res_full is not None and term in res_full.params.index) else np.nan
            draws = boot_betas[term]
            if not draws or np.isnan(beta_full):
                beta_mean, beta_bc = np.nan, np.nan
            else:
                beta_mean = float(np.mean(draws))
                beta_bc = 2 * beta_full - beta_mean
            out[h][term] = {"beta_full": beta_full, "beta_bootstrap_mean": beta_mean,
                            "beta_bc": beta_bc, "se_full": se_full,
                            "n_successful_reps": len(draws)}
        print(f"  [bootstrap] h={h}: done ({sum(len(v) for v in boot_betas.values()) // max(len(regressors),1)} "
              f"successful reps on average across terms).")
    return out



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


def summarize_jackknife(jk_dict):
    """
    Formats split_panel_jackknife()'s output into a long-format
    DataFrame (one row per horizon x term) -- LONG rather than the wide
    per-term-column layout summarize_irf() uses, since this is a
    fundamentally different kind of table (a diagnostic comparison of
    four numbers per term: the full-sample estimate, each half-sample
    estimate, and the bias-corrected estimate) rather than one main
    coefficient table.

    Adds bias_estimate (beta_full - beta_jk, i.e. how much the point
    estimate MOVED once the leading finite-sample bias term was
    removed) and pct_bias_of_se (bias_estimate / se_full, a rough sense
    of whether the correction is large or small RELATIVE to sampling
    noise -- a bias_estimate that is small relative to se_full suggests
    finite-sample bias is not the dominant concern for that term at
    that horizon, even if it's not exactly zero).
    """
    rows = []
    for h, terms in jk_dict.items():
        for term, vals in terms.items():
            bias_estimate = vals["beta_full"] - vals["beta_jk"]
            pct_bias_of_se = (bias_estimate / vals["se_full"]
                              if vals["se_full"] and not np.isnan(vals["se_full"]) and vals["se_full"] != 0
                              else np.nan)
            rows.append({
                "h": h, "term": term,
                "beta_full": vals["beta_full"], "beta_half1": vals["beta_half1"],
                "beta_half2": vals["beta_half2"], "beta_jk": vals["beta_jk"],
                "se_full": vals["se_full"], "bias_estimate": bias_estimate,
                "bias_relative_to_se": pct_bias_of_se,
            })
    return pd.DataFrame(rows)


def summarize_bootstrap(boot_dict):
    """
    Formats bootstrap_bias_correction()'s output into the SAME long-
    format layout as summarize_jackknife(), for direct side-by-side
    comparison of the two bias-correction methods on the same spec.
    """
    rows = []
    for h, terms in boot_dict.items():
        for term, vals in terms.items():
            bias_estimate = vals["beta_full"] - vals["beta_bc"]
            pct_bias_of_se = (bias_estimate / vals["se_full"]
                              if vals["se_full"] and not np.isnan(vals["se_full"]) and vals["se_full"] != 0
                              else np.nan)
            rows.append({
                "h": h, "term": term,
                "beta_full": vals["beta_full"], "beta_bootstrap_mean": vals["beta_bootstrap_mean"],
                "beta_bc": vals["beta_bc"], "se_full": vals["se_full"],
                "bias_estimate": bias_estimate, "bias_relative_to_se": pct_bias_of_se,
                "n_successful_reps": vals["n_successful_reps"],
            })
    return pd.DataFrame(rows)


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
    has_robustness_controls = any("shock_x_trade_openness" in r.params.index for r in results.values())

    # COMPOSABLE design mirroring run_local_projections() -- see that
    # function's comment for why has_boom_bust and has_robustness_controls
    # are independent axes, not a flat 3/4-way choice.
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

    if has_robustness_controls:
        # REFEREE ROBUSTNESS CHECK terms -- see build_panel()'s
        # "*_robustness" branches. Labels here are DELIBERATELY
        # base-count-independent (e.g. "ctrl_trade_openness", not
        # "b4_trade_openness"/"b7_trade_openness" depending on whether
        # 3 or 6 core terms precede them) -- so the same control's
        # coefficient is reported under the SAME column name whether
        # it's attached to ict_robustness (3 core terms before it) or
        # corr_robustness/hs_export_robustness (6 core terms before
        # it), making the four robustness variants directly comparable
        # column-for-column. Focus terms for these 8 extra control
        # terms (e.g. "trade_openness_focus") are NOT real panel
        # columns (deliberately not focus-interacted) -- passing their
        # placeholder names here is safe: _summarize_single_term()
        # already handles a focus_term that never appears in the
        # fitted results gracefully (reports NaN / is_absorbed=True),
        # which correctly communicates "no NL-specific breakdown for
        # this term" rather than requiring special-case code here.
        term_label_triples = term_label_triples + [
            ("trade_openness", "trade_openness_focus", "ctrl_trade_openness"),
            ("shock_x_trade_openness", "shock_x_trade_openness_focus", "ctrl_shock_x_trade_openness"),
            ("population_growth", "population_growth_focus", "ctrl_population_growth"),
            ("shock_x_population_growth", "shock_x_population_growth_focus", "ctrl_shock_x_population_growth"),
            ("productivity_growth", "productivity_growth_focus", "ctrl_productivity_growth"),
            ("shock_x_productivity_growth", "shock_x_productivity_growth_focus", "ctrl_shock_x_productivity_growth"),
            ("gdp_per_capita", "gdp_per_capita_focus", "ctrl_gdp_per_capita"),
            ("shock_x_gdp_per_capita", "shock_x_gdp_per_capita_focus", "ctrl_shock_x_gdp_per_capita"),
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
    The "raw" GDP response to the chosen global shock (WUI, GPR, EPU, TPU,
    or GSCPI, per SHOCK_VARIABLE), with the exposure/mitigation channel switched off
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

    ROBUSTNESS CONTROLS: if `panel` is build_baseline_robustness_panel()'s
    output (detected via the presence of trade_openness) rather than
    the plain baseline panel, the four growth-literature control
    variables are automatically included as ADDITIVE controls alongside
    shock_used/qoq_lag1 -- NO shock interactions are added here (unlike
    the exposure specs' own "*_robustness" checks) -- testing whether
    the baseline shock's average, unconditional effect survives once
    these commonly-cited country characteristics are also controlled
    for, not whether the shock's sensitivity varies with them.
    """
    from linearmodels.panel import PanelOLS

    panel = panel.copy()
    panel["entity"] = panel["country"]
    panel["time"] = panel["quarter"].dt.to_timestamp()

    regressors = ["shock_used", "qoq_lag1"]
    if "trade_openness" in panel.columns:
        regressors = regressors + [
            "trade_openness", "population_growth",
            "productivity_growth", "gdp_per_capita",
        ]

    results = {}
    for h in HORIZONS:
        p = panel.copy()
        # Cumulative IRF -- same construction as run_local_projections();
        # see that function's comment for why this is preferred over
        # summing point estimates from separate per-period regressions.
        p["qoq_cum_lead"] = p.groupby("country")["gdp_level"].transform(
            lambda s, h=h: 100 * (np.log(s.shift(-h)) - np.log(s.shift(1)))
        )
        p = p.dropna(subset=["qoq_cum_lead"] + regressors)
        p = p.set_index(["entity", "time"])

        exog = p[regressors]
        mod = PanelOLS(p["qoq_cum_lead"], exog, entity_effects=True, time_effects=False,
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
    a one-unit increase in the chosen shock (WUI, GPR, EPU, TPU, GSCPI, or PC,
    per SHOCK_VARIABLE) at horizon h, with exposure/mitigation switched off
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
    Plots the unconditional, CUMULATIVE shock IRF (WUI, GPR, EPU, TPU, or
    GSCPI, per SHOCK_VARIABLE) with a 90% CI band, so you can see directly whether
    the cumulative response is statistically significant at each
    horizon (the band crossing zero means "not significant at that h").
    beta_shock_baseline is already a cumulative coefficient on QoQ GDP
    growth -- see
    run_baseline_shock_projections() and _fit_panel_at_horizon()'s
    docstring for the exact dependent-variable definition (qoq_cum_lead).

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
    ax.set_ylabel(f"Cumulative QoQ GDP growth response per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 6. IRF plots
# ----------------------------------------------------------------------

def plot_irf(irf_df, title, focus_country=FOCUS_COUNTRY, shock_std=1.0):
    """
    Plots the panel-average and NL-specific b3 (Shock*F
    interaction) coefficient across horizons, each with a 90% CI band
    (+/- 1.645 SE). This is the ICT spec's mitigating-effect coefficient,
    net of the shock and F main effects (see run_local_projections()),
    on cumulative QoQ GDP growth (0..h).
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
    ax.set_ylabel(f"Cumulative Shock x F interaction effect on QoQ GDP growth per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bias_correction_irf(jk_table, term, title, shock_std=1.0, boot_table=None):
    """
    Plots the ORIGINAL (full-sample) coefficient against its split-panel-
    jackknife bias-corrected counterpart (and, if boot_table is given,
    its cluster-bootstrap bias-corrected counterpart too) across
    horizons, for ONE regressor `term` -- letting the referee-requested
    bias-correction methods be inspected visually, not just as a table.

    jk_table:   long-format DataFrame from summarize_jackknife() (columns
                h, term, beta_full, beta_half1, beta_half2, beta_jk, se_full, ...)
    boot_table: optional, same long format from summarize_bootstrap()
                (columns h, term, beta_full, beta_bootstrap_mean, beta_bc, ...)
                -- only plotted if BOOTSTRAP_BIAS_CORRECTION was enabled.

    The ORIGINAL series is plotted with a 90% CI band (+/- 1.645 *
    se_full, the only standard error available here -- see
    split_panel_jackknife()'s docstring for why no separate, bias-
    corrected SE is derived); the bias-corrected series are plotted as
    plain lines without a band, since no bias-corrected SE is computed
    for them either (this chart is about whether the POINT ESTIMATE
    moves once finite-sample bias is addressed, not a fully corrected
    confidence interval -- see the same docstring).

    shock_std: same rescaling convention as plot_irf() -- "per 1 raw
    unit of shock_used" into "per 1-stdev {SHOCK_VARIABLE} shock".
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    sub = jk_table[jk_table["term"] == term].sort_values("h")
    h = sub["h"]
    b_full = sub["beta_full"] * shock_std
    se_full = sub["se_full"] * shock_std
    b_jk = sub["beta_jk"] * shock_std

    ax.plot(h, b_full, marker="o", label="Original (full sample)", color="#1f77b4")
    ax.fill_between(h, b_full - 1.645 * se_full, b_full + 1.645 * se_full,
                     alpha=0.2, color="#1f77b4")
    ax.plot(h, b_jk, marker="^", label="Split-panel jackknife bias-corrected",
            color="#ff7f0e", linestyle="--")

    if boot_table is not None:
        sub_b = boot_table[boot_table["term"] == term].sort_values("h")
        if not sub_b.empty:
            b_bc = sub_b["beta_bc"] * shock_std
            ax.plot(sub_b["h"], b_bc, marker="s", label="Cluster-bootstrap bias-corrected",
                    color="#2ca02c", linestyle=":")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel(f"Cumulative {term} coefficient on QoQ GDP growth per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
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
    ax.set_ylabel(f"Cumulative Shock x F interaction effect on QoQ GDP growth per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
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
    ax.set_ylabel(f"Cumulative mitigating effect on QoQ GDP growth per 1-stdev {SHOCK_VARIABLE} shock (0..h)")
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
    "qoq_cum_lead": "The regression's DEPENDENT VARIABLE: CUMULATIVE quarter-on-quarter GDP growth from t-1 through t+h -- 100*(ln(gdp[t+h])-ln(gdp[t-1])). By telescoping, this equals the sum of the h+1 individual QoQ growth rates from t through t+h, so this single regression's coefficients ARE the cumulative IRF on QoQ growth directly.",
    "qoq_lag1": "Quarter-on-quarter GDP growth (100*(ln(gdp[t-1])-ln(gdp[t-2]))), lagged so it is measured ONE quarter before the panel's own reference time t -- a GENUINE lagged-dependent-variable-style regressor (the same one-period QoQ growth rate that qoq_cum_lead is built from, just at t-1 instead of t+h). The ONLY control used throughout the plain specs (ICT, Corr, and baseline alike); a second lag is not used. (A separate explicit time_trend control was used here previously; removed -- see build_panel()'s comment at the point it used to be added.)",
    "shock_level": f"Global {SHOCK_VARIABLE} (RAW LEVEL), quarterly -- kept in the panel for reference, but NOT what the model estimates on (see shock_used). See _SHOCK_CITATION for the source paper.",
    "shock_used": f"The {SHOCK_VARIABLE} series actually used in the model: shock_innov, the AR(2)-purified {SHOCK_VARIABLE} shock (see purify_shock()) -- NOT the raw level. Every coefficient, IRF, and chart in this workbook is based on this purified shock.",
    "z_raw":    "Raw AI-exposure state variable before standardization -- ICT investment share of GFCF (Spec A/ict), AI patent applications (Spec A/patent), AI incoming investment counts (Spec A/investment), rolling correlation of the national equity index with the semiconductor index (Spec B/corr), or AI/ICT-related HS export share (Spec C/hs_export, see fetch_hs_export()), depending on which build_panel(exposure=...) produced this panel",
    "ai_patents": "Annual AI-related patent applications per country (ETO/CSET Country Activity Tracker, Patent dataset -- see fetch_ai_patents()). Used as the exposure proxy (z_raw) for the 'patent' spec.",
    "ai_investment": "Annual AI-related INCOMING investment counts per country -- the number of inbound investment deals into AI-related companies, NOT their dollar value (ETO/CSET Country Activity Tracker, Investment dataset -- see fetch_ai_investment()). Used as the exposure proxy (z_raw) for the 'investment' spec.",
    "z":        "Standardized z_raw (STANDARDIZE_MODE='pooled' across the whole panel by default)",
    "F":        "Logistic transform of z: F = 1 / (1 + exp(-THETA * z)), the smooth 0-1 AI-exposure state weight",
    "shock_x_exposure": f"b3 (ICT spec) / b4 (Corr and hs_export specs) interaction regressor: F * shock_used. Its coefficient is the mitigating effect NET OF the {SHOCK_VARIABLE} and F main effects (shock_used, F), which are now separately included in the regression -- not a 'difference' spec.",
    "is_focus": f"1 if country == FOCUS_COUNTRY ({FOCUS_COUNTRY}), 0 otherwise",
    "shock_used_focus": f"is_focus * shock_used -- NL's own b1 ({SHOCK_VARIABLE} main effect) interaction term",
    "F_focus": "is_focus * F -- NL's own b2 (F main effect) interaction term",
    "shock_x_exposure_focus": "is_focus * shock_x_exposure -- NL's own interaction-term deviation (b3 for ICT spec, b4 for Corr and hs_export specs)",
    "is_boom": "1 if the trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) is ABOVE THE MEDIAN of that series over the sample (AI-capex boom, roughly the top half of quarters by this measure), 0 otherwise (bust, roughly the bottom half, the reference level Dum=0) -- median-thresholded so boom and bust each cover close to half the sample by construction, rather than a literal zero threshold (which would skew toward 'boom' given equity indices trend upward over most multi-year windows). Uses its own (shorter) rolling window than the exposure-correlation measure -- see compute_sox_regime() and SOX_REGIME_WINDOW_QUARTERS. Entity-invariant (same for all countries in a given quarter). Also used directly as regressor b3 (Dum) in the moderated regression. Corr and hs_export specs (SAME is_boom classification for both -- see build_panel()).",
    "is_bust": "1 - is_boom. Kept for reference/export only -- NOT used as a regressor (would be perfectly collinear with is_boom + the intercept). Corr and hs_export specs.",
    "is_boom_focus": "is_focus * is_boom -- NL's own b3 (Dum main effect) interaction term. Corr and hs_export specs.",
    "sox_roll_mean_ret": "Trailing SOX_REGIME_WINDOW_QUARTERS-quarter average log return of the semiconductor index (^SOX) -- the continuous series is_boom is thresholded from",
    "shock_x_dum": f"b5 (Corr and hs_export specs): shock_used * is_boom ({SHOCK_VARIABLE} x Dum interaction). Entity-invariant like shock_used and is_boom themselves -- estimable only because this spec uses entity-only fixed effects (see run_local_projections()).",
    "shock_x_dum_focus": "is_focus * shock_x_dum -- NL's own b5 interaction term. Corr and hs_export specs.",
    "shock_x_exposure_x_dum": f"b6 (Corr and hs_export specs): shock_x_exposure * is_boom ({SHOCK_VARIABLE} x F x Dum, the full three-way interaction). Corr and hs_export specs.",
    "shock_x_exposure_x_dum_focus": "is_focus * shock_x_exposure_x_dum -- NL's own b6 interaction term. Corr and hs_export specs.",
    "h": "Local-projection horizon, in quarters",
    "beta_b1_shock_panelavg": f"Coefficient on shock_used (b1, the {SHOCK_VARIABLE} main effect) over horizons 0..h, on cumulative QoQ GDP growth. ICT, Corr, and hs_export specs.",
    "se_b1_shock_panelavg": "Standard error of beta_b1_shock_panelavg (Driscoll-Kraay-style)",
    "beta_b2_F_panelavg": "Coefficient on F (b2, the exposure-weight main effect) over horizons 0..h, on cumulative QoQ GDP growth. ICT, Corr, and hs_export specs.",
    "se_b2_F_panelavg": "Standard error of beta_b2_F_panelavg",
    "beta_b3_shock_x_F_panelavg": f"Coefficient on shock_x_exposure (b3, ICT spec's {SHOCK_VARIABLE}*F interaction) over horizons 0..h, on cumulative QoQ GDP growth -- the ICT spec's mitigating-effect coefficient, net of the {SHOCK_VARIABLE} and F main effects. A POSITIVE value means exposure dampens/offsets the (typically negative) {SHOCK_VARIABLE} shock response; negative means amplification.",
    "se_b3_shock_x_F_panelavg": "Standard error of beta_b3_shock_x_F_panelavg",
    f"beta_b1_shock_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b1 ({SHOCK_VARIABLE} main effect) coefficient (panel-average + deviation)",
    f"se_b1_shock_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b1 coefficient",
    f"beta_b2_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b2 (F main effect) coefficient (panel-average + deviation)",
    f"se_b2_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b2 coefficient",
    f"beta_b3_shock_x_F_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s total b3 (ICT spec's {SHOCK_VARIABLE}*F interaction) coefficient (panel-average + deviation) -- {FOCUS_COUNTRY}'s own mitigating-effect coefficient",
    f"se_b3_shock_x_F_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s total b3 coefficient",
    "beta_b3_dum_panelavg": "Coefficient on is_boom (b3, the Dum/regime main effect) over horizons 0..h, on cumulative QoQ GDP growth. Corr and hs_export specs.",
    "se_b3_dum_panelavg": "Standard error of beta_b3_dum_panelavg",
    "beta_b4_shock_x_F_panelavg": f"Coefficient on shock_x_exposure (b4, Corr/hs_export specs' {SHOCK_VARIABLE}*F interaction) over horizons 0..h, on cumulative QoQ GDP growth -- this is ALSO the mitigating effect in the reference regime (Dum=0, 'bust'); see mitigating_effect_bust_panelavg, which equals this exactly.",
    "se_b4_shock_x_F_panelavg": "Standard error of beta_b4_shock_x_F_panelavg",
    "beta_b5_shock_x_dum_panelavg": f"Coefficient on shock_x_dum (b5, {SHOCK_VARIABLE}*Dum interaction) over horizons 0..h, on cumulative QoQ GDP growth. Corr and hs_export specs.",
    "se_b5_shock_x_dum_panelavg": "Standard error of beta_b5_shock_x_dum_panelavg",
    "beta_b6_shock_x_F_x_dum_panelavg": f"Coefficient on shock_x_exposure_x_dum (b6, the full three-way {SHOCK_VARIABLE}*F*Dum interaction) over horizons 0..h, on cumulative QoQ GDP growth -- this is the ADDITIONAL mitigating effect specific to the boom regime, on top of b4. Corr and hs_export specs.",
    "se_b6_shock_x_F_x_dum_panelavg": "Standard error of beta_b6_shock_x_F_x_dum_panelavg",
    "mitigating_effect_bust_panelavg": f"The mitigating effect of {SHOCK_VARIABLE}*F in the BUST regime (Dum=0, the reference level) = b4 alone. Expected NEGATIVE if shared/concentrated exposure to the AI-chip cycle amplifies the (typically negative) {SHOCK_VARIABLE} shock response during downturns.",
    "se_mitigating_effect_bust_panelavg": "Standard error of mitigating_effect_bust_panelavg (= se_b4_shock_x_F_panelavg exactly, since the bust effect IS b4)",
    "mitigating_effect_boom_panelavg": f"The mitigating effect of {SHOCK_VARIABLE}*F in the BOOM regime (Dum=1) = b4 + b6, with a correctly derived COMBINED standard error (via the full coefficient covariance matrix, NOT b4's and b6's individual SEs summed naively). Expected POSITIVE if the demand/terms-of-trade tailwind dampens the {SHOCK_VARIABLE} shock response during AI-capex booms.",
    "se_mitigating_effect_boom_panelavg": "Correctly derived combined standard error of mitigating_effect_boom_panelavg (b4+b6)",
    f"mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the bust regime (panel-average b4 + {FOCUS_COUNTRY}'s b4 deviation)",
    f"se_mitigating_effect_bust_{FOCUS_COUNTRY}_total": f"Standard error of {FOCUS_COUNTRY}'s bust mitigating effect",
    f"mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"{FOCUS_COUNTRY}'s own mitigating effect in the boom regime (panel-average b4+b6, plus {FOCUS_COUNTRY}'s own b4 and b6 deviations, all four terms combined with a correctly derived SE)",
    f"se_mitigating_effect_boom_{FOCUS_COUNTRY}_total": f"Correctly derived combined standard error of {FOCUS_COUNTRY}'s boom mitigating effect (4-term combination)",
    "beta_shock_baseline": f"Cumulative QoQ GDP growth response, 0..h, to a one-unit {SHOCK_VARIABLE} shock increase, WITHOUT the exposure/mitigation interaction (entity FE only, no time FE -- see run_baseline_shock_projections)",
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


def export_results_to_excel(panels: dict, irfs: dict, figs: dict, raw_data: dict, filepath: str,
                             robustness_figs: dict = None, bias_correction_figs: dict = None):
    """
    Single workbook with everything: one sheet per spec's panel data, one
    sheet per spec's IRF table, one sheet per raw source series, a
    Variable_definitions sheet, an IRF_baseline sheet with the main
    model's embedded charts, an IRF_robust_var sheet with the referee-
    requested robustness-check charts (EXCLUDING any bias-correction
    variants), and (if bias_correction_figs is given) a SEPARATE
    IRF_robust_bias sheet with the split-panel-jackknife and (if
    enabled) cluster-bootstrap bias-corrected robustness charts --
    each sheet positioned in that order, immediately after the
    previous one (charts embedded directly from memory -- no PNG files
    are ever written to disk).

    panels:               {spec_name: panel DataFrame from build_panel()}
    irfs:                 {spec_name: IRF DataFrame from summarize_irf()}
    figs:                 {spec_name: matplotlib Figure from plot_irf()} --
                           MAIN model charts, go on the IRF_baseline sheet
    raw_data:              {sheet_name: raw source DataFrame, e.g. "Raw_GDP": ...}
    robustness_figs:       {spec_name: matplotlib Figure} -- ROBUSTNESS-CHECK
                           charts, EXCLUDING bias correction, go on their own
                           IRF_robust_var sheet. Optional; omit or pass {} if none.
    bias_correction_figs:  {spec_name: matplotlib Figure} -- split-panel-
                           jackknife (and, if enabled, cluster-bootstrap)
                           bias-corrected robustness charts, go on their
                           own IRF_robust_bias sheet, IN ADDITION TO (not
                           instead of) the plain robustness charts above.
                           Optional; omit or pass {} if none.
    """
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from PIL import Image as PILImage

    robustness_figs = robustness_figs or {}
    bias_correction_figs = bias_correction_figs or {}

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

    # Add the IRF plot sheet(s) with embedded images (openpyxl can't embed
    # images via pandas' ExcelWriter context, so reopen the workbook).
    # Images are rendered to an in-memory buffer and embedded directly --
    # no PNG files are ever written to disk.
    #
    # INSERTED AT THE FRONT (index=0, 1, 2 in turn) so the IRF chart
    # sheets appear BEFORE all the data/IRF-table/Raw_* sheets written
    # above -- per the explicit request that chart sheets come first in
    # the workbook, not after the data.
    wb = openpyxl.load_workbook(filepath)

    def _add_plot_sheet(sheet_name, fig_dict, index):
        ws = wb.create_sheet(sheet_name, index)
        row_cursor = 1
        for spec_name, fig in fig_dict.items():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = XLImage(PILImage.open(buf))
            ws.add_image(img, f"A{row_cursor}")
            row_cursor += int(img.height / 15) + 2  # rough row spacing so plots don't overlap

    next_index = 0
    _add_plot_sheet("IRF_baseline", figs, next_index)
    next_index += 1
    if robustness_figs:
        _add_plot_sheet("IRF_robust_var", robustness_figs, next_index)
        next_index += 1
    if bias_correction_figs:
        _add_plot_sheet("IRF_robust_bias", bias_correction_figs, next_index)
        next_index += 1

    wb.save(filepath)

    n_robustness = f" + {len(robustness_figs)} robustness plots" if robustness_figs else ""
    n_bias = f" + {len(bias_correction_figs)} bias-correction plots" if bias_correction_figs else ""
    print(f"Saved all results (data + IRF tables + raw series + "
          f"{len(figs)} main plots{n_robustness}{n_bias}) to '{filepath}'.")



# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(f"SHOCK_VARIABLE = {SHOCK_VARIABLE!r} -- every IRF, coefficient, and")
    print(f"chart below is based on this shock series. Change SHOCK_VARIABLE")
    print(f"near the top of the CONFIG section (currently one of 'WUI', 'GPR',")
    print(f"'EPU', 'TPU', 'GSCPI') to switch the whole script to a different one.")
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
        "Spec B: SOX_index exposure -- BOOM regime mitigating effect (b4+b6)",
        color="#2ca02c", color_nl="#98df8a", shock_std=shock_std,
    )
    fig_corr_bust = plot_regime_irf(
        irf_corr, "mitigating_effect_bust",
        "Spec B: SOX_index exposure -- BUST regime mitigating effect (b4)",
        color="#d62728", color_nl="#ff9896", shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("SPEC C: AI/ICT-related HS export share (HS 847150+847180+847330")
    print("        +848610+848620+848630+848640+848690")
    print(f"        as a share of total merchandise exports) as exposure, global {SHOCK_VARIABLE}")
    print("        -- SAME boom/bust regime split as Spec B (see build_panel() docstring:")
    print("        the is_boom classification is identical, only F itself differs)")
    print("=" * 70)
    panel_hs_export = build_panel(exposure="hs_export")
    res_hs_export = run_local_projections(panel_hs_export)
    irf_hs_export = summarize_irf(res_hs_export)
    print(irf_hs_export.to_string())
    fig_hs_export_boom = plot_regime_irf(
        irf_hs_export, "mitigating_effect_boom",
        "Spec C: AI_ICT export share -- BOOM regime mitigating effect (b4+b6)",
        color="#2ca02c", color_nl="#98df8a", shock_std=shock_std,
    )
    fig_hs_export_bust = plot_regime_irf(
        irf_hs_export, "mitigating_effect_bust",
        "Spec C: AI_ICT export share -- BUST regime mitigating effect (b4)",
        color="#d62728", color_nl="#ff9896", shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK (referee-requested): does the ICT exposure")
    print("mitigating effect (b3, Shock*F_ict) survive once four growth-")
    print("literature control variables -- trade openness, population")
    print("growth, productivity growth, GDP per capita -- and their OWN")
    print("interactions with the shock are also included? Tests whether")
    print("b3 is proxying for one of these more standard channels of")
    print("cross-country heterogeneity in shock sensitivity, rather than")
    print("genuinely reflecting AI/ICT exposure -- see build_panel()'s")
    print("'ict_robustness' docstring for the full specification.")
    print("=" * 70)
    panel_ict_robustness = build_panel(exposure="ict_robustness")
    res_ict_robustness = run_local_projections(panel_ict_robustness)
    irf_ict_robustness = summarize_irf(res_ict_robustness)
    print(irf_ict_robustness.to_string())
    fig_ict_robustness = plot_irf(
        irf_ict_robustness,
        "Robustness check: ICT exposure with growth-literature controls",
        shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK: Spec A (AI patents vs AI investment), same four")
    print("growth-literature controls + their shock-interactions as above.")
    print("=" * 70)
    panel_patent_robustness = build_panel(exposure="patent_robustness")
    res_patent_robustness = run_local_projections(panel_patent_robustness)
    irf_patent_robustness = summarize_irf(res_patent_robustness)
    print(irf_patent_robustness.to_string())

    panel_investment_robustness = build_panel(exposure="investment_robustness")
    res_investment_robustness = run_local_projections(panel_investment_robustness)
    irf_investment_robustness = summarize_irf(res_investment_robustness)
    print(irf_investment_robustness.to_string())

    fig_combined_robustness = plot_combined_mitigating_irf(
        irf_patent_robustness, irf_investment_robustness, shock_std=shock_std,
        title="Robustness check: AI patent vs AI investment, with growth-literature controls",
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK: Spec B (SOX_index exposure), same four growth-")
    print("literature controls + their shock-interactions, on top of the")
    print("SAME boom/bust regime split as the main Spec B model.")
    print("=" * 70)
    panel_corr_robustness = build_panel(exposure="corr_robustness")
    res_corr_robustness = run_local_projections(panel_corr_robustness)
    irf_corr_robustness = summarize_irf(res_corr_robustness)
    print(irf_corr_robustness.to_string())
    fig_corr_robustness_boom = plot_regime_irf(
        irf_corr_robustness, "mitigating_effect_boom",
        "Robustness check: SOX_index exposure -- BOOM (with growth-literature controls)",
        color="#2ca02c", color_nl="#98df8a", shock_std=shock_std,
    )
    fig_corr_robustness_bust = plot_regime_irf(
        irf_corr_robustness, "mitigating_effect_bust",
        "Robustness check: SOX_index exposure -- BUST (with growth-literature controls)",
        color="#d62728", color_nl="#ff9896", shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK: Spec C (AI_ICT export share), same four growth-")
    print("literature controls + their shock-interactions, on top of the")
    print("SAME boom/bust regime split as the main Spec C model.")
    print("=" * 70)
    panel_hs_export_robustness = build_panel(exposure="hs_export_robustness")
    res_hs_export_robustness = run_local_projections(panel_hs_export_robustness)
    irf_hs_export_robustness = summarize_irf(res_hs_export_robustness)
    print(irf_hs_export_robustness.to_string())
    fig_hs_export_robustness_boom = plot_regime_irf(
        irf_hs_export_robustness, "mitigating_effect_boom",
        "Robustness check: AI_ICT export share -- BOOM (with growth-literature controls)",
        color="#2ca02c", color_nl="#98df8a", shock_std=shock_std,
    )
    fig_hs_export_robustness_bust = plot_regime_irf(
        irf_hs_export_robustness, "mitigating_effect_bust",
        "Robustness check: AI_ICT export share -- BUST (with growth-literature controls)",
        color="#d62728", color_nl="#ff9896", shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK: baseline AR(1) shock model, extended with the same")
    print("four growth-literature control variables (TradeOpen, PopGrowth,")
    print("ProdGrowth, GDPPerCapita) as ADDITIVE controls ONLY -- no shock")
    print("interactions -- testing whether the shock's average, UNCONDITIONAL")
    print("effect (no exposure/mitigation channel at all) survives once these")
    print("controls are also included.")
    print("=" * 70)
    panel_baseline_robustness = build_baseline_robustness_panel()
    res_baseline_robustness = run_baseline_shock_projections(panel_baseline_robustness)
    irf_baseline_robustness = summarize_baseline_irf(res_baseline_robustness)
    print(irf_baseline_robustness.to_string())
    fig_baseline_robustness = plot_baseline_irf(
        irf_baseline_robustness,
        title=f"Robustness check: baseline {SHOCK_VARIABLE} AR(1) model (with growth-literature controls)",
        shock_std=shock_std,
    )

    print("\n" + "=" * 70)
    print("FINITE-SAMPLE BIAS CORRECTION (referee-requested): split-panel")
    print("jackknife (Dhaene & Jochmans, 2015) applied to every '*_robustness'")
    print("spec, addressing Nickell-type bias from combining entity FE with")
    print("qoq_lag1. See split_panel_jackknife()'s docstring for the method.")
    if BOOTSTRAP_BIAS_CORRECTION:
        print(f"BOOTSTRAP_BIAS_CORRECTION is ON ({N_BOOTSTRAP_REPS} reps/horizon) --")
        print("this will also run the (much slower) cluster-bootstrap alternative.")
    print("=" * 70)

    jackknife_tables = {}
    bootstrap_tables = {}
    for spec_name, panel_rb in [
        ("ICT_robustness", panel_ict_robustness),
        ("Patent_robustness", panel_patent_robustness),
        ("Investment_robustness", panel_investment_robustness),
        ("Corr_robustness", panel_corr_robustness),
        ("HSExport_robustness", panel_hs_export_robustness),
    ]:
        print(f"\n--- {spec_name}: split-panel jackknife ---")
        jk_result = split_panel_jackknife(panel_rb)
        jk_table = summarize_jackknife(jk_result)
        jackknife_tables[spec_name] = jk_table
        # Print just the mitigating-effect term(s) at a few horizons, not
        # the full (potentially 15-21-term x 9-horizon) table -- the full
        # table still goes to Excel below.
        key_terms = jk_table[jk_table["term"] == "shock_x_exposure"]
        if not key_terms.empty:
            print(key_terms[["h", "beta_full", "beta_jk", "bias_estimate",
                              "bias_relative_to_se"]].to_string(index=False))

        if BOOTSTRAP_BIAS_CORRECTION:
            print(f"\n--- {spec_name}: bootstrap bias correction ---")
            boot_result = bootstrap_bias_correction(panel_rb, n_reps=N_BOOTSTRAP_REPS)
            bootstrap_tables[spec_name] = summarize_bootstrap(boot_result)

    # Bias-correction CHARTS (for the IRF_robust_bias sheet) -- one per
    # robustness spec, all on the "shock_x_exposure" term specifically,
    # since that term is present (with the same meaning: the Shock*F
    # mitigating-effect interaction) across all five spec types,
    # unlike e.g. "mitigating_effect_boom" which is a DERIVED combination
    # (b4+b6) that split_panel_jackknife()/bootstrap_bias_correction()
    # don't separately bias-correct (they operate on raw regressor
    # coefficients only) -- see those functions' docstrings.
    bias_correction_figs = {}
    for spec_name in ["ICT_robustness", "Patent_robustness", "Investment_robustness",
                       "Corr_robustness", "HSExport_robustness"]:
        boot_table_for_spec = bootstrap_tables.get(spec_name) if BOOTSTRAP_BIAS_CORRECTION else None
        bias_correction_figs[spec_name] = plot_bias_correction_irf(
            jackknife_tables[spec_name], "shock_x_exposure",
            f"Bias correction: {spec_name} (Shock*F mitigating effect)",
            shock_std=shock_std, boot_table=boot_table_for_spec,
        )

    print("\n" + "=" * 70)
    print(f"BASELINE: {SHOCK_VARIABLE} shock IRF, exposure/mitigation switched off")
    print("          (entity FE only, no time FE -- see docstring for why;")
    print(f"          shock_used = AR-purified shock, not the raw {SHOCK_VARIABLE} level)")
    print("=" * 70)
    # shock_used and gdp_level are identical across specs (only the
    # exposure proxy differs), so this only needs to be run once, off
    # either panel.
    res_baseline = run_baseline_shock_projections(panel_ict)
    irf_baseline = summarize_baseline_irf(res_baseline)
    print(irf_baseline.to_string())
    fig_baseline = plot_baseline_irf(irf_baseline,
                                      title=f"Baseline {SHOCK_VARIABLE} IRF (AR-purified shock, no exposure)",
                                      shock_std=shock_std)

    print("\n" + "=" * 70)
    print(f"Fetching raw source series for export (GDP, GFCF, AI patents, AI investment, indices, {SHOCK_VARIABLE})")
    print("=" * 70)
    raw_gdp = fetch_gdp_level()[["country", "quarter", "gdp_level"]]
    raw_gfcf = fetch_ict_investment_share()  # country, year, N1132G, N1173G, N11G, ict_share
    raw_ai_patents = fetch_ai_patents()  # country, year, ai_patents
    raw_ai_investment = fetch_ai_investment()  # country, year, ai_investment
    raw_national_index = fetch_national_indices_raw()
    raw_semiconductor = fetch_semiconductor_raw()
    raw_shock_level = fetch_shock_global().reset_index()
    raw_shock_innov, _ar_model = purify_shock(raw_shock_level)  # quarter, shock_level, shock_innov
    raw_sox_regime = compute_sox_regime()  # quarter, sox_roll_mean_ret, is_boom
    raw_hs_export = fetch_hs_export()  # country, year, share
    raw_trade_openness = fetch_trade_openness()  # country, year, trade_openness
    raw_population_growth = fetch_population_growth()  # country, year, population_growth
    raw_productivity_growth = fetch_productivity_growth()  # country, year, productivity_growth
    raw_gdp_per_capita = fetch_gdp_per_capita()  # country, year, gdp_per_capita

    irfs_dict = {
        "ICT_exposure": irf_ict,
        "Patent_exposure": irf_patent,
        "Investment_exposure": irf_investment,
        "Corr_exposure": irf_corr,
        "HSExport_exposure": irf_hs_export,
        "ICT_robustness": irf_ict_robustness,
        "Patent_robustness": irf_patent_robustness,
        "Investment_robustness": irf_investment_robustness,
        "Corr_robustness": irf_corr_robustness,
        "HSExport_robustness": irf_hs_export_robustness,
        "ICT_robustness_Jackknife": jackknife_tables["ICT_robustness"],
        "Patent_robustness_Jackknife": jackknife_tables["Patent_robustness"],
        "Investment_robustness_Jackknife": jackknife_tables["Investment_robustness"],
        "Corr_robustness_Jackknife": jackknife_tables["Corr_robustness"],
        "HSExport_robustness_Jackknife": jackknife_tables["HSExport_robustness"],
        f"Baseline_{SHOCK_VARIABLE}": irf_baseline,
        f"Baseline_{SHOCK_VARIABLE}_robustness": irf_baseline_robustness,
    }
    if BOOTSTRAP_BIAS_CORRECTION:
        irfs_dict.update({
            "ICT_robustness_Bootstrap": bootstrap_tables["ICT_robustness"],
            "Patent_robustness_Bootstrap": bootstrap_tables["Patent_robustness"],
            "Investment_robustness_Bootstrap": bootstrap_tables["Investment_robustness"],
            "Corr_robustness_Bootstrap": bootstrap_tables["Corr_robustness"],
            "HSExport_robustness_Bootstrap": bootstrap_tables["HSExport_robustness"],
        })

    export_results_to_excel(
        panels={"ICT_exposure": panel_ict, "Patent_exposure": panel_patent,
                "Investment_exposure": panel_investment, "Corr_exposure": panel_corr,
                "HSExport_exposure": panel_hs_export,
                "ICT_robustness": panel_ict_robustness,
                "Patent_robustness": panel_patent_robustness,
                "Investment_robustness": panel_investment_robustness,
                "Corr_robustness": panel_corr_robustness,
                "HSExport_robustness": panel_hs_export_robustness,
                f"Baseline_{SHOCK_VARIABLE}_robustness": panel_baseline_robustness},
        irfs=irfs_dict,
        figs={
            "ICT_exposure": fig_ict,
            "Combined_Patent_vs_Investment": fig_combined,
            "Corr_exposure_Boom": fig_corr_boom,
            "Corr_exposure_Bust": fig_corr_bust,
            "HSExport_exposure_Boom": fig_hs_export_boom,
            "HSExport_exposure_Bust": fig_hs_export_bust,
            f"Baseline_{SHOCK_VARIABLE}": fig_baseline,
        },
        robustness_figs={
            "ICT_robustness": fig_ict_robustness,
            "Combined_Patent_vs_Investment_robustness": fig_combined_robustness,
            "Corr_robustness_Boom": fig_corr_robustness_boom,
            "Corr_robustness_Bust": fig_corr_robustness_bust,
            "HSExport_robustness_Boom": fig_hs_export_robustness_boom,
            "HSExport_robustness_Bust": fig_hs_export_robustness_bust,
            f"Baseline_{SHOCK_VARIABLE}_robustness": fig_baseline_robustness,
        },
        bias_correction_figs=bias_correction_figs,
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
            "Raw_HS_Export_Share": raw_hs_export,  # country, year, share
            "Raw_Trade_Openness": raw_trade_openness,  # country, year, trade_openness
            "Raw_Population_Growth": raw_population_growth,  # country, year, population_growth
            "Raw_Productivity_Growth": raw_productivity_growth,  # country, year, productivity_growth
            "Raw_GDP_Per_Capita": raw_gdp_per_capita,  # country, year, gdp_per_capita
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
    3. NL is one of 10 countries; the focus-country interaction terms let
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
       cross-sectional dependence across the 10 countries), with the
       Bartlett-kernel bandwidth at least 4 at every horizon (see
       _dk_bandwidth()) rather than a small fixed constant -- the
       dependent variable (cumulative QoQ GDP growth AT horizon h) is a
       4-quarter-differenced measure, which carries inherent MA(3)-type
       serial correlation by construction regardless of h; a bandwidth
       below ~4 would understate that.
    7. ALL RELEVANT IRF CHARTS (baseline, ICT, Corr, hs_export) are
       rescaled to show
       the cumulative QoQ GDP growth response, AT horizon h, per 1-STANDARD-DEVIATION {SHOCK_VARIABLE} shock
       (shock_std, printed above -- see compute_shock_std()),
       NOT per 1 raw unit of shock_used -- shock_used is an AR(2) residual,
       an arbitrary scale with no natural economic unit, so "per 1
       standard deviation" is the interpretable convention here.
       IMPORTANT: this rescaling applies ONLY to the CHARTS -- the
       underlying Excel tables (ICT_exposure_IRF, Corr_exposure_IRF,
       HSExport_exposure_IRF, Baseline_{SHOCK_VARIABLE}_IRF) still report the raw per-unit coefficients
       exactly as estimated, so any further hand-calculation from the
       tables needs to multiply by shock_std manually to match what
       the charts show. t-statistics are unaffected by this rescaling
       either way (scaling beta and se by the same constant leaves their
       ratio unchanged).
    8. NO EXPLICIT TIME TREND CONTROL is used anywhere in this script
       (an earlier version added one, alongside qoq_lag1, specifically
       because F trends upward over most of the sample for nearly
       every country at once -- a shared, roughly global AI/ICT-
       adoption trend that risked the Shock*F interaction coefficient
       partly reflecting "how the shock-growth relationship changed
       over calendar time in general" rather than something
       specifically tied to exposure). Removed on the reasoning that
       the "*_robustness" specs' growth-literature controls
       (population_growth, gdp_per_capita, etc.) already carry time-
       varying information that substitutes for a generic trend there.
       IMPORTANT CAVEAT: this reasoning applies ONLY to the
       "*_robustness" specs -- the PLAIN specs (ICT, patent,
       investment, Corr, hs_export, and the baseline) have NO such
       substitute and are therefore left with NO explicit protection
       against that confound at all. That is a deliberate, accepted
       trade-off, not an oversight -- but it does mean the plain
       specs' Shock*F coefficients should be read with that specific
       risk back in mind, exactly as before any trend control existed.

    WHAT "MITIGATING EFFECT" MEANS IN ECONOMIC TERMS:

    The baseline spec asks a simple question: does a {SHOCK_VARIABLE}
    shock move GDP growth at all, on average across the panel?
    beta_shock_baseline is that average marginal response -- typically
    expected to be NEGATIVE (higher {SHOCK_VARIABLE} depresses growth),
    though whether it is actually significant here is itself
    informative.

    The ICT, Corr, and hs_export specs ask a follow-up question: does a country's
    AI/ICT exposure change HOW SENSITIVE its growth is to that same
    shock? This is captured by an INTERACTION term ({SHOCK_VARIABLE}
    shock x exposure F), so the coefficients on it (b3 for ICT, b4/b6
    for Corr and hs_export) are not themselves "the effect of the shock" -- they are
    the effect of exposure ON that sensitivity. Concretely: the
    marginal GDP response to a {SHOCK_VARIABLE} shock, as a function of
    exposure F, is

        d(GDP growth) / d(Shock) = b1 + b3*F        (ICT spec)
        d(GDP growth) / d(Shock) = b1 + b4 + b6*Dum  (Corr and hs_export specs, at F=1)

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
