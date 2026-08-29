*=============================================================================
* Global shock propagation and AI-exposure mitigation: NL + euro area panel
* Stata translation of gpr_ai_mitigation_pipeline_WUI_local.py (v39)
*=============================================================================
*
* Estimates a MODERATED-REGRESSION panel local projection (main effects AND
* interaction terms, not a "difference" spec) with ENTITY fixed effects only
* (no time effects):
*
*   ICT/Patent/Investment specs:
*     Delta_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
*                        + b3_h*(Shock_t * F[i,t]) + Gamma_h*X[i,t] + e[i,t+h]
*
*   Corr spec (Dum = is_boom_t; Dum=0/"bust" is the reference level):
*     Delta_gdp[i,t+h] = a[i,h] + b1_h*Shock_t + b2_h*F[i,t]
*                        + b3_h*Dum_t + b4_h*(Shock_t*F[i,t])
*                        + b5_h*(Shock_t*Dum_t) + b6_h*(Shock_t*F[i,t]*Dum_t)
*                        + Gamma_h*X[i,t] + e[i,t+h]
*
* for h = 0..8 quarters, where F[i,t] in (0,1) is a logistic transform of a
* country-level AI-exposure state variable z[i,t], and Shock_t is an
* AR(2)-purified version of a GLOBAL shock series -- either the World
* Uncertainty Index (WUI) or the Geopolitical Risk Index (GPR), whichever
* $shock_variable selects below. EVERY coefficient, IRF table, and chart in
* this do-file is based on whichever shock $shock_variable currently
* selects; change that one line and every downstream result follows.
*
* IDENTIFICATION: Shock_t, Dum_t, and Shock_t*Dum_t are entity-invariant
* (identical across every country at a given quarter). Time fixed effects
* would absorb all three completely, so this uses ENTITY fixed effects only
* -- see the build_panel program below. Cost: common-across-countries
* time-varying confounders (ECB policy, euro-area-wide demand shocks) are
* not swept out of the residual the way two-way FE would.
*
* FOUR versions of z (AI exposure) are estimated as separate models:
*   (A) ICT investment share of GFCF
*   (B) Annual AI patent applications per country
*   (C) Annual AI-related incoming investment counts per country
*       -- (B) and (C) are charted together in a single combined,
*          panel-average-only comparison graph
*   (D) Rolling correlation between each country's national equity index
*       and the global semiconductor index (robustness), with a boom/bust
*       regime split (Channel 1)
*
* ALL INPUT DATA is read from a single LOCAL workbook, `data_file'
* ("ai_data.xlsx"), which must sit in the working directory set by -cd-
* below. No network access of any kind. Expected sheets: gdp, ict_inv,
* ai_patent, ai_inv, index_nat, index_sox, wui, gpr (see the import section
* below for exact columns).
*
* INTERMEDIATE FILES: every intermediate dataset this do-file creates is
* saved as an EXPLICIT, named "tmp_*.dta" file in the working directory
* (NOT via Stata's -tempfile-). This is a deliberate fix, not a style
* choice: passing a -tempfile- local macro's value as a STRING ARGUMENT
* into a -program- (e.g. build_panel ict "`panel_ict'") is fragile --
* Windows temp-folder paths can contain spaces or other characters that
* -args|- (used inside the programs below) does not always parse back
* together correctly, which is exactly what produced the "invalid file
* specification" error this version fixes. Explicit literal filenames
* sidestep that failure mode entirely, at the cost of leaving ~20 small
* tmp_*.dta files behind in the working directory after a run (harmless;
* delete them, or extend section 12 at the very end to -erase- them
* automatically once you've confirmed everything else works).
*
* REQUIRED PACKAGES (installed once, commented out after first run):
*   ssc install xtscc      // Driscoll-Kraay SEs for FE panels (Hoechle 2007)
*   ssc install estout      // (optional) nicer regression tables, unused
*     by the core logic below but handy if you extend this
*
* NOTE: xtscc could not be installed in DSW, so we use xtreg, which produces narrower confidence bands
*=============================================================================

version 17
clear all
set more off
capture log close
log using "gpr_ai_mitigation_pipeline_log.log", replace text

*-----------------------------------------------------------------------
* 0. CONFIG
*-----------------------------------------------------------------------
cd "G:\EBO\MB\vdEnd\DSW_G\risk mngt"
local data_file "ai_data.xlsx"
global shock_variable    "WUI"     // "WUI" or "GPR" -- change this ONE line
                                    // to switch the whole do-file (data
                                    // source, AR-purification, panel
                                    // construction, every IRF table and
                                    // chart) from the World Uncertainty
                                    // Index to the Caldara-Iacoviello
                                    // Geopolitical Risk Index, or back.
if !inlist("$shock_variable", "WUI", "GPR") {
    di as error "shock_variable must be WUI or GPR, got $shock_variable"
    error 198
}
global countries         "NL DE FR IT ES BE AT"   // euro-area panel
global sample_start_q    "2000q1"
global sample_end_q      "2026q2"
global horizons_max      8          // h = 0..8 quarters
global theta             2.0        // logistic transition steepness
global focus_country     "NL"       // country singled out for its own
                                     // interaction term
global standardize_mode  "pooled"   // "pooled" (recommended) or
                                     // "within_country"
global corr_window       8          // rolling window for the correlation
                                     // exposure measure (Spec D)
global sox_window        4          // rolling window for the SOX boom/bust
                                     // regime classification (Dum)
global ar_lags           2          // AR(lags) shock purification

if "$shock_variable" == "WUI" {
    global shock_sheet     "wui"
    global shock_source_col "wui_global"
}
else {
    global shock_sheet     "gpr"
    global shock_source_col "gpr_global"
}

*-----------------------------------------------------------------------
* 1. Import all sheets from the local workbook into separate, EXPLICITLY
*    NAMED .dta files (Stata works with one dataset in memory at a time,
*    unlike pandas -- each sheet is imported once here and saved, then
*    merged/used as needed downstream). Every sheet's exact column set is
*    the same as confirmed against the real workbook used to build this
*    pipeline's Python original.
*-----------------------------------------------------------------------

* --- gdp: country, quarter, gdp_level ---------------------------------
import excel "`data_file'", sheet("gdp") firstrow clear
capture confirm string variable quarter
if _rc {
    tostring quarter, replace
}
gen qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
rename qdate quarter
drop if missing(gdp_level)
sort country quarter
save "tmp_gdp.dta", replace

* --- ict_inv: country, year, N1132G, N1173G, N11G, ict_share ----------
import excel "`data_file'", sheet("ict_inv") firstrow clear
destring year, replace
drop if missing(N1132G) | missing(N11G) | missing(ict_share)
* Diagnostic matching the Python version's N1173G coverage report
count if !missing(N1173G)
local n1173_ok = r(N)
count
di as text "  [diagnostic] N1173G coverage: `n1173_ok'/" r(N) ///
    " country-year rows have a non-null value in the local workbook."
sort country year
save "tmp_ict_inv.dta", replace

* --- ai_patent: country, year, ai_patents ------------------------------
import excel "`data_file'", sheet("ai_patent") firstrow clear
destring year, replace
destring ai_patents, replace
drop if missing(ai_patents)
sort country year
count
di as text "  [diagnostic] AI patent applications: " r(N) " country-year rows."
save "tmp_ai_patent.dta", replace

* --- ai_inv: country, year, ai_investment ------------------------------
import excel "`data_file'", sheet("ai_inv") firstrow clear
destring year, replace
destring ai_investment, replace
drop if missing(ai_investment)
sort country year
count
di as text "  [diagnostic] AI incoming investment counts: " r(N) " country-year rows."
save "tmp_ai_inv.dta", replace

* --- index_nat: country, ticker, quarter, close, log_ret ---------------
import excel "`data_file'", sheet("index_nat") firstrow clear
capture confirm string variable quarter
if _rc {
    tostring quarter, replace
}
gen qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
rename qdate quarter
drop if missing(close)
sort country quarter
save "tmp_index_nat.dta", replace

* --- index_sox: ticker, quarter, close, log_ret -------------------------
import excel "`data_file'", sheet("index_sox") firstrow clear
capture confirm string variable quarter
if _rc {
    tostring quarter, replace
}
gen qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
rename qdate quarter
drop if missing(close)
sort quarter
save "tmp_index_sox.dta", replace

* --- shock series (wui or gpr sheet, per $shock_variable) --------------
import excel "`data_file'", sheet("$shock_sheet") firstrow clear
capture confirm string variable quarter
if _rc {
    tostring quarter, replace
}
gen qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
rename qdate quarter
rename $shock_source_col shock_level
drop if missing(shock_level)
sort quarter
save "tmp_shock.dta", replace

*-----------------------------------------------------------------------
* 2. AR(lags) purification of the chosen global shock series
*    (regress shock_level on its own lags; residual = "identified shock")
*-----------------------------------------------------------------------
use "tmp_shock.dta", clear
tsset quarter
forvalues l = 1/$ar_lags {
    gen shock_lag`l' = L`l'.shock_level
}
* keep only rows where all lags are non-missing, matching the Python
* version's dropna(subset=lag_cols)
egen miss_lags = rowmiss(shock_lag1-shock_lag$ar_lags)
drop if miss_lags > 0
drop miss_lags

regress shock_level shock_lag1-shock_lag$ar_lags
di as text "  [AR($ar_lags) $shock_variable purification] R-squared = " %5.3f e(r2)
predict shock_innov, residual

summarize shock_innov
local shock_std = r(sd)
di as text "  [diagnostic] $shock_variable shock (shock_used) standard deviation = " ///
    %9.4f `shock_std' " -- this is the '1-stdev shock' used to rescale IRF charts."

keep quarter shock_level shock_innov
save "tmp_shock_innov.dta", replace

*-----------------------------------------------------------------------
* 3. GDP growth (YoY log difference, used only as the lag control --
*    NOT the regression's dependent variable, which is a separately-
*    constructed cumulative measure built inside build_panel below)
*-----------------------------------------------------------------------
use "tmp_gdp.dta", clear
encode country, gen(country_id)
xtset country_id quarter
gen dgdp = 100*(ln(gdp_level) - ln(L4.gdp_level))
keep country country_id quarter gdp_level dgdp
save "tmp_gdp_growth.dta", replace

*-----------------------------------------------------------------------
* 4. Semiconductor-correlation exposure proxy (Spec D) and SOX boom/bust
*    regime (Dum) -- both DERIVED from index_nat / index_sox, computed
*    here rather than read from a sheet (they are not raw source data).
*-----------------------------------------------------------------------

* --- 4a. National index log returns (already in the sheet) + SOX -------
use "tmp_index_nat.dta", clear
encode country, gen(country_id)
xtset country_id quarter
save "tmp_nat_ret.dta", replace

use "tmp_index_sox.dta", clear
tsset quarter
rename log_ret sox_log_ret
keep quarter sox_log_ret close
rename close sox_close
save "tmp_sox_ret.dta", replace

* --- 4b. Rolling correlation (Spec D exposure, z_raw) -------------------
* Manual rolling-window correlation, computed explicitly per country via
* an explicit loop over quarters (rather than relying on any third-party
* package's exact syntax/output-column conventions, which are easy to
* get subtly wrong sight-unseen) -- matches pandas'
* groupby("country")[...].rolling($corr_window).corr() exactly: at each
* quarter t, correlate the trailing $corr_window observations of the
* country's own return with the semiconductor index's return.
use "tmp_nat_ret.dta", clear
merge m:1 quarter using "tmp_sox_ret.dta", keep(match) nogenerate
sort country_id quarter
save "tmp_corr_input.dta", replace

quietly levelsof country_id, local(idlist)
clear
save "tmp_corr_exposure.dta", emptyok replace

foreach cid of local idlist {
    use "tmp_corr_input.dta", clear
    keep if country_id == `cid'
    sort quarter
    local N = _N
    gen z_raw = .
    forvalues i = $corr_window/`N' {
        local i0 = `i' - $corr_window + 1
        quietly corr log_ret sox_log_ret in `i0'/`i'
        quietly replace z_raw = r(rho) in `i'
    }
    keep country country_id quarter z_raw
    drop if missing(z_raw)
    append using "tmp_corr_exposure.dta"
    save "tmp_corr_exposure.dta", replace
}
use "tmp_corr_exposure.dta", clear
sort country_id quarter
save "tmp_corr_exposure.dta", replace

* --- 4c. SOX boom/bust regime classification (Dum) -----------------------
* is_boom = 1 if the trailing $sox_window-quarter average log return of
* SOX is ABOVE THE MEDIAN of that same average-return series over the
* sample (median-thresholded so boom/bust each cover ~half the sample,
* rather than a literal-zero threshold, which would skew toward "boom"
* given equity indices trend upward over most multi-year windows).
use "tmp_sox_ret.dta", clear
sort quarter
local N = _N
gen sox_roll_mean_ret = .
forvalues i = $sox_window/`N' {
    local i0 = `i' - $sox_window + 1
    quietly summarize sox_log_ret in `i0'/`i'
    quietly replace sox_roll_mean_ret = r(mean) in `i'
}
* explicit trailing-window average (current obs + the $sox_window-1 prior
* obs) via a plain forvalues loop over physical observation numbers --
* matches pandas' .rolling($sox_window).mean() exactly, without relying
* on the exact argument semantics of -tssmooth ma-'s window() option.
drop if missing(sox_roll_mean_ret)
summarize sox_roll_mean_ret, detail
local median_ret = r(p50)
gen is_boom = (sox_roll_mean_ret > `median_ret')
count if is_boom == 1
local n_boom = r(N)
count
local n_valid = r(N)
local n_bust = `n_valid' - `n_boom'
di as text "  [diagnostic] SOX regime split (median threshold = " %9.6f `median_ret' "): " ///
    "boom=`n_boom' quarters, bust=`n_bust' quarters, out of `n_valid' valid quarters."
keep quarter sox_roll_mean_ret is_boom
save "tmp_sox_regime.dta", replace

*-----------------------------------------------------------------------
* 5. build_panel PROGRAM -- constructs the full regression panel for one
*    exposure spec ("ict", "patent", "investment", or "corr"), mirroring
*    build_panel(exposure=...) in the Python original. Saves the result
*    to the .dta file named in `2' (the second argument) -- ALWAYS an
*    explicit literal filename at the call site (section 9 below), never
*    a -tempfile- macro, which is what previously produced the "invalid
*    file specification" error.
*-----------------------------------------------------------------------
capture program drop build_panel
program define build_panel
    * args: 1 = exposure ("ict"|"patent"|"investment"|"corr")
    *       2 = output .dta path to save the panel to
    args exposure outfile

    capture confirm file "tmp_gdp_growth.dta"
    if _rc {
        di as error "build_panel: tmp_gdp_growth.dta not found -- section 3 must run before this program is called."
        error 601
    }
    use "tmp_gdp_growth.dta", clear

    if "`exposure'" == "ict" {
        * annual ict_share -> quarterly (flat/step interpolation, matching
        * annual_to_quarterly()'s simple step-repeat: broadcast the annual
        * value across all 4 quarters of that calendar year)
        preserve
        use "tmp_ict_inv.dta", clear
        rename ict_share z_raw
        keep country year z_raw
        save "tmp_ict_annual.dta", replace
        restore
        gen year = yofd(dofq(quarter))
        merge m:1 country year using "tmp_ict_annual.dta", keep(master match) nogenerate
        drop year
    }
    else if "`exposure'" == "patent" {
        preserve
        use "tmp_ai_patent.dta", clear
        rename ai_patents z_raw
        keep country year z_raw
        save "tmp_patent_annual.dta", replace
        restore
        gen year = yofd(dofq(quarter))
        merge m:1 country year using "tmp_patent_annual.dta", keep(master match) nogenerate
        drop year
    }
    else if "`exposure'" == "investment" {
        preserve
        use "tmp_ai_inv.dta", clear
        rename ai_investment z_raw
        keep country year z_raw
        save "tmp_investment_annual.dta", replace
        restore
        gen year = yofd(dofq(quarter))
        merge m:1 country year using "tmp_investment_annual.dta", keep(master match) nogenerate
        drop year
    }
    else if "`exposure'" == "corr" {
        * merge on the STRING country (not country_id) -- avoids any
        * dependency on two separate -encode- calls (one for the base
        * panel, one for tmp_corr_exposure.dta, built in different
        * sections) happening to assign identical numeric codes; string
        * keys can't silently mismatch that way.
        merge 1:1 country quarter using "tmp_corr_exposure.dta", ///
            keepusing(z_raw) keep(master match) nogenerate
    }
    else {
        di as error "build_panel: exposure must be ict, patent, investment, or corr"
        error 198
    }

    * merge in the AR-purified shock (entity-invariant, merges on quarter)
    merge m:1 quarter using "tmp_shock_innov.dta", keep(master match) nogenerate
    rename shock_innov shock_used

    *---------------------------------------------------------------
    * STANDARDIZATION OF z -- pooled across the panel by default (NOT
    * within-country); see the long rationale in the Python original's
    * build_panel() docstring: pooled standardization preserves genuine
    * between-country LEVEL differences in exposure.
    *---------------------------------------------------------------
    if "$standardize_mode" == "pooled" {
        summarize z_raw
        gen z = (z_raw - r(mean)) / r(sd)
    }
    else {
        by country_id, sort: egen z_mean = mean(z_raw)
        by country_id, sort: egen z_sd = sd(z_raw)
        gen z = (z_raw - z_mean) / z_sd
        drop z_mean z_sd
    }
    gen F = 1 / (1 + exp(-$theta * z))

    * GDP-growth lag control (year-on-year, see dgdp above)
    xtset country_id quarter
    gen dgdp_lag1 = L.dgdp

    *---------------------------------------------------------------
    * EXPLICIT LINEAR TIME TREND -- an additional control, entity-
    * invariant (same value for every country at a given quarter, just a
    * 0,1,2,... counter over the panel's own quarters). Added because F
    * (the exposure state variable) trends upward over most of the
    * sample for nearly every country at once -- a shared, roughly
    * global AI/ICT-adoption trend -- which risks the Shock*F
    * interaction coefficient (b3/b4) partly picking up "how the
    * economy's growth-shock relationship changed over calendar time in
    * general" rather than "how it changed specifically WITH exposure."
    * A plain additive trend does NOT fully resolve this (it controls
    * for a generic drift in the LEVEL of growth, not specifically for
    * drift in the shock's own sensitivity -- that would require
    * Shock*trend as a further interaction, not implemented here), but
    * it is a first, standard control against confounding by a shared
    * secular trend.
    *
    * IDENTIFICATION CAVEAT: time_trend is entity-invariant, exactly
    * like shock_used and is_boom -- estimable here only because this
    * do-file uses entity-only FE (no time FE, which would absorb it
    * completely, same reasoning as for the shock itself). It is NOT
    * perfectly collinear with shock_used by construction (the AR(2)
    * purification in section 2 already strips out most smooth,
    * trend-like predictability from the shock series), but some
    * residual collinearity is possible -- see the printed correlation
    * diagnostic below, and treat a high correlation there as a signal
    * that b1 (shock_used's own coefficient) and the time_trend
    * coefficient may be poorly separately identified in that run.
    *
    * Stata's %tq quarterly dates are already consecutive integers
    * internally, so a simple quarter-minus-minimum gives the exact
    * same 0,1,2,... counter the Python original builds via an
    * explicit dict-based mapping -- no need to replicate that
    * machinery here.
    *---------------------------------------------------------------
    summarize quarter
    gen time_trend = quarter - r(min)

    quietly corr time_trend shock_used
    di as text "  [diagnostic] corr(time_trend, shock_used) = " %5.3f r(rho) ///
        " -- a magnitude above ~0.5-0.6 suggests b1 and the time_trend " ///
        "coefficient may be weakly separately identified in this run."

    *---------------------------------------------------------------
    * MODERATED-REGRESSION SPECIFICATION (main effects AND interaction
    * terms, not a "difference" spec):
    *   ICT/patent/investment: b1*Shock + b2*F + b3*(Shock*F)
    *   Corr:                  + b3*Dum + b4*(Shock*F) + b5*(Shock*Dum)
    *                          + b6*(Shock*F*Dum)
    * shock_used and F are separate regressors (b1, b2), not folded into
    * an implicit baseline -- so b3/b4 are genuine interaction-effect
    * coefficients net of both main effects.
    *---------------------------------------------------------------
    gen shock_x_exposure = F * shock_used

    * focus-country (NL) interaction terms -- let the LP recover
    * NL-specific deviations from the panel-average coefficients.
    gen is_focus = (country == "$focus_country")
    gen shock_used_focus = is_focus * shock_used
    gen F_focus = is_focus * F
    gen shock_x_exposure_focus = is_focus * shock_x_exposure

    if "`exposure'" == "corr" {
        merge m:1 quarter using "tmp_sox_regime.dta", keepusing(is_boom) ///
            keep(master match) nogenerate
        gen is_bust = 1 - is_boom
        gen shock_x_dum = shock_used * is_boom                 // b5: Shock x Dum
        gen shock_x_exposure_x_dum = shock_x_exposure * is_boom // b6: Shock x F x Dum
        gen is_boom_focus = is_focus * is_boom
        gen shock_x_dum_focus = is_focus * shock_x_dum
        gen shock_x_exposure_x_dum_focus = is_focus * shock_x_exposure_x_dum
    }

    sort country_id quarter
    save "`outfile'", replace
end

*-----------------------------------------------------------------------
* 6. Driscoll-Kraay bandwidth helper (growing with horizon h) -- the
*    cumulative dependent variable's overlapping-window construction
*    induces MA(h)-type serial correlation that grows with h, so a fixed
*    bandwidth understates it at longer horizons. max(4, h+1).
*-----------------------------------------------------------------------
capture program drop dk_bandwidth
program define dk_bandwidth, rclass
    args h
    local bw = max(4, `h' + 1)
    return local bw = `bw'
end

*-----------------------------------------------------------------------
* 7. run_local_projections PROGRAM -- moderated regression at every
*    horizon h=0..$horizons_max, entity FE only, Driscoll-Kraay SEs
*    (xtscc, growing bandwidth). Auto-detects boom/bust columns (is_boom
*    present => Corr spec regressor set; absent => ICT/patent/investment
*    single-interaction regressor set). Results are accumulated via
*    postfile into one long dataset (one row per horizon x coefficient),
*    analogous to summarize_irf()'s output table.
*
*    REQUIRES: ssc install xtscc  (Hoechle 2007 Driscoll-Kraay SEs for
*    unbalanced FE panels)
*-----------------------------------------------------------------------
capture program drop run_local_projections
program define run_local_projections
    * args: 1 = input panel .dta path (explicit filename, see section 9)
    *       2 = output results .dta path (one row per h x coefficient)
    args panelfile resultsfile

    use "`panelfile'", clear
    xtset country_id quarter
    capture confirm variable is_boom
    local has_boom_bust = (_rc == 0)

    if `has_boom_bust' {
        local base_terms  "shock_used F is_boom shock_x_exposure shock_x_dum shock_x_exposure_x_dum"
        local focus_terms "shock_used_focus F_focus is_boom_focus shock_x_exposure_focus shock_x_dum_focus shock_x_exposure_x_dum_focus"
        local labels       "b1_shock b2_F b3_dum b4_shock_x_F b5_shock_x_dum b6_shock_x_F_x_dum"
    }
    else {
        local base_terms  "shock_used F shock_x_exposure"
        local focus_terms "shock_used_focus F_focus shock_x_exposure_focus"
        local labels       "b1_shock b2_F b3_shock_x_F"
    }
    local all_terms "`base_terms' `focus_terms'"

    tempname results_handle
    postfile `results_handle' h str32(label) beta_panelavg se_panelavg ///
        beta_focus_deviation beta_focus_total se_focus_total ///
        mitig_bust mitig_bust_se mitig_boom mitig_boom_se ///
        using "`resultsfile'", replace

    forvalues h = 0/$horizons_max {
        preserve
        gen dgdp_cum_lead = ///
            100*(ln(F`h'.gdp_level) - ln(L.gdp_level))
        local bw
        dk_bandwidth `h'
        local bw = r(bw)

        quietly xtreg dgdp_cum_lead `all_terms' dgdp_lag1 time_trend, fe vce(cluster country_id)
        di as text "--- h=`h' (Driscoll-Kraay bandwidth=`bw') ---"

        local n_terms : word count `base_terms'
        forvalues i = 1/`n_terms' {
            local term  : word `i' of `base_terms'
            local focus : word `i' of `focus_terms'
            local lbl   : word `i' of `labels'

            local b_panel = .
            local se_panel = .
            capture local b_panel = _b[`term']
            capture local se_panel = _se[`term']

            local b_dev = .
            local b_tot = .
            local se_tot = .
            capture {
                local b_dev = _b[`focus']
                lincom `term' + `focus'
                local b_tot = r(estimate)
                local se_tot = r(se)
            }

            post `results_handle' (`h') ("`lbl'") (`b_panel') (`se_panel') ///
                (`b_dev') (`b_tot') (`se_tot') (.) (.) (.) (.)
        }

        if `has_boom_bust' {
            * mitigating effect in BUST (Dum=0, reference level) = b4 alone
            local mb = .
            local mb_se = .
            capture {
                local mb = _b[shock_x_exposure]
                local mb_se = _se[shock_x_exposure]
            }
            * mitigating effect in BOOM (Dum=1) = b4 + b6, correctly
            * combined SE via lincom (uses the full covariance matrix --
            * exactly the "not b4's and b6's SEs summed naively"
            * correction the Python _linear_combination() helper computed
            * manually; Stata's lincom does this natively).
            local mbo = .
            local mbo_se = .
            capture {
                lincom shock_x_exposure + shock_x_exposure_x_dum
                local mbo = r(estimate)
                local mbo_se = r(se)
            }
            post `results_handle' (`h') ("mitigating_effect_bust_boom") (.) (.) ///
                (.) (.) (.) (`mb') (`mb_se') (`mbo') (`mbo_se')
        }
        restore
    }
    postclose `results_handle'
end

*-----------------------------------------------------------------------
* 8. run_baseline_shock_projections PROGRAM -- the "raw" GDP response to
*    the chosen shock, exposure/mitigation switched off entirely (no F,
*    no interaction term). Entity FE only (shock_used is entity-invariant
*    so time FE would absorb it completely). Driscoll-Kraay SEs, same
*    growing bandwidth.
*-----------------------------------------------------------------------
capture program drop run_baseline_shock_projections
program define run_baseline_shock_projections
    args panelfile resultsfile

    use "`panelfile'", clear
    xtset country_id quarter

    tempname bl_handle
    postfile `bl_handle' h beta_shock_baseline se_shock_baseline t_stat_baseline ///
        using "`resultsfile'", replace

    forvalues h = 0/$horizons_max {
        preserve
        gen dgdp_cum_lead = ///
            100*(ln(F`h'.gdp_level) - ln(L.gdp_level))
        local bw
        dk_bandwidth `h'
        local bw = r(bw)

        quietly xtreg dgdp_cum_lead shock_used dgdp_lag1 time_trend, fe vce(cluster country_id)
        di as text "--- baseline h=`h' (Driscoll-Kraay bandwidth=`bw') ---"

        local b = .
        local se = .
        local t = .
        capture {
            local b = _b[shock_used]
            local se = _se[shock_used]
            local t = `b' / `se'
        }
        post `bl_handle' (`h') (`b') (`se') (`t')
        restore
    }
    postclose `bl_handle'
end

*-----------------------------------------------------------------------
* 9. RUN ALL SPECS
*    Every panel/results file below is an EXPLICIT literal filename
*    passed directly to the programs above -- no -tempfile- macros
*    anywhere in this section. This is the actual fix for the "invalid
*    file specification" error: the previous version passed a -tempfile-
*    local's value (e.g. `panel_ict') as a quoted program argument, which
*    is fragile if the underlying OS temp path contains a space or other
*    character -args- can mis-parse; a literal string like "tmp_panel_ict.dta"
*    has no such risk.
*-----------------------------------------------------------------------

di as result _n "{hline 70}"
di as result "SPEC A: ICT investment share as exposure, global $shock_variable"
di as result "{hline 70}"
build_panel ict "tmp_panel_ict.dta"
run_local_projections "tmp_panel_ict.dta" "tmp_irf_ict.dta"

di as result _n "{hline 70}"
di as result "SPEC A (patent): AI patent applications as exposure, global $shock_variable"
di as result "{hline 70}"
build_panel patent "tmp_panel_patent.dta"
run_local_projections "tmp_panel_patent.dta" "tmp_irf_patent.dta"

di as result _n "{hline 70}"
di as result "SPEC A (investment): AI incoming investment counts as exposure, global $shock_variable"
di as result "{hline 70}"
build_panel investment "tmp_panel_investment.dta"
run_local_projections "tmp_panel_investment.dta" "tmp_irf_investment.dta"

di as result _n "{hline 70}"
di as result "SPEC B: Rolling correlation(national index, semiconductor index), global $shock_variable"
di as result "        -- Channel 1 boom/bust split"
di as result "{hline 70}"
build_panel corr "tmp_panel_corr.dta"
run_local_projections "tmp_panel_corr.dta" "tmp_irf_corr.dta"

di as result _n "{hline 70}"
di as result "BASELINE: $shock_variable shock IRF, exposure/mitigation switched off"
di as result "{hline 70}"
run_baseline_shock_projections "tmp_panel_ict.dta" "tmp_irf_baseline.dta"

*-----------------------------------------------------------------------
* 10. GRAPHS
*     (a) Combined AI-patent vs AI-investment mitigating effect, PANEL
*         AVERAGE ONLY -- mirrors plot_combined_mitigating_irf().
*     (b) ICT exposure mitigating effect (panel average + NL total).
*     (c) Corr spec boom/bust mitigating effects.
*     (d) Baseline shock IRF.
*     All rescaled to "per 1-stdev $shock_variable shock" using
*     `shock_std' computed in section 2, matching the Python charts'
*     1-stdev rescaling (charts only -- the underlying .dta/Excel tables
*     stay in raw per-unit-shock terms).
*-----------------------------------------------------------------------

* --- (a) combined patent vs investment, panel average only -------------
use "tmp_irf_patent.dta", clear
keep if label == "b3_shock_x_F"
gen beta_pat = beta_panelavg * `shock_std'
gen se_pat = se_panelavg * `shock_std'
keep h beta_pat se_pat
save "tmp_pat_plot.dta", replace

use "tmp_irf_investment.dta", clear
keep if label == "b3_shock_x_F"
gen beta_inv = beta_panelavg * `shock_std'
gen se_inv = se_panelavg * `shock_std'
keep h beta_inv se_inv
merge 1:1 h using "tmp_pat_plot.dta", nogenerate
gen upper_pat = beta_pat + 1.645*se_pat
gen lower_pat = beta_pat - 1.645*se_pat
gen upper_inv = beta_inv + 1.645*se_inv
gen lower_inv = beta_inv - 1.645*se_inv

twoway (rarea upper_pat lower_pat h, color(blue%15)) ///
       (line beta_pat h, lcolor(blue) lwidth(medthick) mcolor(blue) msymbol(o)) ///
       (rarea upper_inv lower_inv h, color(orange%15)) ///
       (line beta_inv h, lcolor(orange) lwidth(medthick) mcolor(orange) msymbol(s)), ///
    xtitle("Horizon h (quarters)") ///
    ytitle("Cumulative Shock x F interaction effect" "per 1-stdev $shock_variable shock (b3, 0..h)") ///
    title("Spec A mitigating effect ($shock_variable): AI patents vs AI investment (panel average)") ///
    legend(order(2 "AI patent applications (panel avg)" 4 "AI incoming investment counts (panel avg)")) ///
    yline(0, lcolor(black) lpattern(dash))
graph export "combined_patent_vs_investment.png", replace width(1200)

* --- (b) ICT exposure mitigating effect ---------------------------------
use "tmp_irf_ict.dta", clear
keep if label == "b3_shock_x_F"
gen beta_p = beta_panelavg * `shock_std'
gen se_p = se_panelavg * `shock_std'
gen beta_nl = beta_focus_total * `shock_std'
gen se_nl = se_focus_total * `shock_std'
gen upper_p = beta_p + 1.645*se_p
gen lower_p = beta_p - 1.645*se_p
gen upper_nl = beta_nl + 1.645*se_nl
gen lower_nl = beta_nl - 1.645*se_nl

twoway (rarea upper_p lower_p h, color(blue%15)) ///
       (line beta_p h, lcolor(blue) lwidth(medthick) mcolor(blue) msymbol(o)) ///
       (rarea upper_nl lower_nl h, color(red%15)) ///
       (line beta_nl h, lcolor(red) lwidth(medthick) mcolor(red) msymbol(s)), ///
    xtitle("Horizon h (quarters)") ///
    ytitle("Cumulative Shock x F interaction effect" "per 1-stdev $shock_variable shock (b3, 0..h)") ///
    title("Spec A: ICT investment share exposure ($shock_variable)") ///
    legend(order(2 "Panel average" 4 "$focus_country total")) ///
    yline(0, lcolor(black) lpattern(dash))
graph export "ict_exposure_irf.png", replace width(1200)

* --- (c) Corr spec boom/bust mitigating effects -------------------------
use "tmp_irf_corr.dta", clear
keep if label == "mitigating_effect_bust_boom"
gen mb = mitig_bust * `shock_std'
gen mb_se = mitig_bust_se * `shock_std'
gen mbo = mitig_boom * `shock_std'
gen mbo_se = mitig_boom_se * `shock_std'
gen upper_bust = mb + 1.645*mb_se
gen lower_bust = mb - 1.645*mb_se
gen upper_boom = mbo + 1.645*mbo_se
gen lower_boom = mbo - 1.645*mbo_se

twoway (rarea upper_boom lower_boom h, color(green%15)) ///
       (line mbo h, lcolor(green) lwidth(medthick) mcolor(green) msymbol(o)), ///
    xtitle("Horizon h (quarters)") ///
    ytitle("Cumulative mitigating effect" "per 1-stdev $shock_variable shock (0..h)") ///
    title("Spec B: AI-capex-cycle exposure -- BOOM regime mitigating effect (b4+b6)") ///
    yline(0, lcolor(black) lpattern(dash)) legend(off)
graph export "corr_boom_irf.png", replace width(1200)

twoway (rarea upper_bust lower_bust h, color(red%15)) ///
       (line mb h, lcolor(red) lwidth(medthick) mcolor(red) msymbol(o)), ///
    xtitle("Horizon h (quarters)") ///
    ytitle("Cumulative mitigating effect" "per 1-stdev $shock_variable shock (0..h)") ///
    title("Spec B: AI-capex-cycle exposure -- BUST regime mitigating effect (b4)") ///
    yline(0, lcolor(black) lpattern(dash)) legend(off)
graph export "corr_bust_irf.png", replace width(1200)

* --- (d) Baseline shock IRF ----------------------------------------------
use "tmp_irf_baseline.dta", clear
gen b = beta_shock_baseline * `shock_std'
gen se = se_shock_baseline * `shock_std'
gen upper = b + 1.645*se
gen lower = b - 1.645*se

twoway (rarea upper lower h, color(green%15)) ///
       (line b h, lcolor(green) lwidth(medthick) mcolor(green) msymbol(o)), ///
    xtitle("Horizon h (quarters)") ///
    ytitle("Cumulative dGDP response per 1-stdev $shock_variable shock (0..h)") ///
    title("Baseline $shock_variable shock IRF (no exposure interaction)") ///
    yline(0, lcolor(black) lpattern(dash)) legend(off)
graph export "baseline_irf.png", replace width(1200)

*-----------------------------------------------------------------------
* 11. EXPORT RESULTS TO EXCEL
*     One workbook, one sheet per IRF table plus the four raw panels,
*     mirroring export_results_to_excel()'s structure in the Python
*     original (charts are separate .png files above, since Stata cannot
*     embed images into an .xlsx the way openpyxl does).
*-----------------------------------------------------------------------
local outfile "model_results.xlsx"

use "tmp_irf_ict.dta", clear
export excel using "`outfile'", sheet("ICT_exposure_IRF") firstrow(variables) replace

use "tmp_irf_patent.dta", clear
export excel using "`outfile'", sheet("Patent_exposure_IRF") firstrow(variables) sheetmodify

use "tmp_irf_investment.dta", clear
export excel using "`outfile'", sheet("Investment_exposure_IRF") firstrow(variables) sheetmodify

use "tmp_irf_corr.dta", clear
export excel using "`outfile'", sheet("Corr_exposure_IRF") firstrow(variables) sheetmodify

use "tmp_irf_baseline.dta", clear
export excel using "`outfile'", sheet("Baseline_${shock_variable}_IRF") firstrow(variables) sheetmodify

use "tmp_panel_ict.dta", clear
export excel using "`outfile'", sheet("Panel_ICT_exposure") firstrow(variables) sheetmodify

use "tmp_panel_patent.dta", clear
export excel using "`outfile'", sheet("Panel_Patent_exposure") firstrow(variables) sheetmodify

use "tmp_panel_investment.dta", clear
export excel using "`outfile'", sheet("Panel_Investment_exposure") firstrow(variables) sheetmodify

use "tmp_panel_corr.dta", clear
export excel using "`outfile'", sheet("Panel_Corr_exposure") firstrow(variables) sheetmodify

*-----------------------------------------------------------------------
* 12. OPTIONAL CLEANUP -- commented out by default so the tmp_*.dta files
*     stay available for inspection if anything above needs debugging.
*     Once you've confirmed a full run works end-to-end, uncomment this
*     block to delete all intermediate files and keep only
*     model_results.xlsx and the .png charts.
*-----------------------------------------------------------------------
/*
local tmp_files "tmp_gdp.dta tmp_ict_inv.dta tmp_ai_patent.dta tmp_ai_inv.dta ///
    tmp_index_nat.dta tmp_index_sox.dta tmp_shock.dta tmp_shock_innov.dta ///
    tmp_gdp_growth.dta tmp_nat_ret.dta tmp_sox_ret.dta tmp_corr_input.dta ///
    tmp_corr_exposure.dta tmp_sox_regime.dta tmp_ict_annual.dta ///
    tmp_patent_annual.dta tmp_investment_annual.dta tmp_panel_ict.dta ///
    tmp_panel_patent.dta tmp_panel_investment.dta tmp_panel_corr.dta ///
    tmp_irf_ict.dta tmp_irf_patent.dta tmp_irf_investment.dta ///
    tmp_irf_corr.dta tmp_irf_baseline.dta tmp_pat_plot.dta"
foreach f of local tmp_files {
    capture erase "`f'"
}\
*/

di as result _n "{hline 70}"
di as result "Done. Results exported to `outfile'; charts exported as .png files"
di as result "(combined_patent_vs_investment.png, ict_exposure_irf.png,"
di as result " corr_boom_irf.png, corr_bust_irf.png, baseline_irf.png)."
di as result "Shock variable used throughout this run: $shock_variable"
di as result "{hline 70}"

log close
