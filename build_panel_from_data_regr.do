/*=============================================================================
probit_models_matching_python.do

Reproduces the probit specifications in the accompanying Python script.

Data timing
-----------
* growth_surprise(t) comes from ec_forecast_vs_realized.xlsx.
* ICT investment, HS exports, the stock/SOX correlation and AI investment are
  read from ai_data.xlsx and enter every model at t-1.
* The raw ai_data.xlsx observations through 2025 are retained. Consequently,
  a 2025 explanatory observation is assigned to outcome year 2026; outcome
  year 2025 uses the observed 2024 explanatory value. No 2024-to-2025
  missing-value fallback is applied.
* The optional State Aid control is State aid(t) / nominal GDP(t-1): only its
  GDP denominator is lagged. The resulting control is standardized before it
  enters a regression, just as in the Python script.

Models
------
For each of the four explanatory variables this file estimates:
  (1) standardized X;
  (2) standardized X and standardized-X x shock-year interaction;
  (3) standardized above-median magnitude and its shock-year interaction.

All models include country and outcome-year fixed effects. Standard errors are
clustered by country. The binary outcome equals one when growth_surprise > 0.
=============================================================================*/

version 17
clear all
set more off

* ----- USER SETTINGS --------------------------------------------------------
cd "G:\EBO\MB\vdEnd\DSW_G\risk mngt"
global forecast_file "ec_forecast_vs_realized.xlsx"
global ai_data_file  "ai_data.xlsx"

* Mirrors INCLUDE_STATE_AID_CONTROL in the attached Python script.
* 1 = include standardized State aid(t)/GDP(t-1) in every probit.
* 0 = estimate the same probits without the State Aid control.
local include_state_aid_control = 1

if !inlist(`include_state_aid_control', 0, 1) {
    display as error "include_state_aid_control must be 0 or 1."
    exit 198
}


/*=============================================================================
  Outcome data. Multiple forecast vintages for one country/outcome year are
  deliberately retained; each is a separate observation in the Python models.
=============================================================================*/
import excel "$forecast_file", firstrow clear
keep country target_year forecast_growth_annual_pct realized_growth_annual_pct
destring target_year forecast_growth_annual_pct realized_growth_annual_pct, replace force
drop if missing(country) | missing(target_year)
gen double growth_surprise = realized_growth_annual_pct - forecast_growth_annual_pct
tempfile forecast_base
save "`forecast_base'"


/*=============================================================================
  Optional State Aid control: aid at t divided by GDP at t-1.
  The support sheet ends in 2024. When the option is on, a 2025 outcome is
  therefore retained in the working panel but excluded by probit if aid(t) is
  unavailable. There is intentionally no 2024 State Aid fallback for 2025.
=============================================================================*/
tempfile support_control
if `include_state_aid_control' {
    import excel "$ai_data_file", sheet("support") firstrow clear
    keep country year gdp_m_eur
    destring year gdp_m_eur, replace force
    rename year gdp_source_year
    gen int target_year = gdp_source_year + 1
    rename gdp_m_eur gdp_m_eur_lag1
    isid country target_year
    tempfile lagged_gdp
    save "`lagged_gdp'"

    import excel "$ai_data_file", sheet("support") firstrow clear
    keep country year state_aid_m_eur
    destring year state_aid_m_eur, replace force
    rename year target_year
    gen int state_aid_source_year = target_year
    isid country target_year
    merge 1:1 country target_year using "`lagged_gdp'", keep(master match) nogen
    gen double state_aid_over_lagged_gdp = ///
        state_aid_m_eur / gdp_m_eur_lag1 if gdp_m_eur_lag1 > 0
    keep country target_year state_aid_source_year state_aid_m_eur ///
        gdp_source_year gdp_m_eur_lag1 state_aid_over_lagged_gdp
    save "`support_control'"
}


/*=============================================================================
  Lagged annual explanatory series from ai_data.xlsx.
  Re-indexing source year s as target_year=s+1 is the Stata equivalent of the
  Python function align_previous_year_explanatory().
=============================================================================*/

* ICT investment share
import excel "$ai_data_file", sheet("ict_inv") firstrow clear
keep country year ict_share
destring year ict_share, replace force
rename year explanatory_source_year
gen int target_year = explanatory_source_year + 1
isid country target_year
tempfile x_ict
save "`x_ict'"

* AI/ICT-related HS export share
import excel "$ai_data_file", sheet("hs_export") firstrow clear
keep country year share
destring year share, replace force
rename share hs_export_share
rename year explanatory_source_year
gen int target_year = explanatory_source_year + 1
isid country target_year
tempfile x_hs_export
save "`x_hs_export'"

* AI incoming investment share
import excel "$ai_data_file", sheet("ai_inv") firstrow clear
keep country year share
destring year share, replace force
rename share ai_inv_share
rename year explanatory_source_year
gen int target_year = explanatory_source_year + 1
isid country target_year
tempfile x_ai_inv
save "`x_ai_inv'"


/*=============================================================================
  National-index/SOX correlation.

  This reproduces pandas rolling(8).corr(): correlation over the latest eight
  rows within country, requiring eight nonmissing return pairs. The quarterly
  correlations are then averaged within their source calendar year and that
  annual value is shifted one year forward for the regressions.
=============================================================================*/
import excel "$ai_data_file", sheet("index_sox") firstrow clear
keep quarter log_ret
rename log_ret sox_log_ret
gen int qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
isid qdate
tempfile sox_returns
save "`sox_returns'"

import excel "$ai_data_file", sheet("index_nat") firstrow clear
keep country quarter log_ret
gen int qdate = quarterly(quarter, "YQ")
format qdate %tq
drop quarter
merge m:1 qdate using "`sox_returns'", keep(match) nogen
sort country qdate

gen byte pair_nonmissing = !missing(log_ret, sox_log_ret)
gen double pair_x  = cond(pair_nonmissing, log_ret, 0)
gen double pair_y  = cond(pair_nonmissing, sox_log_ret, 0)
gen double pair_x2 = pair_x^2
gen double pair_y2 = pair_y^2
gen double pair_xy = pair_x * pair_y

by country (qdate): gen long   cum_n  = sum(pair_nonmissing)
by country (qdate): gen double cum_x  = sum(pair_x)
by country (qdate): gen double cum_y  = sum(pair_y)
by country (qdate): gen double cum_x2 = sum(pair_x2)
by country (qdate): gen double cum_y2 = sum(pair_y2)
by country (qdate): gen double cum_xy = sum(pair_xy)

foreach v in n x y x2 y2 xy {
    gen double win_`v' = cum_`v'
    by country (qdate): replace win_`v' = cum_`v' - cum_`v'[_n-8] if _n > 8
}

gen double win_cov_num = win_xy - (win_x * win_y / 8)
gen double win_var_x   = win_x2 - (win_x^2 / 8)
gen double win_var_y   = win_y2 - (win_y^2 / 8)
gen double roll_corr_8q = win_cov_num / sqrt(win_var_x * win_var_y) ///
    if _n >= 8 & win_n == 8 & win_var_x > 0 & win_var_y > 0

gen int explanatory_source_year = year(dofq(qdate))
collapse (mean) stock_semis_corr_annual=roll_corr_8q ///
    if !missing(roll_corr_8q), by(country explanatory_source_year)
gen int target_year = explanatory_source_year + 1
isid country target_year
tempfile x_stock_corr
save "`x_stock_corr'"


/*=============================================================================
  Common probit runner. Standardization and sample construction deliberately
  occur in the same order as in Python:
    1. inner-merge forecast rows with lagged X;
    2. standardize X over that merged panel;
    3. construct and standardize the above-median magnitude;
    4. left-merge the optional control and standardize it over the panel;
    5. let probit exclude rows missing a regressor.
=============================================================================*/
capture program drop run_probit_family
program define run_probit_family
    version 17
    syntax, FORECAST(string) XDATA(string) XVAR(name) PANELNAME(string) ///
        INCLUDEAID(integer) [SUPPORTDATA(string)]

    use "`forecast'", clear
    merge m:1 country target_year using "`xdata'", keep(match) nogen

    * pandas Series.std() and egen std() both use the sample SD (N-1).
    egen double x_z = std(`xvar')
    gen byte shock_year_dummy = inlist(target_year, 2020, 2022, 2025)
    gen double x_z_x_shock = x_z * shock_year_dummy

    quietly summarize `xvar', detail
    scalar x_sample_median = r(p50)
    gen double above_median_raw = cond(missing(`xvar'), ., ///
        cond(`xvar' > x_sample_median, `xvar', 0))
    egen double above_median_z = std(above_median_raw)
    gen double above_median_z_x_shock = ///
        above_median_z * shock_year_dummy

    local control_var ""
    if `includeaid' {
        merge m:1 country target_year using "`supportdata'", ///
            keep(master match) nogen
        egen double state_aid_over_lagged_gdp_z = ///
            std(state_aid_over_lagged_gdp)
        local control_var "state_aid_over_lagged_gdp_z"
    }

    encode country, gen(country_id)
    gen byte positive_surprise = growth_surprise > 0 ///
        if !missing(growth_surprise)

    order country country_id target_year explanatory_source_year ///
        growth_surprise `xvar' x_z shock_year_dummy x_z_x_shock ///
        above_median_raw above_median_z above_median_z_x_shock
    save "panel_`panelname'.dta", replace

    display ""
    display as result "`panelname': plain standardized-X probit"
    probit positive_surprise x_z `control_var' ///
        i.country_id i.target_year, vce(cluster country_id)

    display as result "`panelname': standardized-X plus shock interaction"
    probit positive_surprise x_z x_z_x_shock `control_var' ///
        i.country_id i.target_year, vce(cluster country_id)

    display as result "`panelname': above-median magnitude plus interaction"
    probit positive_surprise above_median_z above_median_z_x_shock ///
        `control_var' i.country_id i.target_year, vce(cluster country_id)
end


/*=============================================================================
  Twelve probits: three specifications for each explanatory variable.
=============================================================================*/
local support_option ""
if `include_state_aid_control' {
    local support_option `"supportdata("`support_control'")"'
    display as text "State Aid control ON: standardized Aid(t)/GDP(t-1)."
}
else {
    display as text "State Aid control OFF."
}

run_probit_family, forecast("`forecast_base'") xdata("`x_ict'") ///
    xvar(ict_share) panelname(ict) ///
    includeaid(`include_state_aid_control') `support_option'

run_probit_family, forecast("`forecast_base'") xdata("`x_hs_export'") ///
    xvar(hs_export_share) panelname(hs_export) ///
    includeaid(`include_state_aid_control') `support_option'

run_probit_family, forecast("`forecast_base'") xdata("`x_stock_corr'") ///
    xvar(stock_semis_corr_annual) panelname(stock_corr) ///
    includeaid(`include_state_aid_control') `support_option'

run_probit_family, forecast("`forecast_base'") xdata("`x_ai_inv'") ///
    xvar(ai_inv_share) panelname(ai_inv) ///
    includeaid(`include_state_aid_control') `support_option'

display ""
display as result "Finished: 12 probits estimated with country-clustered SEs."
