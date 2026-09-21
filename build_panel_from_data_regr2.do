/*==============================================================================
build_panel_from_data_regr.do

Reads the "data_regr" sheet of ict_growth_surprise_scatter.xlsx and
reshapes it into panel-format Stata datasets, ready for you to run your
own xtreg / reghdfe / regress commands directly, without rebuilding
anything from the raw Excel file yourself.

IMPORTANT, READ BEFORE USING -- WHY THIS WRITES FOUR SEPARATE .dta FILES,
NOT ONE COMBINED PANEL:
The (country, target_year) pair is NOT a unique row identifier in this
data -- confirmed directly against the workbook: e.g. Austria/2020
appears FOUR times with the SAME ict_share but DIFFERENT
growth_surprise values, because growth_surprise depends on which
EC forecast VINTAGE (Spring/Autumn, published in different years) is
being compared to the eventual realized outcome for that same target
year -- and "data_regr" carries no vintage-identifying column that
would let those rows be told apart. A single combined panel keyed on
(country, target_year) alone would therefore either silently multiply
rows in a meaningless country x-year cross-join when merged, or need an
identifier that simply is not in the source data.

What IS safe, and what this do-file does: WITHIN one explanatory
variable (e.g. ICT), the three specifications (plain X, X+interaction,
above-median value+interaction) come from the exact SAME underlying
rows in the SAME order (confirmed: country/target_year/growth_surprise
match row-for-row across all three blocks) -- so their extra columns
(interaction terms, above-median values) are appended POSITIONALLY
(not merged on a key), which is safe and loses no rows. This gives one
clean, complete panel PER explanatory variable: panel_ict.dta,
panel_hs_export.dta, panel_stock_corr.dta, panel_ai_inv.dta -- each
with every specification's variables as columns, ready for direct use.

CONSEQUENCE FOR YOUR STATA COMMANDS: because (country, target_year)
repeats, Stata's `xtset` (which needs a unique entity-time key) is NOT
used here -- these are not declared as `xt` panels. This does not stop
you from running the exact two-way fixed-effects regressions
build_scatter.py itself runs: use `reghdfe` with `absorb(country
target_year)`, or plain `regress` with `i.country i.target_year`
dummies, both shown in the example commands printed at the end of each
section below. Repeated (country, target_year) rows are ordinary,
valid observations for that kind of regression -- they are only
incompatible with `xtset` specifically.

HONESTY NOTE ON ROW RANGES: the cell ranges below are HARDCODED to this
exact workbook, read directly from its "data_regr" sheet when this
do-file was written. If you regenerate ict_growth_surprise_scatter.xlsx
from a DIFFERENT run of build_scatter.py, the row counts in each block
-- and therefore every range below -- will likely shift, and this
do-file will need updating to match (Stata will error with an "invalid
cell range" or show a visibly wrong row count if so).

USAGE: edit the `cd` and `global xlsxfile` lines below, then run this
whole do-file. It writes four .dta files in the same folder.
==============================================================================*/
adopath + "C:\Program Files\StataADO\ftoolspackage"
adopath + "C:\Program Files\StataADO\reghdfepackage"

clear all
set more off
cd "G:\EBO\MB\vdEnd\DSW_G\risk mngt"
*local data_file "ict_growth_surprise_scatter.xlsx"

* ----------------------------------------------------------------------------
import excel "ict_growth_surprise_scatter_hs.xlsx", ///
sheet("data_regr") ///
cellrange(A2134:F2682) ///
firstrow clear
/*==============================================================================
  ICT investment share -- blocks at rows 1 (plain), 2133 (interaction),
  4265 (above-median).
==============================================================================*/
import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A2134:F2682) firstrow clear
rename interaction ict_x_shock
tempfile ict_interact
save "`ict_interact'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A4266:G4814) firstrow clear
rename above_median ict_above_median
rename above_x_shock ict_above_x_shock
keep ict_above_median ict_above_x_shock
tempfile ict_abovemed
save "`ict_abovemed'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A2:D550) firstrow clear
* Positional append (NOT a key-based merge) -- see the note at the top
* of this file for why: confirmed country/target_year/growth_surprise
* match row-for-row across all three ICT blocks, in the same order.
merge 1:1 _n using "`ict_interact'", nogenerate update replace
merge 1:1 _n using "`ict_abovemed'", nogenerate
order country target_year growth_surprise ict_share shock_year_dummy ///
    ict_x_shock ict_above_median ict_above_x_shock
label variable growth_surprise "realized - forecast annual GDP growth, pct points"
label variable ict_share "ICT investment share"
label variable shock_year_dummy "1 if target_year in {2020, 2022, 2025}"
label variable ict_x_shock "ict_share * shock_year_dummy"
label variable ict_above_median "ict_share where > its own median, else 0 (NOT a 0/1 dummy)"
label variable ict_above_x_shock "ict_above_median * shock_year_dummy"
save "panel_ict.dta", replace
display as result "panel_ict.dta written: " _N " rows."

/*==============================================================================
  AI/ICT-related HS export share -- blocks at rows 552, 2684, 4816.
==============================================================================*/
import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A2685:F3265) firstrow clear
rename interaction hs_export_x_shock
tempfile hs_interact
save "`hs_interact'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A4817:G5397) firstrow clear
rename above_median hs_export_above_median
rename above_x_shock hs_export_above_x_shock
keep hs_export_above_median hs_export_above_x_shock
tempfile hs_abovemed
save "`hs_abovemed'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A553:D1133) firstrow clear
merge 1:1 _n using "`hs_interact'", nogenerate update replace
merge 1:1 _n using "`hs_abovemed'", nogenerate
order country target_year growth_surprise hs_export_share shock_year_dummy ///
    hs_export_x_shock hs_export_above_median hs_export_above_x_shock
label variable growth_surprise "realized - forecast annual GDP growth, pct points"
label variable hs_export_share "AI/ICT-related HS export share"
label variable shock_year_dummy "1 if target_year in {2020, 2022, 2025}"
label variable hs_export_x_shock "hs_export_share * shock_year_dummy"
label variable hs_export_above_median "hs_export_share where > its own median, else 0 (NOT a 0/1 dummy)"
label variable hs_export_above_x_shock "hs_export_above_median * shock_year_dummy"
save "panel_hs_export.dta", replace
display as result "panel_hs_export.dta written: " _N " rows."

/*==============================================================================
  National vs. semiconductor index correlation -- blocks at rows 1135,
  3267, 5399.
==============================================================================*/
import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A3268:F3820) firstrow clear
rename interaction stock_corr_x_shock
tempfile sc_interact
save "`sc_interact'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A5400:G5952) firstrow clear
rename above_median stock_corr_above_median
rename above_x_shock stock_corr_above_x_shock
keep stock_corr_above_median stock_corr_above_x_shock
tempfile sc_abovemed
save "`sc_abovemed'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A1136:D1688) firstrow clear
merge 1:1 _n using "`sc_interact'", nogenerate update replace
merge 1:1 _n using "`sc_abovemed'", nogenerate
order country target_year growth_surprise stock_semis_corr_annual shock_year_dummy ///
    stock_corr_x_shock stock_corr_above_median stock_corr_above_x_shock
label variable growth_surprise "realized - forecast annual GDP growth, pct points"
label variable stock_semis_corr_annual "National vs. semiconductor index correlation"
label variable shock_year_dummy "1 if target_year in {2020, 2022, 2025}"
label variable stock_corr_x_shock "stock_semis_corr_annual * shock_year_dummy"
label variable stock_corr_above_median "stock_semis_corr_annual where > its own median, else 0 (NOT a 0/1 dummy)"
label variable stock_corr_above_x_shock "stock_corr_above_median * shock_year_dummy"
save "panel_stock_corr.dta", replace
display as result "panel_stock_corr.dta written: " _N " rows."

/*==============================================================================
  AI incoming investment share -- blocks at rows 1690, 3822, 5954.
==============================================================================*/
import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A3823:F4263) firstrow clear
rename interaction ai_inv_x_shock
tempfile ai_interact
save "`ai_interact'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A5955:G6395) firstrow clear
rename above_median ai_inv_above_median
rename above_x_shock ai_inv_above_x_shock
keep ai_inv_above_median ai_inv_above_x_shock
tempfile ai_abovemed
save "`ai_abovemed'"

import excel "ict_growth_surprise_scatter.xlsx", sheet("data_regr") cellrange(A1691:D2131) firstrow clear
merge 1:1 _n using "`ai_interact'", nogenerate update replace
merge 1:1 _n using "`ai_abovemed'", nogenerate
order country target_year growth_surprise ai_inv_share shock_year_dummy ///
    ai_inv_x_shock ai_inv_above_median ai_inv_above_x_shock
label variable growth_surprise "realized - forecast annual GDP growth, pct points"
label variable ai_inv_share "AI incoming investment share (per avg. quarterly GDP)"
label variable shock_year_dummy "1 if target_year in {2020, 2022, 2025}"
label variable ai_inv_x_shock "ai_inv_share * shock_year_dummy"
label variable ai_inv_above_median "ai_inv_share where > its own median, else 0 (NOT a 0/1 dummy)"
label variable ai_inv_above_x_shock "ai_inv_above_median * shock_year_dummy"
save "panel_ai_inv.dta", replace
display as result "panel_ai_inv.dta written: " _N " rows."

/*==============================================================================
  Example commands (reghdfe needs installing once: ssc install reghdfe
  -- or use the plain regress/i. version, which needs nothing extra).
==============================================================================*/
display ""
display as text "Example commands on panel_ict.dta (same pattern for the other three files):"
display as text `"  use panel_ict.dta, clear"'
display as text `"  reghdfe growth_surprise ict_share, absorb(country target_year) vce(cluster country)"'
display as text `"  reghdfe growth_surprise ict_share ict_x_shock, absorb(country target_year) vce(cluster country)"'
display as text `"  reghdfe growth_surprise ict_above_median ict_above_x_shock, absorb(country target_year) vce(cluster country)"'
display as text `"  * -- or, without reghdfe: --"'
display as text `"  regress growth_surprise ict_share i.country i.target_year, vce(cluster country)"'

/*=============================================================================
  Probit checks matching build_scatter.py

  For every explanatory variable below:
    1. If an EXISTING 2025 observation has a missing X, replace it with that
       same country's mean 2024 X value. (The mean is innocuous here because
       X is constant across forecast vintages within a country-year.)
    2. Standardize X only AFTER applying that fallback: z=(X-mean)/SD.
    3. Rebuild the interaction as standardized_X * shock_year_dummy. The
       interaction is not separately standardized, matching the Python code.
    4. Build the above-median variant AFTER the fallback: raw X where X is
       strictly above the full-sample median, 0 otherwise (not a dummy), then
       standardize that derived variable and interact it with shock_year_dummy.
    5. Cluster probit standard errors by country_id.

  Important: this fallback can repair missing X values on 2025 rows that are
  present in the panel. It cannot recreate 2025 observations that were already
  removed upstream by an inner merge. Regenerate the Excel workbook with the
  attached Python script to include such rows (notably HS exports, for which
  Python creates country-specific 2025 values from 2024 before merging).
=============================================================================*/

* ----- Probit model 1: ICT investment share -------------------------------
use "panel_ict.dta", clear
bysort country: egen double ict_share_2024 = ///
    mean(cond(target_year == 2024, ict_share, .))
replace ict_share = ict_share_2024 if target_year == 2025 & missing(ict_share)
drop ict_share_2024
egen double ict_share_z = std(ict_share)
replace shock_year_dummy = inlist(target_year, 2020, 2022, 2025)
gen double ict_z_x_shock = ict_share_z * shock_year_dummy
quietly summarize ict_share, detail
local ict_median = r(p50)
gen double ict_above_raw = cond(missing(ict_share), ., ///
    cond(ict_share > `ict_median', ict_share, 0))
egen double ict_above_z = std(ict_above_raw)
gen double ict_above_z_x_shock = ict_above_z * shock_year_dummy
encode country, gen(country_id)
gen byte positive_surprise = growth_surprise > 0 if !missing(growth_surprise)

probit positive_surprise ict_share_z i.country_id i.target_year, ///
    vce(cluster country_id)
probit positive_surprise ict_share_z ict_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)
probit positive_surprise ict_above_z ict_above_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)

* ----- Probit model 2: AI/ICT-related HS export share ---------------------
use "panel_hs_export.dta", clear
bysort country: egen double hs_export_2024 = ///
    mean(cond(target_year == 2024, hs_export_share, .))
replace hs_export_share = hs_export_2024 ///
    if target_year == 2025 & missing(hs_export_share)
drop hs_export_2024
egen double hs_export_share_z = std(hs_export_share)
replace shock_year_dummy = inlist(target_year, 2020, 2022, 2025)
gen double hs_export_z_x_shock = hs_export_share_z * shock_year_dummy
quietly summarize hs_export_share, detail
local hs_export_median = r(p50)
gen double hs_export_above_raw = cond(missing(hs_export_share), ., ///
    cond(hs_export_share > `hs_export_median', hs_export_share, 0))
egen double hs_export_above_z = std(hs_export_above_raw)
gen double hs_export_above_z_x_shock = hs_export_above_z * shock_year_dummy
encode country, gen(country_id)
gen byte positive_surprise = growth_surprise > 0 if !missing(growth_surprise)

probit positive_surprise hs_export_share_z i.country_id i.target_year, ///
    vce(cluster country_id)
probit positive_surprise hs_export_share_z hs_export_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)
probit positive_surprise hs_export_above_z hs_export_above_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)

* ----- Probit model 3: national/semiconductor-index correlation -----------
use "panel_stock_corr.dta", clear
bysort country: egen double stock_corr_2024 = ///
    mean(cond(target_year == 2024, stock_semis_corr_annual, .))
replace stock_semis_corr_annual = stock_corr_2024 ///
    if target_year == 2025 & missing(stock_semis_corr_annual)
drop stock_corr_2024
egen double stock_corr_z = std(stock_semis_corr_annual)
replace shock_year_dummy = inlist(target_year, 2020, 2022, 2025)
gen double stock_corr_z_x_shock = stock_corr_z * shock_year_dummy
quietly summarize stock_semis_corr_annual, detail
local stock_corr_median = r(p50)
gen double stock_corr_above_raw = cond(missing(stock_semis_corr_annual), ., ///
    cond(stock_semis_corr_annual > `stock_corr_median', ///
    stock_semis_corr_annual, 0))
egen double stock_corr_above_z = std(stock_corr_above_raw)
gen double stock_corr_above_z_x_shock = stock_corr_above_z * shock_year_dummy
encode country, gen(country_id)
gen byte positive_surprise = growth_surprise > 0 if !missing(growth_surprise)

probit positive_surprise stock_corr_z i.country_id i.target_year, ///
    vce(cluster country_id)
probit positive_surprise stock_corr_z stock_corr_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)
probit positive_surprise stock_corr_above_z stock_corr_above_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)

* ----- Probit model 4: AI incoming investment share -----------------------
use "panel_ai_inv.dta", clear
bysort country: egen double ai_inv_2024 = ///
    mean(cond(target_year == 2024, ai_inv_share, .))
replace ai_inv_share = ai_inv_2024 if target_year == 2025 & missing(ai_inv_share)
drop ai_inv_2024
egen double ai_inv_share_z = std(ai_inv_share)
replace shock_year_dummy = inlist(target_year, 2020, 2022, 2025)
gen double ai_inv_z_x_shock = ai_inv_share_z * shock_year_dummy
quietly summarize ai_inv_share, detail
local ai_inv_median = r(p50)
gen double ai_inv_above_raw = cond(missing(ai_inv_share), ., ///
    cond(ai_inv_share > `ai_inv_median', ai_inv_share, 0))
egen double ai_inv_above_z = std(ai_inv_above_raw)
gen double ai_inv_above_z_x_shock = ai_inv_above_z * shock_year_dummy
encode country, gen(country_id)
gen byte positive_surprise = growth_surprise > 0 if !missing(growth_surprise)

* Retain the do-file's original AI-investment OLS check, now using the
* standardized explanatory variable and its standardized-X interaction.
reghdfe growth_surprise ai_inv_share_z ai_inv_z_x_shock, ///
    absorb(country target_year) vce(cluster country_id)

probit positive_surprise ai_inv_share_z i.country_id i.target_year, ///
    vce(cluster country_id)
probit positive_surprise ai_inv_share_z ai_inv_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)
probit positive_surprise ai_inv_above_z ai_inv_above_z_x_shock ///
    i.country_id i.target_year, vce(cluster country_id)
