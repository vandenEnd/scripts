import pandas as pd
import numpy as np
import openpyxl
from openpyxl.chart import ScatterChart, BarChart, LineChart, Series, Reference
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.styles import Font, Alignment, PatternFill

# --- Load source data ---
forecast = pd.read_excel("ec_forecast_vs_realized.xlsx")
ict = pd.read_excel("ai_data.xlsx", sheet_name="ict_inv")[["country", "year", "ict_share"]]
ict = ict.rename(columns={"year": "target_year"})

# AI/ICT-related export exposure now comes from the "eur_export" sheet's
# "share" column (CPA 2.2-classified EU trade data, Comext DS-059366),
# per explicit instruction -- REPLACING the previous "hs_export" sheet's
# "share" column (HS/CN-classified OECD BIMTS data) everywhere in this
# script: charts, unconditional bivariate tests, and every regression.
eur_export = pd.read_excel("ai_data.xlsx", sheet_name="eur_export")[["country", "year", "share"]]
eur_export = eur_export.rename(columns={"year": "target_year", "share": "eur_export_share"})

ai_inv = pd.read_excel("ai_data.xlsx", sheet_name="ai_inv")[["country", "year", "share"]]
ai_inv = ai_inv.rename(columns={"year": "target_year", "share": "ai_inv_share"})

# Contemporaneous State Aid control. Unlike the four focal explanatory
# variables, support_share uses the SAME year t as growth_surprise. If a
# country's 2025 value is unavailable, use that country's 2024 value; an
# observed 2025 value always takes precedence.
support = pd.read_excel("ai_data.xlsx", sheet_name="support")[[
    "country", "year", "support_share"
]].rename(columns={"year": "target_year"})
support["support_source_year"] = support["target_year"]
support_2025_fallback = support.loc[
    support["target_year"].eq(2024)
    & ~support["country"].isin(
        support.loc[support["target_year"].eq(2025), "country"])
].copy()
support_2025_fallback["target_year"] = 2025
support_control = pd.concat([support, support_2025_fallback], ignore_index=True)
support_control = support_control.drop_duplicates(
    ["country", "target_year"], keep="first")

# National-vs-semiconductor index correlation: rolling 8-QUARTER
# correlation between each country's national index log-return and the
# global semiconductor index (^SOX) log-return, THEN annualized by
# averaging the (up to 4) quarterly rolling-correlation values that
# fall within each calendar year -- mathematically verified separately
# (rolling-window correlation matched a manual same-window
# calculation exactly; the annual average matched a manual mean of
# that year's quarterly values exactly).
index_nat = pd.read_excel("ai_data.xlsx", sheet_name="index_nat")[["country", "quarter", "log_ret"]]
index_sox = pd.read_excel("ai_data.xlsx", sheet_name="index_sox")[["quarter", "log_ret"]]
index_sox = index_sox.rename(columns={"log_ret": "sox_log_ret"})
idx_merged = index_nat.merge(index_sox, on="quarter", how="inner")
idx_merged["quarter"] = pd.PeriodIndex(idx_merged["quarter"], freq="Q")
idx_merged = idx_merged.sort_values(["country", "quarter"])
idx_merged["roll_corr_8q"] = idx_merged.groupby("country").apply(
    lambda g: g["log_ret"].rolling(8).corr(g["sox_log_ret"])
).reset_index(level=0, drop=True)
idx_merged["target_year"] = idx_merged["quarter"].dt.year
stock_corr_annual = (idx_merged.dropna(subset=["roll_corr_8q"])
                      .groupby(["country", "target_year"])["roll_corr_8q"]
                      .mean().reset_index()
                      .rename(columns={"roll_corr_8q": "stock_semis_corr_annual"}))

def align_previous_year_explanatory(df):
    """Re-index explanatory year s to outcome year t=s+1."""
    lagged = df.copy()
    lagged["explanatory_source_year"] = lagged["target_year"]
    lagged["target_year"] = lagged["target_year"] + 1
    return lagged


ict_lagged = align_previous_year_explanatory(ict)
eur_export_lagged = align_previous_year_explanatory(eur_export)
stock_corr_annual_lagged = align_previous_year_explanatory(stock_corr_annual)
ai_inv_lagged = align_previous_year_explanatory(ai_inv)

merged_ict = forecast.merge(ict_lagged, on=["country", "target_year"], how="inner")
merged_eur = forecast.merge(eur_export_lagged, on=["country", "target_year"], how="inner")
merged_corr = forecast.merge(
    stock_corr_annual_lagged, on=["country", "target_year"], how="inner")
merged_ai_inv = forecast.merge(ai_inv_lagged, on=["country", "target_year"], how="inner")


def add_support_control(df):
    """Attach same-year support_share (2024 fallback only for missing 2025)."""
    return df.merge(
        support_control,
        on=["country", "target_year"],
        how="left",
        validate="many_to_one",
    )


merged_ict = add_support_control(merged_ict)
merged_eur = add_support_control(merged_eur)
merged_corr = add_support_control(merged_corr)
merged_ai_inv = add_support_control(merged_ai_inv)
print(f"previous-year ict_share merge: {len(merged_ict)} rows (from {len(forecast)} forecast rows)")
print(f"previous-year eur_export share merge: {len(merged_eur)} rows (from {len(forecast)} forecast rows)")
print(f"previous-year stock/semis correlation merge: {len(merged_corr)} rows (from {len(forecast)} forecast rows)")
print(f"previous-year ai_inv share merge: {len(merged_ai_inv)} rows (from {len(forecast)} forecast rows)")

wb = openpyxl.Workbook()
wb.remove(wb.active)  # remove the default empty sheet; we add our own below

bold = Font(bold=True)


def write_data_sheet(wb, sheet_name, df, explanatory_col, explanatory_header):
    """
    Writes one data sheet: country, vintage, vintage_round, target_year,
    forecast_growth_annual_pct, realized_growth_annual_pct,
    growth_surprise_pct (LIVE FORMULA = realized - forecast),
    <explanatory_header>, explanatory_source_year. Returns
    (worksheet, last_data_row).
    """
    ws = wb.create_sheet(sheet_name)
    headers = ["country", "vintage", "vintage_round", "target_year",
               "forecast_growth_annual_pct", "realized_growth_annual_pct",
               "growth_surprise_pct", explanatory_header,
               "explanatory_source_year"]
    for col_idx, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col_idx, value=h)
        c.font = bold

    df_sorted = df.sort_values(["country", "target_year", "vintage_round"]).reset_index(drop=True)
    for row_idx, row in enumerate(df_sorted.itertuples(index=False), start=2):
        ws.cell(row=row_idx, column=1, value=row.country)
        ws.cell(row=row_idx, column=2, value=row.vintage)
        ws.cell(row=row_idx, column=3, value=row.vintage_round)
        ws.cell(row=row_idx, column=4, value=int(row.target_year))
        ws.cell(row=row_idx, column=5, value=row.forecast_growth_annual_pct)
        ws.cell(row=row_idx, column=6, value=row.realized_growth_annual_pct)
        ws.cell(row=row_idx, column=7,
                value=f'=IF(OR(F{row_idx}="",E{row_idx}=""),"",F{row_idx}-E{row_idx})')
        ws.cell(row=row_idx, column=8, value=getattr(row, explanatory_col))
        ws.cell(row=row_idx, column=9,
                value=int(row.explanatory_source_year))

    for col_idx in range(1, 10):
        ws.column_dimensions[chr(64 + col_idx)].width = 18

    last_row = len(df_sorted) + 1
    return ws, last_row


def write_chart_sheet(wb, sheet_name, data_ws, last_row, title, x_title, marker_color,
                       trendline=False, show_legend=False,
                       x_min=None, x_max=None, x_major=None,
                       y_min=None, y_max=None, y_major=None):
    """Native, editable Excel scatter chart: X=explanatory column (H),
    Y=growth_surprise_pct (G), points only (no connecting line).
    trendline=True adds a linear regression line PLUS live
    SLOPE/INTERCEPT/RSQ formulas next to the chart (columns J:K), so
    the visual trend is backed by an exact number, not eyeballing.

    show_legend: defaults to False -- with only ONE series on this
    chart, a legend showing just "Country-year observations" adds
    nothing (and was the setting in place when the "list of individual
    point values" rendering bug was reported); set True only if you
    genuinely want that single-entry legend shown.

    x_min/x_max/x_major, y_min/y_max/y_major: OPTIONAL explicit axis
    bounds/step -- left at None (auto-scale) by default, since this
    function is REUSED for several very different explanatory
    variables (ict_share ~0-0.2, eur_export_share ~0.0004, stock_corr
    ~-1 to 1, target_year ~2011-2026) that cannot share one hardcoded
    scale. Pass explicit values only for a SPECIFIC call site where
    you want to force a fixed scale (e.g. to exactly match another
    chart's axes for visual comparison) -- every other call site stays
    auto-scaled unaffected.
    """
    from openpyxl.chart.trendline import Trendline

    chart = ScatterChart()
    # EXPLICIT scatterStyle: openpyxl leaves this unset (None) by
    # default, which means the raw saved XML has NO <scatterStyle>
    # element at all -- confirmed by inspecting a genuinely
    # unrecalculated save. Without this hint, Excel's own native
    # rendering (as opposed to LibreOffice's recalc pass, which adds
    # a scatterStyle of its own while rewriting the file) can
    # misinterpret the chart, which is the likely cause of a "list of
    # individual point values" rendering artifact instead of a clean
    # continuous X/Y scatter with a normal single-entry legend.
    chart.scatterStyle = "marker"
    chart.title = title
    chart.style = 13
    chart.x_axis.title = x_title
    chart.y_axis.title = "Growth surprise (realized - forecast, pp)"
    # EXPLICIT axis positions: openpyxl's ScatterChart does not reliably
    # default x_axis to the bottom and y_axis to the left on its own --
    # confirmed from a real saved file's chart XML, where BOTH axes had
    # axPos="l" (left), an invalid/ambiguous configuration that made
    # Excel unable to render tick labels on either axis correctly (it
    # fell back to listing every point in the legend instead). Setting
    # this explicitly on every chart built below fixes that.
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"
    # Also make sure axis tick labels and lines are switched on (not
    # left at whatever ambiguous default resulted from the axPos bug) --
    # belt-and-braces alongside the axPos fix above.
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    if x_min is not None:
        chart.x_axis.scaling.min = x_min
    if x_max is not None:
        chart.x_axis.scaling.max = x_max
    if x_major is not None:
        chart.x_axis.majorUnit = x_major
    if y_min is not None:
        chart.y_axis.scaling.min = y_min
    if y_max is not None:
        chart.y_axis.scaling.max = y_max
    if y_major is not None:
        chart.y_axis.majorUnit = y_major
    chart.height = 12
    chart.width = 22

    x_values = Reference(data_ws, min_col=8, min_row=2, max_row=last_row)
    y_values = Reference(data_ws, min_col=7, min_row=2, max_row=last_row)
    series = Series(y_values, x_values, title="Country-year observations")
    series.marker = Marker(symbol="circle", size=6)
    series.marker.graphicalProperties = GraphicalProperties(solidFill=marker_color)
    series.marker.graphicalProperties.line.solidFill = marker_color
    series.graphicalProperties = GraphicalProperties()
    series.graphicalProperties.line.noFill = True
    if trendline:
        series.trendline = Trendline(trendlineType="linear", dispEq=True, dispRSqr=True)
    chart.series.append(series)

    if not show_legend:
        # With only ONE series, a legend showing just its single name
        # ("Country-year observations") adds nothing -- and was present
        # when the "list of individual point values" rendering bug was
        # reported. Removing it entirely, rather than trying to keep it
        # in a fixed/correct state, is the most direct guarantee that
        # nothing unexpected can render in that area.
        chart.legend = None

    ws_chart = wb.create_sheet(sheet_name)
    ws_chart.add_chart(chart, "B2")

    if trendline:
        g_range = f"'{data_ws.title}'!$G$2:$G${last_row}"
        h_range = f"'{data_ws.title}'!$H$2:$H${last_row}"
        labels = ["Slope (pp per year)", "Intercept", "R-squared", "Correlation"]
        formulas = [f"=SLOPE({g_range},{h_range})", f"=INTERCEPT({g_range},{h_range})",
                    f"=RSQ({g_range},{h_range})", f"=CORREL({g_range},{h_range})"]
        for i, (label, formula) in enumerate(zip(labels, formulas)):
            c1 = ws_chart.cell(row=2 + i, column=13, value=label)
            c1.font = bold
            c2 = ws_chart.cell(row=2 + i, column=14, value=formula)
            c2.number_format = "0.0000"
        ws_chart.column_dimensions["M"].width = 22
        ws_chart.column_dimensions["N"].width = 14

    return ws_chart


# --- Summary sheet, created FIRST (before everything else) so it is
# the workbook's leftmost tab; every section below writes into it as
# the script progresses (correlation table at the top, then the
# below/above-median blocks further down), rather than each having its
# own separate sheet.
ws_summary = wb.create_sheet("summary", 0)
title_cell = ws_summary.cell(row=1, column=2, value="Correlation ICT/AI & growth surprises")
title_cell.font = Font(bold=True, size=14)

# --- Six (data, chart) sheet pairs: 3 explanatory variables x 3 periods ---
specs = [
    ("ict_share", merged_ict, "ict_share", "Previous-year ICT investment share", "1F77B4",
     "vs. Previous-year ICT investment share"),
    ("eur_export", merged_eur, "eur_export_share", "Previous-year AI/ICT-related EU export share", "D62728",
     "vs. Previous-year AI/ICT-related EU export share"),
    ("stock_corr", merged_corr, "stock_semis_corr_annual",
     "Previous-year national vs. semiconductor index correlation (8Q rolling, annualized)", "2CA02C",
     "vs. national-semiconductor index correlation"),
    ("ai_inv_share", merged_ai_inv, "ai_inv_share",
     "Previous-year AI incoming investment share (per avg. quarterly GDP)", "E377C2",
     "vs. Previous-year AI incoming investment share"),
]
periods = [
    ("full", None, None, "full sample"),
    ("since2020", 2020, None, "since 2020"),
    ("till2019", None, 2019, "through 2019"),
]

sheet_registry = {}  # var_label -> {period_label: (data_sheet_name, last_row)}
period_labels_ordered = [p[3] for p in periods]  # preserves the periods list's order

for var_key, df, col, header, color, title_suffix in specs:
    sheet_registry.setdefault(header, {})
    for period_key, min_year, max_year, period_label in periods:
        df_period = df
        if min_year is not None:
            df_period = df_period[df_period["target_year"] >= min_year]
        if max_year is not None:
            df_period = df_period[df_period["target_year"] <= max_year]
        data_sheet_name = f"data_{var_key}_{period_key}"
        chart_sheet_name = f"chart_{var_key}_{period_key}"
        # Excel caps sheet names at 31 characters and openpyxl SILENTLY
        # TRUNCATES anything longer (it only emits a UserWarning that is
        # easy to miss in a long run) -- which previously produced a
        # sheet actually named "chart_..._since202", quietly losing the
        # final character and breaking any name-based cross-reference.
        # Fail loudly here instead, so a future rename of var_key that
        # pushes a name over the limit is caught immediately rather than
        # corrupting sheet names unnoticed.
        for _name in (data_sheet_name, chart_sheet_name):
            if len(_name) > 31:
                raise SystemExit(
                    f"\nSheet name '{_name}' is {len(_name)} characters, over "
                    f"Excel's 31-character limit -- it would be silently "
                    f"truncated. Shorten the var_key '{var_key}' in the "
                    f"`specs` list above."
                )
        ws_data, last_row = write_data_sheet(wb, data_sheet_name, df_period, col, header)
        chart_title = f"Growth surprise {title_suffix} -- {period_label}"
        # Fixed axis scale ONLY for ict_share/full, to exactly match a
        # previously reviewed reference chart for that specific
        # combination -- every other chart stays auto-scaled (passing
        # these bounds into every call would badly distort
        # eur_export_share and stock_corr, which sit on completely
        # different scales).
        axis_kwargs = {}
        if var_key == "ict_share" and period_key == "full":
            axis_kwargs = dict(x_min=0, x_max=0.20, x_major=0.02,
                                y_min=-20, y_max=25, y_major=5)
        write_chart_sheet(wb, chart_sheet_name, ws_data, last_row, chart_title, header, color,
                           **axis_kwargs)
        sheet_registry[header][period_label] = (data_sheet_name, last_row)
        print(f"  {data_sheet_name}: {last_row - 1} rows -> {chart_sheet_name}")

# --- Correlation table, written into the TOP of the shared "summary"
# sheet (rows 3-6: header + one row per explanatory variable) -- one
# column per period, each cell a LIVE =CORREL(...) formula referencing
# that variable/period's own data sheet (growth_surprise_pct in col G,
# the explanatory variable in col H) -- recalculates automatically if
# the underlying data changes, rather than a Python-computed snapshot.
# This used to be its own separate "correlation_summary" sheet; it now
# lives at the top of "summary" instead, per the reviewed reference
# layout this function reproduces.
corr_header_row = 3
c = ws_summary.cell(row=corr_header_row, column=1, value="explanatory_variable")
c.font = bold
for col_idx, period_label in enumerate(period_labels_ordered, start=2):
    c = ws_summary.cell(row=corr_header_row, column=col_idx, value=period_label)
    c.font = bold

for row_offset, (var_label, period_dict) in enumerate(sheet_registry.items()):
    row_idx = corr_header_row + 1 + row_offset
    c = ws_summary.cell(row=row_idx, column=1, value=var_label)
    c.font = bold
    for col_idx, period_label in enumerate(period_labels_ordered, start=2):
        data_sheet, last_row = period_dict[period_label]
        g_range = f"'{data_sheet}'!$G$2:$G${last_row}"
        h_range = f"'{data_sheet}'!$H$2:$H${last_row}"
        cell = ws_summary.cell(row=row_idx, column=col_idx,
                                value=f"=CORREL({g_range},{h_range})")
        cell.number_format = "0.000"

ws_summary.column_dimensions["A"].width = 45
for col_idx in range(2, 2 + len(period_labels_ordered)):
    ws_summary.column_dimensions[chr(64 + col_idx)].width = 16


def significance_stars(x1, n1, x2, n2, symbol="*"):
    """
    Two-proportion z-test (pooled variance) between two independent
    "share positive" proportions (p1=x1/n1, p2=x2/n2) -- used to test
    whether two "share positive surprises" cells genuinely differ, not
    just numerically but statistically. Returns symbol/symbol*2/
    symbol*3 for p<0.10/0.05/0.01 respectively, "" if not significant
    or if either n is 0 (comparison undefined, matching this project's
    existing "n/a" handling for zero-count cells elsewhere).

    symbol: "*" (default) for the WITHIN-period below-vs-above
    comparisons (D/H columns); "+" for the ACROSS-period (all years
    vs. shock years) comparisons (F/G/H columns) -- two different
    symbols so a cell needing BOTH (H, which is both an above-vs-below
    AND an across-period comparison site) can show both distinctly
    combined, e.g. "above +++ **", rather than an ambiguous "*****".
    """
    from scipy import stats as _stats
    if n1 == 0 or n2 == 0:
        return ""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)
    if denom <= 0:
        return ""
    z = (p1 - p2) / (denom ** 0.5)
    p_value = 2 * (1 - _stats.norm.cdf(abs(z)))
    if p_value < 0.01:
        return symbol * 3
    if p_value < 0.05:
        return symbol * 2
    if p_value < 0.10:
        return symbol
    return ""


def write_summary_block(ws_summary, next_row, source_sheet_name, data_sheet_title,
                         below_start, below_end, above_start, above_end, col_offset=0):
    """
    Writes ONE "full sample / below / above / share positive surprises"
    block into the SHARED summary sheet (ws_summary), starting at
    next_row -- preceded by a label row naming which data sheet this
    block belongs to (data_sheet_title), so multiple blocks stacked in
    one sheet stay traceable to their source. LIVE formulas reference
    the ORIGINAL data sheet (data_sheet_title) directly via a
    cross-sheet reference -- the underlying numbers are never copied,
    so editing the source data sheet still updates this summary
    automatically.

    col_offset: 0 for the LEFT block (label/values start at column B),
    4 for the RIGHT block (label/values start at column F, i.e. one
    blank column E of separation from the left block) -- lets two
    related blocks (e.g. a variable's "cluster" and "yrs" sheets) sit
    SIDE BY SIDE on the same rows instead of stacked vertically. The
    "share positive surprises" row LABEL (column A) is only written
    for the LEFT block (col_offset=0) -- the right block reuses that
    same row without repeating the label, since both blocks already
    share the same row.

    Returns the NEXT available row in ws_summary (after this block
    plus one blank separator row), so the caller can thread multiple
    calls together without recomputing row math itself.
    """
    label_row = next_row
    header_row = label_row + 1
    data_row = header_row + 1
    label_col = 2 + col_offset
    value_cols = [label_col, label_col + 1, label_col + 2]

    label_cell = ws_summary.cell(row=label_row, column=label_col, value=source_sheet_name)
    label_cell.font = Font(bold=True, underline="single")

    for col_idx, text in zip(value_cols, ("full sample", "below", "above")):
        ws_summary.cell(row=header_row, column=col_idx, value=text).font = bold
    if col_offset == 0:
        ws_summary.cell(row=data_row, column=1, value="share positive surprises").font = bold

    full_range = f"'{data_sheet_title}'!$G${below_start}:$G${above_end}"
    below_range = f"'{data_sheet_title}'!$G${below_start}:$G${below_end}"
    above_range = f"'{data_sheet_title}'!$G${above_start}:$G${above_end}"
    cell_full = ws_summary.cell(row=data_row, column=value_cols[0],
                                 value=f'=COUNTIF({full_range},">0")/COUNT({full_range})')
    cell_below = ws_summary.cell(row=data_row, column=value_cols[1],
                                  value=f'=COUNTIF({below_range},">0")/COUNT({below_range})')
    cell_above = ws_summary.cell(row=data_row, column=value_cols[2],
                                  value=f'=COUNTIF({above_range},">0")/COUNT({above_range})')
    for c in (cell_full, cell_below, cell_above):
        c.number_format = "0%"

    # NL-only row, directly below the all-countries row -- same
    # "share positive surprises" statistic, but COUNTIFS-restricted to
    # country=="NL" (column A of the data sheet) as well as the value/
    # cluster-range condition, so it reuses the SAME row ranges
    # (below_start:below_end, above_start:above_end) without needing a
    # separate NL-specific data sheet.
    nl_row = data_row + 1
    if col_offset == 0:
        ws_summary.cell(row=nl_row, column=1, value="share positive surprises (NL)").font = bold

    country_full = f"'{data_sheet_title}'!$A${below_start}:$A${above_end}"
    country_below = f"'{data_sheet_title}'!$A${below_start}:$A${below_end}"
    country_above = f"'{data_sheet_title}'!$A${above_start}:$A${above_end}"
    # IFERROR-wrapped: if NL happens to have ZERO rows in a given
    # range (e.g. NL's own value never fell below the median for this
    # particular variable/period), COUNTIFS(country_range,"NL") is 0,
    # and dividing by it would raise a genuine #DIV/0! -- confirmed
    # from a live run (ict_share and eur_export's "below" columns both
    # hit this for NL). "n/a" makes that explicit rather than leaving
    # a raw Excel error in the cell.
    nl_cell_full = ws_summary.cell(
        row=nl_row, column=value_cols[0],
        value=(f'=IFERROR(COUNTIFS({country_full},"NL",{full_range},">0")'
               f'/COUNTIFS({country_full},"NL"),"n/a")'))
    nl_cell_below = ws_summary.cell(
        row=nl_row, column=value_cols[1],
        value=(f'=IFERROR(COUNTIFS({country_below},"NL",{below_range},">0")'
               f'/COUNTIFS({country_below},"NL"),"n/a")'))
    nl_cell_above = ws_summary.cell(
        row=nl_row, column=value_cols[2],
        value=(f'=IFERROR(COUNTIFS({country_above},"NL",{above_range},">0")'
               f'/COUNTIFS({country_above},"NL"),"n/a")'))
    for c in (nl_cell_full, nl_cell_below, nl_cell_above):
        c.number_format = "0%"

    return nl_row + 1  # the NL row itself replaces the old blank separator --
    # this keeps each block's total height at 4 rows (label/header/
    # data/NL), exactly matching the original spacing before this NL
    # row was added, rather than growing every block by one extra row.


def write_clustered_time_chart(wb, cluster_key, merged_df, cluster_col, cluster_header,
                                color_below, color_above, ws_summary, summary_next_row,
                                col_offset=0):
    """
    Splits merged_df into two clusters by cluster_col's MEDIAN (rows
    with cluster_col <= median = "Below median", the rest = "Above
    median"), then plots growth_surprise_pct (Y, live formula, same
    blank-safe =IF(OR(...)) guard as write_data_sheet) against
    target_year (X) as TWO differently-colored series on one chart --
    so the two groups' patterns over time can be compared directly on
    the same axes, rather than as two separate charts.

    Rows are sorted so each cluster occupies a CONTIGUOUS row range
    (below-median rows first, then above-median rows) -- the same
    technique used earlier for the "excluding 2020" comparison chart --
    which lets two plain, contiguous Reference() ranges feed the two
    series without needing per-row filtering logic in the chart itself.
    """
    df = merged_df.dropna(subset=[cluster_col]).copy()
    median_val = df[cluster_col].median()
    df["cluster"] = df[cluster_col].apply(
        lambda v: "Below median" if v <= median_val else "Above median")
    df_sorted = pd.concat([
        df[df["cluster"] == "Below median"].sort_values(["target_year", "country"]),
        df[df["cluster"] == "Above median"].sort_values(["target_year", "country"]),
    ]).reset_index(drop=True)

    data_sheet_name = f"data_cluster_{cluster_key}"
    ws = wb.create_sheet(data_sheet_name)
    headers = ["country", "vintage", "vintage_round", "target_year",
               "forecast_growth_annual_pct", "realized_growth_annual_pct",
               "growth_surprise_pct", cluster_header, "cluster",
               "explanatory_source_year"]
    for col_idx, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col_idx, value=h)
        c.font = bold

    for row_idx, row in enumerate(df_sorted.itertuples(index=False), start=2):
        ws.cell(row=row_idx, column=1, value=row.country)
        ws.cell(row=row_idx, column=2, value=row.vintage)
        ws.cell(row=row_idx, column=3, value=row.vintage_round)
        ws.cell(row=row_idx, column=4, value=int(row.target_year))
        ws.cell(row=row_idx, column=5, value=row.forecast_growth_annual_pct)
        ws.cell(row=row_idx, column=6, value=row.realized_growth_annual_pct)
        ws.cell(row=row_idx, column=7,
                value=f'=IF(OR(F{row_idx}="",E{row_idx}=""),"",F{row_idx}-E{row_idx})')
        ws.cell(row=row_idx, column=8, value=getattr(row, cluster_col))
        ws.cell(row=row_idx, column=9, value=row.cluster)
        ws.cell(row=row_idx, column=10,
                value=int(row.explanatory_source_year))

    for col_idx in range(1, 11):
        ws.column_dimensions[chr(64 + col_idx)].width = 18

    n_below = int((df_sorted["cluster"] == "Below median").sum())
    n_total = len(df_sorted)
    below_start, below_end = 2, 1 + n_below
    above_start, above_end = 2 + n_below, 1 + n_total

    # Growth surprise and "positive surprise" success counts, computed
    # directly in Python from the source data (not read back from the
    # live Excel formula, which has no cached value at this point in
    # the script) -- returned alongside summary_next_row so the caller
    # can run significance tests between this block's proportions and
    # another (paired) block's proportions.
    df_sorted["growth_surprise"] = (df_sorted["realized_growth_annual_pct"]
                                     - df_sorted["forecast_growth_annual_pct"])
    below_gs = df_sorted.iloc[:n_below]["growth_surprise"].dropna()
    above_gs = df_sorted.iloc[n_below:]["growth_surprise"].dropna()
    full_gs = df_sorted["growth_surprise"].dropna()
    proportions = {
        "n_full": len(full_gs), "x_full": int((full_gs > 0).sum()),
        "n_below": len(below_gs), "x_below": int((below_gs > 0).sum()),
        "n_above": len(above_gs), "x_above": int((above_gs > 0).sum()),
    }

    # Summary block now written into the SHARED "summary" sheet (not
    # below this sheet's own data anymore) -- see write_summary_block().
    summary_next_row = write_summary_block(
        ws_summary, summary_next_row, data_sheet_name, data_sheet_name,
        below_start, below_end, above_start, above_end, col_offset=col_offset,
    )

    chart = ScatterChart()
    chart.scatterStyle = "marker"
    chart.title = f"Growth surprise over time -- clustered by {cluster_header} (median split)"
    chart.style = 13
    chart.x_axis.title = "Target year (time)"
    chart.y_axis.title = "Growth surprise (realized - forecast, pp)"
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.height = 12
    chart.width = 22

    def add_series(min_row, max_row, label, color):
        if max_row < min_row:
            return
        x_vals = Reference(ws, min_col=4, min_row=min_row, max_row=max_row)
        y_vals = Reference(ws, min_col=7, min_row=min_row, max_row=max_row)
        s = Series(y_vals, x_vals, title=label)
        s.marker = Marker(symbol="circle", size=6)
        s.marker.graphicalProperties = GraphicalProperties(solidFill=color)
        s.marker.graphicalProperties.line.solidFill = color
        s.graphicalProperties = GraphicalProperties()
        s.graphicalProperties.line.noFill = True
        chart.series.append(s)

    add_series(below_start, below_end, f"Below median ({cluster_header} <= {median_val:.4f})",
               color_below)
    add_series(above_start, above_end, f"Above median ({cluster_header} > {median_val:.4f})",
               color_above)

    chart_sheet_name = f"chart_cluster_{cluster_key}"
    ws_chart = wb.create_sheet(chart_sheet_name)
    ws_chart.add_chart(chart, "B2")
    print(f"  {data_sheet_name}: {n_total} rows ({n_below} below / {n_total - n_below} above "
          f"median={median_val:.4f}) -> {chart_sheet_name}")
    return summary_next_row, proportions


# --- Alternative chart, ONE PER main explanatory variable, next to
# each variable's regular charts: growth surprise restricted to ONLY
# the SPRING-forecast-based surprise (realized[year] -
# Spring[year]_forecast -- the Autumn round for these years is
# deliberately excluded) for target years 2020, 2022, and 2025
# specifically, all countries pooled -- a fixed, hand-picked 3-year
# slice rather than a continuous period range like "since2020"/
# "till2019", so it needed its own dedicated filter rather than fitting
# the existing `periods` list.
SPECIAL_YEARS = [2020, 2022, 2025]
# Both forecast rounds now included (was Spring-only before) -- for a
# given target_year, horizon==0 correctly picks out THAT year's own
# Spring AND Autumn forecast alike (each round's own "current-year"
# forecast), so each country now contributes up to 2 rows per target
# year instead of 1.
SPECIAL_YEARS_ROUNDS = ["Spring", "Autumn"]


def write_special_years_chart(wb, var_key, raw_forecast, raw_explanatory, col, header, color,
                               ws_summary, summary_next_row, col_offset=0):
    """
    For every shock-year outcome t, use the explanatory observation from
    exactly t-1. There is no same-year lookup and no fallback: if country i
    has no value in t-1, that outcome observation is excluded.
    """

    # horizon==0 is essential here, not just vintage_round=="Spring":
    # without it, an EARLIER round's horizon=1 forecast (e.g.
    # Spring2019 forecasting 2020 one year ahead) would ALSO match
    # "target_year=2020", duplicating each country -- confirmed from a
    # real run producing 34 rows instead of the expected 30 (10
    # countries x 3 years) back when this was Spring-only. horizon==0
    # restricts this to specifically the CURRENT-year forecast made in
    # that same year's own round (i.e. vintage=="Spring{target_year}"
    # or "Autumn{target_year}"), matching "that year's own forecast,
    # both rounds" as (up to) two rows per country per target year.
    df = raw_forecast[(raw_forecast["vintage_round"].isin(SPECIAL_YEARS_ROUNDS))
                       & (raw_forecast["horizon"] == 0)
                       & (raw_forecast["target_year"].isin(SPECIAL_YEARS))].copy()

    # Build a (country, source_year) -> explanatory value lookup for t-1.
    lookup_years_needed = {year - 1 for year in SPECIAL_YEARS}
    explanatory_lookup = raw_explanatory[
        raw_explanatory["target_year"].isin(lookup_years_needed)
        & raw_explanatory[col].notna()
    ].set_index(["country", "target_year"])[col]

    explanatory_values, explanatory_years_used = [], []
    keep_mask = []
    for _, row in df.iterrows():
        ty = row["target_year"]
        chosen_year = ty - 1
        key = (row["country"], chosen_year)
        chosen_value = explanatory_lookup.loc[key] if key in explanatory_lookup.index else None
        if chosen_value is not None and not pd.isna(chosen_value):
            explanatory_values.append(chosen_value)
            explanatory_years_used.append(chosen_year)
            keep_mask.append(True)
        else:
            explanatory_values.append(None)
            explanatory_years_used.append(None)
            keep_mask.append(False)
    df[col] = explanatory_values
    df["explanatory_year_used"] = explanatory_years_used
    df = df[keep_mask].copy()

    # Below/above-median classification, computed WITHIN this specific
    # years-restricted subset (2020/2022/2025 Spring only) -- NOT the
    # same median as the full-sample cluster sheets, since this is a
    # much smaller, differently-composed sample. Same technique as
    # write_clustered_time_chart(): rows sorted so each cluster is a
    # CONTIGUOUS block, letting simple COUNTIF row-ranges (no COUNTIFS
    # needed) drive the summary formulas below.
    df = df.dropna(subset=[col])
    median_val = df[col].median()
    df["cluster"] = df[col].apply(
        lambda v: "Below median" if v <= median_val else "Above median")
    df_sorted = pd.concat([
        df[df["cluster"] == "Below median"].sort_values(["target_year", "country"]),
        df[df["cluster"] == "Above median"].sort_values(["target_year", "country"]),
    ]).reset_index(drop=True)

    data_sheet_name = f"data_{var_key}_yrs"
    chart_sheet_name = f"chart_{var_key}_yrs"
    ws_data = wb.create_sheet(data_sheet_name)
    headers = ["country", "vintage", "vintage_round", "target_year",
               "forecast_growth_annual_pct", "realized_growth_annual_pct",
               "growth_surprise_pct", header, "cluster", "explanatory_year_used"]
    for col_idx, h in enumerate(headers, start=1):
        c = ws_data.cell(row=1, column=col_idx, value=h)
        c.font = bold

    for row_idx, row in enumerate(df_sorted.itertuples(index=False), start=2):
        ws_data.cell(row=row_idx, column=1, value=row.country)
        ws_data.cell(row=row_idx, column=2, value=row.vintage)
        ws_data.cell(row=row_idx, column=3, value=row.vintage_round)
        ws_data.cell(row=row_idx, column=4, value=int(row.target_year))
        ws_data.cell(row=row_idx, column=5, value=row.forecast_growth_annual_pct)
        ws_data.cell(row=row_idx, column=6, value=row.realized_growth_annual_pct)
        ws_data.cell(row=row_idx, column=7,
                      value=f'=IF(OR(F{row_idx}="",E{row_idx}=""),"",F{row_idx}-E{row_idx})')
        ws_data.cell(row=row_idx, column=8, value=getattr(row, col))
        ws_data.cell(row=row_idx, column=9, value=row.cluster)
        ws_data.cell(row=row_idx, column=10, value=int(row.explanatory_year_used))
    for col_idx in range(1, 11):
        ws_data.column_dimensions[chr(64 + col_idx)].width = 20

    n_below = int((df_sorted["cluster"] == "Below median").sum())
    n_total = len(df_sorted)
    last_row = n_total + 1
    below_start, below_end = 2, 1 + n_below
    above_start, above_end = 2 + n_below, 1 + n_total

    df_sorted["growth_surprise"] = (df_sorted["realized_growth_annual_pct"]
                                     - df_sorted["forecast_growth_annual_pct"])
    below_gs = df_sorted.iloc[:n_below]["growth_surprise"].dropna()
    above_gs = df_sorted.iloc[n_below:]["growth_surprise"].dropna()
    full_gs = df_sorted["growth_surprise"].dropna()
    proportions = {
        "n_full": len(full_gs), "x_full": int((full_gs > 0).sum()),
        "n_below": len(below_gs), "x_below": int((below_gs > 0).sum()),
        "n_above": len(above_gs), "x_above": int((above_gs > 0).sum()),
    }

    # Summary block now written into the SHARED "summary" sheet (not
    # below this sheet's own data anymore) -- see write_summary_block().
    summary_next_row = write_summary_block(
        ws_summary, summary_next_row, data_sheet_name, data_sheet_name,
        below_start, below_end, above_start, above_end, col_offset=col_offset,
    )

    years_str = "/".join(str(y) for y in SPECIAL_YEARS)
    rounds_str = " & ".join(SPECIAL_YEARS_ROUNDS)
    chart_title = (f"Growth surprise ({rounds_str} forecast) vs. {header} "
                    f"-- {years_str}")
    write_chart_sheet(wb, chart_sheet_name, ws_data, last_row, chart_title, header, color)
    print(f"  {data_sheet_name}: {last_row - 1} rows ({years_str}, "
          f"{rounds_str}) -> {chart_sheet_name}")
    return summary_next_row, proportions, df_sorted


last_corr_row = corr_header_row + len(sheet_registry)  # last variable's correlation row
summary_row = last_corr_row + 1  # one blank separator row before the group headers below

# --- "All years" (cluster sheets, full sample) vs "Shock years" (the
# 3 special-years sheets) group headers, written ONCE, directly above
# the first pair of blocks below.
group_header_row = summary_row + 1
ws_summary.cell(row=group_header_row, column=2, value="All years").font = Font(bold=True, size=12)
ws_summary.cell(row=group_header_row, column=6, value="Shock years").font = Font(bold=True, size=12)
summary_row = group_header_row + 2
def build_significance_formula(range1, range2, symbol="*"):
    """
    Builds a LIVE Excel formula that reproduces significance_stars()'s
    two-proportion z-test entirely in worksheet formulas (COUNTIF/
    COUNT/NORM.S.DIST), for the requested J12:J27 verification column.
    range1/range2: Excel range strings (e.g.
    "'data_cluster_ict_share'!$G$2:$G$245") for the two groups being
    compared. symbol: "*" or "+", matching significance_stars()'s own
    convention. Wrapped in IFERROR(...,"") for the same zero-count
    edge case significance_stars() itself guards against (COUNT=0).
    """
    x1 = f'COUNTIF({range1},">0")'
    n1 = f'COUNT({range1})'
    x2 = f'COUNTIF({range2},">0")'
    n2 = f'COUNT({range2})'
    p1 = f'({x1}/{n1})'
    p2 = f'({x2}/{n2})'
    p_pool = f'(({x1}+{x2})/({n1}+{n2}))'
    denom = f'({p_pool}*(1-{p_pool})*(1/{n1}+1/{n2}))'
    z = f'(({p1}-{p2})/SQRT({denom}))'
    p_value = f'(2*(1-_xlfn.NORM.S.DIST(ABS({z}),TRUE)))'
    stars3, stars2, stars1 = symbol * 3, symbol * 2, symbol
    mapped = (f'IF({p_value}<0.01,"{stars3}",IF({p_value}<0.05,"{stars2}",'
              f'IF({p_value}<0.1,"{stars1}","")))')
    return f'=IFERROR({mapped},"")'


# --- Each variable's "cluster" (full-sample, below/above median) block
# and "yrs" (2020/2022/2025 shock-years) block are written SIDE BY SIDE
# on the SAME rows (col_offset=0 for cluster on the left, col_offset=4
# for yrs on the right) -- not stacked vertically like before -- so the
# two views of the same variable are directly comparable at a glance.
# Only the LEFT (col_offset=0) call's returned row is used to advance
# summary_row for the next pair; the RIGHT call's return is identical
# (both blocks occupy the same rows) and is intentionally discarded.
def add_significance_stars(ws_summary, header_row, cluster_props, yrs_props):
    """
    Two DIFFERENT kinds of comparison, using two different symbols so
    they stay visually distinct when a cell needs both:

    WITHIN-period comparisons (symbol="*", unchanged from before):
      - D{header_row}: cluster "above" vs. cluster "below" (all years)
      - H{header_row}: yrs "above" vs. yrs "below" (shock years)

    ACROSS-period comparisons (symbol="+", NEW): comparing the SAME
    column (full/below/above) between the two blocks -- i.e. F13:H26
    against B13:D26, row by row:
      - F{header_row}: yrs "full" vs. cluster "full"
      - G{header_row}: yrs "below" vs. cluster "below"
      - H{header_row}: yrs "above" vs. cluster "above"

    H therefore can receive stars from BOTH tests -- if both are
    significant, they are combined on that one cell, e.g.
    "above +++ **" (the "+++" from the across-period test, "**" from
    the within-period test), rather than overwriting one with the
    other.
    """
    # Within-period (below vs above), symbol="*"
    d_star = significance_stars(cluster_props["x_above"], cluster_props["n_above"],
                                 cluster_props["x_below"], cluster_props["n_below"],
                                 symbol="*")
    h_star = significance_stars(yrs_props["x_above"], yrs_props["n_above"],
                                 yrs_props["x_below"], yrs_props["n_below"],
                                 symbol="*")
    # Across-period (shock years vs all years), symbol="+"
    f_plus = significance_stars(yrs_props["x_full"], yrs_props["n_full"],
                                 cluster_props["x_full"], cluster_props["n_full"],
                                 symbol="+")
    g_plus = significance_stars(yrs_props["x_below"], yrs_props["n_below"],
                                 cluster_props["x_below"], cluster_props["n_below"],
                                 symbol="+")
    h_plus = significance_stars(yrs_props["x_above"], yrs_props["n_above"],
                                 cluster_props["x_above"], cluster_props["n_above"],
                                 symbol="+")

    combined = {
        "D": [d_star],
        "F": [f_plus],
        "G": [g_plus],
        "H": [h_plus, h_star],  # both kinds can apply to H
    }
    for col_letter, markers in combined.items():
        markers = [m for m in markers if m]
        if not markers:
            continue
        cell = ws_summary[f"{col_letter}{header_row}"]
        cell.value = f"{cell.value} {' '.join(markers)}"


all_props = {}  # var_label -> (cluster_props, yrs_props), used later for the
                 # staging table -- written as PLAIN VALUES, not formulas, so
                 # the charts don't depend on a multi-layer uncached-formula
                 # chain (see the chat discussion this fix came from).
block_header_rows = {}  # var_label -> its summary block's header row (12/16/20/24),
                         # used later to build live-formula references from the
                         # staging table (rows 61-67) back to these blocks.

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "ict_share", merged_ict, "ict_share", "Previous-year ICT investment share", "AEC7E8", "1F77B4",
    ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_ict = write_special_years_chart(
    wb, "ict_share", forecast, ict, "ict_share", "Previous-year ICT investment share", "FF7F0E",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
block_header_rows["Previous-year ICT investment share"] = header_row
all_props["Previous-year ICT investment share"] = (cluster_props, yrs_props)

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "eur_export", merged_eur, "eur_export_share", "Previous-year AI/ICT-related EU export share",
    "FFBB78", "D62728", ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_hs = write_special_years_chart(
    wb, "eur_export_share", forecast, eur_export, "eur_export_share",
    "Previous-year AI/ICT-related EU export share", "9467BD",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["Previous-year AI/ICT-related EU export share"] = (cluster_props, yrs_props)
block_header_rows["Previous-year AI/ICT-related EU export share"] = header_row

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "stock_corr", merged_corr, "stock_semis_corr_annual",
    "Previous-year national vs. semiconductor index correlation", "98DF8A", "2CA02C",
    ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_corr = write_special_years_chart(
    wb, "stock_corr", forecast, stock_corr_annual, "stock_semis_corr_annual",
    "Previous-year national vs. semiconductor index correlation", "8C564B",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["Previous-year national vs. semiconductor index correlation"] = (cluster_props, yrs_props)
block_header_rows["Previous-year national vs. semiconductor index correlation"] = header_row

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "ai_inv_share", merged_ai_inv, "ai_inv_share",
    "Previous-year AI incoming investment share (per avg. quarterly GDP)", "F7B6D2", "E377C2",
    ws_summary, summary_row, col_offset=0)
# The special-year comparison also aligns every exposure to t-1.
_, yrs_props, _ = write_special_years_chart(
    wb, "ai_inv_share", forecast, ai_inv, "ai_inv_share",
    "Previous-year AI incoming investment share (per avg. quarterly GDP)", "17BECF",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["Previous-year AI incoming investment share"] = (cluster_props, yrs_props)
block_header_rows["Previous-year AI incoming investment share"] = header_row

# --- Explanation of the significance markers used throughout the
# blocks above, at the EXPLICITLY requested row 28.
ws_summary.cell(
    row=28, column=1,
    value="Significance legend: */**/*** = below vs. above median differs at "
          "10%/5%/1% (two-proportion z-test, within the SAME period -- compare "
          "D and H columns to C and G). +/++/+++ = shock years differs from "
          "all years at 10%/5%/1% (SAME column compared ACROSS periods -- "
          "compare F, G, and H columns to B, C, D; a cell can show both kinds "
          "combined, e.g. \"above +++ **\", if both tests are significant)."
).font = Font(italic=True, size=9)

# --- J12:J27: LIVE Excel formulas reproducing the significance
# markers (stars/plusses) from scratch, using COUNTIF/COUNT/
# NORM.S.DIST directly on the underlying data sheets -- so the D/F/G/H
# markers above can be checked/verified independently. 4 rows per
# variable block (12-15 ICT, 16-19 Export, 20-23 stock_corr, 24-27
# ai_inv), one formula type per row within each block:
#   row+0 (header row):  D-type -- above vs below, all years
#   row+1 (data row):    F-type -- full vs full, shock vs all years
#   row+2 (NL row):      G-type -- below vs below, shock vs all years
#   row+3 (blank row):   H-type -- above vs above, shock vs all years
# Data-sheet names/row ranges are reconstructed from n_below/n_above
# (each sheet's rows are sorted contiguously: below rows first, then
# above rows, starting at row 2 -- exactly matching how
# write_clustered_time_chart()/write_special_years_chart() built them).
j_col_specs = [
    ("Previous-year ICT investment share", "data_cluster_ict_share", "data_ict_share_yrs"),
    ("Previous-year AI/ICT-related EU export share", "data_cluster_eur_export", "data_eur_export_share_yrs"),
    ("Previous-year national vs. semiconductor index correlation", "data_cluster_stock_corr",
     "data_stock_corr_yrs"),
    ("Previous-year AI incoming investment share", "data_cluster_ai_inv_share", "data_ai_inv_share_yrs"),
]

ws_summary.cell(row=11, column=10,
                 value="J: live-formula check of D/F/G/H (see row 28)").font = Font(
    bold=True, italic=True, size=9)

for var_label, cluster_sheet, yrs_sheet in j_col_specs:
    header_row_j = block_header_rows[var_label]
    cp, yp = all_props[var_label]

    def rng(sheet, start, end):
        return f"'{sheet}'!$G${start}:$G${end}"

    c_below_start, c_below_end = 2, 1 + cp["n_below"]
    c_above_start, c_above_end = 2 + cp["n_below"], 1 + cp["n_below"] + cp["n_above"]
    y_below_start, y_below_end = 2, 1 + yp["n_below"]
    y_above_start, y_above_end = 2 + yp["n_below"], 1 + yp["n_below"] + yp["n_above"]

    c_below_rng = rng(cluster_sheet, c_below_start, c_below_end)
    c_above_rng = rng(cluster_sheet, c_above_start, c_above_end)
    c_full_rng = rng(cluster_sheet, c_below_start, c_above_end)
    y_below_rng = rng(yrs_sheet, y_below_start, y_below_end)
    y_above_rng = rng(yrs_sheet, y_above_start, y_above_end)
    y_full_rng = rng(yrs_sheet, y_below_start, y_above_end)

    # D-type: cluster above vs. cluster below (symbol="*")
    ws_summary.cell(row=header_row_j, column=10,
                     value=build_significance_formula(c_above_rng, c_below_rng, symbol="*"))
    # F-type: yrs full vs. cluster full (symbol="+")
    ws_summary.cell(row=header_row_j + 1, column=10,
                     value=build_significance_formula(y_full_rng, c_full_rng, symbol="+"))
    # G-type: yrs below vs. cluster below (symbol="+")
    ws_summary.cell(row=header_row_j + 2, column=10,
                     value=build_significance_formula(y_below_rng, c_below_rng, symbol="+"))
    # H-type: yrs above vs. cluster above (symbol="+")
    ws_summary.cell(row=header_row_j + 3, column=10,
                     value=build_significance_formula(y_above_rng, c_above_rng, symbol="+"))

ws_summary.column_dimensions["J"].width = 14

ws_summary.column_dimensions["A"].width = 30
for col_letter in ("B", "C", "D", "F", "G", "H"):
    ws_summary.column_dimensions[col_letter].width = 18

# --- Panel regressions: growth_surprise = const + country FE + time
# FE + beta*explanatory_var, for each of the three explanatory
# variables, using the SAME data that feeds the "_full" sheets
# (merged_ict/merged_eur/merged_corr, unfiltered by period). Computed
# directly from the in-memory DataFrames (not by reading back the
# Excel formulas), since growth_surprise_pct is itself a live Excel
# formula with no cached value until the file is opened/recalculated --
# computing it here in Python from realized/forecast avoids that
# entirely and guarantees this regression uses the exact same numbers
# the "_full" sheets display.
#
# All OLS, logit, and probit regressions below use country-clustered
# sandwich standard errors with the same finite-sample correction,
# equivalent to Stata's vce(cluster country_id).
from scipy import stats as _stats


def _country_clustered_cov(bread, score_obs, cluster_labels, estimator_name):
    """
    Country-clustered sandwich covariance used by OLS, logit, and probit.
    `bread` is the inverse information matrix appropriate to the estimator;
    `score_obs` contains one score vector per observation. Scores are summed
    within country before forming the sandwich meat. The finite-sample factor
    matches the standard Stata-style clustered covariance correction.
    """
    cluster_labels = np.asarray(cluster_labels)
    clusters = pd.unique(cluster_labels)
    n, k = score_obs.shape
    n_clusters = len(clusters)
    if n_clusters < 2:
        raise ValueError(
            f"Country-clustered {estimator_name} SEs require at least two countries."
        )
    if n <= k:
        raise ValueError(
            f"Country-clustered {estimator_name} SEs require N greater than model rank."
        )
    cluster_scores = np.vstack([
        score_obs[cluster_labels == cluster].sum(axis=0)
        for cluster in clusters
    ])
    meat = cluster_scores.T @ cluster_scores
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * (bread @ meat @ bread)
    return cov, n_clusters


def panel_ols_two_way_fe(df, y_col, x_col, entity_col="country", time_col="target_year",
                          extra_cols=None):
    """
    extra_cols: optional list of ADDITIONAL regressor column names
    already present in df (e.g. an interaction term X*shock_dummy) --
    included alongside x_col, country FE, and time FE. Coefficients
    for every column in extra_cols are returned in "extra" (a dict
    keyed by column name), in the same shape as the main beta/se/
    t_stat/p_value.

    Standard errors are country-clustered by entity_col, using the same
    sandwich estimator and finite-sample correction as logit and probit.
    """
    extra_cols = extra_cols or []
    needed_cols = [y_col, x_col, entity_col, time_col] + extra_cols
    d = df[needed_cols].dropna().copy()
    n_obs = len(d)
    entities = sorted(d[entity_col].unique())
    periods = sorted(d[time_col].unique())
    entity_dummies = pd.get_dummies(d[entity_col], prefix="ent", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(d[time_col], prefix="t", drop_first=True, dtype=float)
    X = pd.concat([
        pd.Series(1.0, index=d.index, name="const"),
        d[[x_col] + extra_cols].astype(float),
        entity_dummies,
        time_dummies,
    ], axis=1)
    y = d[y_col].astype(float).values
    X_mat = X.values

    beta_hat, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    y_hat = X_mat @ beta_hat
    resid = y - y_hat
    n, k = X_mat.shape
    bread = np.linalg.pinv(X_mat.T @ X_mat)
    score_obs = X_mat * resid[:, None]
    cov, n_clusters = _country_clustered_cov(
        bread, score_obs, d[entity_col].to_numpy(), "OLS"
    )
    se_all = np.sqrt(np.abs(np.diag(cov)))
    cluster_dof = n_clusters - 1

    def _coef_stats(col_name):
        idx = list(X.columns).index(col_name)
        b, s = beta_hat[idx], se_all[idx]
        t = b / s
        p = 2 * (1 - _stats.t.cdf(abs(t), cluster_dof))
        return {"beta": b, "se": s, "t_stat": t, "p_value": p}

    main = _coef_stats(x_col)
    extra = {col: _coef_stats(col) for col in extra_cols}
    const_idx = list(X.columns).index("const")
    const_hat = beta_hat[const_idx]

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    return {
        **main, "extra": extra,
        "const": const_hat, "r_squared": r_squared, "n_obs": n_obs,
        "n_entities": len(entities), "n_periods": len(periods),
        "n_clusters": n_clusters,
    }


def panel_logit_two_way_fe(df, y_col, x_col, entity_col="country", time_col="target_year",
                            extra_cols=None):
    """
    Two-way fixed-effects LOGIT, estimating P(y_col > 0 | X) instead of
    OLS's E[y_col | X] -- i.e. does X make a POSITIVE growth surprise
    more LIKELY, as opposed to panel_ols_two_way_fe()'s "does X raise
    the average SIZE of the surprise". y_col is binarized internally
    (1 if > 0, else 0) -- the function itself does the split, so the
    caller passes the SAME continuous growth_surprise column used for
    the OLS tables above.

    Standard errors are country-clustered by entity_col, using the same
    sandwich estimator and finite-sample correction as OLS and probit.

    HONESTY NOTE on why logit (not probit) and what this specification
    does and does NOT solve, stated plainly because the choice was
    made explicitly for this reason: the classic "incidental parameters
    problem" (Neyman-Scott 1948) is that, in nonlinear panel models
    with a fixed effect per entity, the number of entity-specific
    parameters grows with the number of entities (here, 10 countries)
    even as each entity contributes only a few observations -- this
    contaminates the COMMON parameter (beta) estimate too, not just
    the fixed effects themselves. Chamberlain (1980) showed that LOGIT
    specifically has a special property PROBIT does not: conditioning
    the likelihood on each entity's own total count of successes (a
    sufficient statistic) lets the entity fixed effects be eliminated
    from the likelihood ENTIRELY, giving a consistent beta regardless
    of how few observations each entity has. This is what people mean
    when they say "logit avoids the incidental parameters problem
    where probit doesn't."

    BUT -- and this is the crucial caveat -- that specific property
    belongs to CONDITIONAL (fixed-effects) logit, which conditions out
    the entity effects analytically and DROPS entities with no
    within-entity variation in the outcome (an entity that is always
    positive or always negative contributes nothing to the conditional
    likelihood). What is implemented HERE is the more straightforward
    LSDV-style logit -- entity and time fixed effects as ordinary
    dummy variables in a standard (unconditional) MLE, matching the
    exact same dummy-variable specification style already used for
    the OLS tables above, for consistency and because a genuine
    two-way (country AND year) conditional logit is a substantially
    larger, non-standard undertaking (the classical conditional-logit
    result is built for a SINGLE fixed effect, not two crossed ones).
    Unconditional dummy-variable logit does NOT have Chamberlain's
    sufficient-statistic property -- it can still be biased in exactly
    the way probit-with-dummies is, especially with 10 countries x
    ~15-20 years and correspondingly few observations per country. Any
    country with an outcome that is entirely 0s or entirely 1s across
    all its observations is a visible symptom of this (the model wants
    to push that country's own dummy coefficient toward +/-infinity) --
    flagged explicitly below via a diagnostic warning rather than left
    to silently produce a huge, meaningless coefficient and standard
    error.
    """
    extra_cols = extra_cols or []
    needed_cols = [y_col, x_col, entity_col, time_col] + extra_cols
    d = df[needed_cols].dropna().copy()
    n_obs = len(d)
    entities = sorted(d[entity_col].unique())
    periods = sorted(d[time_col].unique())

    y_binary = (d[y_col] > 0).astype(float)
    # Diagnostic: entities with NO within-entity variation in the
    # binary outcome (always positive or always negative surprises)
    # are the visible symptom of the incidental-parameters problem
    # described above -- their own dummy coefficient is poorly
    # identified (pushed toward +/-infinity in the unconditional MLE).
    per_entity_variation = y_binary.groupby(d[entity_col]).nunique()
    degenerate_entities = per_entity_variation[per_entity_variation < 2].index.tolist()
    if degenerate_entities:
        print(f"    [!] WARNING (logit): {degenerate_entities} have a "
              f"constant (all-positive or all-negative) growth_surprise sign "
              f"across every observation in this sample -- their own entity "
              f"dummy is poorly identified in this unconditional (LSDV-style) "
              f"logit. Coefficients/SEs below may be unstable; treat with "
              f"extra caution (see this function's own docstring for why).")

    entity_dummies = pd.get_dummies(d[entity_col], prefix="ent", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(d[time_col], prefix="t", drop_first=True, dtype=float)
    X = pd.concat([
        pd.Series(1.0, index=d.index, name="const"),
        d[[x_col] + extra_cols].astype(float),
        entity_dummies,
        time_dummies,
    ], axis=1)
    X_mat = X.values
    y_arr = y_binary.values
    n, k = X_mat.shape

    def neg_log_lik(beta):
        z = X_mat @ beta
        # numerically stable log(1+exp(z)) via logaddexp(0, z)
        return -(y_arr * z - np.logaddexp(0.0, z)).sum()

    def neg_log_lik_grad(beta):
        z = X_mat @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        return -(X_mat.T @ (y_arr - p))

    from scipy.optimize import minimize
    beta0 = np.zeros(k)
    opt_result = minimize(neg_log_lik, beta0, jac=neg_log_lik_grad, method="BFGS")
    if not opt_result.success:
        print(f"    [!] WARNING (logit): optimizer did not report success "
              f"({opt_result.message}) -- results may not be a genuine "
              f"maximum-likelihood solution.")
    beta_hat = opt_result.x

    # Country-clustered sandwich covariance. The bread is the inverse
    # observed information and the observation scores are x_i*(y_i-p_i).
    z_fitted = X_mat @ beta_hat
    p_fitted = 1.0 / (1.0 + np.exp(-z_fitted))
    w = p_fitted * (1.0 - p_fitted)
    hessian = X_mat.T @ (X_mat * w[:, None])
    bread = np.linalg.pinv(hessian)
    score_obs = X_mat * (y_arr - p_fitted)[:, None]
    cov, n_clusters = _country_clustered_cov(
        bread, score_obs, d[entity_col].to_numpy(), "logit"
    )
    se_all = np.sqrt(np.abs(np.diag(cov)))

    def _coef_stats(col_name):
        idx = list(X.columns).index(col_name)
        b, s = beta_hat[idx], se_all[idx]
        zstat = b / s if s > 0 else np.nan
        pval = 2 * (1 - _stats.norm.cdf(abs(zstat))) if not np.isnan(zstat) else np.nan
        return {"beta": b, "se": s, "z_stat": zstat, "p_value": pval}

    main = _coef_stats(x_col)
    extra = {col: _coef_stats(col) for col in extra_cols}

    # McFadden's pseudo-R^2: 1 - (log-likelihood of the fitted model /
    # log-likelihood of an intercept-only model predicting the sample
    # mean for every observation).
    ll_full = -opt_result.fun
    p_bar = y_arr.mean()
    ll_null = (y_arr * np.log(p_bar) + (1 - y_arr) * np.log(1 - p_bar)).sum()
    pseudo_r2 = 1 - ll_full / ll_null if ll_null != 0 else np.nan

    return {
        **main, "extra": extra,
        "pseudo_r2": pseudo_r2, "n_obs": n_obs,
        "n_entities": len(entities), "n_periods": len(periods),
        "n_clusters": n_clusters,
    }


def panel_probit_two_way_fe(df, y_col, x_col, entity_col="country", time_col="target_year",
                             extra_cols=None):
    """
    Two-way fixed-effects PROBIT -- a robustness check alongside
    panel_logit_two_way_fe() above, estimating the SAME P(y_col > 0 | X)
    specification with the normal (instead of logistic) link function.
    Same LSDV-style (dummy-variable) fixed effects as the logit version
    -- and therefore the SAME incidental-parameters caveat applies here
    too (see panel_logit_two_way_fe()'s own docstring): probit has no
    conditional-MLE escape from that problem at all (that property is
    specific to logit), so this comparison is honestly a check of
    "does the functional-form choice (logistic vs. normal link) change
    the substantive conclusions", not a check that resolves the
    incidental-parameters issue either way.

    Standard errors use a country-clustered sandwich covariance matrix,
    matching Stata's ``vce(cluster country_id)``: the score contributions
    are summed within country before forming the meat of the sandwich, and
    the usual finite-sample cluster correction is applied. Thus inference
    is robust to arbitrary heteroskedasticity and within-country dependence;
    countries (clusters) are assumed independent of one another.
    """
    extra_cols = extra_cols or []
    needed_cols = [y_col, x_col, entity_col, time_col] + extra_cols
    d = df[needed_cols].dropna().copy()
    n_obs = len(d)
    entities = sorted(d[entity_col].unique())
    periods = sorted(d[time_col].unique())

    y_binary = (d[y_col] > 0).astype(float)
    per_entity_variation = y_binary.groupby(d[entity_col]).nunique()
    degenerate_entities = per_entity_variation[per_entity_variation < 2].index.tolist()
    if degenerate_entities:
        print(f"    [!] WARNING (probit): {degenerate_entities} have a "
              f"constant (all-positive or all-negative) growth_surprise sign "
              f"across every observation in this sample -- their own entity "
              f"dummy is poorly identified in this unconditional (LSDV-style) "
              f"probit. Coefficients/SEs below may be unstable; treat with "
              f"extra caution.")

    entity_dummies = pd.get_dummies(d[entity_col], prefix="ent", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(d[time_col], prefix="t", drop_first=True, dtype=float)
    X = pd.concat([
        pd.Series(1.0, index=d.index, name="const"),
        d[[x_col] + extra_cols].astype(float),
        entity_dummies,
        time_dummies,
    ], axis=1)
    X_mat = X.values
    y_arr = y_binary.values
    n, k = X_mat.shape

    # Numerical floor/ceiling on Phi(z) to avoid log(0)/division-by-
    # zero for observations far out in the tails during optimization.
    eps = 1e-10

    def neg_log_lik(beta):
        z = X_mat @ beta
        Phi = np.clip(_stats.norm.cdf(z), eps, 1 - eps)
        return -(y_arr * np.log(Phi) + (1 - y_arr) * np.log(1 - Phi)).sum()

    def neg_log_lik_grad(beta):
        z = X_mat @ beta
        Phi = np.clip(_stats.norm.cdf(z), eps, 1 - eps)
        phi = _stats.norm.pdf(z)
        # d(LL)/dz = phi(z) * [y/Phi(z) - (1-y)/(1-Phi(z))]
        grad_z = phi * (y_arr / Phi - (1 - y_arr) / (1 - Phi))
        return -(X_mat.T @ grad_z)

    from scipy.optimize import minimize
    beta0 = np.zeros(k)
    opt_result = minimize(neg_log_lik, beta0, jac=neg_log_lik_grad, method="BFGS")
    if not opt_result.success:
        print(f"    [!] WARNING (probit): optimizer did not report success "
              f"({opt_result.message}) -- results may not be a genuine "
              f"maximum-likelihood solution.")
    beta_hat = opt_result.x

    # Country-clustered sandwich covariance, corresponding to Stata's
    # vce(cluster country_id). The bread is the inverse observed
    # information for the probit likelihood. The meat is formed from
    # country-level sums of the individual score vectors, allowing every
    # observation within a country to be arbitrarily correlated.
    z_fitted = X_mat @ beta_hat
    Phi_fitted = np.clip(_stats.norm.cdf(z_fitted), eps, 1 - eps)
    phi_fitted = _stats.norm.pdf(z_fitted)
    inverse_mills_pos = phi_fitted / Phi_fitted
    inverse_mills_neg = phi_fitted / (1 - Phi_fitted)
    observed_info_weight = (
        y_arr * inverse_mills_pos * (inverse_mills_pos + z_fitted)
        + (1 - y_arr) * inverse_mills_neg * (inverse_mills_neg - z_fitted)
    )
    observed_info = X_mat.T @ (X_mat * observed_info_weight[:, None])
    bread = np.linalg.pinv(observed_info)

    score_scalar = phi_fitted * (
        y_arr / Phi_fitted - (1 - y_arr) / (1 - Phi_fitted)
    )
    score_obs = X_mat * score_scalar[:, None]
    cov, n_clusters = _country_clustered_cov(
        bread, score_obs, d[entity_col].to_numpy(), "probit"
    )
    se_all = np.sqrt(np.abs(np.diag(cov)))

    def _coef_stats(col_name):
        idx = list(X.columns).index(col_name)
        b, s = beta_hat[idx], se_all[idx]
        zstat = b / s if s > 0 else np.nan
        pval = 2 * (1 - _stats.norm.cdf(abs(zstat))) if not np.isnan(zstat) else np.nan
        return {"beta": b, "se": s, "z_stat": zstat, "p_value": pval}

    main = _coef_stats(x_col)
    extra = {col: _coef_stats(col) for col in extra_cols}

    ll_full = -opt_result.fun
    p_bar = y_arr.mean()
    ll_null = (y_arr * np.log(p_bar) + (1 - y_arr) * np.log(1 - p_bar)).sum()
    pseudo_r2 = 1 - ll_full / ll_null if ll_null != 0 else np.nan

    return {
        **main, "extra": extra,
        "pseudo_r2": pseudo_r2, "n_obs": n_obs,
        "n_entities": len(entities), "n_periods": len(periods),
        "n_clusters": n_clusters,
    }


def _standardize(series):
    """
    Z-score standardization (x - mean) / sd, per explicit instruction
    to include X, AboveMedian, and the support_share control as standardized
    variables in all regressions -- NOT the dummy/derived variables (shock_year_dummy,
    interaction terms), which stay in their original 0/1 or product
    scale. Mean/SD are computed on the series' own non-missing values
    (skipna, pandas default), so a NaN entry stays NaN after
    standardizing rather than being silently dropped or zeroed here --
    each regression's own dropna() still handles missing values
    exactly as before.
    """
    return (series - series.mean()) / series.std()


def _clean_var_label(label):
    """Strips both the "(data_..._full)" sheet-reference suffix and
    the ", ABOVE MEDIAN value" suffix from a regression spec's label,
    so all six models' results key into summary_table_results under
    the SAME 4 clean variable names, regardless of which of the three
    specifications (plain/interaction/above-median) produced the
    label."""
    return label.split(" (")[0].split(", ABOVE MEDIAN")[0]


def _academic_stars(p_value):
    """Standard academic significance stars from a coefficient's own
    p-value -- ***/**/* for p<0.01/0.05/0.10, "" otherwise. A DIFFERENT
    convention from significance_stars() above (which uses +/* for a
    two-proportion z-test between two CELLS, not a single coefficient's
    own p-value) -- kept separate since they answer different
    questions and would be confusing to conflate."""
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


# var_label -> {model_number: (coef_value, p_value)} -- filled in as
# each of the six regression tables below is built, then read out at
# the very end to build the academic-style summary table.
summary_table_results = {}


regression_specs = [
    ("Previous-year ICT investment share (data_ict_share_full)", merged_ict, "ict_share"),
    ("Previous-year AI/ICT-related EU export share (data_eur_export_full)", merged_eur, "eur_export_share"),
    ("Previous-year national vs. semiconductor index correlation (data_stock_corr_full)",
     merged_corr, "stock_semis_corr_annual"),
    ("Previous-year AI incoming investment share (data_ai_inv_share_full)",
     merged_ai_inv, "ai_inv_share"),
]

ws_data_regr = wb.create_sheet("data_regr")
data_regr_row = [1]  # mutable cursor (list so the helper can mutate it in place)


def write_regression_data_block(regression_number, label, df_src, x_col, extra_cols=None):
    """
    Writes the EXACT data panel_ols_two_way_fe() will run on for this
    regression -- country, target_year, growth_surprise, the
    explanatory column, and any extra regressor columns (shock dummy /
    interaction terms) -- as one labelled block in the shared
    "data_regr" sheet, in the SAME order the regressions themselves
    run in, so each regression's output can be checked by hand against
    its own exact input rows. extra_cols: list of column names (beyond
    country/target_year/growth_surprise/x_col) already present in
    df_src to include, e.g. ["shock_year_dummy", "interaction"].
    """
    extra_cols = extra_cols or []
    cols = ["country", "target_year"]
    if "explanatory_source_year" in df_src.columns:
        cols.append("explanatory_source_year")
    cols += ["growth_surprise", x_col] + extra_cols
    d = df_src[cols].dropna().copy()

    title_row = data_regr_row[0]
    ws_data_regr.cell(row=title_row, column=1,
                       value=f"Regression {regression_number}: {label}").font = Font(
        bold=True, size=12)
    header_row_d = title_row + 1
    for col_idx, col_name in enumerate(cols, start=1):
        ws_data_regr.cell(row=header_row_d, column=col_idx, value=col_name).font = bold

    for row_offset, row in enumerate(d.itertuples(index=False), start=1):
        r = header_row_d + row_offset
        for col_idx, val in enumerate(row, start=1):
            ws_data_regr.cell(row=r, column=col_idx, value=val)

    data_regr_row[0] = header_row_d + len(d) + 2  # +1 header consumed, +2 blank separator
    return len(d)


ws_data_regr.column_dimensions["A"].width = 45
for col_letter in "BCDEFGHIJKL":
    ws_data_regr.column_dimensions[col_letter].width = 16
chart_data_row = 30  # fixed early start row -- charts now come FIRST

# --- Bar charts + interaction (line) plots, reproducing the HORIZONTAL
# staging layout the user built by hand in Excel on top of an earlier
# version of this script's output (confirmed by inspecting that
# file's actual chart XML: categories/series run ACROSS a row, e.g.
# summary!$E$31:$F$31 for categories, not down a column like this
# script originally used) -- this version matches that layout exactly,
# plus adds the Previous-year AI incoming investment share block (not present in
# that manually-built example) for consistency with every other
# section of this sheet, and adds the interaction (line) plots below,
# which were not built by hand.
#
# All staging values are PLAIN PYTHON NUMBERS (not formulas) -- a
# prior version used live formulas here, which chained through OTHER
# uncached formula cells (the "share positive surprises" cells
# themselves) and left charts blank in the raw, un-recalculated file
# every user actually gets (confirmed from a live run: 0 of the
# intended charts survived when built on that formula chain). Reading
# straight from the already-computed cluster_props/yrs_props
# dictionaries sidesteps that entirely.
def _safe_share(x, n):
    """x successes out of n -- None (a genuinely blank cell) if n=0,
    matching this project's existing "n/a" handling elsewhere."""
    return (x / n) if n else None


ict_cluster_props, ict_yrs_props = all_props["Previous-year ICT investment share"]

variable_chart_specs = [
    ("Previous-year ICT investment share", "ICT/AI invest", "ICT/AI invest"),
    ("Previous-year AI/ICT-related EU export share", "ICT/AI  export", "ICTAI  export"),
    ("Previous-year national vs. semiconductor index correlation", "SOX correl", "SOX correl"),
    ("Previous-year AI incoming investment share", "AI equity inv", "AI equity inv"),
]
ws_summary.cell(row=chart_data_row, column=1,
                 value="Chart data (LIVE formulas referencing the summary blocks below "
                       "-- see the comment above row 28)").font = Font(bold=True, italic=True)

# --- Chart 1 data: "full sample" comparison, all years vs. shock
# years -- columns A-C, ONE row (full-sample share is identical across
# variables, so ICT's own value represents it).
full_row = chart_data_row + 2
ws_summary.cell(row=full_row, column=1, value="Full sample")
ws_summary.cell(row=full_row, column=2, value="All years - full")
ws_summary.cell(row=full_row, column=3, value="Shock years - full")
full_data_row = full_row + 1
ws_summary.cell(row=full_data_row, column=1, value="Share positive surprises")
# LIVE FORMULAS throughout this whole staging table, per explicit
# instruction -- referencing each variable's own "share positive
# surprises" cells (B/C/D/F/G/H at that variable's own header_row+1).
# HONESTY NOTE, kept from an earlier version of this comment: a
# formula referencing ANOTHER formula cell (e.g. B13 is itself
# "=COUNTIF(...)/COUNT(...)") has no cached value in the raw,
# un-recalculated file every user actually gets until Excel fully
# recalculates once -- confirmed earlier in this project to leave
# charts appearing blank until that first recalculation. This is now
# accepted deliberately across the whole table (not just rows 61-67
# as an earlier, narrower version of this script did), per explicit
# instruction; open and save the file once in Excel (or run the
# mandatory recalc step) before relying on the charts looking right.
ict_header_row = block_header_rows["Previous-year ICT investment share"]
ict_data_row = ict_header_row + 1
ws_summary.cell(row=full_data_row, column=2, value=f"=B{ict_data_row}")
ws_summary.cell(row=full_data_row, column=3, value=f"=F{ict_data_row}")
for c in (2, 3):
    ws_summary.cell(row=full_data_row, column=c).number_format = "0%"

# --- Charts 2-5 data: one 4-column block per variable (label col +
# All-years col + Shock-years col + 1 blank separator col), starting
# at column D, H, L, P -- matching the 4-column spacing the manually-
# built example used (D-F used, G blank; H-J used, K blank; etc.).
lowhigh_block_starts = [4, 8, 12, 16]  # D, H, L, P (1-indexed column numbers)
lowhigh_row_map = {}  # var_label -> (header_col, row_all, row_shock)
header_row_for_blocks = chart_data_row + 5
for var_idx, ((var_label, low_label, high_label), block_start_col) in enumerate(
        zip(variable_chart_specs, lowhigh_block_starts)):
    cp, yp = all_props[var_label]
    label_col = block_start_col
    all_col = block_start_col + 1
    shock_col = block_start_col + 2

    ws_summary.cell(row=header_row_for_blocks, column=all_col, value="All years").font = bold
    ws_summary.cell(row=header_row_for_blocks, column=shock_col, value="Shock years").font = bold

    low_row = header_row_for_blocks + 1
    ws_summary.cell(row=low_row, column=label_col, value=f"Low {low_label}")
    high_row = header_row_for_blocks + 2
    ws_summary.cell(row=high_row, column=label_col, value=f"High {high_label}")

    # LIVE FORMULAS for every variable now (not just ICT), per
    # explicit instruction -- see the comment above full_data_row.
    var_data_row = block_header_rows[var_label] + 1
    ws_summary.cell(row=low_row, column=all_col, value=f"=C{var_data_row}")
    ws_summary.cell(row=low_row, column=shock_col, value=f"=G{var_data_row}")
    ws_summary.cell(row=high_row, column=all_col, value=f"=D{var_data_row}")
    ws_summary.cell(row=high_row, column=shock_col, value=f"=H{var_data_row}")

    for r in (low_row, high_row):
        for c in (all_col, shock_col):
            ws_summary.cell(row=r, column=c).number_format = "0%"

    lowhigh_row_map[var_label] = (header_row_for_blocks, low_row, high_row,
                                   label_col, all_col, shock_col)


def build_bar_chart_horizontal(title, cats_ref, series_specs, y_title, x_title):
    """series_specs: list of (values_ref, series_name, color_hex) --
    series_name may be a plain string (literal series label) OR an
    openpyxl Reference (a LIVE cell reference for the series legend
    text, e.g. pointing at D36) -- a Reference is rendered as a
    strRef so the legend updates automatically if that cell's text
    ever changes, instead of being baked in as a fixed string."""
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.style = 10
    # NOTE: axis titles intentionally NOT set here, per explicit
    # instruction to remove all axis titles from these charts.
    # Explicit axis position/visibility -- confirmed necessary from the
    # earlier scatter-chart investigation in this project: openpyxl
    # does not reliably default the category axis to "b" (bottom) on
    # its own, and a chart with an ambiguous/undeclared axis position
    # can render with missing tick labels.
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.height = 7
    chart.width = 11
    for values_ref, name, color in series_specs:
        chart.add_data(values_ref, titles_from_data=False, from_rows=True)
        s = chart.series[-1]
        if isinstance(name, str):
            s.tx = openpyxl.chart.series.SeriesLabel(v=name)
        else:
            s.tx = openpyxl.chart.series.SeriesLabel(
                strRef=openpyxl.chart.data_source.StrRef(f=str(name)))
        s.graphicalProperties.solidFill = color
    chart.set_categories(cats_ref)
    return chart


def build_line_chart_horizontal(title, cats_ref, series_specs, y_title, x_title,
                                 point_stars=None):
    """
    series_specs: list of (values_ref, series_name, color_hex).
    point_stars: optional list of significance-star strings ("***"/
    "**"/"*"/""), one per category point (same length/order as
    cats_ref).

    HONESTY NOTE: an earlier version tried to show these as per-point
    CUSTOM DATA LABEL TEXT (DataLabel.tx) -- confirmed, by inspecting
    openpyxl's own DataLabel class directly, that this attribute does
    not exist in this openpyxl version's schema at all (only txPr,
    text FORMATTING, is supported; there is no tx, text CONTENT,
    override) -- so it was silently dropped and never appeared in the
    saved file's XML, verified directly. This version instead appends
    the stars to the CATEGORY axis label text itself (e.g. "Shock
    years ***"), which IS reliably rendered, since it uses the
    already-working category-reference mechanism rather than a
    per-point label override.
    """
    chart = LineChart()
    # NOTE: no chart.title assigned here, per explicit instruction --
    # the interaction plots are deliberately untitled (matching the
    # reviewed reference workbook's own chart6-9.xml, each confirmed
    # to have <c:autoTitleDeleted val="1"/> and no <c:title> element).
    # "title" is still accepted as a parameter (for the call sites'
    # own bookkeeping/potential future use) but intentionally unused
    # here. The bar charts built by build_bar_chart_horizontal() above
    # are a SEPARATE function and keep their own titles unchanged.
    chart.style = 12
    # NOTE: axis titles intentionally NOT set here either, per
    # explicit instruction to remove all axis titles from these
    # charts (matching the bar-chart change above).
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.height = 7
    chart.width = 11
    for values_ref, name, color in series_specs:
        chart.add_data(values_ref, titles_from_data=False, from_rows=True)
        s = chart.series[-1]
        if isinstance(name, str):
            s.tx = openpyxl.chart.series.SeriesLabel(v=name)
        else:
            s.tx = openpyxl.chart.series.SeriesLabel(
                strRef=openpyxl.chart.data_source.StrRef(f=str(name)))
        s.graphicalProperties.line.solidFill = color
        s.graphicalProperties.line.width = 25000
        s.marker = Marker(symbol="circle", size=7)
        s.marker.graphicalProperties = GraphicalProperties(solidFill=color)
        s.smooth = False
    chart.set_categories(cats_ref)
    return chart


chart_anchor_row = header_row_for_blocks + 5

# Chart 1: full-sample comparison (one series, two categories in a row)
full_cats = Reference(ws_summary, min_col=2, max_col=3, min_row=full_row, max_row=full_row)
full_vals = Reference(ws_summary, min_col=2, max_col=3, min_row=full_data_row, max_row=full_data_row)
chart1 = build_bar_chart_horizontal(
    "Share positive surprises: all years vs. shock years (full sample)",
    full_cats, [(full_vals, "Full sample", "4472C4")],
    "Share of positive surprises", "Period")
ws_summary.add_chart(chart1, f"A{chart_anchor_row}")

# Charts 2-5: one per variable, two series (Low/High) each spanning a
# row of 2 values (All years, Shock years).
bar_chart_anchor_cols = ["D", "I", "P", "Y"]
for i, (var_label, low_label, high_label) in enumerate(variable_chart_specs):
    header_row_b, low_row, high_row, label_col, all_col, shock_col = lowhigh_row_map[var_label]
    cats = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                      min_row=header_row_b, max_row=header_row_b)
    low_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                          min_row=low_row, max_row=low_row)
    high_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                           min_row=high_row, max_row=high_row)
    # Legend text now refers LIVE to the label cells themselves (e.g.
    # D36/D37), per explicit instruction -- NOT a plain string copy of
    # their text, so the legend stays in sync if that cell is ever
    # edited directly in Excel afterward.
    low_label_ref = Reference(ws_summary, min_col=label_col, max_col=label_col,
                               min_row=low_row, max_row=low_row)
    high_label_ref = Reference(ws_summary, min_col=label_col, max_col=label_col,
                                min_row=high_row, max_row=high_row)
    chart = build_bar_chart_horizontal(
        f"Share positive surprises: {var_label}", cats,
        [(low_vals, low_label_ref, "BDD7EE"), (high_vals, high_label_ref, "4472C4")],
        "Share of positive surprises", "Period")
    ws_summary.add_chart(chart, f"{bar_chart_anchor_cols[i]}{chart_anchor_row}")

# --- Interaction (line) plots, directly below the bar charts -- same
# staging data, plotted as two lines (Low/High) across the two
# periods. CONVERGING lines = the X effect shrinks in shock years;
# DIVERGING lines = it strengthens (the visual test discussed in chat
# for whether "high X" amplifies positive surprises specifically
# during shocks). Each point is annotated with significance stars
# (***/**/* at the 1%/5%/10% level) for whether High vs. Low differs
# significantly AT THAT SPECIFIC point (all years / shock years) --
# reusing the exact same two-proportion z-test already computed for
# the D/H header-cell stars earlier in this script, applied here to
# BOTH points instead of just one.
line_anchor_row = chart_anchor_row + 16
for i, (var_label, low_label, high_label) in enumerate(variable_chart_specs):
    header_row_b, low_row, high_row, label_col, all_col, shock_col = lowhigh_row_map[var_label]

    cp, yp = all_props[var_label]
    # Same-time (within-period) above-vs-below comparison, symbol="*"
    # -- an overall "is there a Low/High gap at all, at this point"
    # indicator, shown once per point (shared by both lines, since
    # both are annotated on the shared x-axis category text).
    all_years_stars = significance_stars(cp["x_above"], cp["n_above"],
                                          cp["x_below"], cp["n_below"], symbol="*")
    shock_years_stars = significance_stars(yp["x_above"], yp["n_above"],
                                            yp["x_below"], yp["n_below"], symbol="*")
    # HORIZONTAL (same-colour-dot) comparison, symbol="+": does the
    # Low line's shock-years point differ significantly from its OWN
    # all-years point, and likewise for the High line -- this is the
    # exact same two-proportion z-test already computed for the G/H
    # header-cell "+" markers above, reused here per-line.
    low_horizontal_plus = significance_stars(yp["x_below"], yp["n_below"],
                                              cp["x_below"], cp["n_below"], symbol="+")
    high_horizontal_plus = significance_stars(yp["x_above"], yp["n_above"],
                                               cp["x_above"], cp["n_above"], symbol="+")

    # Dedicated category row for the INTERACTION PLOT specifically
    # (its own row, not shared with the bar chart's categories above),
    # with significance stars appended directly to the category text
    # itself -- e.g. "Shock years *** (Low+++ High++)". This is
    # deliberately NOT implemented via per-point DataLabel text
    # overrides: confirmed, by inspecting openpyxl's own DataLabel
    # class directly, that this openpyxl version's DataLabel schema
    # has NO "tx" (text content) attribute at all (only "txPr", text
    # formatting) -- any such override is silently dropped and never
    # appears in the saved file's XML (verified directly against a
    # real saved file). The category-label approach below uses the
    # already-reliable category-reference mechanism instead. Since
    # Low and High SHARE one x-axis category point per period, their
    # separate "+" markers are combined into one label, tagged by
    # series so they stay distinguishable.
    interaction_cat_row = header_row_b + 3
    ws_summary.cell(row=interaction_cat_row, column=all_col,
                     value=f"All years {all_years_stars}".strip())
    shock_label_parts = ["Shock years"]
    if shock_years_stars:
        shock_label_parts.append(shock_years_stars)
    horiz_bits = []
    if low_horizontal_plus:
        horiz_bits.append(f"Low{low_horizontal_plus}")
    if high_horizontal_plus:
        horiz_bits.append(f"High{high_horizontal_plus}")
    if horiz_bits:
        shock_label_parts.append(f"({' '.join(horiz_bits)})")
    ws_summary.cell(row=interaction_cat_row, column=shock_col,
                     value=" ".join(shock_label_parts))
    interaction_cats = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                                  min_row=interaction_cat_row, max_row=interaction_cat_row)

    low_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                          min_row=low_row, max_row=low_row)
    high_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                           min_row=high_row, max_row=high_row)
    # Legend text refers LIVE to the same D36/D37-style label cells,
    # per explicit instruction -- same mechanism as the bar charts
    # above.
    low_label_ref = Reference(ws_summary, min_col=label_col, max_col=label_col,
                               min_row=low_row, max_row=low_row)
    high_label_ref = Reference(ws_summary, min_col=label_col, max_col=label_col,
                                min_row=high_row, max_row=high_row)

    chart = build_line_chart_horizontal(
        f"Interaction plot: {var_label}", interaction_cats,
        [(low_vals, low_label_ref, "BDD7EE"), (high_vals, high_label_ref, "4472C4")],
        "Share of positive surprises", "Period")
    anchor_col = bar_chart_anchor_cols[i]
    ws_summary.add_chart(chart, f"{anchor_col}{line_anchor_row}")

# --- Explanatory notes below the interaction charts, explaining what
# they show and how to read them.
notes_row = line_anchor_row + 16
notes = [
    "How to read the interaction plots above:",
    "Each plot shows two lines -- \"Low\" (below-median) and \"High\" "
    "(above-median) countries for that explanatory variable -- with "
    "their share of positive growth surprises plotted at two points: "
    "\"All years\" (the full sample) and \"Shock years\" (2020/2022/2025).",
    "PARALLEL lines mean the Low/High gap is roughly the same size in "
    "both periods -- i.e. the explanatory variable's effect does NOT "
    "change during shocks.",
    "DIVERGING lines (further apart at \"Shock years\" than at \"All "
    "years\") mean the High group's advantage over Low STRENGTHENS "
    "specifically during shocks.",
    "CONVERGING lines (closer together, or crossing, at \"Shock years\") "
    "mean the advantage SHRINKS or reverses during shocks.",
    "Significance markers: * on the shared axis label test whether Low "
    "and High differ at that SAME point (see the legend at row 28); "
    "+ markers next to \"Low\"/\"High\" in the \"Shock years\" label test "
    "whether THAT line's own shock-years point differs from its own "
    "all-years point (the horizontal, same-colour-line comparison).",
]
for offset, line in enumerate(notes):
    cell = ws_summary.cell(row=notes_row + offset, column=1, value=line)
    cell.font = Font(bold=True, size=10) if offset == 0 else Font(italic=True, size=9)
charts_end_row = notes_row + len(notes)

# --- IMPORTANT INTERPRETATION NOTE, applying to ALL panel regressions
# below (OLS, interaction, above-median-dummy alike): growth_surprise
# is used here as a CONTINUOUS variable, with BOTH its positive and
# negative observations included -- these are ordinary OLS
# regressions on the full, unsplit surprise magnitude, not on a
# positive/negative split. A positive beta therefore means X raises
# the AVERAGE/EXPECTED SIZE of the surprise (E[growth_surprise|X]
# shifts upward) -- it does NOT mean X makes a positive surprise MORE
# LIKELY. That second, genuinely different question -- does X raise
# the PROBABILITY that growth_surprise > 0 -- is what the "share
# positive surprises" blocks above (and their D/F/G/H significance
# markers) test via a two-proportion comparison, and what the LOGIT
# regressions further below test directly via a binary outcome model.
# These two kinds of results can legitimately diverge (e.g. a variable
# can shift the average size of surprises without changing how often
# they are positive, or vice versa, if a few large outliers dominate
# the mean without moving the majority of observations across zero).
reg_start_row = charts_end_row + 3  # charts now come first; regression tables follow
ws_summary.cell(
    row=reg_start_row, column=1,
    value="NOTE: the regressions below use CONTINUOUS growth_surprise (both "
          "positive and negative values) -- beta/gamma estimate whether X "
          "raises the AVERAGE SIZE of the surprise, NOT whether X makes a "
          "positive surprise more LIKELY (that is what the LOGIT models "
          "further below, and the 'share positive surprises' blocks above, "
          "test instead). Every X is aligned strictly at t-1 relative to the "
          "outcome year t; observations without that prior-year value are excluded. "
          "The standardized State Aid control is contemporaneous at t (using the "
          "country's 2024 value only when its 2025 value is missing)."
).font = Font(italic=True, size=9, color="800000")
reg_start_row += 2
ws_summary.cell(row=reg_start_row, column=1,
    value="Panel regressions: growth_surprise(t) = const + country FE + "
                       "time FE + beta*X(t-1) + phi*support_share_std(t)").font = Font(
    bold=True, size=12)
ws_summary.cell(
    row=reg_start_row + 1, column=1,
    value="(two-way fixed effects, OLS via dummy variables; standard errors "
          "clustered by country)"
).font = Font(italic=True, size=9)

reg_header_row = reg_start_row + 3
reg_headers = ["Explanatory variable", "beta", "SE", "t-stat", "p-value",
               "phi (std support_share)", "SE(phi)", "p(phi)",
               "R-squared", "N (obs)", "N (countries)", "N (years)"]
for col_idx, h in enumerate(reg_headers, start=1):
    ws_summary.cell(row=reg_header_row, column=col_idx, value=h).font = bold

for offset, (label, df_src, x_col) in enumerate(regression_specs):
    df_src = df_src.copy()
    df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                  - df_src["forecast_growth_annual_pct"])
    # X is standardized (z-score) before entering the regression, per
    # explicit instruction -- a new column, NOT overwriting x_col
    # itself, so the raw value is still available (and still shown,
    # via extra_cols below) in the data_regr sheet for verification.
    x_col_std = x_col + "_std"
    df_src[x_col_std] = _standardize(df_src[x_col])
    df_src["support_share_std"] = _standardize(df_src["support_share"])
    result = panel_ols_two_way_fe(
        df_src, "growth_surprise", x_col_std, extra_cols=["support_share_std"])
    phi = result["extra"]["support_share_std"]
    write_regression_data_block(
        1, label, df_src, x_col_std,
        extra_cols=[x_col, "support_share", "support_share_std",
                    "support_source_year"])

    row_idx = reg_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["t_stat"], 3))
    ws_summary.cell(row=row_idx, column=5, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(phi["beta"], 4))
    ws_summary.cell(row=row_idx, column=7, value=round(phi["se"], 4))
    ws_summary.cell(row=row_idx, column=8, value=round(phi["p_value"], 4))
    ws_summary.cell(row=row_idx, column=9, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=10, value=result["n_obs"])
    ws_summary.cell(row=row_idx, column=11, value=result["n_entities"])
    ws_summary.cell(row=row_idx, column=12, value=result["n_periods"])
    print(f"  Panel regression ({label}): beta={result['beta']:.4f}, "
          f"p={result['p_value']:.4f}, N={result['n_obs']}")
    summary_table_results.setdefault(_clean_var_label(label), {})[4] = (result["beta"], result["p_value"])

# Widths for B-H (shared with the "full sample"/"below"/"above" summary
# blocks above, incl. their significance-star suffixes like "full
# sample ***") kept wide enough for BOTH uses -- this section used to
# set B-H narrower again (as low as 10), overriding the wider setting
# applied earlier and effectively hiding the star suffixes in Excel
# (text overflowing into an adjacent non-empty cell gets visually
# truncated, not shown). Column I (only used by the regression table,
# not the summary blocks) can stay narrower.
for col_letter, width in zip(
        "BCDEFGHIJKL", [18, 18, 18, 18, 18, 18, 18, 18, 12, 12, 12]):
    ws_summary.column_dimensions[col_letter].width = width

# --- Interaction regression: growth_surprise = const + country FE +
# time FE + beta*X + gamma*(X * shock_year_dummy) -- run on the FULL
# sample (NOT restricted to shock years only), with shock_year_dummy=1
# for target_year in {2020, 2022, 2025} and 0 otherwise. This
# estimates whether the SLOPE of X on growth_surprise is genuinely
# DIFFERENT during shock years specifically (gamma), on top of the
# already-controlled-for country and year fixed effects -- a more
# direct test of "does ICT/export/correlation matter MORE during
# shocks" than comparing two separately-run regressions on different
# sub-samples (the shock-years-only table this replaces), since gamma
# here is estimated jointly with beta on the full, larger sample.
SHOCK_YEARS_SET = set(SPECIAL_YEARS)

interaction_specs = [
    ("Previous-year ICT investment share (data_ict_share_full)", merged_ict, "ict_share"),
    ("Previous-year AI/ICT-related EU export share (data_eur_export_full)", merged_eur, "eur_export_share"),
    ("Previous-year national vs. semiconductor index correlation (data_stock_corr_full)",
     merged_corr, "stock_semis_corr_annual"),
    ("Previous-year AI incoming investment share (data_ai_inv_share_full)",
     merged_ai_inv, "ai_inv_share"),
]

reg2_start_row = reg_header_row + len(regression_specs) + 3
ws_summary.cell(
    row=reg2_start_row, column=1,
    value="Panel regressions WITH SHOCK-YEAR INTERACTION: growth_surprise = const + "
          "country FE + time FE + beta*X + gamma*(X*shock_year_dummy) + "
          "phi*support_share_std(t)"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=reg2_start_row + 1, column=1,
    value="(full sample; shock_year_dummy=1 for target_year in 2020/2022/2025 -- gamma "
          "estimates X's ADDITIONAL effect specifically during shock years, on top of "
          "beta's baseline effect)"
).font = Font(italic=True, size=9)

reg2_header_row = reg2_start_row + 3
reg2_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "p(beta)",
                "gamma (X*shock)", "SE(gamma)", "p(gamma)",
                "phi (std support_share)", "SE(phi)", "p(phi)",
                "R-squared", "N (obs)"]
for col_idx, h in enumerate(reg2_headers, start=1):
    ws_summary.cell(row=reg2_header_row, column=col_idx, value=h).font = bold

for offset, (label, df_src, x_col) in enumerate(interaction_specs):
    df_src = df_src.copy()
    df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                  - df_src["forecast_growth_annual_pct"])
    # X standardized (z-score) BEFORE building the interaction term, so
    # "interaction" is standardized_X * shock_dummy, not raw_X *
    # shock_dummy -- the interaction term itself is a product with a
    # 0/1 dummy, so it is NOT separately standardized (per explicit
    # instruction: dummies/derived product terms stay as-is).
    x_col_std = x_col + "_std"
    df_src[x_col_std] = _standardize(df_src[x_col])
    df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
    df_src["interaction"] = df_src[x_col_std] * df_src["shock_year_dummy"]
    df_src["support_share_std"] = _standardize(df_src["support_share"])
    result = panel_ols_two_way_fe(
        df_src, "growth_surprise", x_col_std,
        extra_cols=["interaction", "support_share_std"])
    gamma = result["extra"]["interaction"]
    phi = result["extra"]["support_share_std"]
    write_regression_data_block(2, label, df_src, x_col_std,
                                 extra_cols=[x_col, "shock_year_dummy", "interaction",
                                             "support_share", "support_share_std",
                                             "support_source_year"])

    row_idx = reg2_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
    ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
    ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
    ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
    ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
    ws_summary.cell(row=row_idx, column=11, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
    print(f"  Interaction regression ({label}): beta={result['beta']:.4f} "
          f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} (p={gamma['p_value']:.4f}), "
          f"N={result['n_obs']}")
    summary_table_results.setdefault(_clean_var_label(label), {})[5] = (gamma["beta"], gamma["p_value"])

# --- THIRD regression table: same interaction specification, but with
# X replaced by a BINARY "above median" indicator (1 if that
# country-year's X was above the variable's OWN overall median, 0
# otherwise) instead of the continuous X value -- i.e. it directly
# tests the SAME "above/below median" classification already shown
# throughout the summary blocks above, via a regression instead of a
# simple proportion comparison. gamma here estimates the ADDITIONAL
# effect of being ABOVE MEDIAN specifically during shock years, on top
# of beta's baseline above-median effect.
above_median_specs = [
    ("Previous-year ICT investment share, ABOVE MEDIAN value (data_ict_share_full)", merged_ict, "ict_share"),
    ("Previous-year AI/ICT-related EU export share, ABOVE MEDIAN value (data_eur_export_full)",
     merged_eur, "eur_export_share"),
    ("Previous-year national vs. semiconductor index correlation, ABOVE MEDIAN value "
     "(data_stock_corr_full)", merged_corr, "stock_semis_corr_annual"),
    ("Previous-year AI incoming investment share, ABOVE MEDIAN value (data_ai_inv_share_full)",
     merged_ai_inv, "ai_inv_share"),
]

reg3_start_row = reg2_header_row + len(interaction_specs) + 3
ws_summary.cell(
    row=reg3_start_row, column=1,
    value="Panel regressions WITH ABOVE-MEDIAN VARIABLE AND SHOCK-YEAR INTERACTION: "
          "growth_surprise = const + country FE + time FE + beta*AboveMedian + "
          "gamma*(AboveMedian*shock_year_dummy) + phi*support_share_std(t)"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=reg3_start_row + 1, column=1,
    value="(full sample; AboveMedian is NOT a binary dummy -- it takes X's own "
          "CONTINUOUS value for observations above that variable's overall median, and "
          "0 otherwise, so beta captures the marginal effect of X's own MAGNITUDE "
          "specifically among above-median observations, not merely a level shift from "
          "crossing the median threshold; gamma estimates the ADDITIONAL effect of that "
          "same above-median magnitude specifically during shock years, on top of beta's "
          "baseline above-median effect)"
).font = Font(italic=True, size=9)

reg3_header_row = reg3_start_row + 3
reg3_headers = ["Explanatory variable", "beta (AboveMedian)", "SE(beta)", "p(beta)",
                "gamma (AboveMedian*shock)", "SE(gamma)", "p(gamma)",
                "phi (std support_share)", "SE(phi)", "p(phi)",
                "R-squared", "N (obs)"]
for col_idx, h in enumerate(reg3_headers, start=1):
    ws_summary.cell(row=reg3_header_row, column=col_idx, value=h).font = bold

for offset, (label, df_src, x_col) in enumerate(above_median_specs):
    df_src = df_src.copy()
    df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                  - df_src["forecast_growth_annual_pct"])
    # Median computed on the SAME (full-sample) data this regression
    # runs on, matching how write_clustered_time_chart() computes its
    # own median for the summary blocks above (dropna first, so a
    # missing X doesn't skew the threshold).
    median_val = df_src[x_col].dropna().median()
    # BUG FIX, found via the data_regr sheet cross-check below: a plain
    # "(df_src[x_col] > median_val)" comparison silently evaluates to
    # False (not NaN) when x_col itself is NaN in pandas -- meaning a
    # country-year with genuinely MISSING X was being mis-classified
    # as "below median" (0) instead of excluded from the regression
    # entirely, inflating N by exactly the count of missing X rows (4
    # extra rows for ICT, confirmed: reported N=488 vs. the true
    # complete-case N=484 once this is fixed). np.where here explicitly
    # preserves NaN so panel_ols_two_way_fe()'s own dropna() correctly
    # excludes these rows, matching every other regression in this
    # script.
    #
    # CONTINUOUS version, per explicit instruction: "above_median" here
    # is NOT the binary 0/1 dummy anymore -- it takes X's own
    # CONTINUOUS value for observations above the median, and 0
    # otherwise (a "kinked"/censored-at-the-median variable), so beta_3
    # now captures the marginal effect of X's own MAGNITUDE specifically
    # among above-median observations, rather than just the level shift
    # from crossing the median threshold. The variable/column name
    # "above_median" is kept unchanged (models 3/6's table row/column
    # labelling and write_regression_data_block()'s column list both
    # reference this exact name) -- only its CONTENT changed.
    df_src["above_median_raw"] = np.where(
        df_src[x_col].isna(), np.nan,
        np.where(df_src[x_col] > median_val, df_src[x_col], 0.0))
    # AboveMedian is standardized (z-score) AFTER being built (it's a
    # derived variable -- X's value where above the median, else 0 --
    # not X itself), per explicit instruction; the median threshold
    # above stays computed on the RAW X (a monotonic transform doesn't
    # change which observations fall above/below it). "above_x_shock"
    # is then built from the STANDARDIZED above_median, matching how
    # interaction_specs' own "interaction" is built from standardized
    # X above -- the dummy/product term itself is not separately
    # standardized.
    df_src["above_median"] = _standardize(df_src["above_median_raw"])
    df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
    df_src["above_x_shock"] = df_src["above_median"] * df_src["shock_year_dummy"]
    df_src["support_share_std"] = _standardize(df_src["support_share"])
    result = panel_ols_two_way_fe(
        df_src, "growth_surprise", "above_median",
        extra_cols=["above_x_shock", "support_share_std"])
    gamma = result["extra"]["above_x_shock"]
    phi = result["extra"]["support_share_std"]
    write_regression_data_block(3, label, df_src, "above_median",
                                 extra_cols=[x_col, "above_median_raw",
                                             "shock_year_dummy", "above_x_shock",
                                             "support_share", "support_share_std",
                                             "support_source_year"])

    row_idx = reg3_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
    ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
    ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
    ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
    ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
    ws_summary.cell(row=row_idx, column=11, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
    print(f"  Above-median interaction regression ({label}): beta={result['beta']:.4f} "
          f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} (p={gamma['p_value']:.4f}), "
          f"N={result['n_obs']}, median={median_val:.4f}")
    summary_table_results.setdefault(_clean_var_label(label), {})[6] = (gamma["beta"], gamma["p_value"])


# --- LOGIT versions of all three specifications above (plain X, X +
# shock-year interaction, above-median value + interaction), estimating
# P(growth_surprise > 0 | X) instead of E[growth_surprise | X] -- see
# panel_logit_two_way_fe()'s own docstring for the full explanation of
# why logit (not probit) was chosen here, and the important caveat that
# this specific LSDV-style (dummy-variable) implementation does NOT
# fully solve the incidental parameters problem the way a genuine
# conditional (fixed-effects) logit would.
probit_primary_start_row = reg3_header_row + len(above_median_specs) + 3
ws_summary.cell(
    row=probit_primary_start_row, column=1,
    value="PROBIT versions of the three panel regressions above: "
          "P(growth_surprise > 0) = Phi(const + country FE + time FE + beta*X "
          "[+ gamma*interaction] + phi*support_share_std(t))"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=probit_primary_start_row + 1, column=1,
    value="(estimates whether X makes a POSITIVE surprise MORE LIKELY, not whether "
          "X raises the average SIZE of the surprise -- see the note above the OLS "
          "tables. PRIMARY functional form for this project (rather than logit) -- "
          "probit is the more common choice in the macro/growth-forecasting "
          "literature and aligns conceptually with OLS's implicit normal-error "
          "assumption; a logit robustness check is reported further below)"
).font = Font(italic=True, size=9)

probit_primary_spec_groups = [
    ("Plain X", regression_specs, None),
    ("X + shock-year interaction", interaction_specs, "interaction"),
    ("Above-median value + interaction", above_median_specs, "above_x_shock"),
]

probit_primary_current_row = probit_primary_start_row + 3
for group_label, specs, interaction_col_name in probit_primary_spec_groups:
    ws_summary.cell(row=probit_primary_current_row, column=1, value=group_label).font = Font(
        bold=True, size=10, italic=True)
    probit_primary_header_row = probit_primary_current_row + 1
    if interaction_col_name is None:
        probit_primary_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "z(beta)",
                          "p(beta)", "phi (std support_share)", "SE(phi)", "p(phi)",
                          "Pseudo R-sq", "N (obs)"]
    else:
        probit_primary_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "p(beta)",
                          f"gamma ({interaction_col_name})", "SE(gamma)", "p(gamma)",
                          "phi (std support_share)", "SE(phi)", "p(phi)",
                          "Pseudo R-sq", "N (obs)"]
    for col_idx, h in enumerate(probit_primary_headers, start=1):
        ws_summary.cell(row=probit_primary_header_row, column=col_idx, value=h).font = bold

    for offset, (label, df_src, x_col) in enumerate(specs):
        df_src = df_src.copy()
        df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                      - df_src["forecast_growth_annual_pct"])
        row_idx = probit_primary_header_row + 1 + offset

        if interaction_col_name is None:
            x_col_std = x_col + "_std"
            df_src[x_col_std] = _standardize(df_src[x_col])
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_probit_two_way_fe(
                df_src, "growth_surprise", x_col_std,
                extra_cols=["support_share_std"])
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["z_stat"], 3))
            ws_summary.cell(row=row_idx, column=5, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=10, value=result["n_obs"])
            print(f"  Probit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), N={result['n_obs']}")
            summary_table_results.setdefault(_clean_var_label(label), {})[1] = (
                result["beta"], result["p_value"])
        elif interaction_col_name == "interaction":
            x_col_std = x_col + "_std"
            df_src[x_col_std] = _standardize(df_src[x_col])
            df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
            df_src["interaction"] = df_src[x_col_std] * df_src["shock_year_dummy"]
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_probit_two_way_fe(
                df_src, "growth_surprise", x_col_std,
                extra_cols=["interaction", "support_share_std"])
            gamma = result["extra"]["interaction"]
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=11, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
            print(f"  Probit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} "
                  f"(p={gamma['p_value']:.4f}), N={result['n_obs']}")
            summary_table_results.setdefault(_clean_var_label(label), {})[2] = (
                gamma["beta"], gamma["p_value"])
        else:  # above_x_shock
            median_val = df_src[x_col].dropna().median()
            # CONTINUOUS version, matching the OLS specification (model
            # 6) above exactly: X's own value where above median, 0
            # otherwise -- not the binary 0/1 dummy. Standardized
            # AFTER being built, same reasoning as the OLS version.
            df_src["above_median_raw"] = np.where(
                df_src[x_col].isna(), np.nan,
                np.where(df_src[x_col] > median_val, df_src[x_col], 0.0))
            df_src["above_median"] = _standardize(df_src["above_median_raw"])
            df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
            df_src["above_x_shock"] = df_src["above_median"] * df_src["shock_year_dummy"]
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_probit_two_way_fe(
                df_src, "growth_surprise", "above_median",
                extra_cols=["above_x_shock", "support_share_std"])
            gamma = result["extra"]["above_x_shock"]
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=11, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
            print(f"  Probit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} "
                  f"(p={gamma['p_value']:.4f}), N={result['n_obs']}, median={median_val:.4f}")
            summary_table_results.setdefault(_clean_var_label(label), {})[3] = (
                gamma["beta"], gamma["p_value"])

    probit_primary_current_row = probit_primary_header_row + len(specs) + 3

# --- Academic-style summary table, below ALL regression output above:
# rows = the 4 explanatory variables, columns = the 6 models (1)-(6),
# cells = the requested coefficient (beta for models 1 & 4, gamma for
# models 2/3/5/6) with significance stars -- collected in
# summary_table_results as each of the six regression loops ran above.
# Layout matches the reviewed reference workbook exactly: model-number
# row centred, a second "beta"/"gamma" sub-header row (italic, centred)
# identifying which coefficient each column shows, one blank row before
# the data rows, and centred coefficient cells.
model_labels = {
    1: "(1)", 2: "(2)", 3: "(3)", 4: "(4)", 5: "(5)", 6: "(6)",
}
model_estimator = {
    1: "Probit", 2: "Probit", 3: "Probit", 4: "OLS", 5: "OLS", 6: "OLS",
}
model_coef_kind = {
    1: "beta", 2: "gamma", 3: "gamma", 4: "beta", 5: "gamma", 6: "gamma",
}
center_align = Alignment(horizontal="center")
from openpyxl.styles import Side, Border
bottom_border = Border(bottom=Side(style="thin"))

summary_table_start_row = probit_primary_current_row + 2
white_fill = PatternFill(start_color="FFFFFFFF", end_color="FFFFFFFF", fill_type="solid")

ws_summary.cell(row=summary_table_start_row, column=1,
                 value="Regression results").font = Font(
    bold=True, size=12)

table_header_row = summary_table_start_row + 2
ws_summary.cell(row=table_header_row, column=1, value="").font = bold
for model_num in range(1, 7):
    cell = ws_summary.cell(row=table_header_row, column=1 + model_num,
                            value=model_labels[model_num])
    cell.font = bold
    cell.alignment = center_align

# NEW row identifying each column's estimator (OLS vs. Logit), between
# the model-number row and the beta/gamma row.
estimator_row = table_header_row + 1
for model_num in range(1, 7):
    cell = ws_summary.cell(row=estimator_row, column=1 + model_num,
                            value=model_estimator[model_num])
    cell.font = bold
    cell.alignment = center_align

subheader_row = estimator_row + 2
for model_num in range(1, 7):
    cell = ws_summary.cell(row=subheader_row, column=1 + model_num,
                            value=model_coef_kind[model_num])
    cell.font = Font(italic=True, size=11)
    cell.alignment = center_align
    cell.border = bottom_border
ws_summary.cell(row=subheader_row, column=1).border = bottom_border

# Row order and display names, per explicit instruction, matching the
# reviewed reference workbook exactly -- NOT the order regression_specs
# itself lists these four variables in (that order stays ICT/Export/
# stock_corr/ai_inv, unchanged, everywhere else in this script; only
# THIS table's display order/names differ).
table_display_names = [
    ("Previous-year ICT investment share", "ICT/AI-related investment"),
    ("Previous-year AI incoming investment share", "Equity investments in AI"),
    ("Previous-year AI/ICT-related EU export share", "ICT/AI-related export"),
    ("Previous-year national vs. semiconductor index correlation", "Correlation semiconductor index"),
]
data_first_row = subheader_row + 2
for row_offset, (internal_label, display_label) in enumerate(table_display_names):
    row_idx = data_first_row + row_offset
    is_last_row = (row_offset == len(table_display_names) - 1)
    label_cell = ws_summary.cell(row=row_idx, column=1, value=display_label)
    if is_last_row:
        label_cell.border = bottom_border
    coefs_for_var = summary_table_results.get(internal_label, {})
    for model_num in range(1, 7):
        coef_p = coefs_for_var.get(model_num)
        cell = ws_summary.cell(row=row_idx, column=1 + model_num)
        cell.alignment = center_align
        if is_last_row:
            cell.border = bottom_border
        if coef_p is None:
            cell.value = "n/a"
            continue
        coef_val, p_val = coef_p
        stars = _academic_stars(p_val)
        # 2 decimals (was 4), per explicit instruction, matching the
        # reviewed reference workbook.
        cell.value = f"{coef_val:.2f}{stars}"

table_note_row = data_first_row + len(table_display_names) + 1
note_lines = [
    "Note: Coefficient shown is beta for models (1) and (4) (the plain-X "
    "specification), and gamma (the interaction-term coefficient) for models "
    "(2), (3), (5), and (6). Significance: * p<0.10, ** p<0.05, *** p<0.01.",
    "(4) OLS, plain X: growth_surprise = const + country FE + time FE + beta*X "
    "+ phi*support_share_std.",
    "(5) OLS, X + shock-year interaction: adds gamma*(X*shock_year_dummy) to (4); "
    "gamma is the ADDITIONAL effect of X specifically during shock years.",
    # Split across two lines, same reasoning/layout as (1) below: this
    # note is noticeably longer than the others (266 characters as one
    # line) and was wrapping/overflowing awkwardly as a single line.
    "(6) OLS, above-median value + interaction: same as (5), but X is replaced "
    "by a variable equal to X's own value where X is above its overall median, "
    "and 0 otherwise (NOT a binary indicator);",
    "     gamma is that above-median magnitude's additional effect during shock years.",
    # Split across two lines (matching the reviewed reference workbook's
    # own rows 154-155) -- this one note is noticeably longer than the
    # others and was wrapping/overflowing awkwardly as a single line.
    "(1) PROBIT, plain X: same specification as (4), but P(growth_surprise > 0) "
    "instead of E[growth_surprise] -- see the note above the OLS tables and "
    "panel_probit_two_way_fe()'s docstring",
    "     for the beta-vs-gamma interpretation and the incidental-parameters caveat.",
    "(2) PROBIT, X + shock-year interaction: probit counterpart of (5).",
    "(3) PROBIT, above-median value + interaction: probit counterpart of (6).",
    "Standard errors in every OLS, probit, and logit regression are COUNTRY-CLUSTERED "
    "sandwich standard errors, equivalent to Stata's vce(cluster country_id).",
    "     This allows arbitrary heteroskedasticity and dependence among observations "
    "within the same country; inference assumes independence across countries. The "
    "same finite-sample cluster correction is used for all three estimators.",
    "     OLS p-values use a t distribution with number-of-countries minus one degrees "
    "of freedom; probit and logit report their conventional cluster-robust z tests.",
    "Timing: each focal explanatory variable X is the country's observed value in t-1 "
    "for an outcome in year t; missing t-1 values are excluded from the relevant "
    "chart, test, and regression.",
    "     AboveMedian_raw is constructed from that same t-1 explanatory value, and "
    "shock interactions multiply the lagged exposure by the outcome-year shock dummy; "
    "support_share is contemporaneous at t (2024 if 2025 is missing), standardized, "
    "and its phi is omitted only from the compact Regression results table.",
    # Explanatory note on standardization, per explicit instruction --
    # split across several lines, same wrapping reasoning as the notes
    # above.
    "Explanatory variables are STANDARDIZED (z-score), the dependent variable is NOT: "
    "growth_surprise (the left-hand-side variable) is left in its original, "
    "non-standardized units in every regression above; X and AboveMedian are each",
    "     standardized before entering the regression (mean 0, SD 1) -- but "
    "shock_year_dummy and the interaction terms themselves (X*shock_year_dummy, "
    "AboveMedian*shock_year_dummy) are NOT separately standardized, since they are",
    "     dummy/product terms, not the continuous explanatory variables the "
    "standardization was requested for; support_share is also standardized before entry, "
    "with phi, SE(phi), and p(phi) reported in the detailed tables.",
    "For OLS (models 4-6): this means beta is directly interpretable as \"a "
    "one-standard-deviation increase in X is associated with a beta-unit change in "
    "growth_surprise (in its own original units), holding other variables constant.\"",
    "For PROBIT (models 1-3): the SAME one-standard-deviation-increase framing "
    "applies to the underlying latent index (X*beta), but NOT directly to the predicted "
    "PROBABILITY of a positive surprise -- because of the normal (Gaussian) link function,",
    "     you cannot get the change in probability simply by reading off beta: a "
    "one-standard-deviation increase in X is associated with an increase in the "
    "predicted probability of a positive surprise of some number of percentage",
    "     points, but that number is NOT beta itself, and it is NOT constant -- it "
    "depends on the STARTING VALUES of all the covariates (the marginal effect of X "
    "on probability varies along the normal CDF curve, largest near p=0.5 and",
    "     smaller near p=0 or p=1). Computing an actual percentage-point figure would "
    "require evaluating the model's predicted probability at a specific combination "
    "of covariate values (e.g. at their means) both before and after a "
    "one-standard-deviation shift in X, which is not done in this table.",
]
for offset, line in enumerate(note_lines):
    cell = ws_summary.cell(row=table_note_row + offset, column=1, value=line)
    cell.font = Font(italic=True, size=9)

# White background across the ENTIRE table (title row through the last
# note line, columns A-G), matching the reviewed reference workbook --
# applied as a final pass over the whole range rather than per-cell
# above, so no cell in this section is accidentally missed.
table_last_row = table_note_row + len(note_lines) - 1
for r in range(summary_table_start_row, table_last_row + 1):
    for c in range(1, 8):
        ws_summary.cell(row=r, column=c).fill = white_fill

# --- PROBIT robustness check, at the very bottom of the summary
# sheet, in the SAME table format as the OLS/logit tables above.
# Estimates the same three specifications (plain X, X+shock-year
# interaction, above-median value+interaction) as probit models
# instead of logit -- see panel_probit_two_way_fe()'s own docstring
# for why this is honestly a functional-form robustness check, not a
# fix for the incidental-parameters problem discussed for logit above
# (probit has no conditional-MLE escape from that problem at all).
logit_robustness_start_row = table_last_row + 3
ws_summary.cell(
    row=logit_robustness_start_row, column=1,
    value="LOGIT robustness check: same three specifications as the PROBIT tables "
          "above, re-estimated with a logistic (instead of normal) link function"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=logit_robustness_start_row + 1, column=1,
    value="(compares whether the LOGISTIC vs. NORMAL functional-form choice changes "
          "sign/significance. LSDV-style dummy-variable logit, NOT conditional/"
          "fixed-effects logit -- does not fully solve the incidental parameters "
          "problem despite logit's theoretical advantage over probit in the "
          "CONDITIONAL case; see panel_logit_two_way_fe()'s docstring)"
).font = Font(italic=True, size=9)

logit_robustness_spec_groups = [
    ("Plain X", regression_specs, None),
    ("X + shock-year interaction", interaction_specs, "interaction"),
    ("Above-median value + interaction", above_median_specs, "above_x_shock"),
]

logit_robustness_current_row = logit_robustness_start_row + 3
for group_label, specs, interaction_col_name in logit_robustness_spec_groups:
    ws_summary.cell(row=logit_robustness_current_row, column=1, value=group_label).font = Font(
        bold=True, size=10, italic=True)
    logit_robustness_header_row = logit_robustness_current_row + 1
    if interaction_col_name is None:
        logit_robustness_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "z(beta)",
                           "p(beta)", "phi (std support_share)", "SE(phi)", "p(phi)",
                           "Pseudo R-sq", "N (obs)"]
    else:
        logit_robustness_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "p(beta)",
                           f"gamma ({interaction_col_name})", "SE(gamma)", "p(gamma)",
                           "phi (std support_share)", "SE(phi)", "p(phi)",
                           "Pseudo R-sq", "N (obs)"]
    for col_idx, h in enumerate(logit_robustness_headers, start=1):
        ws_summary.cell(row=logit_robustness_header_row, column=col_idx, value=h).font = bold

    for offset, (label, df_src, x_col) in enumerate(specs):
        df_src = df_src.copy()
        df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                      - df_src["forecast_growth_annual_pct"])
        row_idx = logit_robustness_header_row + 1 + offset

        if interaction_col_name is None:
            x_col_std = x_col + "_std"
            df_src[x_col_std] = _standardize(df_src[x_col])
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_logit_two_way_fe(
                df_src, "growth_surprise", x_col_std,
                extra_cols=["support_share_std"])
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["z_stat"], 3))
            ws_summary.cell(row=row_idx, column=5, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=10, value=result["n_obs"])
            print(f"  Logit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), N={result['n_obs']}")
        elif interaction_col_name == "interaction":
            x_col_std = x_col + "_std"
            df_src[x_col_std] = _standardize(df_src[x_col])
            df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
            df_src["interaction"] = df_src[x_col_std] * df_src["shock_year_dummy"]
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_logit_two_way_fe(
                df_src, "growth_surprise", x_col_std,
                extra_cols=["interaction", "support_share_std"])
            gamma = result["extra"]["interaction"]
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=11, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
            print(f"  Logit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} "
                  f"(p={gamma['p_value']:.4f}), N={result['n_obs']}")
        else:  # above_x_shock
            median_val = df_src[x_col].dropna().median()
            df_src["above_median_raw"] = np.where(
                df_src[x_col].isna(), np.nan,
                np.where(df_src[x_col] > median_val, df_src[x_col], 0.0))
            df_src["above_median"] = _standardize(df_src["above_median_raw"])
            df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
            df_src["above_x_shock"] = df_src["above_median"] * df_src["shock_year_dummy"]
            df_src["support_share_std"] = _standardize(df_src["support_share"])
            result = panel_logit_two_way_fe(
                df_src, "growth_surprise", "above_median",
                extra_cols=["above_x_shock", "support_share_std"])
            gamma = result["extra"]["above_x_shock"]
            phi = result["extra"]["support_share_std"]
            ws_summary.cell(row=row_idx, column=1, value=label)
            ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
            ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
            ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
            ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
            ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
            ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
            ws_summary.cell(row=row_idx, column=8, value=round(phi["beta"], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(phi["se"], 4))
            ws_summary.cell(row=row_idx, column=10, value=round(phi["p_value"], 4))
            ws_summary.cell(row=row_idx, column=11, value=round(result["pseudo_r2"], 4))
            ws_summary.cell(row=row_idx, column=12, value=result["n_obs"])
            print(f"  Logit ({group_label}, {label}): beta={result['beta']:.4f} "
                  f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} "
                  f"(p={gamma['p_value']:.4f}), N={result['n_obs']}, median={median_val:.4f}")

    logit_robustness_current_row = logit_robustness_header_row + len(specs) + 3

# --- Move "data_regr" to right after "summary" in the sheet tab order,
# per explicit instruction -- it's currently created mid-script (after
# all the per-variable data/chart sheets already exist), so its natural
# creation-order position is much later; this repositions it without
# touching any of those other sheets' own order relative to each other.
summary_idx = wb.sheetnames.index("summary")
data_regr_idx = wb.sheetnames.index("data_regr")
wb.move_sheet("data_regr", offset=(summary_idx + 1) - data_regr_idx)

wb.save("ict_growth_surprise_scatter.xlsx")
print("Saved ict_growth_surprise_scatter.xlsx")


# --- MANDATORY recalculation step ---
# openpyxl writes formulas (growth_surprise_pct, and every CORREL/COUNT
# in correlation_summary) WITHOUT a cached result, and each chart's OWN
# embedded copy of its plotted points (numCache) is built from whatever
# was in those cells at save time -- i.e. empty. This is why a chart
# can still look wrong even after Excel recalculates the plain cell
# values on open: the chart's own cached points need refreshing too.
#
# This step below (Claude's sandbox-only LibreOffice helper) does that
# automatically WHEN THIS SCRIPT RUNS INSIDE CLAUDE -- but that helper
# does not exist on a normal machine, so on your own computer this
# block will harmlessly do nothing (caught below), and you instead
# need the ONE manual step described in the printed message.
import subprocess
recalc_script = "/mnt/skills/public/xlsx/scripts/recalc.py"
try:
    result = subprocess.run(
        ["python3", recalc_script, "ict_growth_surprise_scatter.xlsx"],
        capture_output=True, text=True, timeout=60,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
    print("\n" + "=" * 70)
    print("IMPORTANT -- one manual step still needed (this recalculation")
    print("helper only exists inside Claude's own sandbox, not here):")
    print("  1. Open ict_growth_surprise_scatter.xlsx in Excel.")
    print("  2. Press Ctrl+S once (or just make any edit and undo it) --")
    print("     this forces Excel to recompute every formula AND rebuild")
    print("     every chart's own cached data points.")
    print("Without this step, growth_surprise_pct and every correlation")
    print("cell read back as blank, and every chart plots nothing/garbage")
    print("until you do it -- this is the most likely cause of 'weird'")
    print("chart output.")
    print("=" * 70)
