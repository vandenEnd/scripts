import pandas as pd
import numpy as np
import openpyxl
from openpyxl.chart import ScatterChart, BarChart, LineChart, Series, Reference
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.styles import Font

# --- Load source data ---
forecast = pd.read_excel("ec_forecast_vs_realized.xlsx")
ict = pd.read_excel("ai_data.xlsx", sheet_name="ict_inv")[["country", "year", "ict_share"]]
ict = ict.rename(columns={"year": "target_year"})
hs_export = pd.read_excel("ai_data.xlsx", sheet_name="hs_export")[["country", "year", "share"]]
hs_export = hs_export.rename(columns={"year": "target_year", "share": "hs_export_share"})

# hs_export's source data ends at 2024 (confirmed: no 2025 rows at all,
# for any country) -- one year short of ict_inv/forecast's own coverage.
# Per explicit instruction, each country's missing 2025 row is filled
# with that SAME country's 2024 hs_export_share value, so the "_full"/
# "_since2020"/"_till2019" period sheets (which all derive from this
# same hs_export DataFrame, merged once below) include 2025 instead of
# silently missing it entirely. This is a genuine, deliberate
# substitution -- flagged clearly here and in the printed diagnostic
# below, not hidden.
hs_export_years = set(hs_export["target_year"].unique())
if 2025 not in hs_export_years and 2024 in hs_export_years:
    hs_2024 = hs_export[hs_export["target_year"] == 2024].copy()
    hs_2025_substitute = hs_2024.copy()
    hs_2025_substitute["target_year"] = 2025
    hs_export = pd.concat([hs_export, hs_2025_substitute], ignore_index=True)
    print(f"  [diagnostic] hs_export: 2025 missing from source data -- filled "
          f"{len(hs_2025_substitute)} countries' 2025 hs_export_share with their "
          f"own 2024 value.")

ai_inv = pd.read_excel("ai_data.xlsx", sheet_name="ai_inv")[["country", "year", "share"]]
ai_inv = ai_inv.rename(columns={"year": "target_year", "share": "ai_inv_share"})

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

merged_ict = forecast.merge(ict, on=["country", "target_year"], how="inner")
merged_hs = forecast.merge(hs_export, on=["country", "target_year"], how="inner")
merged_corr = forecast.merge(stock_corr_annual, on=["country", "target_year"], how="inner")
merged_ai_inv = forecast.merge(ai_inv, on=["country", "target_year"], how="inner")
print(f"ict_share merge: {len(merged_ict)} rows (from {len(forecast)} forecast rows)")
print(f"hs_export share merge: {len(merged_hs)} rows (from {len(forecast)} forecast rows)")
print(f"stock/semis correlation merge: {len(merged_corr)} rows (from {len(forecast)} forecast rows)")
print(f"ai_inv share merge: {len(merged_ai_inv)} rows (from {len(forecast)} forecast rows)")

wb = openpyxl.Workbook()
wb.remove(wb.active)  # remove the default empty sheet; we add our own below

bold = Font(bold=True)


def write_data_sheet(wb, sheet_name, df, explanatory_col, explanatory_header):
    """
    Writes one data sheet: country, vintage, vintage_round, target_year,
    forecast_growth_annual_pct, realized_growth_annual_pct,
    growth_surprise_pct (LIVE FORMULA = realized - forecast),
    <explanatory_header>. Returns (worksheet, last_data_row).
    """
    ws = wb.create_sheet(sheet_name)
    headers = ["country", "vintage", "vintage_round", "target_year",
               "forecast_growth_annual_pct", "realized_growth_annual_pct",
               "growth_surprise_pct", explanatory_header]
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

    for col_idx in range(1, 9):
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
    variables (ict_share ~0-0.2, hs_export_share ~0.0004, stock_corr
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
    ("ict_share", merged_ict, "ict_share", "ICT investment share", "1F77B4",
     "vs. ICT investment share"),
    ("hs_export_share", merged_hs, "hs_export_share", "AI/ICT-related HS export share", "D62728",
     "vs. AI/ICT-related HS export share"),
    ("stock_corr", merged_corr, "stock_semis_corr_annual",
     "National vs. semiconductor index correlation (8Q rolling, annualized)", "2CA02C",
     "vs. national-semiconductor index correlation"),
    ("ai_inv_share", merged_ai_inv, "ai_inv_share",
     "AI incoming investment share (per avg. quarterly GDP)", "E377C2",
     "vs. AI incoming investment share"),
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
        ws_data, last_row = write_data_sheet(wb, data_sheet_name, df_period, col, header)
        chart_title = f"Growth surprise {title_suffix} -- {period_label}"
        # Fixed axis scale ONLY for ict_share/full, to exactly match a
        # previously reviewed reference chart for that specific
        # combination -- every other chart stays auto-scaled (passing
        # these bounds into every call would badly distort
        # hs_export_share and stock_corr, which sit on completely
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
    # from a live run (ict_share and hs_export's "below" columns both
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
               "growth_surprise_pct", cluster_header, "cluster"]
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

    for col_idx in range(1, 10):
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
                               ws_summary, summary_next_row, substitute_year_map=None,
                               col_offset=0):
    """
    substitute_year_map: e.g. {2025: 2024} -- for a target_year with no
    genuine explanatory-variable data of its own (2025's ict_share/
    hs_export_share is not yet published by Eurostat), use that OTHER
    year's value instead, FOR CLASSIFICATION PURPOSES ONLY. This is
    made fully transparent: an extra "explanatory_year_used" column
    shows exactly which year's value was actually used for each row,
    so a substituted row is never silently indistinguishable from a
    genuine same-year one.
    """
    substitute_year_map = substitute_year_map or {}

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

    # Build a (country, target_year) -> explanatory value lookup that
    # substitutes an EARLIER year's value wherever substitute_year_map
    # says to (e.g. 2025 -> 2024), instead of the plain inner-join this
    # project used before -- an inner join would have silently DROPPED
    # every 2025 row entirely, since Eurostat has no 2025 ict_share/
    # hs_export_share data yet (see fetch_ict_investment_share()'s own
    # coverage diagnostics in collect_ai_data.py).
    lookup_years_needed = set(SPECIAL_YEARS) | set(substitute_year_map.values())
    explanatory_lookup = raw_explanatory[
        raw_explanatory["target_year"].isin(lookup_years_needed)
    ].set_index(["country", "target_year"])[col]

    explanatory_values, explanatory_years_used = [], []
    keep_mask = []
    for _, row in df.iterrows():
        ty = row["target_year"]
        lookup_year = substitute_year_map.get(ty, ty)
        key = (row["country"], lookup_year)
        if key in explanatory_lookup.index:
            explanatory_values.append(explanatory_lookup.loc[key])
            explanatory_years_used.append(lookup_year)
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
    wb, "ict_share", merged_ict, "ict_share", "ICT investment share", "AEC7E8", "1F77B4",
    ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_ict = write_special_years_chart(
    wb, "ict_share", forecast, ict, "ict_share", "ICT investment share", "FF7F0E",
    ws_summary, header_row - 1, substitute_year_map={2025: 2024}, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
block_header_rows["ICT investment share"] = header_row
all_props["ICT investment share"] = (cluster_props, yrs_props)

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "hs_export", merged_hs, "hs_export_share", "AI/ICT-related HS export share",
    "FFBB78", "D62728", ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_hs = write_special_years_chart(
    wb, "hs_export_share", forecast, hs_export, "hs_export_share",
    "AI/ICT-related HS export share", "9467BD",
    ws_summary, header_row - 1, substitute_year_map={2025: 2024}, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["AI/ICT-related HS export share"] = (cluster_props, yrs_props)
block_header_rows["AI/ICT-related HS export share"] = header_row

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "stock_corr", merged_corr, "stock_semis_corr_annual",
    "National vs. semiconductor index correlation", "98DF8A", "2CA02C",
    ws_summary, summary_row, col_offset=0)
_, yrs_props, yrs_df_corr = write_special_years_chart(
    wb, "stock_corr", forecast, stock_corr_annual, "stock_semis_corr_annual",
    "National vs. semiconductor index correlation", "8C564B",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["National vs. semiconductor index correlation"] = (cluster_props, yrs_props)
block_header_rows["National vs. semiconductor index correlation"] = header_row

header_row = summary_row + 1
summary_row, cluster_props = write_clustered_time_chart(
    wb, "ai_inv_share", merged_ai_inv, "ai_inv_share",
    "AI incoming investment share (per avg. quarterly GDP)", "F7B6D2", "E377C2",
    ws_summary, summary_row, col_offset=0)
# No substitute_year_map here -- unlike ict_share/hs_export_share,
# ai_inv already has genuine 2025 data of its own (confirmed directly
# against the source sheet), so no prior-year substitution is needed.
_, yrs_props, _ = write_special_years_chart(
    wb, "ai_inv_share", forecast, ai_inv, "ai_inv_share",
    "AI incoming investment share (per avg. quarterly GDP)", "17BECF",
    ws_summary, header_row - 1, col_offset=4)
add_significance_stars(ws_summary, header_row, cluster_props, yrs_props)
all_props["AI incoming investment share"] = (cluster_props, yrs_props)
block_header_rows["AI incoming investment share"] = header_row

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
    ("ICT investment share", "data_cluster_ict_share", "data_ict_share_yrs"),
    ("AI/ICT-related HS export share", "data_cluster_hs_export", "data_hs_export_share_yrs"),
    ("National vs. semiconductor index correlation", "data_cluster_stock_corr",
     "data_stock_corr_yrs"),
    ("AI incoming investment share", "data_cluster_ai_inv_share", "data_ai_inv_share_yrs"),
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
# (merged_ict/merged_hs/merged_corr, unfiltered by period). Computed
# directly from the in-memory DataFrames (not by reading back the
# Excel formulas), since growth_surprise_pct is itself a live Excel
# formula with no cached value until the file is opened/recalculated --
# computing it here in Python from realized/forecast avoids that
# entirely and guarantees this regression uses the exact same numbers
# the "_full" sheets display.
#
# HONESTY NOTE: this is a two-way fixed-effects (country + year) OLS
# regression via the standard dummy-variable (LSDV) approach, with
# PLAIN (homoskedastic) OLS standard errors -- NOT the Driscoll-Kraay
# robust standard errors this project's underlying panel-LP modeling
# scripts use elsewhere. No external regression package (statsmodels/
# linearmodels) was available to install in this environment (no
# network access), so this is a from-scratch numpy implementation,
# validated separately against synthetic data with a known true
# coefficient (recovered to within 0.001 of the true value) before
# being applied here.
from scipy import stats as _stats


def panel_ols_two_way_fe(df, y_col, x_col, entity_col="country", time_col="target_year",
                          extra_cols=None):
    """
    extra_cols: optional list of ADDITIONAL regressor column names
    already present in df (e.g. an interaction term X*shock_dummy) --
    included alongside x_col, country FE, and time FE. Coefficients
    for every column in extra_cols are returned in "extra" (a dict
    keyed by column name), in the same shape as the main beta/se/
    t_stat/p_value.
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
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    se_all = np.sqrt(np.diag(sigma2 * XtX_inv))

    def _coef_stats(col_name):
        idx = list(X.columns).index(col_name)
        b, s = beta_hat[idx], se_all[idx]
        t = b / s
        p = 2 * (1 - _stats.t.cdf(abs(t), dof))
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
    }


regression_specs = [
    ("ICT investment share (data_ict_share_full)", merged_ict, "ict_share"),
    ("AI/ICT-related HS export share (data_hs_export_share_full)", merged_hs, "hs_export_share"),
    ("National vs. semiconductor index correlation (data_stock_corr_full)",
     merged_corr, "stock_semis_corr_annual"),
    ("AI incoming investment share (data_ai_inv_share_full)",
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
    cols = ["country", "target_year", "growth_surprise", x_col] + extra_cols
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
for col_letter in "BCDEFG":
    ws_data_regr.column_dimensions[col_letter].width = 16

reg_start_row = 30
ws_summary.cell(row=reg_start_row, column=1,
                 value="Panel regressions: growth_surprise = const + country FE + "
                       "time FE + beta*X").font = Font(bold=True, size=12)
ws_summary.cell(
    row=reg_start_row + 1, column=1,
    value="(two-way fixed effects, OLS via dummy variables; plain/homoskedastic "
          "standard errors -- NOT Driscoll-Kraay robust)"
).font = Font(italic=True, size=9)

reg_header_row = reg_start_row + 3
reg_headers = ["Explanatory variable", "beta", "SE", "t-stat", "p-value",
               "R-squared", "N (obs)", "N (countries)", "N (years)"]
for col_idx, h in enumerate(reg_headers, start=1):
    ws_summary.cell(row=reg_header_row, column=col_idx, value=h).font = bold

for offset, (label, df_src, x_col) in enumerate(regression_specs):
    df_src = df_src.copy()
    df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                  - df_src["forecast_growth_annual_pct"])
    result = panel_ols_two_way_fe(df_src, "growth_surprise", x_col)
    write_regression_data_block(1, label, df_src, x_col)

    row_idx = reg_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["t_stat"], 3))
    ws_summary.cell(row=row_idx, column=5, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=7, value=result["n_obs"])
    ws_summary.cell(row=row_idx, column=8, value=result["n_entities"])
    ws_summary.cell(row=row_idx, column=9, value=result["n_periods"])
    print(f"  Panel regression ({label}): beta={result['beta']:.4f}, "
          f"p={result['p_value']:.4f}, N={result['n_obs']}")

# Widths for B-H (shared with the "full sample"/"below"/"above" summary
# blocks above, incl. their significance-star suffixes like "full
# sample ***") kept wide enough for BOTH uses -- this section used to
# set B-H narrower again (as low as 10), overriding the wider setting
# applied earlier and effectively hiding the star suffixes in Excel
# (text overflowing into an adjacent non-empty cell gets visually
# truncated, not shown). Column I (only used by the regression table,
# not the summary blocks) can stay narrower.
for col_letter, width in zip("BCDEFGHI", [18, 18, 18, 18, 18, 18, 12, 12]):
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
    ("ICT investment share (data_ict_share_full)", merged_ict, "ict_share"),
    ("AI/ICT-related HS export share (data_hs_export_share_full)", merged_hs, "hs_export_share"),
    ("National vs. semiconductor index correlation (data_stock_corr_full)",
     merged_corr, "stock_semis_corr_annual"),
    ("AI incoming investment share (data_ai_inv_share_full)",
     merged_ai_inv, "ai_inv_share"),
]

reg2_start_row = reg_header_row + len(regression_specs) + 3
ws_summary.cell(
    row=reg2_start_row, column=1,
    value="Panel regressions WITH SHOCK-YEAR INTERACTION: growth_surprise = const + "
          "country FE + time FE + beta*X + gamma*(X*shock_year_dummy)"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=reg2_start_row + 1, column=1,
    value="(full sample; shock_year_dummy=1 for target_year in 2020/2022/2025 -- gamma "
          "estimates X's ADDITIONAL effect specifically during shock years, on top of "
          "beta's baseline effect)"
).font = Font(italic=True, size=9)

reg2_header_row = reg2_start_row + 3
reg2_headers = ["Explanatory variable", "beta (X)", "SE(beta)", "p(beta)",
                "gamma (X*shock)", "SE(gamma)", "p(gamma)", "R-squared", "N (obs)"]
for col_idx, h in enumerate(reg2_headers, start=1):
    ws_summary.cell(row=reg2_header_row, column=col_idx, value=h).font = bold

for offset, (label, df_src, x_col) in enumerate(interaction_specs):
    df_src = df_src.copy()
    df_src["growth_surprise"] = (df_src["realized_growth_annual_pct"]
                                  - df_src["forecast_growth_annual_pct"])
    df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
    df_src["interaction"] = df_src[x_col] * df_src["shock_year_dummy"]
    result = panel_ols_two_way_fe(df_src, "growth_surprise", x_col, extra_cols=["interaction"])
    gamma = result["extra"]["interaction"]
    write_regression_data_block(2, label, df_src, x_col,
                                 extra_cols=["shock_year_dummy", "interaction"])

    row_idx = reg2_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
    ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
    ws_summary.cell(row=row_idx, column=8, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=9, value=result["n_obs"])
    print(f"  Interaction regression ({label}): beta={result['beta']:.4f} "
          f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} (p={gamma['p_value']:.4f}), "
          f"N={result['n_obs']}")

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
    ("ICT investment share, ABOVE MEDIAN dummy (data_ict_share_full)", merged_ict, "ict_share"),
    ("AI/ICT-related HS export share, ABOVE MEDIAN dummy (data_hs_export_share_full)",
     merged_hs, "hs_export_share"),
    ("National vs. semiconductor index correlation, ABOVE MEDIAN dummy "
     "(data_stock_corr_full)", merged_corr, "stock_semis_corr_annual"),
    ("AI incoming investment share, ABOVE MEDIAN dummy (data_ai_inv_share_full)",
     merged_ai_inv, "ai_inv_share"),
]

reg3_start_row = reg2_header_row + len(interaction_specs) + 3
ws_summary.cell(
    row=reg3_start_row, column=1,
    value="Panel regressions WITH ABOVE-MEDIAN DUMMY AND SHOCK-YEAR INTERACTION: "
          "growth_surprise = const + country FE + time FE + beta*AboveMedian + "
          "gamma*(AboveMedian*shock_year_dummy)"
).font = Font(bold=True, size=12)
ws_summary.cell(
    row=reg3_start_row + 1, column=1,
    value="(full sample; AboveMedian=1 if X > that variable's own overall median, else "
          "0 -- the SAME above/below classification used throughout the blocks above, "
          "now tested via regression; gamma estimates the ADDITIONAL effect of being "
          "above median specifically during shock years, on top of beta's baseline "
          "above-median effect)"
).font = Font(italic=True, size=9)

reg3_header_row = reg3_start_row + 3
reg3_headers = ["Explanatory variable", "beta (AboveMedian)", "SE(beta)", "p(beta)",
                "gamma (AboveMedian*shock)", "SE(gamma)", "p(gamma)", "R-squared", "N (obs)"]
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
    df_src["above_median"] = np.where(df_src[x_col].isna(), np.nan,
                                       (df_src[x_col] > median_val).astype(float))
    df_src["shock_year_dummy"] = df_src["target_year"].isin(SHOCK_YEARS_SET).astype(float)
    df_src["above_x_shock"] = df_src["above_median"] * df_src["shock_year_dummy"]
    result = panel_ols_two_way_fe(df_src, "growth_surprise", "above_median",
                                   extra_cols=["above_x_shock"])
    gamma = result["extra"]["above_x_shock"]
    write_regression_data_block(3, label, df_src, "above_median",
                                 extra_cols=[x_col, "shock_year_dummy", "above_x_shock"])

    row_idx = reg3_header_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=label)
    ws_summary.cell(row=row_idx, column=2, value=round(result["beta"], 4))
    ws_summary.cell(row=row_idx, column=3, value=round(result["se"], 4))
    ws_summary.cell(row=row_idx, column=4, value=round(result["p_value"], 4))
    ws_summary.cell(row=row_idx, column=5, value=round(gamma["beta"], 4))
    ws_summary.cell(row=row_idx, column=6, value=round(gamma["se"], 4))
    ws_summary.cell(row=row_idx, column=7, value=round(gamma["p_value"], 4))
    ws_summary.cell(row=row_idx, column=8, value=round(result["r_squared"], 4))
    ws_summary.cell(row=row_idx, column=9, value=result["n_obs"])
    print(f"  Above-median interaction regression ({label}): beta={result['beta']:.4f} "
          f"(p={result['p_value']:.4f}), gamma={gamma['beta']:.4f} (p={gamma['p_value']:.4f}), "
          f"N={result['n_obs']}, median={median_val:.4f}")

# --- Bar charts + interaction (line) plots, reproducing the HORIZONTAL
# staging layout the user built by hand in Excel on top of an earlier
# version of this script's output (confirmed by inspecting that
# file's actual chart XML: categories/series run ACROSS a row, e.g.
# summary!$E$31:$F$31 for categories, not down a column like this
# script originally used) -- this version matches that layout exactly,
# plus adds the AI incoming investment share block (not present in
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


ict_cluster_props, ict_yrs_props = all_props["ICT investment share"]

variable_chart_specs = [
    ("ICT investment share", "ICT-AI  invest"),
    ("AI/ICT-related HS export share", "ICT-AI  export"),
    ("National vs. semiconductor index correlation", "SOX correl"),
    ("AI incoming investment share", "AI invest"),
]

chart_data_row = reg3_header_row + len(above_median_specs) + 3
ws_summary.cell(row=chart_data_row, column=1,
                 value="Chart data (plain computed values, not formulas -- "
                       "see the comment above)").font = Font(bold=True, italic=True)

# --- Chart 1 data: "full sample" comparison, all years vs. shock
# years -- columns A-C, ONE row (full-sample share is identical across
# variables, so ICT's own value represents it).
full_row = chart_data_row + 2
ws_summary.cell(row=full_row, column=1, value="Full sample")
ws_summary.cell(row=full_row, column=2, value="All years - full")
ws_summary.cell(row=full_row, column=3, value="Shock years - full")
full_data_row = full_row + 1
ws_summary.cell(row=full_data_row, column=1, value="Share positive surprises")
# LIVE FORMULAS here (rows 61-67 specifically, per explicit request) --
# referencing the ICT block's own "full sample" cells (B/F at its
# header_row+1). Everywhere ELSE in this staging table stays PLAIN
# VALUES (see _safe_share() below), since a formula referencing
# ANOTHER formula cell (B13 is itself "=COUNTIF(...)/COUNT(...)") has
# no cached value in the raw, un-recalculated file every user actually
# gets -- confirmed earlier in this project to leave charts blank
# until Excel fully recalculates. Rows 61-67 accept that tradeoff
# deliberately, for verification purposes; the rest of the table keeps
# the safer plain-value approach so the OTHER charts stay reliable.
ict_header_row = block_header_rows["ICT investment share"]
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
for var_idx, ((var_label, short_label), block_start_col) in enumerate(
        zip(variable_chart_specs, lowhigh_block_starts)):
    cp, yp = all_props[var_label]
    label_col = block_start_col
    all_col = block_start_col + 1
    shock_col = block_start_col + 2
    use_formulas = (var_idx == 0)  # ICT only -- see the comment above

    ws_summary.cell(row=header_row_for_blocks, column=all_col, value="All years").font = bold
    ws_summary.cell(row=header_row_for_blocks, column=shock_col, value="Shock years").font = bold

    low_row = header_row_for_blocks + 1
    ws_summary.cell(row=low_row, column=label_col, value=f"Low {short_label}")
    high_row = header_row_for_blocks + 2
    ws_summary.cell(row=high_row, column=label_col, value=f"High {short_label}")

    if use_formulas:
        var_data_row = block_header_rows[var_label] + 1
        ws_summary.cell(row=low_row, column=all_col, value=f"=C{var_data_row}")
        ws_summary.cell(row=low_row, column=shock_col, value=f"=G{var_data_row}")
        ws_summary.cell(row=high_row, column=all_col, value=f"=D{var_data_row}")
        ws_summary.cell(row=high_row, column=shock_col, value=f"=H{var_data_row}")
    else:
        ws_summary.cell(row=low_row, column=all_col,
                         value=_safe_share(cp["x_below"], cp["n_below"]))
        ws_summary.cell(row=low_row, column=shock_col,
                         value=_safe_share(yp["x_below"], yp["n_below"]))
        ws_summary.cell(row=high_row, column=all_col,
                         value=_safe_share(cp["x_above"], cp["n_above"]))
        ws_summary.cell(row=high_row, column=shock_col,
                         value=_safe_share(yp["x_above"], yp["n_above"]))

    for r in (low_row, high_row):
        for c in (all_col, shock_col):
            ws_summary.cell(row=r, column=c).number_format = "0%"

    lowhigh_row_map[var_label] = (header_row_for_blocks, low_row, high_row,
                                   label_col, all_col, shock_col)


def build_bar_chart_horizontal(title, cats_ref, series_specs, y_title, x_title):
    """series_specs: list of (values_ref, series_name, color_hex)."""
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.style = 10
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title
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
        s.tx = openpyxl.chart.series.SeriesLabel(v=name)
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
    chart.title = title
    chart.style = 12
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title
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
        s.tx = openpyxl.chart.series.SeriesLabel(v=name)
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
for i, (var_label, short_label) in enumerate(variable_chart_specs):
    header_row_b, low_row, high_row, label_col, all_col, shock_col = lowhigh_row_map[var_label]
    cats = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                      min_row=header_row_b, max_row=header_row_b)
    low_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                          min_row=low_row, max_row=low_row)
    high_vals = Reference(ws_summary, min_col=all_col, max_col=shock_col,
                           min_row=high_row, max_row=high_row)
    chart = build_bar_chart_horizontal(
        f"Share positive surprises: {var_label}", cats,
        [(low_vals, f"Low {short_label}", "BDD7EE"), (high_vals, f"High {short_label}", "4472C4")],
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
for i, (var_label, short_label) in enumerate(variable_chart_specs):
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

    chart = build_line_chart_horizontal(
        f"Interaction plot: {var_label}", interaction_cats,
        [(low_vals, f"Low {short_label}", "BDD7EE"), (high_vals, f"High {short_label}", "4472C4")],
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
