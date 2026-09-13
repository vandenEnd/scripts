import pandas as pd
import openpyxl
from openpyxl.chart import ScatterChart, BarChart, Series, Reference
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
    return summary_next_row


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
    return summary_next_row


last_corr_row = corr_header_row + len(sheet_registry)  # last variable's correlation row
summary_row = last_corr_row + 1  # one blank separator row before the group headers below

# --- "All years" (cluster sheets, full sample) vs "Shock years" (the
# 3 special-years sheets) group headers, written ONCE, directly above
# the first pair of blocks below.
group_header_row = summary_row + 1
ws_summary.cell(row=group_header_row, column=2, value="All years").font = Font(bold=True, size=12)
ws_summary.cell(row=group_header_row, column=6, value="Shock years").font = Font(bold=True, size=12)
summary_row = group_header_row + 2

# --- Each variable's "cluster" (full-sample, below/above median) block
# and "yrs" (2020/2022/2025 shock-years) block are written SIDE BY SIDE
# on the SAME rows (col_offset=0 for cluster on the left, col_offset=4
# for yrs on the right) -- not stacked vertically like before -- so the
# two views of the same variable are directly comparable at a glance.
# Only the LEFT (col_offset=0) call's returned row is used to advance
# summary_row for the next pair; the RIGHT call's return is identical
# (both blocks occupy the same rows) and is intentionally discarded.
summary_row = write_clustered_time_chart(wb, "ict_share", merged_ict, "ict_share",
                                          "ICT investment share", "AEC7E8", "1F77B4",
                                          ws_summary, summary_row, col_offset=0)
write_special_years_chart(wb, "ict_share", forecast, ict, "ict_share",
                           "ICT investment share", "FF7F0E",
                           ws_summary, summary_row - 4, substitute_year_map={2025: 2024},
                           col_offset=4)

summary_row = write_clustered_time_chart(wb, "hs_export", merged_hs, "hs_export_share",
                                          "AI/ICT-related HS export share", "FFBB78", "D62728",
                                          ws_summary, summary_row, col_offset=0)
write_special_years_chart(wb, "hs_export_share", forecast, hs_export, "hs_export_share",
                           "AI/ICT-related HS export share", "9467BD",
                           ws_summary, summary_row - 4, substitute_year_map={2025: 2024},
                           col_offset=4)

summary_row = write_clustered_time_chart(wb, "stock_corr", merged_corr, "stock_semis_corr_annual",
                                          "National vs. semiconductor index correlation",
                                          "98DF8A", "2CA02C", ws_summary, summary_row, col_offset=0)
write_special_years_chart(wb, "stock_corr", forecast, stock_corr_annual, "stock_semis_corr_annual",
                           "National vs. semiconductor index correlation", "8C564B",
                           ws_summary, summary_row - 4, col_offset=4)

summary_row = write_clustered_time_chart(wb, "ai_inv_share", merged_ai_inv, "ai_inv_share",
                                          "AI incoming investment share (per avg. quarterly GDP)",
                                          "F7B6D2", "E377C2", ws_summary, summary_row, col_offset=0)
# No substitute_year_map here -- unlike ict_share/hs_export_share,
# ai_inv already has genuine 2025 data of its own (confirmed directly
# against the source sheet), so no prior-year substitution is needed.
write_special_years_chart(wb, "ai_inv_share", forecast, ai_inv, "ai_inv_share",
                           "AI incoming investment share (per avg. quarterly GDP)", "17BECF",
                           ws_summary, summary_row - 4, col_offset=4)

ws_summary.column_dimensions["A"].width = 30
for col_letter in ("B", "C", "D", "F", "G", "H"):
    ws_summary.column_dimensions[col_letter].width = 14

# --- Supporting charts, visualizing the economic interpretation
# discussed in chat: (1) the "adverse shocks overestimated" resilience
# pattern (positive-surprise share is far higher in shock years than
# in the full sample), and (2) whether being ABOVE each explanatory
# variable's own median is associated with a higher positive-surprise
# share, both in normal years and specifically in shock years.
#
# A small staging table is written first (rows 30-34), with LIVE
# same-sheet cell references (not copied numbers) into the existing
# "share positive surprises" rows (13/17/21/25 = all-countries "full
# sample"/"below"/"above" for the cluster sheets; the SAME rows'
# columns F/G/H = the equivalent "yrs" (shock-years) sheet) -- this
# gives each chart a clean, contiguous range to plot from, without
# duplicating any underlying data.
chart_data_header_row = 30
chart_data_start_row = 31

ws_summary.cell(row=chart_data_header_row, column=1,
                 value="Chart data (live references to the rows above)").font = Font(
    bold=True, italic=True)

chart_headers = ["Variable", "All years - full", "All years - below", "All years - above",
                 "Shock years - full", "Shock years - below", "Shock years - above"]
for col_idx, h in enumerate(chart_headers, start=1):
    ws_summary.cell(row=chart_data_start_row, column=col_idx, value=h).font = bold

# (variable label, all-countries "share positive surprises" row number)
variable_rows = [
    ("ICT investment share", 13),
    ("AI/ICT-related HS export share", 17),
    ("National vs. semiconductor index correlation", 21),
    ("AI incoming investment share", 25),
]
for offset, (var_label, src_row) in enumerate(variable_rows):
    row_idx = chart_data_start_row + 1 + offset
    ws_summary.cell(row=row_idx, column=1, value=var_label)
    ws_summary.cell(row=row_idx, column=2, value=f"=B{src_row}")
    ws_summary.cell(row=row_idx, column=3, value=f"=C{src_row}")
    ws_summary.cell(row=row_idx, column=4, value=f"=D{src_row}")
    ws_summary.cell(row=row_idx, column=5, value=f"=F{src_row}")
    ws_summary.cell(row=row_idx, column=6, value=f"=G{src_row}")
    ws_summary.cell(row=row_idx, column=7, value=f"=H{src_row}")
    for col_idx in range(2, 8):
        ws_summary.cell(row=row_idx, column=col_idx).number_format = "0%"

chart_data_first_row = chart_data_start_row + 1
chart_data_last_row = chart_data_start_row + len(variable_rows)


def build_bar_chart(title, y_axis_title, series_specs):
    """series_specs: list of (column_index, series_name, color_hex)."""
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.style = 10
    chart.y_axis.title = y_axis_title
    chart.x_axis.title = "Explanatory variable"
    chart.height = 9
    chart.width = 18
    cats = Reference(ws_summary, min_col=1, min_row=chart_data_first_row,
                      max_row=chart_data_last_row)
    for col_idx, series_name, color_hex in series_specs:
        data_ref = Reference(ws_summary, min_col=col_idx, min_row=chart_data_start_row,
                              max_row=chart_data_last_row)
        chart.add_data(data_ref, titles_from_data=True)
        series = chart.series[-1]
        series.graphicalProperties.solidFill = color_hex
    chart.set_categories(cats)
    return chart


# Chart 1: "adverse shocks overestimated" -- full-sample positive-
# surprise share, all years vs. shock years, per variable.
chart1 = build_bar_chart(
    "Positive surprises: all years vs. shock years (full sample)",
    "Share of positive surprises",
    [(2, "All years", "1F77B4"), (5, "Shock years", "D62728")],
)
ws_summary.add_chart(chart1, "J1")

# Chart 2: below vs. above median, ALL years -- does being above the
# median predict a higher positive-surprise share, in normal times?
chart2 = build_bar_chart(
    "Below vs. above median (all years)",
    "Share of positive surprises",
    [(3, "Below median", "AEC7E8"), (4, "Above median", "1F77B4")],
)
ws_summary.add_chart(chart2, "J20")

# Chart 3: below vs. above median, SHOCK years specifically -- does the
# above-median advantage strengthen during shocks (as argued for
# Export and stock_corr in the chat discussion)?
chart3 = build_bar_chart(
    "Below vs. above median (shock years only)",
    "Share of positive surprises",
    [(6, "Below median", "FFBB78"), (7, "Above median", "D62728")],
)
ws_summary.add_chart(chart3, "J39")

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
