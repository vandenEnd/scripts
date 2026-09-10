import pandas as pd
import openpyxl
from openpyxl.chart import ScatterChart, Series, Reference
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.styles import Font

# --- Load source data ---
forecast = pd.read_excel("ec_forecast_vs_realized.xlsx")
ict = pd.read_excel("ai_data.xlsx", sheet_name="ict_inv")[["country", "year", "ict_share"]]
ict = ict.rename(columns={"year": "target_year"})
hs_export = pd.read_excel("ai_data.xlsx", sheet_name="hs_export")[["country", "year", "share"]]
hs_export = hs_export.rename(columns={"year": "target_year", "share": "hs_export_share"})

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
print(f"ict_share merge: {len(merged_ict)} rows (from {len(forecast)} forecast rows)")
print(f"hs_export share merge: {len(merged_hs)} rows (from {len(forecast)} forecast rows)")
print(f"stock/semis correlation merge: {len(merged_corr)} rows (from {len(forecast)} forecast rows)")

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


# --- Nine (data, chart) sheet pairs: 4 explanatory variables x
# (3 for the first three; a dedicated full-sample-plus-trendline chart
# is added separately below for time) ---
specs = [
    ("ict_share", merged_ict, "ict_share", "ICT investment share", "1F77B4",
     "vs. ICT investment share"),
    ("hs_export_share", merged_hs, "hs_export_share", "AI/ICT-related HS export share", "D62728",
     "vs. AI/ICT-related HS export share"),
    ("stock_corr", merged_corr, "stock_semis_corr_annual",
     "National vs. semiconductor index correlation (8Q rolling, annualized)", "2CA02C",
     "vs. national-semiconductor index correlation"),
    ("time_trend", forecast, "target_year", "Target year (time)", "9467BD",
     "vs. time (target year)"),
]
periods = [
    ("full", None, None, "full sample"),
    ("since2021", 2021, None, "since 2021"),
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
        want_trendline = (var_key == "time_trend" and period_key == "full")
        # Fixed axis scale ONLY for ict_share/full, to exactly match a
        # previously reviewed reference chart for that specific
        # combination -- every other chart stays auto-scaled (passing
        # these bounds into every call would badly distort
        # hs_export_share, stock_corr, and time_trend, which sit on
        # completely different scales).
        axis_kwargs = {}
        if var_key == "ict_share" and period_key == "full":
            axis_kwargs = dict(x_min=0, x_max=0.20, x_major=0.02,
                                y_min=-20, y_max=25, y_major=5)
        write_chart_sheet(wb, chart_sheet_name, ws_data, last_row, chart_title, header, color,
                           trendline=want_trendline, **axis_kwargs)
        sheet_registry[header][period_label] = (data_sheet_name, last_row)
        print(f"  {data_sheet_name}: {last_row - 1} rows -> {chart_sheet_name}")

# --- Correlation summary sheet: PIVOTED to match the summary table
# shown in chat -- one row per explanatory variable, one column per
# period, each cell a LIVE =CORREL(...) formula referencing that
# variable/period's own data sheet (growth_surprise_pct in col G,
# the explanatory variable in col H) -- recalculates automatically if
# the underlying data changes, rather than a Python-computed snapshot.
ws_corr = wb.create_sheet("correlation_summary", 0)
c = ws_corr.cell(row=1, column=1, value="explanatory_variable")
c.font = bold
for col_idx, period_label in enumerate(period_labels_ordered, start=2):
    c = ws_corr.cell(row=1, column=col_idx, value=period_label)
    c.font = bold

for row_idx, (var_label, period_dict) in enumerate(sheet_registry.items(), start=2):
    c = ws_corr.cell(row=row_idx, column=1, value=var_label)
    c.font = bold
    for col_idx, period_label in enumerate(period_labels_ordered, start=2):
        data_sheet, last_row = period_dict[period_label]
        g_range = f"'{data_sheet}'!$G$2:$G${last_row}"
        h_range = f"'{data_sheet}'!$H$2:$H${last_row}"
        cell = ws_corr.cell(row=row_idx, column=col_idx, value=f"=CORREL({g_range},{h_range})")
        cell.number_format = "0.000"

ws_corr.column_dimensions["A"].width = 45
for col_idx in range(2, 2 + len(period_labels_ordered)):
    ws_corr.column_dimensions[chr(64 + col_idx)].width = 16


def write_clustered_time_chart(wb, cluster_key, merged_df, cluster_col, cluster_header,
                                color_below, color_above):
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


write_clustered_time_chart(wb, "ict_share", merged_ict, "ict_share",
                            "ICT investment share", "AEC7E8", "1F77B4")
write_clustered_time_chart(wb, "hs_export", merged_hs, "hs_export_share",
                            "AI/ICT-related HS export share", "FFBB78", "D62728")
write_clustered_time_chart(wb, "stock_corr", merged_corr, "stock_semis_corr_annual",
                            "National vs. semiconductor index correlation", "98DF8A", "2CA02C")


# --- Alternative chart, ONE PER main explanatory variable, next to
# each variable's regular charts: growth surprise restricted to ONLY
# the SPRING-forecast-based surprise (realized[year] -
# Spring[year]_forecast -- the Autumn round for these years is
# deliberately excluded) for target years 2020, 2022, and 2025
# specifically, all countries pooled -- a fixed, hand-picked 3-year
# slice rather than a continuous period range like "since2021"/
# "till2019", so it needed its own dedicated filter rather than fitting
# the existing `periods` list.
SPECIAL_YEARS = [2020, 2022, 2025]
SPECIAL_YEARS_ROUND = "Spring"


def write_special_years_chart(wb, var_key, merged_df, col, header, color):
    # horizon==0 is essential here, not just vintage_round=="Spring":
    # without it, an EARLIER Spring round's horizon=1 forecast (e.g.
    # Spring2019 forecasting 2020 one year ahead) would ALSO match
    # "Spring round, target_year=2020", duplicating each country --
    # confirmed from a real run producing 34 rows instead of the
    # expected 30 (10 countries x 3 years). horizon==0 restricts this
    # to specifically the CURRENT-year forecast made in that same
    # year's own Spring round (i.e. vintage=="Spring{target_year}"),
    # matching "Spring forecast 2020" as one thing per country.
    df = merged_df[(merged_df["vintage_round"] == SPECIAL_YEARS_ROUND)
                    & (merged_df["horizon"] == 0)
                    & (merged_df["target_year"].isin(SPECIAL_YEARS))].copy()
    data_sheet_name = f"data_{var_key}_yrs"
    chart_sheet_name = f"chart_{var_key}_yrs"
    ws_data, last_row = write_data_sheet(wb, data_sheet_name, df, col, header)
    years_str = "/".join(str(y) for y in SPECIAL_YEARS)
    chart_title = (f"Growth surprise ({SPECIAL_YEARS_ROUND} forecast only) vs. {header} "
                    f"-- {years_str}")
    write_chart_sheet(wb, chart_sheet_name, ws_data, last_row, chart_title, header, color)
    print(f"  {data_sheet_name}: {last_row - 1} rows ({years_str}, "
          f"{SPECIAL_YEARS_ROUND} only) -> {chart_sheet_name}")


write_special_years_chart(wb, "ict_share", merged_ict, "ict_share",
                           "ICT investment share", "FF7F0E")
write_special_years_chart(wb, "hs_export_share", merged_hs, "hs_export_share",
                           "AI/ICT-related HS export share", "9467BD")
write_special_years_chart(wb, "stock_corr", merged_corr, "stock_semis_corr_annual",
                           "National vs. semiconductor index correlation", "8C564B")

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
