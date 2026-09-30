#!/usr/bin/env python3
"""Fetch monthly HS exports from Eurostat Comext DS-059341.

The script requests exports to the world, in euros, from 2020-01 onward for
eight six-digit HS product categories plus total merchandise exports.  It first
checks whether the Eurostat euro-area aggregate reporter contains observations.
If it does not, it uses the ten-country panel from the reference workbook:
AT, BE, DE, ES, FI, FR, IE, IT, NL and PT.

Output: one XLSX workbook with the ``hs_export_monthly`` sheet. Chart-support
formulas are kept in visible rows on that same sheet, in columns L and M, so
every chart series refers directly to those two percentage columns. Each run
performs fresh Eurostat requests and overwrites the requested output workbook.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DATASET = "DS-059341"
API_URL = (
    "https://ec.europa.eu/eurostat/api/comext/dissemination/"
    f"statistics/1.0/data/{DATASET}"
)
HS_CODES = [
    "847150",
    "847180",
    "847330",
    "848610",
    "848620",
    "848630",
    "848640",
    "848690",
]
PRODUCTS = HS_CODES + ["TOTAL"]
FALLBACK_COUNTRIES = ["AT", "BE", "DE", "ES", "FI", "FR", "IE", "IT", "NL", "PT"]
EURO_AREA_REPORTER = "EA"
NETHERLANDS_REPORTER = "NL"


def _request_json(params: dict[str, str], attempts: int = 5) -> dict[str, Any]:
    """Download one JSON-stat response with retries and exponential backoff."""
    url = f"{API_URL}?{urlencode(params)}"
    headers = {
        "Accept": "application/json",
        "Cache-Control": "no-cache, no-store, max-age=0",
        "Pragma": "no-cache",
        "User-Agent": "monthly-hs-export-fetcher/1.0 (Eurostat public API)",
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=180) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(30.0, (2 ** (attempt - 1)) + random.random())
            print(f"  transient request error ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"Eurostat request failed after {attempts} attempts: {url}") from last_error


def _flatten_jsonstat(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Eurostat JSON-stat 2 dataset into non-null long-form rows."""
    required = {"dimension", "id", "size", "value"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Eurostat response lacks keys {sorted(missing)}")

    dimension_ids = payload["id"]
    category_lists: list[list[str]] = []
    for dimension_id in dimension_ids:
        category = payload["dimension"][dimension_id]["category"]
        ordered = sorted(category["index"].items(), key=lambda item: item[1])
        category_lists.append([code for code, _ in ordered])

    values = payload["value"]
    if isinstance(values, list):
        values = {str(i): value for i, value in enumerate(values) if value is not None}

    rows: list[dict[str, Any]] = []
    for flat_index, combination in enumerate(itertools.product(*category_lists)):
        value = values.get(str(flat_index))
        if value is None:
            continue
        row = dict(zip(dimension_ids, combination))
        row["value"] = value
        rows.append(row)
    return rows


def _fetch_product(product: str, start: str, end: str) -> list[dict[str, Any]]:
    # Reporter is deliberately omitted: one compact response retrieves all
    # reporters and lets us test the EA aggregate without extra requests.
    params = {
        "format": "JSON",
        "freq": "M",
        "partner": "WORLD",
        "product": product,
        "flow": "2",  # exports
        "indicators": "VALUE_EUR",
        "sinceTimePeriod": start,
        "untilTimePeriod": end,
    }
    rows = _flatten_jsonstat(_request_json(params))
    print(f"  {product}: {len(rows):,} non-null observations")
    return rows


def fetch_monthly_exports(start: str, end: str) -> tuple[list[dict[str, Any]], str]:
    """Fetch all products and return wide rows plus the reporter strategy."""
    product_rows: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_product, product, start, end): product
            for product in PRODUCTS
        }
        for future in as_completed(futures):
            product = futures[future]
            product_rows[product] = future.result()

    # Direct EA use requires at least one non-null observation for every
    # requested series. Otherwise use the ten-country panel requested by user.
    ea_has_every_series = all(
        any(row.get("reporter") == EURO_AREA_REPORTER for row in product_rows[p])
        for p in PRODUCTS
    )
    if ea_has_every_series:
        reporters = [NETHERLANDS_REPORTER, EURO_AREA_REPORTER]
        strategy = "NL and Eurostat euro-area aggregate (EA)"
    else:
        reporters = FALLBACK_COUNTRIES
        strategy = "10-country fallback panel (EA aggregate has no observations)"

    wide: dict[tuple[str, str], dict[str, Any]] = {}
    for product, rows in product_rows.items():
        output_column = "total_export" if product == "TOTAL" else f"HS_{product}"
        for row in rows:
            reporter = str(row.get("reporter", ""))
            month = str(row.get("time", ""))
            if reporter not in reporters or not month:
                continue
            key = (reporter, month)
            record = wide.setdefault(key, {"country": reporter, "month": month})
            record[output_column] = row["value"]

    columns = ["country", "month"] + [f"HS_{code}" for code in HS_CODES] + ["total_export"]
    output_rows = []
    for key in sorted(wide):
        record = wide[key]
        if any(record.get(column) is not None for column in columns[2:]):
            output_rows.append({column: record.get(column) for column in columns})

    if not output_rows:
        raise RuntimeError("No requested observations were returned by Eurostat")
    return output_rows, strategy


def _month_sequence(start: str, end: str) -> list[str]:
    """Return every YYYY-MM month from start through end, inclusive."""
    start_date = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")
    months = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def _style_chart(chart: LineChart, title: str) -> None:
    chart.title = title
    chart.y_axis.title = "Percentage of total goods exports"
    chart.x_axis.title = None
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.y_axis.numFmt = "0.0%"
    chart.x_axis.tickLblSkip = 12
    chart.x_axis.tickMarkSkip = 12
    chart.x_axis.tickLblPos = "nextTo"
    chart.y_axis.tickLblPos = "nextTo"
    chart.legend.position = "b"
    chart.legend.overlay = True
    chart.display_blanks = "gap"
    chart.height = 7.5
    chart.width = 15.0
    chart.style = None
    chart.visible_cells_only = False


def _add_chart_series(
    chart: LineChart,
    worksheet,
    column: int,
    min_row: int,
    max_row: int,
    title: str,
    color: str,
) -> None:
    values = Reference(
        worksheet, min_col=column, min_row=min_row, max_row=max_row
    )
    series = Series(values, title=title)
    series.graphicalProperties.line.solidFill = color
    series.graphicalProperties.line.width = 24000
    series.marker.symbol = "none"
    chart.series.append(series)


def _add_chart_source_rows(
    worksheet,
    months: list[str],
    source_rows: dict[str, dict[str, int]],
) -> dict[str, dict[str, tuple[int, int]]]:
    """Add hidden raw/MA chart sources in main-sheet columns L and M."""
    entities = ["NL", "EA", "EA excl NL"]
    ranges: dict[str, dict[str, tuple[int, int]]] = {"raw": {}, "ma6": {}}

    for series_type in ["raw", "ma6"]:
        for entity in entities:
            start_row = worksheet.max_row + 1
            for month_index, month in enumerate(months):
                output_row = worksheet.max_row + 1
                worksheet.cell(output_row, 1, f"{entity} chart {series_type}")
                worksheet.cell(output_row, 2, month)
                if series_type == "raw":
                    source_row = source_rows[entity].get(month)
                    for column, letter in [(12, "L"), (13, "M")]:
                        formula = f"={letter}{source_row}" if source_row else "=NA()"
                        worksheet.cell(output_row, column, formula)
                else:
                    rolling_months = months[max(0, month_index - 5):month_index + 1]
                    rolling_rows = [source_rows[entity].get(m) for m in rolling_months]
                    for column, letter in [(12, "L"), (13, "M")]:
                        if len(rolling_rows) == 6 and all(rolling_rows):
                            refs = ",".join(f"{letter}{row}" for row in rolling_rows)
                            formula = f"=IF(COUNT({refs})=6,AVERAGE({refs}),NA())"
                        else:
                            formula = "=NA()"
                        worksheet.cell(output_row, column, formula)
            ranges[series_type][entity] = (start_row, worksheet.max_row)
    return ranges


def _add_charts(
    worksheet,
    category_range: tuple[int, int],
    source_ranges: dict[str, dict[str, tuple[int, int]]],
) -> None:
    """Add four charts whose series formulas all point to columns L or M."""
    entities = ["NL", "EA", "EA excl NL"]
    categories = Reference(
        worksheet,
        min_col=2,
        min_row=category_range[0],
        max_row=category_range[1],
    )
    chart_specs = [
        ("8 product groups, by month", "raw", 12, "P2"),
        ("8 product groups, by 6 months rolling average", "ma6", 12, "AB2"),
        ("5 product groups, by month", "raw", 13, "P20"),
        ("5 product groups, by 6 months rolling average", "ma6", 13, "AB20"),
    ]
    colors = ["4472C4", "ED7D31", "A5A5A5"]
    for title, series_type, column, anchor in chart_specs:
        chart = LineChart()
        _style_chart(chart, title)
        for entity, color in zip(entities, colors):
            min_row, max_row = source_ranges[series_type][entity]
            _add_chart_series(
                chart, worksheet, column, min_row, max_row, entity, color
            )
        chart.set_categories(categories)
        worksheet.add_chart(chart, anchor)


def write_xlsx(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write detailed data, live aggregates/formulas, and charts to XLSX."""
    headers = (
        ["country", "month"]
        + [f"HS_{code}" for code in HS_CODES]
        + ["total_export", "8 groups", "5 groups"]
    )
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "hs_export_monthly"
    worksheet.append(headers)
    for row in rows:
        worksheet.append([row.get(header) for header in headers[:11]])

    data_last_row = worksheet.max_row
    source_rows: dict[str, dict[str, int]] = {"NL": {}, "EA": {}, "EA excl NL": {}}
    for row_number, row in enumerate(rows, start=2):
        if row["country"] == "NL":
            source_rows["NL"][row["month"]] = row_number
    for row_number in range(2, data_last_row + 1):
        worksheet.cell(row_number, 12, f"=SUM(C{row_number}:J{row_number})/K{row_number}")
        worksheet.cell(row_number, 13, f"=SUM(F{row_number}:J{row_number})/K{row_number}")

    months = _month_sequence(min(row["month"] for row in rows), max(row["month"] for row in rows))
    for entity in ["EA", "EA excl NL"]:
        for month in months:
            output_row = worksheet.max_row + 1
            worksheet.cell(output_row, 1, entity)
            worksheet.cell(output_row, 2, month)
            source_rows[entity][month] = output_row
            for column_number in range(3, 12):
                column_letter = get_column_letter(column_number)
                if entity == "EA":
                    formula = (
                        f'=SUMIF($B$2:$B${data_last_row},$B{output_row},'
                        f'{column_letter}$2:{column_letter}${data_last_row})'
                    )
                else:
                    formula = (
                        f'=SUMIFS({column_letter}$2:{column_letter}${data_last_row},'
                        f'$B$2:$B${data_last_row},$B{output_row},'
                        f'$A$2:$A${data_last_row},"<>NL")'
                    )
                worksheet.cell(output_row, column_number, formula)
            worksheet.cell(output_row, 12, f"=SUM(C{output_row}:J{output_row})/K{output_row}")
            worksheet.cell(output_row, 13, f"=SUM(F{output_row}:J{output_row})/K{output_row}")

    visible_last_row = worksheet.max_row
    ea_category_range = (
        min(source_rows["EA"].values()),
        max(source_rows["EA"].values()),
    )
    chart_source_ranges = _add_chart_source_rows(worksheet, months, source_rows)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
    worksheet.freeze_panes = "C2"
    worksheet.auto_filter.ref = f"A1:M{visible_last_row}"
    worksheet.column_dimensions["A"].width = 12
    worksheet.column_dimensions["B"].width = 12
    for column_index in range(3, 12):
        letter = get_column_letter(column_index)
        worksheet.column_dimensions[letter].width = 17
        for cell in worksheet[letter][1:]:
            cell.number_format = "#,##0"
    for column_index in (12, 13):
        letter = get_column_letter(column_index)
        worksheet.column_dimensions[letter].width = 13
        for cell in worksheet[letter][1:]:
            cell.number_format = "0.0000%"

    _add_charts(worksheet, ea_category_range, chart_source_ranges)
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.calculation.calcMode = "auto"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def _valid_month(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must use YYYY-MM") from exc
    return value


def parse_args() -> argparse.Namespace:
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_valid_month, default="2020-01")
    parser.add_argument("--end", type=_valid_month, default=current_month)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monthly_hs_exports_2020_onward_with_charts.xlsx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not precede --start")
    print(f"Fetching {DATASET} monthly exports for {args.start} through {args.end}...")
    rows, strategy = fetch_monthly_exports(args.start, args.end)
    write_xlsx(rows, args.output)
    print(f"Reporter selection: {strategy}")
    print(
        f"Saved {len(rows):,} rows ({rows[0]['month']} through {rows[-1]['month']}) "
        f"to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
