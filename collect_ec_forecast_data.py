"""
STANDALONE data collection: European Commission GDP growth FORECASTS
(from AMECO's archived Spring/Autumn vintages, 2011-2025) matched
against the subsequently REALIZED ANNUAL GDP growth (Eurostat),
for the same 10 countries this project's other scripts use (NL, DE,
FR, IT, ES, BE, AT, IE, FI, PT).

WHAT THIS ACTUALLY PROVIDES, stated precisely (see the chat discussion
this script came out of): the European Commission does NOT produce
quarterly forecasts -- DG ECFIN publishes ANNUAL GDP growth forecasts
twice a year (Spring, Autumn), each forecasting the CURRENT year and
1-2 years ahead. This script:
  1. Downloads each Spring/Autumn AMECO vintage archive (real GDP
     level, variable OVGD) and computes that vintage's own IMPLIED
     annual growth forecast for its target year(s), from the LEVELS
     as reported in that specific vintage (not a later, revised
     figure).
  2. Compares that ANNUAL forecast directly against the REALIZED
     ANNUAL GDP growth for that SAME target year (Eurostat quarterly
     levels, summed to an annual aggregate, then year-on-year growth
     on that aggregate -- see fetch_realized_annual_growth()) -- ONE
     row per (country, vintage, horizon), matching the genuinely
     annual nature of the forecast itself (no quarterly broadcasting).

DATA AVAILABILITY, stated honestly:
  - Vintage coverage: Spring/Autumn 2011 through Autumn 2025 -- REAL,
    verified download URLs (fetched directly from the EC's current
    AMECO Archive page). Vintages before 2011 existed historically but
    are NOT listed on the CURRENT archive page (likely dropped during
    a website migration) -- NOT included here.
  - 2026 (the current/live vintage, Spring 2026) is NOT in the archive
    (the archive only holds PAST vintages) and the EC's documented
    "Web API" for querying the live database
    (ec.europa.eu/economy_finance/ameco/wq/series) returned an EMPTY
    result even for the LITERAL example from the EC's own official
    user manual when tested live -- this API appears to be non-
    functional or deprecated (the EC's own archive page notes the
    entire "old interface" is being phased out before Autumn 2026).
    2026 is therefore NOT covered by this script.
  - HONESTY NOTE on the archive ZIP files' internal format: AMECO's
    text-file convention (documented in EC reference material) uses
    semicolon-separated rows with a series code combining a country
    code and a variable code (e.g. "NLD.1.0.0.0.OVGD" for a level
    variable), one column per year -- this parsing logic follows that
    documented convention but was NOT verified against an actual
    downloaded, unzipped file (this sandbox could not reach
    ec.europa.eu directly; only web_fetch, which does not preserve
    raw ZIP bytes, was usable for verification). If parsing fails on a
    real run, the diagnostic output (raw file list, first lines of
    each candidate file) is printed specifically so the actual format
    can be inspected and the parser adjusted.

OUTPUT: ec_forecast_vs_realized.xlsx, one sheet "forecast_vs_realized"
with columns: country, vintage (e.g. "Spring2015"), vintage_round
(Spring/Autumn), vintage_year, target_year, horizon (0=current-year
forecast, 1=next-year forecast), forecast_growth_annual_pct,
realized_growth_annual_pct.
"""

import sys
import subprocess
import io
import re
import zipfile
from pathlib import Path


def install_if_needed(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_if_needed("requests")
install_if_needed("pandas")
install_if_needed("openpyxl")

import numpy as np
import pandas as pd
import requests

OUTPUT_FILE = Path(__file__).resolve().parent / "ec_forecast_vs_realized.xlsx"

COUNTRIES_ISO2 = ["NL", "DE", "FR", "IT", "ES", "BE", "AT", "IE", "FI", "PT"]
# AMECO's own country codes for these 10 (confirmed via AMECO documentation
# and consistent with this project's other scripts' ISO3 mapping).
ISO2_TO_AMECO = {
    "NL": "NLD", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "BE": "BEL", "AT": "AUT", "IE": "IRL", "FI": "FIN", "PT": "PRT",
}
AMECO_TO_ISO2 = {v: k for k, v in ISO2_TO_AMECO.items()}

AMECO_GDP_VARIABLE = "OVGD"  # Gross domestic product at constant (reference-year) prices, level

# REAL, verified download URLs -- fetched directly from the EC's current
# AMECO Archive page (economy-finance.ec.europa.eu/.../ameco-archive_en)
# on the date this script was written. These are permanent per-document
# GUIDs, not a predictable/guessable pattern -- if the EC reorganizes
# the archive again, these will need re-fetching from that page.
AMECO_VINTAGE_URLS = {
    "Spring2011": "https://economy-finance.ec.europa.eu/document/download/e8b9a0ee-8f28-4c86-999d-c8dd1a166e60_en?filename=ameco_spring2011.zip",
    "Autumn2011": "https://economy-finance.ec.europa.eu/document/download/af1d61bb-6a83-4bc2-ad9a-b00b4f551b8e_en?filename=ameco_autumn2011.zip",
    "Spring2012": "https://economy-finance.ec.europa.eu/document/download/6b88702e-dbea-41da-af92-3beda9cc5e43_en?filename=ameco_spring2012.zip",
    "Autumn2012": "https://economy-finance.ec.europa.eu/document/download/6a98c5a3-5b99-4fd2-ac1b-d7255293a38a_en?filename=ameco_autumn2012.zip",
    "Spring2013": "https://economy-finance.ec.europa.eu/document/download/af0377ee-ab3e-4fdf-b147-b81183728402_en?filename=ameco_spring2013.zip",
    "Autumn2013": "https://economy-finance.ec.europa.eu/document/download/3b4daf03-9de1-41e6-b260-89a10de26f0a_en?filename=ameco_autumn2013.zip",
    "Spring2014": "https://economy-finance.ec.europa.eu/document/download/e5dfa67a-695a-4535-9615-e594cf88fd85_en?filename=ameco_spring2014.zip",
    "Autumn2014": "https://economy-finance.ec.europa.eu/document/download/e18a0d7f-bb2f-41fa-92ad-34b6d55dd80a_en?filename=ameco_autumn2014.zip",
    "Spring2015": "https://economy-finance.ec.europa.eu/document/download/af291ff7-1b5d-4343-805f-061c8362ae60_en?filename=ameco_spring2015.zip",
    "Autumn2015": "https://economy-finance.ec.europa.eu/document/download/8ff15496-dc11-45e8-8711-820eae40f27e_en?filename=ameco_autumn2015.zip",
    "Spring2016": "https://economy-finance.ec.europa.eu/document/download/ac6e8f81-3ae8-4cdf-9776-e1fd34449cb3_en?filename=ameco_spring2016.zip",
    "Autumn2016": "https://economy-finance.ec.europa.eu/document/download/2bad1334-5866-404f-9f60-ffe2f8ee951c_en?filename=ameco_autumn20161.zip",
    "Spring2017": "https://economy-finance.ec.europa.eu/document/download/ada26fc5-3ce0-412c-af3c-6937376ae6af_en?filename=ameco_spring20171.zip",
    "Autumn2017": "https://economy-finance.ec.europa.eu/document/download/b7b692aa-44e3-4a75-9501-7379dfaafa50_en?filename=ameco_autumn20171.zip",
    "Spring2018": "https://economy-finance.ec.europa.eu/document/download/66a10fb5-0c73-4fcf-831a-e58d25ca79bf_en?filename=ameco_spring20181.zip",
    "Autumn2018": "https://economy-finance.ec.europa.eu/document/download/5cfc8121-138b-4c42-9411-90df559b21db_en?filename=ameco_autumn20181.zip",
    "Spring2019": "https://economy-finance.ec.europa.eu/document/download/2dbfb168-f2f9-43b8-bfb7-c6efa0e4da88_en?filename=ameco_spring2019.zip",
    "Autumn2019": "https://economy-finance.ec.europa.eu/document/download/d7f7871e-fa02-46f2-b9b8-4fed822bd72d_en?filename=ameco_autumn2019.zip",
    "Spring2020": "https://economy-finance.ec.europa.eu/document/download/e6e8f547-0d9b-40f6-b9d8-98d11b502696_en?filename=ameco_spring2020.zip",
    "Autumn2020": "https://economy-finance.ec.europa.eu/document/download/e058b7c7-a9c6-4340-abe1-a6669c469592_en?filename=ameco_autumn2020.zip",
    "Spring2021": "https://economy-finance.ec.europa.eu/document/download/17372dca-211c-42b6-9a1d-89559b985d23_en?filename=ameco_spring2021.zip",
    "Autumn2021": "https://economy-finance.ec.europa.eu/document/download/a522fc1a-c386-4c3e-80dd-50de123fb773_en?filename=ameco_autumn2021.zip",
    "Spring2022": "https://economy-finance.ec.europa.eu/document/download/c5be576e-fbea-45f3-98a5-3cf59d16d70a_en?filename=ameco_spring2022.zip",
    "Autumn2022": "https://economy-finance.ec.europa.eu/document/download/b16e473a-04ef-4ec8-b2f8-1e0c30a39d3c_en?filename=ameco_autumn2022.zip",
    "Spring2023": "https://economy-finance.ec.europa.eu/document/download/adbcba6f-8169-4d6d-80e7-7a365ff6e87d_en?filename=ameco_spring2023.zip",
    "Autumn2023": "https://economy-finance.ec.europa.eu/document/download/6a48b097-bd09-496b-908f-404d0e85cff8_en?filename=AMECO_2023_Autumn_Revised.zip",
    "Spring2024": "https://economy-finance.ec.europa.eu/document/download/ad8d7bd0-c4d9-4860-b597-bb613408d6bd_en?filename=AMECO_2024_Spring_final.zip",
    "Autumn2024": "https://economy-finance.ec.europa.eu/document/download/9c2630c3-6c25-456d-b484-6b05c202e8ce_en?filename=ameco0.zip",
    "Spring2025": "https://economy-finance.ec.europa.eu/document/download/b4b375e7-80a6-4581-b8fd-9c207790cd01_en?filename=ameco0.zip",
    "Autumn2025": "https://economy-finance.ec.europa.eu/document/download/92bd1d5e-35e8-4a94-a8b1-8d919ced4cdb_en?filename=ameco-autumn2025.zip",
}
# NOTE: 2026 (current/live vintage) is deliberately NOT included here --
# see the module docstring's "DATA AVAILABILITY" section for why.


def _download_and_extract_zip(url, max_attempts=4):
    """
    Downloads an AMECO vintage ZIP with retry-with-backoff (same pattern
    as this project's other network fetches, e.g.
    fetch_commodity_price_index() in collect_ai_data.py), and returns a
    zipfile.ZipFile object opened from the in-memory bytes (never
    written to disk).
    """
    r = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_attempts:
                raise SystemExit(
                    f"\n_download_and_extract_zip({url}): network request "
                    f"failed after {max_attempts} attempts: "
                    f"{e.__class__.__name__}: {e}"
                )
            wait_s = 5 * (3 ** (attempt - 1))
            print(f"  [!] Attempt {attempt}/{max_attempts} failed "
                  f"({e.__class__.__name__}: {e}) -- retrying in {wait_s}s...")
            import time
            time.sleep(wait_s)

    try:
        return zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile as e:
        raise ValueError(
            f"_download_and_extract_zip({url}): downloaded content is not "
            f"a valid ZIP file ({e}) -- the URL may have changed; re-fetch "
            f"the current link from the AMECO Archive page."
        )


def _parse_ameco_gdp_level(zf, ameco_codes, variable_code=AMECO_GDP_VARIABLE):
    """
    Searches EVERY text file inside the given AMECO vintage ZIP for rows
    matching the requested country codes and variable code, and returns
    a long DataFrame (ameco_country, year, gdp_level).

    HONESTY NOTE: AMECO's documented text-file convention is semicolon-
    separated, with a series-code column formatted as
    "{COUNTRY}.{one or more numeric sub-codes}.{VARIABLE}" (e.g.
    "NLD.1.0.0.0.OVGD"), followed by one column per year. This function
    searches ALL text-like files in the archive (rather than assuming a
    specific chapter-file name/number) specifically because that exact
    naming was not independently confirmed -- if the real files use a
    different exact structure, the diagnostic prints below (file list,
    first lines of each candidate) will show what needs adjusting.

    NESTED-ZIP HANDLING (found from a real run's diagnostic output):
    three specific vintages -- Autumn 2020, Autumn 2021, Spring 2022 --
    package each chapter as its OWN nested .zip file (ameco0.zip,
    ameco1.zip, ... ameco18.zip) INSIDE the outer vintage zip, instead
    of plain .txt files directly at the top level like every other
    vintage. This also explains why those three vintages' outer zip is
    roughly DOUBLE the file size of their neighbours (extra zip-within-
    zip compression overhead) -- a red herring that earlier looked
    like it might be a different series-code format, but was actually
    one level of nesting the parser never looked inside. One level of
    .zip nesting is unwrapped below before the normal text-file search
    runs, covering both the plain (majority of vintages) and nested
    (these three vintages) packaging.
    """
    def _collect_text_candidates(zip_file, prefix=""):
        """Returns [(display_name, raw_bytes), ...] for every
        .txt/.csv/.dat file directly in zip_file, PLUS (one level
        only) every .txt/.csv/.dat file inside any .zip member of
        zip_file."""
        found = []
        for name in zip_file.namelist():
            display = f"{prefix}{name}"
            lower = name.lower()
            if lower.endswith((".txt", ".csv", ".dat")):
                try:
                    found.append((display, zip_file.read(name)))
                except Exception as e:
                    print(f"  [!] Could not read {display} from archive: {e}")
            elif lower.endswith(".zip"):
                try:
                    nested_bytes = zip_file.read(name)
                    nested_zf = zipfile.ZipFile(io.BytesIO(nested_bytes))
                except Exception as e:
                    print(f"  [!] Could not open nested zip {display}: {e}")
                    continue
                for inner_name in nested_zf.namelist():
                    if inner_name.lower().endswith((".txt", ".csv", ".dat")):
                        try:
                            found.append((f"{display}/{inner_name}",
                                          nested_zf.read(inner_name)))
                        except Exception as e:
                            print(f"  [!] Could not read {display}/{inner_name}: {e}")
        return found

    candidates = _collect_text_candidates(zf)
    if not candidates:
        print(f"  [!] No .txt/.csv/.dat files found (directly or inside nested .zip "
              f"members) in the archive. Actual top-level contents: {zf.namelist()}")
        return pd.DataFrame(columns=["ameco_country", "year", "gdp_level"])

    rows = []
    code_pattern = re.compile(
        r"^(" + "|".join(ameco_codes) + r")(\.\d+)+\." + variable_code + r"$"
    )

    for name, raw in candidates:
        text = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            print(f"  [!] Could not decode {name} with any tried encoding -- skipping.")
            continue

        lines = text.splitlines()
        if not lines:
            continue
        header = lines[0]
        delimiter = ";" if header.count(";") >= header.count(",") else ","
        header_cols = header.split(delimiter)

        year_col_idx = {}
        for i, col in enumerate(header_cols):
            col_clean = col.strip().strip('"')
            if re.fullmatch(r"(19|20)\d{2}", col_clean):
                year_col_idx[int(col_clean)] = i

        if not year_col_idx:
            continue  # this file has no year columns -- not a data chapter file

        matched_any = False
        sample_codes = []
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split(delimiter)
            if not fields:
                continue
            series_code = fields[0].strip().strip('"')
            if len(sample_codes) < 5:
                sample_codes.append(series_code)
            m = code_pattern.match(series_code)
            if not m:
                continue
            matched_any = True
            ameco_country = m.group(1)
            for year, idx in year_col_idx.items():
                if idx >= len(fields):
                    continue
                raw_val = fields[idx].strip().strip('"').replace(",", ".")
                if raw_val in ("", "-", "n/a", "N/A"):
                    continue
                try:
                    val = float(raw_val)
                except ValueError:
                    continue
                rows.append({"ameco_country": ameco_country, "year": year, "gdp_level": val})

        if matched_any:
            print(f"  [diagnostic] Found {variable_code} rows in '{name}'.")
        elif sample_codes:
            # This file DID look like a real data chapter (it has year
            # columns), but NOTHING matched our country+variable pattern
            # -- print actual series codes seen here so a genuinely
            # different naming convention (not just a different sub-code
            # count, which the relaxed pattern above already covers) is
            # immediately visible rather than silently producing zero
            # rows for the whole vintage.
            print(f"  [diagnostic] '{name}' has year columns but matched zero rows -- "
                  f"sample series codes actually present: {sample_codes}")

    if not rows:
        candidate_display_names = [n for n, _ in candidates]
        print(f"  [!] No rows matched variable '{variable_code}' for the requested "
              f"countries in any file. Files inspected: {candidate_display_names}")
        print(f"  [!] First 3 lines of first candidate file, for manual inspection:")
        if candidates:
            sample = candidates[0][1].decode("latin-1", errors="replace")
            for line in sample.splitlines()[:3]:
                print(f"      {line[:200]}")

    return pd.DataFrame(rows).drop_duplicates(subset=["ameco_country", "year"])


def fetch_ec_gdp_forecasts(countries=COUNTRIES_ISO2):
    """
    For every Spring/Autumn AMECO vintage in AMECO_VINTAGE_URLS,
    downloads the archive, extracts real GDP level (OVGD) for the 10
    project countries, and computes THAT VINTAGE's own implied annual
    growth forecast for its target year(s) -- current year (horizon=0)
    and next year (horizon=1) -- from the levels AS REPORTED in that
    specific vintage (not a later, revised figure).

    Returns a long DataFrame: country, vintage, vintage_round,
    vintage_year, target_year, horizon, forecast_growth_annual_pct.
    """
    ameco_codes = [ISO2_TO_AMECO[c] for c in countries]
    all_forecasts = []

    for vintage_name, url in AMECO_VINTAGE_URLS.items():
        m = re.match(r"(Spring|Autumn)(\d{4})", vintage_name)
        round_name, vintage_year = m.group(1), int(m.group(2))

        print(f"\n  Fetching {vintage_name}...")
        try:
            zf = _download_and_extract_zip(url)
        except SystemExit:
            print(f"  [!] Skipping {vintage_name} -- download failed after retries.")
            continue

        levels = _parse_ameco_gdp_level(zf, ameco_codes)
        if levels.empty:
            print(f"  [!] No GDP level data extracted for {vintage_name} -- skipping.")
            continue

        # A Spring vintage forecasts vintage_year (horizon 0) and
        # vintage_year+1 (horizon 1); Autumn additionally often covers
        # vintage_year+2, but this script only uses horizons 0 and 1 for
        # consistency across both rounds.
        for horizon, target_year in enumerate([vintage_year, vintage_year + 1]):
            prior_year = target_year - 1
            for ameco_c in ameco_codes:
                lvl_t = levels[(levels["ameco_country"] == ameco_c)
                               & (levels["year"] == target_year)]["gdp_level"]
                lvl_t1 = levels[(levels["ameco_country"] == ameco_c)
                                & (levels["year"] == prior_year)]["gdp_level"]
                if lvl_t.empty or lvl_t1.empty or lvl_t1.iloc[0] == 0:
                    continue
                growth_pct = 100 * (lvl_t.iloc[0] / lvl_t1.iloc[0] - 1)
                all_forecasts.append({
                    "country": AMECO_TO_ISO2[ameco_c],
                    "vintage": vintage_name,
                    "vintage_round": round_name,
                    "vintage_year": vintage_year,
                    "target_year": target_year,
                    "horizon": horizon,
                    "forecast_growth_annual_pct": growth_pct,
                })

    result = pd.DataFrame(all_forecasts)
    print(f"\n  [diagnostic] EC GDP growth forecasts: {len(result)} "
          f"(country x vintage x horizon) rows across "
          f"{result['vintage'].nunique() if not result.empty else 0} vintages.")
    return result


def fetch_realized_annual_growth(countries=COUNTRIES_ISO2):
    """
    Realized ANNUAL, year-on-year GDP growth (%), per country, from
    Eurostat (dataset namq_10_gdp, na_item=B1GQ, unit=CLV10_MEUR,
    s_adj=SCA) -- the SAME quarterly source and construction this
    project's other scripts use for GDP level (see fetch_gdp_level()
    in collect_ai_data.py), aggregated here to annual by SUMMING each
    year's 4 quarterly levels (the standard practical approximation to
    an official annual chain-linked figure when a separately-published
    annual series isn't fetched directly -- chain-linked quarterly
    volumes don't sum perfectly additively to the official annual
    figure, but the discrepancy is minor and this keeps the realized
    side on the exact same underlying data as this project's other
    quarterly fetches). Annual growth is then the plain year-on-year
    log-difference of that annual aggregate: 100*ln(level[t]/level[t-1]).

    This annual REALIZED growth is what gets compared against the
    (already annual) EC forecast, year for year -- replacing the
    earlier quarterly-broadcast design (one forecast row per quarter)
    with a single row per (country, vintage, horizon), since the EC's
    forecast itself is annual and never was genuinely quarterly.
    """
    EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    params = {
        "format": "JSON",
        "na_item": "B1GQ",
        "unit": "CLV10_MEUR",
        "s_adj": "SCA",
        "geo": countries,
        "sinceTimePeriod": "2010-Q1",
    }
    r = requests.get(f"{EUROSTAT_BASE}/namq_10_gdp", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    import itertools as _itertools
    dims = js["dimension"]
    dim_ids = js["id"]
    values = js["value"]
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}

    idx_lists = []
    for d in dim_ids:
        cat = dims[d]["category"]
        order = sorted(cat["index"].items(), key=lambda kv: kv[1])
        idx_lists.append([k for k, _ in order])

    rows = []
    for flat_i, combo in enumerate(_itertools.product(*idx_lists)):
        v = values.get(str(flat_i))
        if v is None:
            continue
        rows.append(dict(zip(dim_ids, combo), value=v))

    df = pd.DataFrame(rows)
    df = df.rename(columns={"geo": "country", "time": "quarter"})
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["gdp_level"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["country", "quarter", "gdp_level"]].dropna()
    df["target_year"] = df["quarter"].dt.year

    # Only keep years with all 4 quarters present, so the annual sum is
    # a genuine full-year aggregate, not a partial-year undercount.
    counts = df.groupby(["country", "target_year"])["quarter"].transform("count")
    df = df[counts == 4]

    annual = (df.groupby(["country", "target_year"])["gdp_level"]
              .sum().reset_index().sort_values(["country", "target_year"]))
    annual["realized_growth_annual_pct"] = annual.groupby("country")["gdp_level"].transform(
        lambda s: 100 * np.log(s / s.shift(1))
    )
    annual = annual.dropna(subset=["realized_growth_annual_pct"])

    print(f"  [diagnostic] Realized annual GDP growth: {len(annual)} country-year "
          f"rows across {annual['country'].nunique()} countries.")
    return annual[["country", "target_year", "realized_growth_annual_pct"]]


def build_forecast_vs_realized(countries=COUNTRIES_ISO2):
    """
    Combines fetch_ec_gdp_forecasts() and fetch_realized_annual_growth()
    into one ANNUAL-frequency table: ONE row per (country, vintage,
    horizon), pairing that vintage's forecast_growth_annual_pct for
    target_year directly against the realized_growth_annual_pct for
    that SAME year -- no quarterly broadcasting (the earlier design put
    4 rows, one per quarter, under each forecast; both sides of the
    comparison are genuinely annual here, matching what a Spring or
    Autumn EC round actually forecasts: year t and year t+1 as a whole,
    not any individual quarter within it).
    """
    forecasts = fetch_ec_gdp_forecasts(countries)
    realized = fetch_realized_annual_growth(countries)
    if forecasts.empty:
        raise RuntimeError(
            "build_forecast_vs_realized: zero forecast rows were extracted -- "
            "see the [!] diagnostic messages above for why (likely the ZIP "
            "internal format did not match what this parser expected)."
        )

    merged = forecasts.merge(realized, on=["country", "target_year"], how="left")
    merged = merged.sort_values(["country", "vintage_year", "vintage_round", "horizon"])

    print(f"\n  [diagnostic] Combined annual forecast-vs-realized table: {len(merged)} rows.")
    return merged[["country", "vintage", "vintage_round", "vintage_year", "target_year",
                    "horizon", "forecast_growth_annual_pct", "realized_growth_annual_pct"]]


if __name__ == "__main__":
    print("=" * 70)
    print("Collecting EC Spring/Autumn GDP growth forecasts (AMECO archive,")
    print("2011-2025) vs. realized ANNUAL GDP growth (Eurostat)")
    print("=" * 70)

    result = build_forecast_vs_realized()
    result_out = result.copy()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result_out.to_excel(writer, sheet_name="forecast_vs_realized", index=False)

    print(f"\n{OUTPUT_FILE.name} written -- {len(result_out)} rows, "
          f"sheet 'forecast_vs_realized'.")
    print("Covers Spring/Autumn 2011-2025 vintages only -- see the module")
    print("docstring for why 2000-2010 and 2026 are not included.")
    print("Upload this file back to Claude, or open it directly.")
