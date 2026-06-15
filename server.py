"""
CRE Intelligence MCP Server
Live market data layer for commercial real estate analysis.

Powered by:
- FRED (Federal Reserve) — live rates, CPI, CRE price index (FREE)
- US Census Bureau — address-level demographics, income, vacancy (FREE)
- Claude AI — document parsing, risk analysis, memo generation

No data licensing fees. Every number comes from verified public sources.
"""

from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Optional
from datetime import date
from starlette.requests import Request
from starlette.responses import JSONResponse
import anthropic
import anyio
import requests
import json
import math
import os
import re
import time
import concurrent.futures

load_dotenv()

# ─── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="CRE Intelligence",
    instructions="""
    You are a commercial real estate analyst with access to live market data.
    Use these tools to provide accurate, data-driven CRE analysis.

    All interest rate data comes directly from the Federal Reserve (FRED) — never guess at rates.
    All demographic data comes from the US Census Bureau — never estimate demographics.
    Always use get_current_rates() before building any DCF model.
    Always use get_market_demographics() when analyzing a specific property location.
    Use get_radius_demographics() for 1/3/5-mile trade-area analysis around a property.
    """
)

# ─── Claude helper ────────────────────────────────────────────────────────────

def ask_claude(data: dict, instructions: str, max_tokens: int = 2000) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": f"Here is the data:\n\n{json.dumps(data, indent=2)}\n\n{instructions}"
        }]
    )
    return msg.content[0].text

# ─── FRED API helper ──────────────────────────────────────────────────────────

def get_fred_series(series_id: str, limit: int = 1) -> dict:
    """Fetch latest observations from FRED (Federal Reserve Economic Data)."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {"error": "FRED_API_KEY not set. Free key at fred.stlouisfed.org/docs/api/api_key.html"}

    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "sort_order": "desc",
                "limit": limit,
                "file_type": "json"
            },
            timeout=10
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return {
            "series_id": series_id,
            "observations": [
                {"date": o["date"], "value": float(o["value"]) if o["value"] not in (".", "") else None}
                for o in obs
            ]
        }
    except Exception as e:
        return {"error": f"FRED fetch failed for {series_id}: {str(e)}"}


def parse_json_from_claude(text: str) -> dict:
    """Extract JSON from Claude's response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    # Find the JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in response")


# ─── Geo helpers ──────────────────────────────────────────────────────────────

def geocode_address(address: str) -> dict:
    """Geocode a US address via the Census Geocoder.
    Returns matched address, lat/lng, and tract FIPS codes — or {'error': ...}."""
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 3:
        return {"error": "Provide full address: '123 Main St, City, ST 12345'"}

    street, city, state_zip = parts[0], parts[1], parts[2]

    try:
        # Extract state and optional zip from "ST ZIPCODE" or just "ST"
        state_parts = state_zip.split()
        state_code = state_parts[0][:2]
        zip_code = state_parts[1] if len(state_parts) > 1 else ""

        geo_params = {
            "street": street,
            "city": city,
            "state": state_code,
            "benchmark": "Public_AR_Census2020",
            "vintage": "Census2020_Census2020",
            "layers": "all",
            "format": "json"
        }
        if zip_code:
            geo_params["zip"] = zip_code

        geo_r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/address",
            params=geo_params,
            timeout=15
        )
        matches = geo_r.json().get("result", {}).get("addressMatches", [])

        if not matches:
            return {"error": f"Could not geocode: '{address}'. Try including ZIP code."}

        match = matches[0]
        tracts = match.get("geographies", {}).get("Census Tracts", [])
        if not tracts:
            return {"error": "No census tract found for this address."}

        tract = tracts[0]
        coords = match.get("coordinates", {})
        return {
            "matched_address": match.get("matchedAddress", address),
            "lat": coords.get("y"),
            "lng": coords.get("x"),
            "state_fips": tract["STATE"],
            "county_fips": tract["COUNTY"],
            "tract_fips": tract["TRACT"],
        }
    except Exception as e:
        return {"error": f"Geocoding failed: {str(e)}"}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * 3958.7613 * math.asin(math.sqrt(a))


# ─── Tool 1: Live Interest Rates ──────────────────────────────────────────────

@mcp.tool()
def get_current_rates() -> dict:
    """
    Get live interest rates from the Federal Reserve (FRED).
    Returns SOFR, 10-year Treasury, 5-year Treasury, Fed Funds Rate, and 30-day SOFR average.
    Also calculates implied cap rate ranges based on current treasury spreads.

    Use this BEFORE any DCF model or loan underwriting. These are real-time numbers
    Claude cannot access on its own.
    """
    series_map = {
        "sofr":           "SOFR",
        "sofr_30day_avg": "SOFR30DAYAVG",
        "treasury_10yr":  "DGS10",
        "treasury_5yr":   "DGS5",
        "treasury_2yr":   "DGS2",
        "fed_funds_rate": "DFF",
    }

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(get_fred_series, sid): key for key, sid in series_map.items()}
        for future, key in futures.items():
            data = future.result()
            if "observations" in data and data["observations"]:
                obs = data["observations"][0]
                results[key] = {"rate_pct": obs["value"], "date": obs["date"]}
            else:
                results[key] = data  # Pass through error

    # Implied cap rate context based on 10yr Treasury spread
    t10 = results.get("treasury_10yr", {}).get("rate_pct")
    if t10:
        results["cap_rate_context"] = {
            "ten_year_treasury": t10,
            "typical_spreads": {
                "core_multifamily":   f"{t10 + 0.75:.2f}%–{t10 + 1.50:.2f}%",
                "core_industrial":    f"{t10 + 1.00:.2f}%–{t10 + 1.75:.2f}%",
                "core_office":        f"{t10 + 1.50:.2f}%–{t10 + 2.50:.2f}%",
                "value_add":          f"{t10 + 1.50:.2f}%–{t10 + 2.50:.2f}%",
                "opportunistic":      f"{t10 + 2.50:.2f}%+",
            },
            "note": "Traditional cap rate spreads over 10yr Treasury — verify with local comps"
        }

    # Typical loan rate guidance
    sofr = results.get("sofr", {}).get("rate_pct")
    if sofr:
        results["loan_rate_guidance"] = {
            "floating_rate": f"SOFR ({sofr:.2f}%) + 150–250bps spread = ~{sofr + 1.75:.2f}%–{sofr + 2.25:.2f}%",
            "fixed_rate_proxy": f"10yr Treasury ({t10:.2f}%) + 150–200bps = ~{t10 + 1.5:.2f}%–{t10 + 2.0:.2f}%" if t10 else "N/A",
            "note": "Actual rates vary by borrower, LTV, property type, and lender"
        }

    return results


# ─── Tool 2: Market Demographics ─────────────────────────────────────────────

@mcp.tool()
def get_market_demographics(address: str) -> dict:
    """
    Get Census Bureau demographics for any US property address.
    Returns median income, population, employment rate, housing vacancy,
    median rents, and education levels for the census tract.

    This is address-specific data from the actual Census tract — not estimates.
    Claude cannot access this without the MCP.
    For 1/3/5-mile trade-area rings, use get_radius_demographics instead.

    Args:
        address: Full US property address (e.g. "1234 Main St, Charlotte, NC 28202")
    """
    # Step 1: Geocode via Census Geocoder
    geo = geocode_address(address)
    if "error" in geo:
        return geo

    state_fips  = geo["state_fips"]
    county_fips = geo["county_fips"]
    tract_fips  = geo["tract_fips"]

    # Step 2: Pull ACS 5-year data
    variables = {
        "B19013_001E": "median_household_income",
        "B01003_001E": "total_population",
        "B23025_004E": "employed_civilians",
        "B23025_003E": "civilian_labor_force",
        "B25001_001E": "total_housing_units",
        "B25002_002E": "occupied_housing_units",
        "B25002_003E": "vacant_housing_units",
        "B25064_001E": "median_gross_rent",
        "B25058_001E": "median_contract_rent",
        "B15003_022E": "bachelors_degree",
        "B15003_001E": "education_population_total",
    }

    def safe_int(v):
        try:
            n = int(v)
            return None if n < 0 else n
        except:
            return None

    census_key = os.getenv("CENSUS_API_KEY", "")
    if not census_key:
        return {"error": "CENSUS_API_KEY not set. Free key at api.census.gov/data/key_signup.html"}

    # Try most recent ACS first, fall back if needed
    for acs_year in ["2023", "2022"]:
        try:
            census_key = os.getenv("CENSUS_API_KEY", "")
            params = {
                "get": ",".join(variables.keys()),
                "for": f"tract:{tract_fips}",
                "in": f"state:{state_fips} county:{county_fips}",
            }
            if census_key:
                params["key"] = census_key

            acs_r = requests.get(
                f"https://api.census.gov/data/{acs_year}/acs/acs5",
                params=params,
                timeout=15
            )

            if acs_r.status_code != 200:
                continue

            acs_data = acs_r.json()
            if len(acs_data) < 2:
                continue

            headers, values = acs_data[0], acs_data[1]
            raw = {headers[i]: values[i] for i in range(len(headers))}

            pop         = safe_int(raw.get("B01003_001E"))
            employed    = safe_int(raw.get("B23025_004E"))
            labor_force = safe_int(raw.get("B23025_003E"))
            housing     = safe_int(raw.get("B25001_001E"))
            occupied    = safe_int(raw.get("B25002_002E"))
            vacant      = safe_int(raw.get("B25002_003E"))
            bachelors   = safe_int(raw.get("B15003_022E"))
            edu_total   = safe_int(raw.get("B15003_001E"))
            income      = safe_int(raw.get("B19013_001E"))
            gross_rent  = safe_int(raw.get("B25064_001E"))
            contract_rent = safe_int(raw.get("B25058_001E"))

            return {
                "address_matched": geo["matched_address"],
                "coordinates": {"lat": geo["lat"], "lng": geo["lng"]},
                "census_tract": f"{state_fips}-{county_fips}-{tract_fips}",
                "acs_vintage": acs_year,
                "demographics": {
                    "median_household_income":  income,
                    "total_population":         pop,
                    "employment_rate_pct":       round(employed / labor_force * 100, 1) if employed and labor_force else None,
                    "college_educated_pct":      round(bachelors / edu_total * 100, 1) if bachelors and edu_total else None,
                },
                "housing": {
                    "total_units":       housing,
                    "occupied_units":    occupied,
                    "vacant_units":      vacant,
                    "vacancy_rate_pct":  round(vacant / housing * 100, 1) if vacant and housing else None,
                },
                "rents": {
                    "median_gross_rent":    gross_rent,
                    "median_contract_rent": contract_rent,
                    "note": "ACS 5-year estimate for this census tract"
                }
            }

        except Exception:
            continue

    return {"error": "Census ACS data unavailable for this location. Try a more specific address."}


# ─── Tool 2b: Radius Demographics (1/3/5-mile rings) ─────────────────────────

RADIUS_ACS_VARS = {
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B23025_004E": "employed_civilians",
    "B23025_003E": "civilian_labor_force",
    "B25001_001E": "total_housing_units",
    "B25002_002E": "occupied_housing_units",
    "B25002_003E": "vacant_housing_units",
    "B25064_001E": "median_gross_rent",
    "B15003_022E": "bachelors_degree",
    "B15003_001E": "education_population_total",
    "B25003_001E": "tenure_total",
    "B25003_003E": "renter_occupied",
}


def _point_radius_demographics(lat: float, lng: float, radii: list, census_key: str) -> dict:
    """Core ring-aggregation engine: tracts around a point → aggregated ACS demographics.
    Shared by the get_radius_demographics tool (address) and the map API (coordinates)."""
    # Step 1: All tracts intersecting the largest circle (TIGERweb, 2020 tract vintage)
    try:
        feats = []
        offset = 0
        while True:
            r = requests.get(
                "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/10/query",
                params={
                    "geometry": f"{lng},{lat}",
                    "geometryType": "esriGeometryPoint",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                    "distance": radii[-1] * 1609.344,
                    "units": "esriSRUnit_Meter",
                    "outFields": "GEOID,STATE,COUNTY,TRACT,CENTLAT,CENTLON",
                    "returnGeometry": "false",
                    "resultOffset": offset,
                    "f": "json",
                },
                timeout=20,
            )
            d = r.json()
            batch = d.get("features", [])
            feats.extend(batch)
            if d.get("exceededTransferLimit") and batch:
                offset += len(batch)
            else:
                break
    except Exception as e:
        return {"error": f"Tract lookup failed: {str(e)}"}

    if not feats:
        return {"error": "No census tracts found around this address."}

    tracts = []
    for f in feats:
        a = f.get("attributes", {})
        try:
            clat, clon = float(a["CENTLAT"]), float(a["CENTLON"])
        except (KeyError, TypeError, ValueError):
            continue
        tracts.append({
            "state": a["STATE"], "county": a["COUNTY"], "tract": a["TRACT"],
            "dist": _haversine_miles(lat, lng, clat, clon),
        })

    # Step 2: Batch ACS pull — one call per county covered by the largest ring
    counties = sorted({(t["state"], t["county"]) for t in tracts})
    acs_by_tract, acs_vintage = {}, None
    for acs_year in ["2023", "2022"]:
        try:
            tmp, ok = {}, True
            for st, co in counties:
                resp = requests.get(
                    f"https://api.census.gov/data/{acs_year}/acs/acs5",
                    params={
                        "get": ",".join(RADIUS_ACS_VARS.keys()),
                        "for": "tract:*",
                        "in": f"state:{st} county:{co}",
                        "key": census_key,
                    },
                    timeout=20,
                )
                if resp.status_code != 200:
                    ok = False
                    break
                rows = resp.json()
                hdr = rows[0]
                for row in rows[1:]:
                    rec = dict(zip(hdr, row))
                    tmp[(rec["state"], rec["county"], rec["tract"])] = rec
            if ok and tmp:
                acs_by_tract, acs_vintage = tmp, acs_year
                break
        except Exception:
            continue

    if not acs_by_tract:
        return {"error": "Census ACS data unavailable for this area."}

    def _i(rec, var):
        try:
            n = int(rec.get(var))
            return None if n < 0 else n  # Census uses negative sentinels for suppressed data
        except (TypeError, ValueError):
            return None

    # Step 3: Aggregate per ring
    rings = {}
    for radius in radii:
        in_ring = [t for t in tracts if t["dist"] <= radius]
        ring_note = None
        if not in_ring:
            in_ring = [min(tracts, key=lambda t: t["dist"])]
            ring_note = "No tract centroid within radius — using nearest tract (rural area)."

        recs = []
        for t in in_ring:
            rec = acs_by_tract.get((t["state"], t["county"], t["tract"]))
            if rec:
                recs.append(rec)

        def ssum(var):
            vals = [v for v in (_i(r, var) for r in recs) if v is not None]
            return sum(vals) if vals else None

        def wmedian(value_var, weight_var):
            num = den = 0
            for rec in recs:
                v, w = _i(rec, value_var), _i(rec, weight_var)
                if v is not None and w:
                    num += v * w
                    den += w
            return round(num / den) if den else None

        pop      = ssum("B01003_001E")
        employed = ssum("B23025_004E")
        labor    = ssum("B23025_003E")
        bach     = ssum("B15003_022E")
        edu      = ssum("B15003_001E")
        units    = ssum("B25001_001E")
        occ      = ssum("B25002_002E")
        vac      = ssum("B25002_003E")
        tenure   = ssum("B25003_001E")
        renters  = ssum("B25003_003E")

        ring = {
            "radius_miles": radius,
            "tract_count": len(recs),
            "population": pop,
            "median_household_income": wmedian("B19013_001E", "B25002_002E"),
            "employment_rate_pct": round(employed / labor * 100, 1) if employed and labor else None,
            "college_educated_pct": round(bach / edu * 100, 1) if bach and edu else None,
            "housing": {
                "total_units": units,
                "occupied_units": occ,
                "vacant_units": vac,
                "vacancy_rate_pct": round(vac / units * 100, 1) if vac and units else None,
                "renter_share_pct": round(renters / tenure * 100, 1) if renters and tenure else None,
            },
            "median_gross_rent": wmedian("B25064_001E", "B25003_003E"),
        }
        if ring_note:
            ring["note"] = ring_note
        rings[f"{radius:g}_mile"] = ring

    return {
        "coordinates": {"lat": lat, "lng": lng},
        "acs_vintage": acs_vintage,
        "rings": rings,
        "methodology": (
            "Each ring aggregates all census tracts whose centroid falls within the radius "
            "(2020 tract boundaries). Median income and rent are household-weighted averages "
            "of tract medians — the standard free-data approximation. Verify against a licensed "
            "demographics provider for institutional reporting."
        ),
        "source": "US Census Bureau ACS 5-year + TIGERweb",
    }


@mcp.tool()
def get_radius_demographics(address: str, radii_miles: str = "1,3,5") -> dict:
    """
    Get aggregated Census demographics for radius rings around a US property address —
    the standard 1/3/5-mile trade-area format used in CRE site analysis.
    Aggregates every census tract whose centroid falls within each radius:
    population, household-weighted median income, employment rate, college attainment,
    housing vacancy, renter share, and median rent.

    Use this for trade-area / site analysis. Use get_market_demographics for the
    single census tract immediately around the address.

    Args:
        address:     Full US property address (e.g. "1234 Main St, Charlotte, NC 28202")
        radii_miles: Comma-separated radii in miles (default "1,3,5", each capped at 15)
    """
    census_key = os.getenv("CENSUS_API_KEY", "")
    if not census_key:
        return {"error": "CENSUS_API_KEY not set. Free key at api.census.gov/data/key_signup.html"}

    try:
        radii = sorted({min(float(r.strip()), 15.0) for r in radii_miles.split(",") if float(r.strip()) > 0})[:4]
    except Exception:
        return {"error": "radii_miles must be comma-separated numbers, e.g. '1,3,5'"}
    if not radii:
        return {"error": "No valid radii provided."}

    geo = geocode_address(address)
    if "error" in geo:
        return geo

    result = _point_radius_demographics(geo["lat"], geo["lng"], radii, census_key)
    if "error" in result:
        return result
    return {"address_matched": geo["matched_address"], **result}


# ─── Tool 2c: Land Market Screener (for land investing) ──────────────────────

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


@mcp.tool()
def screen_land_market(state: str, county: str) -> dict:
    """
    Screen a US county as a LAND-INVESTING market (raw-land flip / Podolsky style).
    Grades the county on the signals that matter for buying cheap rural land and
    reselling on terms: population growth, demographics, owner share, and affordability.

    IMPORTANT: This screens on FREE Census data only (growth + demographics + a
    home-value affordability proxy). It does NOT include actual land sale prices or
    comps — those require county records or a paid service, and must be verified
    per-parcel before buying. Use this to rank/shortlist markets, not to buy.

    Args:
        state:  2-letter state abbreviation (e.g. "AZ") or 2-digit state FIPS
        county: County name (e.g. "Mohave" or "Mohave County")
    """
    census_key = os.getenv("CENSUS_API_KEY", "")
    if not census_key:
        return {"error": "CENSUS_API_KEY not set."}

    st = state.strip().upper()
    st_fips = STATE_FIPS.get(st) or (st if st.isdigit() and len(st) == 2 else None)
    if not st_fips:
        return {"error": f"Unknown state '{state}'. Use a 2-letter abbreviation like 'AZ'."}

    target = county.strip().lower().replace(" county", "")
    variables = "NAME,B01003_001E,B19013_001E,B25077_001E,B25003_001E,B25003_002E"

    try:
        r = requests.get(
            "https://api.census.gov/data/2023/acs/acs5",
            params={"get": variables, "for": "county:*", "in": f"state:{st_fips}", "key": census_key},
            timeout=20,
        )
        if r.status_code != 200:
            return {"error": f"Census query failed ({r.status_code})."}
        rows = r.json()
        hdr = rows[0]
        match = None
        for row in rows[1:]:
            rec = dict(zip(hdr, row))
            if target in rec["NAME"].lower():
                match = rec
                break
        if not match:
            return {"error": f"County '{county}' not found in {st}. Try just the county name, e.g. 'Mohave'."}

        co_fips = match["county"]

        r19 = requests.get(
            "https://api.census.gov/data/2019/acs/acs5",
            params={"get": "B01003_001E", "for": f"county:{co_fips}", "in": f"state:{st_fips}", "key": census_key},
            timeout=15,
        )
        p19 = int(r19.json()[1][0]) if r19.status_code == 200 else None

        def _i(v):
            try:
                n = int(match.get(v))
                return None if n < 0 else n
            except (TypeError, ValueError):
                return None

        pop = _i("B01003_001E")
        inc = _i("B19013_001E")
        hval = _i("B25077_001E")
        tenure = _i("B25003_001E")
        owners = _i("B25003_002E")
        owner_pct = round(owners / tenure * 100, 1) if owners and tenure else None
        growth = round((pop - p19) / p19 * 100, 1) if pop and p19 else None

        # Transparent screening heuristic (0-100). Growth weighted highest; cheaper = better for 25c buys.
        score = 50.0
        if growth is not None:
            score += max(-15, min(20, growth * 2.5))
        if hval:
            score += max(-12, min(12, (300000 - hval) / 300000 * 12))
        if owner_pct is not None:
            score += max(-8, min(8, (owner_pct - 55) * 0.4))
        if inc:
            score += max(-6, min(6, (inc - 45000) / 45000 * 6))
        score = round(max(0, min(100, score)))
        grade = ("A" if score >= 72 else "A-" if score >= 66 else "B+" if score >= 60
                 else "B" if score >= 52 else "C+" if score >= 44 else "C" if score >= 36 else "D")

        bits = []
        if growth is not None:
            bits.append(f"{'strong' if growth >= 4 else 'modest' if growth >= 1.5 else 'flat/declining'} growth ({growth:+.1f}% 2019-2023)")
        if hval:
            bits.append(f"{'affordable' if hval < 250000 else 'mid-priced' if hval < 400000 else 'expensive'} (median home ${hval:,})")
        if owner_pct is not None:
            bits.append(f"{owner_pct:.0f}% owner-occupied")

        return {
            "county": match["NAME"],
            "fips": f"{st_fips}{co_fips}",
            "land_market_score": score,
            "grade": grade,
            "read": "; ".join(bits) + ".",
            "signals": {
                "population": pop,
                "population_growth_2019_2023_pct": growth,
                "median_household_income": inc,
                "median_home_value_proxy": hval,
                "owner_occupancy_pct": owner_pct,
            },
            "next_step": "Verify cheap parcels actually exist here (LandWatch / Land.com / county records) and run per-parcel due diligence. This screen ranks markets; it does not price land.",
            "source": "US Census Bureau ACS 5-year (2019 & 2023)",
        }
    except Exception as e:
        return {"error": f"Land market screen failed: {str(e)}"}


# ─── Tool 2d: Parcel DD Pre-Screen (land due diligence) ──────────────────────

@mcp.tool()
def screen_parcel_dd(lat: float, lng: float) -> dict:
    """
    Pre-screen a land parcel's location for the AUTOMATABLE due-diligence red flags:
    FEMA flood zone and federal wetlands. Pulls live from FEMA's National Flood Hazard
    Layer and the US Fish & Wildlife National Wetlands Inventory.

    Use this to kill obviously-bad parcels (flood zone, wetlands) at scale BEFORE
    spending time on manual due diligence.

    IMPORTANT: Checks flood + wetlands only. It does NOT check legal ACCESS
    (landlocked — the #1 land deal-killer), title/liens, or zoning — those stay MANUAL,
    per-parcel checks via county records. A clean screen here is necessary, NOT sufficient.

    Args:
        lat: Parcel latitude (decimal degrees)
        lng: Parcel longitude (decimal degrees)
    """
    flags = []
    out = {"coordinates": {"lat": lat, "lng": lng}}

    # ── FEMA flood zone ──
    try:
        r = requests.get(
            "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query",
            params={"geometry": f"{lng},{lat}", "geometryType": "esriGeometryPoint", "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects", "outFields": "FLD_ZONE,ZONE_SUBTY",
                    "returnGeometry": "false", "f": "json"},
            timeout=15,
        )
        feats = r.json().get("features", [])
        if feats:
            z = feats[0]["attributes"].get("FLD_ZONE")
            sub = feats[0]["attributes"].get("ZONE_SUBTY")
            high = z in ("A", "AE", "AH", "AO", "AR", "A99", "V", "VE")
            out["flood"] = {"zone": z, "detail": sub, "risk": "HIGH" if high else "low/minimal"}
            if high:
                flags.append(f"FLOOD — Zone {z} (high-risk floodplain): tanks value and buildability")
        else:
            out["flood"] = {"zone": "X (likely)", "detail": "No mapped flood hazard at this point", "risk": "low/minimal"}
    except Exception as e:
        out["flood"] = {"error": f"FEMA check failed: {str(e)}"}

    # ── USFWS National Wetlands Inventory ──
    try:
        r = requests.get(
            "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query",
            params={"geometry": f"{lng},{lat}", "geometryType": "esriGeometryPoint", "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
                    "returnGeometry": "false", "f": "json"},
            timeout=15,
        )
        feats = r.json().get("features", [])
        if feats:
            attrs = feats[0]["attributes"]
            wt = next((v for k, v in attrs.items() if k.endswith("WETLAND_TYPE")), "Wetland")
            out["wetlands"] = {"present": True, "type": wt}
            flags.append(f"WETLAND — '{wt}' mapped on the parcel: likely unbuildable/unusable")
        else:
            out["wetlands"] = {"present": False, "type": None}
    except Exception as e:
        out["wetlands"] = {"error": f"Wetlands check failed: {str(e)}"}

    out["red_flags"] = flags
    errored = [k for k in ("flood", "wetlands") if "error" in out.get(k, {})]
    if errored:
        out["verdict"] = (f"⚠️ {' & '.join(errored)} check FAILED — do NOT treat as clean, recheck manually"
                          + (" (plus red flags found below)" if flags else ""))
    elif flags:
        out["verdict"] = "RED FLAGS — investigate hard or pass"
    else:
        out["verdict"] = "No flood/wetland red flags — but still verify access, title & zoning manually"
    out["manual_dd_still_required"] = [
        "Legal ACCESS — is it landlocked? (#1 deal-killer — check county GIS / plat map)",
        "Title & liens — county recorder + title search",
        "Zoning & allowed use — county zoning department",
        "Exact back taxes owed — county treasurer",
    ]
    out["source"] = "FEMA NFHL + US Fish & Wildlife National Wetlands Inventory (free public data)"
    return out


# ─── Tool 3: Inflation & Rent Growth Data ────────────────────────────────────

@mcp.tool()
def get_inflation_data() -> dict:
    """
    Get current CPI and rent inflation data from the Federal Reserve.
    Returns overall inflation, shelter inflation, and rent-specific CPI with YoY changes.
    Use this to calibrate rent growth assumptions in your DCF model — don't guess.
    """
    series_map = {
        "cpi_all_items": ("CPIAUCSL", "All Items CPI"),
        "cpi_shelter":   ("CUSR0000SAH1", "Shelter CPI (housing costs)"),
        "cpi_rent":      ("CUSR0000SEHA", "Rent of Primary Residence CPI"),
    }

    results = {}
    for key, (sid, label) in series_map.items():
        data = get_fred_series(sid, limit=14)  # 13 months for YoY
        if "observations" in data:
            obs = [o for o in data["observations"] if o["value"] is not None]
            if len(obs) >= 2:
                latest   = obs[0]["value"]
                year_ago = obs[min(12, len(obs) - 1)]["value"]
                yoy = round((latest - year_ago) / year_ago * 100, 2) if year_ago else None
                results[key] = {
                    "label":        label,
                    "current_index": latest,
                    "date":         obs[0]["date"],
                    "yoy_change_pct": yoy
                }
        else:
            results[key] = {"label": label, **data}

    # DCF modeling guidance
    rent_yoy = results.get("cpi_rent", {}).get("yoy_change_pct")
    if rent_yoy:
        conservative = max(2.0, round(rent_yoy * 0.75, 1))
        base = max(2.5, round(rent_yoy * 0.90, 1))
        results["dcf_rent_growth_guidance"] = {
            "conservative_assumption": f"{conservative}%",
            "base_case_assumption":    f"{base}%",
            "trailing_rent_cpi_yoy":   f"{rent_yoy}%",
            "note": "Conservative = 75% of trailing rent CPI. Verify with local market comps."
        }

    return results


# ─── Tool 4: CRE Market Data ──────────────────────────────────────────────────

@mcp.tool()
def get_cre_market_data() -> dict:
    """
    Get Commercial Real Estate price index and broader market data from the Federal Reserve.
    Returns CRE price trends, office/retail/industrial vacancy proxies, and credit spreads.
    Provides macro context for deal underwriting and cap rate analysis.
    """
    series_map = {
        "cre_price_index":       ("BOGZ1FL075035503Q", "CRE Price Index (Fed Flow of Funds)"),
        "cre_loans_outstanding": ("CREACBM027NBOG", "CRE Loans Outstanding, All Commercial Banks"),
        "commercial_industrial_loans": ("BUSLOANS", "Commercial & Industrial Loans Outstanding"),
        "cre_delinquency":       ("DRCRELEXFACBS", "Delinquency Rate on CRE Loans (Excl. Farmland)"),
        "credit_spread_bbb":     ("BAMLC0A4CBBB", "BBB Corporate Bond Spread (credit proxy)"),
    }

    results = {}
    for key, (sid, label) in series_map.items():
        data = get_fred_series(sid, limit=5)
        if "observations" in data:
            obs = [o for o in data["observations"] if o["value"] is not None]
            results[key] = {
                "label":  label,
                "latest": obs[0] if obs else None,
                "trend":  obs[:4] if len(obs) >= 4 else obs
            }
        else:
            results[key] = {"label": label, **data}

    results["data_source"] = "Federal Reserve FRED — updated per respective release schedules"
    results["note"] = "CRE price index is quarterly. Credit spreads are daily. Loan data is monthly."
    return results


# ─── Tool 5: Analyze Rent Roll ────────────────────────────────────────────────

@mcp.tool()
def analyze_rent_roll(text: str, property_name: Optional[str] = None) -> dict:
    """
    Extract structured tenant and lease data from a rent roll document.
    Paste the text content of your rent roll PDF here (copy-paste from PDF reader).
    Returns tenant list, suite/SF, lease dates, monthly rent, escalations, and options.

    Args:
        text: Raw text copied from a rent roll PDF
        property_name: Optional property name for context
    """
    instructions = """You are a CRE lease abstraction specialist. Extract all tenant data from this rent roll.

Return ONLY a valid JSON object with this structure:
{
  "property": "property name or null",
  "as_of_date": "date shown on rent roll or null",
  "total_sf": total rentable SF as number or null,
  "total_tenants": number of tenants (excluding vacant),
  "occupied_sf": occupied SF as number,
  "occupancy_rate_pct": occupancy percentage as number,
  "annual_base_rent": total annual base rent as number,
  "average_rent_psf": annual rent per SF as number,
  "tenants": [
    {
      "name": "tenant name",
      "suite": "suite/unit identifier",
      "sf": square footage as number,
      "lease_start": "YYYY-MM or YYYY-MM-DD",
      "lease_end": "YYYY-MM or YYYY-MM-DD",
      "months_remaining": months from the provided current_date to lease end as number,
      "monthly_rent": number,
      "annual_rent": number,
      "rent_psf_annual": number,
      "lease_type": "NNN/Gross/Modified Gross/Full Service/Other",
      "escalations": "description or null",
      "options": "renewal options description or null",
      "is_vacant": false
    }
  ],
  "vacant_units": [
    {"suite": "suite id", "sf": number}
  ],
  "lease_expiration_schedule": {
    "within_12mo": {"count": X, "sf": Y},
    "within_24mo": {"count": X, "sf": Y},
    "within_36mo": {"count": X, "sf": Y}
  }
}
Return ONLY the JSON. No explanation."""

    data = {
        "rent_roll_text": text[:14000],
        "current_date": date.today().strftime("%B %Y")
    }
    if property_name:
        data["property_name"] = property_name

    raw = ask_claude(data, instructions, max_tokens=3500)
    try:
        return parse_json_from_claude(raw)
    except Exception:
        return {"raw_extraction": raw, "parse_error": "Could not auto-parse — review raw_extraction"}


# ─── Tool 6: Abstract Lease ───────────────────────────────────────────────────

@mcp.tool()
def abstract_lease(text: str) -> dict:
    """
    Extract all key terms from a commercial lease document.
    Returns term, base rent schedule, escalations, TI allowance, CAM structure,
    renewal options, termination rights, exclusivity, co-tenancy, and red flags.

    Args:
        text: Raw text copied from a commercial lease PDF
    """
    instructions = """You are a senior CRE attorney reviewing a commercial lease. Extract all key terms.

Return ONLY a valid JSON object:
{
  "tenant": "name",
  "landlord": "name",
  "property_address": "address",
  "suite": "suite/space identifier",
  "rentable_sf": number or null,
  "lease_type": "NNN/Gross/Modified Gross/Full Service",
  "term": {
    "commencement": "date",
    "expiration": "date",
    "total_months": number
  },
  "base_rent": {
    "initial_monthly": number,
    "initial_annual": number,
    "initial_psf_annual": number,
    "schedule": [{"period": "Year 1", "monthly": X, "annual": Y, "psf_annual": Z}]
  },
  "escalations": {
    "type": "Fixed/CPI/Greater of Fixed or CPI/Other",
    "rate_or_formula": "e.g. 3% per annum",
    "detail": "full description"
  },
  "expenses": {
    "cam_responsibility": "description",
    "cam_cap": "cap if any or null",
    "real_estate_taxes": "description",
    "insurance": "description",
    "utilities": "description",
    "management_fee_included": true/false
  },
  "tenant_improvements": {
    "ti_allowance": "dollar amount or null",
    "delivery_condition": "as-is/warm shell/cold dark/white box/turnkey",
    "construction_allowance_detail": "description or null"
  },
  "options": {
    "renewal_options": "description or null",
    "expansion_rights": "description or null",
    "termination_rights": "description or null",
    "purchase_option": "description or null",
    "right_of_first_refusal": "description or null"
  },
  "other_key_provisions": {
    "exclusivity_clause": "description or null",
    "co_tenancy_clause": "description or null",
    "assignment_subletting": "description",
    "personal_guarantee": "description or null",
    "subordination_nondisturbance": "SNDA status",
    "estoppel_requirement": "description or null"
  },
  "red_flags": ["list concerning provisions"],
  "landlord_favorable_provisions": ["list"],
  "tenant_favorable_provisions": ["list"],
  "plain_english_summary": "2-3 sentence summary"
}
Return ONLY the JSON."""

    raw = ask_claude({"lease_text": text[:16000]}, instructions, max_tokens=3500)
    try:
        return parse_json_from_claude(raw)
    except Exception:
        return {"raw_extraction": raw, "parse_error": "Could not auto-parse — review raw_extraction"}


# ─── Tool 7: Flag Lease Risks ─────────────────────────────────────────────────

@mcp.tool()
def flag_lease_risks(rent_roll_json: str) -> dict:
    """
    Analyze a parsed rent roll for investment risks.
    Feed the output from analyze_rent_roll directly into this tool.
    Returns: rollover risk, tenant concentration, credit risk, and actionable recommendations.

    Args:
        rent_roll_json: JSON string from the analyze_rent_roll tool output
    """
    try:
        data = json.loads(rent_roll_json) if isinstance(rent_roll_json, str) else rent_roll_json
    except Exception:
        data = {"raw": rent_roll_json}

    instructions = """You are a CRE investment risk analyst. Analyze this rent roll data for risks.

Return ONLY a valid JSON object:
{
  "overall_risk": "Low/Medium/High/Critical",
  "rollover_risk": {
    "score": "Low/Medium/High",
    "tenants_expiring_12mo": [{"name": "", "sf": X, "annual_rent": Y}],
    "tenants_expiring_24mo": [{"name": "", "sf": X, "annual_rent": Y}],
    "weighted_avg_lease_term_years": number,
    "pct_revenue_expiring_24mo": number,
    "commentary": "analysis"
  },
  "concentration_risk": {
    "score": "Low/Medium/High",
    "top_tenant": {"name": "", "pct_of_revenue": X, "pct_of_sf": Y},
    "top_3_tenants_pct_revenue": number,
    "herfindahl_index": number,
    "commentary": "analysis"
  },
  "credit_risk": {
    "score": "Low/Medium/High",
    "tenants_of_concern": ["names of any questionable credits"],
    "commentary": "analysis"
  },
  "vacancy_analysis": {
    "current_vacancy_pct": number,
    "vacant_units": [],
    "commentary": "analysis"
  },
  "rent_analysis": {
    "avg_rent_psf": number,
    "rent_range": {"min": X, "max": Y},
    "commentary": "Note any outliers — verify against market comps separately"
  },
  "top_risks": ["ranked list, most critical first"],
  "recommendations": ["actionable items for buyer"],
  "due_diligence_checklist": ["items to verify before closing"]
}
Return ONLY the JSON."""

    raw = ask_claude(data, instructions, max_tokens=2500)
    try:
        return parse_json_from_claude(raw)
    except Exception:
        return {"raw_analysis": raw, "parse_error": "Could not auto-parse — review raw_analysis"}


# ─── Tool 8: DCF Model ────────────────────────────────────────────────────────

@mcp.tool()
def build_dcf_model(
    noi_year1: float,
    purchase_price: float,
    hold_years: int = 10,
    noi_growth_rate: float = 3.0,
    exit_cap_rate: Optional[float] = None,
    equity_pct: float = 35.0,
    loan_rate: Optional[float] = None,
    amortization_years: int = 30,
) -> dict:
    """
    Build a levered DCF model using live Federal Reserve rates.
    Automatically fetches current SOFR to derive the loan rate if not provided.
    Returns: annual cash flows, IRR, equity multiple, cash-on-cash, DSCR, and exit analysis.

    Args:
        noi_year1:          Year 1 Net Operating Income ($)
        purchase_price:     Acquisition price ($)
        hold_years:         Hold period in years (default 10)
        noi_growth_rate:    Annual NOI growth rate % (default 3.0)
        exit_cap_rate:      Exit cap rate % — if None, uses entry cap + 25bps (conservative)
        equity_pct:         Equity as % of purchase price (default 35%)
        loan_rate:          Loan interest rate % — if None, fetches live SOFR + 175bps
        amortization_years: Loan amortization period (default 30 years)
    """
    # ── Pull live rates if loan_rate not provided ──────────────────────────────
    live_rates_note = {}
    if loan_rate is None:
        rates = get_current_rates()
        sofr = rates.get("sofr", {}).get("rate_pct")
        t10  = rates.get("treasury_10yr", {}).get("rate_pct")
        if sofr:
            loan_rate = round(sofr + 1.75, 2)
            live_rates_note = {"source": "FRED SOFR", "sofr": sofr, "spread_bps": 175, "derived_rate": loan_rate}
        elif t10:
            loan_rate = round(t10 + 1.5, 2)
            live_rates_note = {"source": "FRED 10yr Treasury", "t10": t10, "spread_bps": 150, "derived_rate": loan_rate}
        else:
            loan_rate = 6.5
            live_rates_note = {"source": "Fallback (FRED unavailable)", "rate": loan_rate}

    # ── Core inputs ────────────────────────────────────────────────────────────
    entry_cap = (noi_year1 / purchase_price) * 100
    if exit_cap_rate is None:
        exit_cap_rate = round(entry_cap + 0.25, 2)  # Conservative: slight cap expansion

    equity       = purchase_price * (equity_pct / 100)
    loan_amount  = purchase_price - equity
    ltv          = (loan_amount / purchase_price) * 100

    # ── Mortgage math ──────────────────────────────────────────────────────────
    monthly_r = loan_rate / 100 / 12
    n_pmts    = amortization_years * 12
    if monthly_r > 0:
        monthly_pmt = loan_amount * (monthly_r * (1 + monthly_r)**n_pmts) / ((1 + monthly_r)**n_pmts - 1)
    else:
        monthly_pmt = loan_amount / n_pmts
    annual_ds = monthly_pmt * 12

    def loan_balance(years):
        pmts_made = years * 12
        if monthly_r > 0:
            bal = loan_amount * ((1 + monthly_r)**n_pmts - (1 + monthly_r)**pmts_made) / ((1 + monthly_r)**n_pmts - 1)
        else:
            bal = loan_amount * (1 - pmts_made / n_pmts)
        return max(0, bal)

    # ── Annual cash flows ──────────────────────────────────────────────────────
    cash_flows = []
    for yr in range(1, hold_years + 1):
        noi   = noi_year1 * (1 + noi_growth_rate / 100) ** (yr - 1)
        cfads = noi - annual_ds
        dscr  = noi / annual_ds if annual_ds > 0 else None
        coc   = cfads / equity * 100 if equity > 0 else None
        cash_flows.append({
            "year":         yr,
            "noi":          round(noi),
            "debt_service": round(annual_ds),
            "cash_flow":    round(cfads),
            "dscr":         round(dscr, 2) if dscr else None,
            "coc_pct":      round(coc, 2) if coc else None,
        })

    # ── Exit / reversion ───────────────────────────────────────────────────────
    noi_exit      = noi_year1 * (1 + noi_growth_rate / 100) ** hold_years
    exit_value    = noi_exit / (exit_cap_rate / 100)
    loan_payoff   = loan_balance(hold_years)
    net_proceeds  = exit_value - loan_payoff

    # ── IRR (bisection method) ─────────────────────────────────────────────────
    equity_flows = [-equity] + [f["cash_flow"] for f in cash_flows]
    equity_flows[-1] += net_proceeds

    def npv(r, flows):
        return sum(f / (1 + r) ** i for i, f in enumerate(flows))

    def calc_irr(flows):
        lo, hi = -0.99, 10.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if npv(mid, flows) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
        return (lo + hi) / 2

    irr_val = calc_irr(equity_flows) * 100
    em      = (sum(f["cash_flow"] for f in cash_flows) + net_proceeds) / equity
    avg_coc = sum(f["coc_pct"] for f in cash_flows) / len(cash_flows)

    return {
        "inputs": {
            "purchase_price":    f"${purchase_price:,.0f}",
            "noi_year1":         f"${noi_year1:,.0f}",
            "entry_cap_rate":    f"{entry_cap:.2f}%",
            "exit_cap_rate":     f"{exit_cap_rate:.2f}%",
            "equity":            f"${equity:,.0f}",
            "loan_amount":       f"${loan_amount:,.0f}",
            "ltv":               f"{ltv:.1f}%",
            "loan_rate":         f"{loan_rate:.2f}%",
            "hold_years":        hold_years,
            "noi_growth_rate":   f"{noi_growth_rate:.1f}%/yr",
        },
        "live_rate_used": live_rates_note,
        "returns": {
            "irr":              f"{irr_val:.1f}%",
            "equity_multiple":  f"{em:.2f}x",
            "avg_coc":          f"{avg_coc:.1f}%",
            "year1_coc":        f"{cash_flows[0]['coc_pct']:.1f}%",
            "year1_dscr":       cash_flows[0]["dscr"],
        },
        "exit_analysis": {
            "year": hold_years,
            "exit_noi":       f"${noi_exit:,.0f}",
            "exit_value":     f"${exit_value:,.0f}",
            "loan_payoff":    f"${loan_payoff:,.0f}",
            "net_proceeds":   f"${net_proceeds:,.0f}",
        },
        "annual_cash_flows": cash_flows,
        "loan_summary": {
            "amount":           f"${loan_amount:,.0f}",
            "rate":             f"{loan_rate:.2f}%",
            "annual_ds":        f"${annual_ds:,.0f}",
            "amortization_yrs": amortization_years,
        }
    }


# ─── Tool 9: Generate Deal Memo ───────────────────────────────────────────────

@mcp.tool()
def generate_deal_memo(
    property_address: str,
    property_type: str,
    noi: float,
    asking_price: float,
    rent_roll_summary: Optional[str] = None,
    additional_context: Optional[str] = None
) -> str:
    """
    Generate a formatted CRE acquisition memo / Investment Committee memo.
    Automatically pulls live rates from FRED and demographics from Census Bureau
    to provide real market context — not guesses.

    Args:
        property_address:   Full property address
        property_type:      Multifamily / Office / Retail / Industrial / Mixed-Use
        noi:                Net Operating Income ($)
        asking_price:       Asking price ($)
        rent_roll_summary:  Optional: paste output from analyze_rent_roll or flag_lease_risks
        additional_context: Any additional deal notes, seller info, market color
    """
    # Pull live data
    rates = get_current_rates()
    demo  = get_market_demographics(property_address)
    dcf   = build_dcf_model(noi_year1=noi, purchase_price=asking_price)

    deal_data = {
        "property_address":  property_address,
        "property_type":     property_type,
        "asking_price":      asking_price,
        "noi":               noi,
        "cap_rate_pct":      round(noi / asking_price * 100, 2),
        "live_rates":        rates,
        "market_demographics": demo,
        "dcf_returns":       dcf.get("returns"),
        "dcf_exit":          dcf.get("exit_analysis"),
        "dcf_inputs":        dcf.get("inputs"),
        "rent_roll_summary": rent_roll_summary,
        "additional_context": additional_context,
        "memo_date":         date.today().strftime("%B %d, %Y")
    }

    instructions = """Generate a professional CRE Investment Committee (IC) memo in markdown format.

Use ONLY the data provided — do not invent numbers or make up market statistics.
Reference actual figures from the live_rates and market_demographics sections.

Structure:
# [Property Address] — Acquisition Memo
**Date:** [use memo_date from data] | **Type:** [property_type] | **Status:** For Review

---

## Executive Summary
2-3 sentences on the deal thesis and whether this is a GO/NO-GO/NEEDS MORE INFO.

## Deal Snapshot
| Metric | Value |
|--------|-------|
(table: asking price, NOI, cap rate, IRR, equity multiple, Year 1 DSCR, Year 1 CoC)

## Market Context
Using the LIVE data provided (FRED rates + Census demographics), discuss:
- Current rate environment and cost of capital for this deal
- Local demographics: income, employment, vacancy
- Where entry cap rate sits relative to current treasury spreads

## Financial Returns
Summarize DCF results. Comment on IRR vs. a reasonable hurdle rate.
Discuss DSCR adequacy and debt coverage cushion.

## Risk Factors
Bulleted list of 3-5 specific risks for this deal.

## Recommendation
**GO / NO-GO / NEEDS MORE INFO** — one clear sentence with rationale.

Keep it tight and professional. Every statistic must come from the provided data."""

    return ask_claude(deal_data, instructions, max_tokens=2500)


# ─── Tool 11: Export Excel Underwriting Model ────────────────────────────────

_DOWNLOADS: dict = {}  # file_id -> (path, expires_at)


def _generate_dcf_workbook(path: str, inputs: dict, rates: dict, sens: dict, rings_full: Optional[dict], property_name: str) -> None:
    """Write a formula-driven .xlsx underwriting model (editable assumptions, live PMT/FV/IRR)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Model"

    TITLE = Font(bold=True, size=14, color="1D4ED8")
    H = Font(bold=True, size=11)
    HDR_FILL = PatternFill("solid", fgColor="1E293B")
    HDR_FONT = Font(bold=True, color="FFFFFF", size=10)
    INPUT = Font(color="2563EB", bold=True)  # blue = editable input (industry convention)
    MUTED = Font(color="64748B", size=9)
    THIN = Border(bottom=Side(style="thin", color="CBD5E1"))
    MONEY = '#,##0'
    PCT = '0.00%'

    hold = inputs["hold_years"]

    ws["A1"] = f"CRE INTELLIGENCE — DCF UNDERWRITING MODEL"
    ws["A1"].font = TITLE
    ws["A2"] = f"{property_name} · Generated {date.today().strftime('%B %d, %Y')} · Live Fed rates as of {rates.get('sofr', {}).get('date', 'n/a')} · cre-intelligence-mcp.vercel.app"
    ws["A2"].font = MUTED

    # ── Assumptions (blue cells are editable inputs) ──
    ws["A4"] = "ASSUMPTIONS  (blue cells are inputs — edit freely)"
    ws["A4"].font = H
    rows = [
        ("Purchase Price ($)", inputs["price"], MONEY),
        ("Year 1 NOI ($)", inputs["noi"], MONEY),
        ("NOI Growth (annual)", inputs["growth"] / 100, PCT),
        ("Hold Period (years)", hold, '0'),
        ("Exit Cap Rate", inputs["exit_cap"] / 100, PCT),
        ("Equity (% of price)", inputs["equity_pct"] / 100, PCT),
        ("Loan Interest Rate", inputs["loan_rate"] / 100, PCT),
        ("Amortization (years)", inputs["amort"], '0'),
    ]
    for i, (label, val, fmt) in enumerate(rows, start=5):
        ws[f"A{i}"] = label
        ws[f"B{i}"] = val
        ws[f"B{i}"].font = INPUT
        ws[f"B{i}"].number_format = fmt

    # ── Derived ──
    ws["A14"] = "DERIVED"
    ws["A14"].font = H
    derived = [
        ("Entry Cap Rate", "=B6/B5", PCT),
        ("Equity ($)", "=B5*B10", MONEY),
        ("Loan Amount ($)", "=B5-B16", MONEY),
        ("Monthly Payment ($)", "=PMT(B11/12,B12*12,-B17)", MONEY),
        ("Annual Debt Service ($)", "=B18*12", MONEY),
    ]
    for i, (label, formula, fmt) in enumerate(derived, start=15):
        ws[f"A{i}"] = label
        ws[f"B{i}"] = formula
        ws[f"B{i}"].number_format = fmt

    # ── Pro forma ──
    hdr_row = 21
    ws[f"A{hdr_row - 1}"] = "ANNUAL CASH FLOWS"
    ws[f"A{hdr_row - 1}"].font = H
    headers = ["Year", "NOI", "Debt Service", "Cash Flow", "DSCR", "Cash-on-Cash", "Loan Balance", "Equity CF"]
    for c, name in enumerate(headers, start=1):
        cell = ws.cell(row=hdr_row, column=c, value=name)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center")

    yr0 = hdr_row + 1
    ws.cell(row=yr0, column=1, value=0)
    ws.cell(row=yr0, column=8, value="=-B16").number_format = MONEY

    first = yr0 + 1
    last = yr0 + hold
    for k in range(1, hold + 1):
        r = yr0 + k
        ws.cell(row=r, column=1, value=k)
        noi_cell = ws.cell(row=r, column=2)
        noi_cell.value = "=B6" if k == 1 else f"=B{r - 1}*(1+$B$7)"
        ws.cell(row=r, column=3, value="=$B$19")
        ws.cell(row=r, column=4, value=f"=B{r}-C{r}")
        ws.cell(row=r, column=5, value=f"=B{r}/C{r}").number_format = '0.00"x"'
        ws.cell(row=r, column=6, value=f"=D{r}/$B$16").number_format = PCT
        ws.cell(row=r, column=7, value=f"=FV($B$11/12,A{r}*12,$B$18,-$B$17)")
        eq = ws.cell(row=r, column=8)
        eq.value = f"=D{r}" if k < hold else f"=D{r}+$B${last + 6}"
        for col, fmt in ((2, MONEY), (3, MONEY), (4, MONEY), (7, MONEY), (8, MONEY)):
            ws.cell(row=r, column=col).number_format = fmt
        for c in range(1, 9):
            ws.cell(row=r, column=c).border = THIN

    # ── Exit ──
    ex = last + 3
    ws[f"A{ex - 1}"] = "EXIT / REVERSION"
    ws[f"A{ex - 1}"].font = H
    exit_rows = [
        (f"Exit NOI (Year {hold + 1})", f"=B{last}*(1+$B$7)", MONEY),
        ("Gross Sale Value", f"=B{ex}/$B$9", MONEY),
        ("Loan Payoff", f"=G{last}", MONEY),
        ("Net Sale Proceeds", f"=B{ex + 1}-B{ex + 2}", MONEY),
    ]
    for i, (label, formula, fmt) in enumerate(exit_rows):
        ws[f"A{ex + i}"] = label
        ws[f"B{ex + i}"] = formula
        ws[f"B{ex + i}"].number_format = fmt

    # ── Returns ──
    rt = ex + 6
    ws[f"A{rt - 1}"] = "RETURNS"
    ws[f"A{rt - 1}"].font = H
    ret_rows = [
        ("IRR (levered)", f"=IRR(H{yr0}:H{last})", PCT),
        ("Equity Multiple", f"=SUM(H{first}:H{last})/B16", '0.00"x"'),
        ("Average Cash-on-Cash", f"=AVERAGE(F{first}:F{last})", PCT),
        ("Year 1 DSCR", f"=E{first}", '0.00"x"'),
    ]
    for i, (label, formula, fmt) in enumerate(ret_rows):
        ws[f"A{rt + i}"] = label
        cell = ws[f"B{rt + i}"]
        cell.value = formula
        cell.number_format = fmt
        cell.font = Font(bold=True, color="16A34A")

    ws.column_dimensions["A"].width = 26
    for c in range(2, 9):
        ws.column_dimensions[get_column_letter(c)].width = 14

    # ── Sensitivity sheet (values computed at generation) ──
    s = wb.create_sheet("Sensitivity")
    s["A1"] = "IRR SENSITIVITY — exit cap rate × NOI growth"
    s["A1"].font = TITLE
    s["A2"] = "Computed at generation from the Model assumptions. Edit the Model sheet for live what-ifs."
    s["A2"].font = MUTED
    s["A4"] = "Exit Cap ↓ / Growth →"
    s["A4"].font = HDR_FONT
    s["A4"].fill = HDR_FILL
    for j, g in enumerate(sens["noi_growth_pct"]):
        cell = s.cell(row=4, column=2 + j, value=g / 100)
        cell.number_format = PCT
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
    for i, ec in enumerate(sens["exit_cap_pct"]):
        cell = s.cell(row=5 + i, column=1, value=ec / 100)
        cell.number_format = PCT
        cell.font = H
        for j in range(len(sens["noi_growth_pct"])):
            v = sens["irr_grid"][i][j]
            c = s.cell(row=5 + i, column=2 + j, value=(v / 100) if v is not None else None)
            c.number_format = PCT
    s.column_dimensions["A"].width = 22

    # ── Market data sheet ──
    m = wb.create_sheet("Market Data")
    m["A1"] = "LIVE MARKET DATA — Federal Reserve (FRED)"
    m["A1"].font = TITLE
    r = 3
    for key, label in [("sofr", "SOFR"), ("sofr_30day_avg", "SOFR 30-day avg"), ("treasury_10yr", "10yr Treasury"),
                       ("treasury_5yr", "5yr Treasury"), ("treasury_2yr", "2yr Treasury"), ("fed_funds_rate", "Fed Funds (daily)")]:
        d = rates.get(key, {})
        if isinstance(d, dict) and d.get("rate_pct") is not None:
            m[f"A{r}"] = label
            m[f"B{r}"] = d["rate_pct"] / 100
            m[f"B{r}"].number_format = PCT
            m[f"C{r}"] = f"as of {d.get('date')}"
            m[f"C{r}"].font = MUTED
            r += 1
    if rings_full and rings_full.get("rings"):
        r += 2
        m[f"A{r}"] = "TRADE-AREA DEMOGRAPHICS — US Census ACS " + str(rings_full.get("acs_vintage", ""))
        m[f"A{r}"].font = H
        r += 1
        for c, name in enumerate(["Ring", "Population", "Median HHI", "Renter %", "Median Rent", "Tracts"], start=1):
            cell = m.cell(row=r, column=c, value=name)
            cell.fill = HDR_FILL
            cell.font = HDR_FONT
        for ring_name, ring in rings_full["rings"].items():
            r += 1
            m.cell(row=r, column=1, value=ring_name.replace("_", " "))
            m.cell(row=r, column=2, value=ring.get("population")).number_format = MONEY
            m.cell(row=r, column=3, value=ring.get("median_household_income")).number_format = MONEY
            rs = (ring.get("housing") or {}).get("renter_share_pct")
            m.cell(row=r, column=4, value=(rs / 100) if rs is not None else None).number_format = PCT
            m.cell(row=r, column=5, value=ring.get("median_gross_rent")).number_format = MONEY
            m.cell(row=r, column=6, value=ring.get("tract_count"))
    m.column_dimensions["A"].width = 24
    for c in range(2, 7):
        m.column_dimensions[get_column_letter(c)].width = 14

    wb.save(path)


@mcp.tool()
def export_dcf_excel(
    noi_year1: float,
    purchase_price: float,
    address: Optional[str] = None,
    property_name: Optional[str] = None,
    hold_years: int = 10,
    noi_growth_rate: float = 3.0,
    exit_cap_rate: Optional[float] = None,
    equity_pct: float = 35.0,
    loan_rate: Optional[float] = None,
    amortization_years: int = 30,
) -> dict:
    """
    Generate a downloadable Excel (.xlsx) underwriting model with LIVE formulas —
    editable assumptions, PMT/FV amortization, IRR, equity multiple, a sensitivity
    grid, live Fed rates, and (if an address is given) Census trade-area demographics.
    Returns a download link valid for 60 minutes.

    Args:
        noi_year1:          Year 1 Net Operating Income ($)
        purchase_price:     Acquisition price ($)
        address:            Optional property address — adds a demographics sheet
        property_name:      Optional label for the model header
        hold_years:         Hold period (default 10)
        noi_growth_rate:    Annual NOI growth % (default 3.0)
        exit_cap_rate:      Exit cap % — default entry cap + 25bps
        equity_pct:         Equity as % of price (default 35)
        loan_rate:          Loan rate % — default live SOFR + 175bps
        amortization_years: Amortization (default 30)
    """
    import uuid

    rates = _cached("rates", 3600, get_current_rates)
    if loan_rate is None:
        sofr = rates.get("sofr", {}).get("rate_pct")
        t10 = rates.get("treasury_10yr", {}).get("rate_pct")
        loan_rate = round(sofr + 1.75, 2) if sofr else (round(t10 + 1.5, 2) if t10 else 6.5)

    entry_cap = noi_year1 / purchase_price * 100
    if exit_cap_rate is None:
        exit_cap_rate = round(entry_cap + 0.25, 2)

    # Sensitivity grid (reuses the DCF engine)
    growth_steps = [2.0, 2.5, 3.0, 3.5, 4.0]
    exit_steps = [round(entry_cap + d, 2) for d in (-0.25, 0.0, 0.25, 0.5, 0.75)]
    irr_grid = []
    for ec in exit_steps:
        row = []
        for g in growth_steps:
            d = build_dcf_model(noi_year1=noi_year1, purchase_price=purchase_price, loan_rate=loan_rate,
                                exit_cap_rate=ec, noi_growth_rate=g, hold_years=hold_years,
                                equity_pct=equity_pct, amortization_years=amortization_years)
            row.append(_pct(d["returns"]["irr"]))
        irr_grid.append(row)
    sens = {"noi_growth_pct": growth_steps, "exit_cap_pct": exit_steps, "irr_grid": irr_grid}

    rings_full = None
    display = property_name or "Untitled Property"
    if address:
        rings_full = _cached(f"rings:{address.lower()}", 86400, lambda: get_radius_demographics(address))
        if "error" in rings_full:
            rings_full = None
        elif not property_name:
            display = rings_full.get("address_matched", address)

    file_id = uuid.uuid4().hex[:12]
    path = f"/tmp/cre_model_{file_id}.xlsx"
    _generate_dcf_workbook(
        path,
        {"price": purchase_price, "noi": noi_year1, "growth": noi_growth_rate, "hold_years": hold_years,
         "exit_cap": exit_cap_rate, "equity_pct": equity_pct, "loan_rate": loan_rate, "amort": amortization_years},
        rates, sens, rings_full, display,
    )
    _DOWNLOADS[file_id] = (path, time.time() + 3600)

    base = os.getenv("PUBLIC_BASE_URL", "https://cre-intelligence-mcp.onrender.com")
    headline = build_dcf_model(noi_year1=noi_year1, purchase_price=purchase_price, loan_rate=loan_rate,
                               exit_cap_rate=exit_cap_rate, noi_growth_rate=noi_growth_rate,
                               hold_years=hold_years, equity_pct=equity_pct, amortization_years=amortization_years)

    return {
        "download_url": f"{base}/download/{file_id}",
        "filename": f"cre_model_{file_id}.xlsx",
        "expires_in_minutes": 60,
        "model": {
            "property": display,
            "entry_cap": f"{entry_cap:.2f}%",
            "loan_rate_used": f"{loan_rate:.2f}% (live SOFR + 175bps)" if loan_rate else None,
            "returns_preview": headline["returns"],
        },
        "sheets": ["Model (live formulas — edit blue cells)", "Sensitivity (IRR grid)", "Market Data (live FRED + Census)"],
        "note": "Open in Excel or Google Sheets. All assumptions are editable; IRR/DSCR/amortization recalculate live.",
    }


@mcp.custom_route("/download/{file_id}", methods=["GET"])
async def download_file(request: Request):
    from starlette.responses import FileResponse

    now = time.time()
    for fid, (p, exp) in list(_DOWNLOADS.items()):
        if exp < now:
            _DOWNLOADS.pop(fid, None)
            try:
                os.remove(p)
            except OSError:
                pass

    file_id = request.path_params["file_id"]
    entry = _DOWNLOADS.get(file_id)
    if not entry or not os.path.exists(entry[0]):
        return JSONResponse({"error": "File expired or not found. Generate a fresh model."}, status_code=404)
    return FileResponse(entry[0], filename="cre_underwriting_model.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─── Public demo API (landing page "Try it live") ────────────────────────────
# Free-data only (FRED + Census) — no Claude calls, so zero marginal cost.
# Per-IP rate limiting + short-TTL caching protect the upstream APIs.

_RATE_BUCKET: dict = {}
_API_CACHE: dict = {}
_STATS = {"since": time.time(), "analyze": 0, "analyze_point": 0, "ips": set()}

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _rate_limited(ip: str, limit: int = 8, window_s: int = 60) -> bool:
    now = time.time()
    hits = [t for t in _RATE_BUCKET.get(ip, []) if now - t < window_s]
    if len(hits) >= limit:
        _RATE_BUCKET[ip] = hits
        return True
    hits.append(now)
    _RATE_BUCKET[ip] = hits
    if len(_RATE_BUCKET) > 5000:
        _RATE_BUCKET.clear()  # crude flush; fine for a demo endpoint
    return False


def _cached(key: str, ttl_s: int, fn):
    now = time.time()
    hit = _API_CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    val = fn()
    # Don't cache errors
    if not (isinstance(val, dict) and "error" in val):
        _API_CACHE[key] = (now + ttl_s, val)
    if len(_API_CACHE) > 800:
        for k in [k for k, v in list(_API_CACHE.items()) if v[0] <= now]:
            _API_CACHE.pop(k, None)
    return val


def _pct(s) -> Optional[float]:
    """Parse '13.2%' / '3.11x' style strings back to floats."""
    try:
        return float(str(s).rstrip("%x"))
    except (TypeError, ValueError):
        return None


def _market_grade(rings: dict) -> dict:
    """Rule-of-thumb market quality grade from 3-mile ring demographics vs national medians."""
    r3 = rings.get("3_mile") or rings.get("1_mile") or (next(iter(rings.values())) if rings else {})
    hhi = r3.get("median_household_income")
    emp = r3.get("employment_rate_pct")
    vac = (r3.get("housing") or {}).get("vacancy_rate_pct")
    if not hhi:
        return {"grade": "N/A", "score": None}
    score = 50.0
    score += max(-25, min(25, (hhi / 78000 - 1) * 40))        # income vs ~US median HHI
    if emp is not None:
        score += max(-10, min(10, (emp - 95) * 2))             # employment strength
    if vac is not None:
        score += max(-10, min(10, (8 - vac) * 1.5))            # vacancy (lower is better)
    for grade, floor_ in [("A", 68), ("A-", 62), ("B+", 56), ("B", 50), ("B-", 44), ("C+", 38), ("C", 30)]:
        if score >= floor_:
            return {"grade": grade, "score": round(score)}
    return {"grade": "D", "score": round(score)}


def _reverse_county_name(lat: float, lng: float) -> Optional[str]:
    """Coordinates → 'County, ST' display name via Census reverse geocoder."""
    try:
        r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
            params={
                "x": lng, "y": lat,
                "benchmark": "Public_AR_Census2020",
                "vintage": "Census2020_Census2020",
                "layers": "Counties,States",
                "format": "json",
            },
            timeout=8,
        )
        g = r.json().get("result", {}).get("geographies", {})
        county = (g.get("Counties") or [{}])[0].get("BASENAME")
        state = (g.get("States") or [{}])[0].get("STUSAB")
        if county and state:
            return f"{county} County, {state}"
    except Exception:
        pass
    return None


def _demo_payload(rings_full: dict, display_name: str, noi: float, price: float) -> dict:
    """Shared analysis payload: live rates + DCF + sensitivity + verdict + ring summary."""
    rates = _cached("rates", 3600, get_current_rates)
    sofr = rates.get("sofr", {}).get("rate_pct")
    t10 = rates.get("treasury_10yr", {}).get("rate_pct")
    if sofr:
        loan_rate = round(sofr + 1.75, 2)
    elif t10:
        loan_rate = round(t10 + 1.5, 2)
    else:
        loan_rate = 6.5

    base = build_dcf_model(noi_year1=noi, purchase_price=price, loan_rate=loan_rate)
    entry_cap = noi / price * 100

    # Sensitivity: IRR across exit cap (rows) x NOI growth (cols) — pure math, instant
    growth_steps = [2.0, 2.5, 3.0, 3.5, 4.0]
    exit_steps = [round(entry_cap + d, 2) for d in (-0.25, 0.0, 0.25, 0.5, 0.75)]
    irr_grid = []
    for ec in exit_steps:
        row = []
        for g in growth_steps:
            d = build_dcf_model(noi_year1=noi, purchase_price=price, loan_rate=loan_rate,
                                exit_cap_rate=ec, noi_growth_rate=g)
            row.append(_pct(d["returns"]["irr"]))
        irr_grid.append(row)

    irr = _pct(base["returns"]["irr"])
    dscr = base["returns"]["year1_dscr"]
    if irr is not None and dscr is not None and irr >= 13 and dscr >= 1.25:
        verdict = "GO"
        verdict_note = "Clears a 13% IRR hurdle with adequate debt coverage — pending rent roll and comp diligence."
    elif irr is not None and irr >= 10:
        verdict = "MARGINAL"
        verdict_note = "Returns are workable but thin — pricing or terms need to move."
    else:
        verdict = "NO-GO"
        verdict_note = "Does not pencil at this basis with current market rates."

    grade = _market_grade(rings_full.get("rings", {}))

    ring_summary = {}
    for name, ring in rings_full.get("rings", {}).items():
        ring_summary[name] = {
            "population": ring.get("population"),
            "median_household_income": ring.get("median_household_income"),
            "renter_share_pct": (ring.get("housing") or {}).get("renter_share_pct"),
            "median_gross_rent": ring.get("median_gross_rent"),
            "tract_count": ring.get("tract_count"),
        }

    return {
        "address": display_name,
        "market_grade": grade,
        "market": {
            "sofr_pct": sofr,
            "treasury_10yr_pct": t10,
            "loan_rate_pct": loan_rate,
            "loan_rate_basis": "SOFR + 175bps" if sofr else ("10yr T + 150bps" if t10 else "fallback"),
            "as_of": rates.get("sofr", {}).get("date"),
        },
        "deal": {
            "noi": noi,
            "price": price,
            "entry_cap_pct": round(entry_cap, 2),
            "spread_over_10yr_bps": round((entry_cap - t10) * 100) if t10 else None,
        },
        "returns": base["returns"],
        "demographics": {"acs_vintage": rings_full.get("acs_vintage"), "rings": ring_summary},
        "sensitivity": {
            "noi_growth_pct": growth_steps,
            "exit_cap_pct": exit_steps,
            "irr_grid": irr_grid,
            "base_case": {"exit_cap_pct": exit_steps[2], "noi_growth_pct": 3.0},
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
        "disclaimer": "Rule-of-thumb screen on live FRED/Census data — not investment advice. Full analysis: connect the MCP.",
    }


def _demo_analyze(address: str, noi: float, price: float) -> dict:
    rings_full = _cached(f"rings:{address.lower()}", 86400, lambda: get_radius_demographics(address))
    if "error" in rings_full:
        return {"error": rings_full["error"]}
    return _demo_payload(rings_full, rings_full.get("address_matched", address), noi, price)


def _demo_analyze_point(lat: float, lng: float, noi: float, price: float) -> dict:
    census_key = os.getenv("CENSUS_API_KEY", "")
    if not census_key:
        return {"error": "Server misconfigured: CENSUS_API_KEY not set."}
    key = f"rings:pt:{round(lat, 4)},{round(lng, 4)}"
    rings_full = _cached(key, 86400, lambda: _point_radius_demographics(lat, lng, [1.0, 3.0, 5.0], census_key))
    if "error" in rings_full:
        return {"error": rings_full["error"]}
    name = _cached(f"county:{round(lat, 3)},{round(lng, 3)}", 86400,
                   lambda: _reverse_county_name(lat, lng)) or f"{lat:.4f}, {lng:.4f}"
    return _demo_payload(rings_full, name, noi, price)


@mcp.custom_route("/api/analyze", methods=["GET", "OPTIONS"])
async def api_analyze(request: Request) -> JSONResponse:
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS_HEADERS)

    client_ip = request.headers.get("x-forwarded-for", "")
    if not client_ip and request.client:
        client_ip = request.client.host
    client_ip = client_ip.split(",")[0].strip() or "unknown"

    if _rate_limited(client_ip):
        return JSONResponse(
            {"error": "Rate limit reached (8/min). Connect the MCP in Claude for unlimited access."},
            status_code=429, headers=_CORS_HEADERS,
        )

    address = (request.query_params.get("address") or "").strip()[:200]
    try:
        noi = float(request.query_params.get("noi", ""))
        price = float(request.query_params.get("price", ""))
    except ValueError:
        return JSONResponse({"error": "noi and price must be numbers"}, status_code=400, headers=_CORS_HEADERS)

    if not address or noi <= 0 or price <= 0 or noi >= price:
        return JSONResponse(
            {"error": "Provide a full address, NOI > 0, and price > NOI. Format: '123 Main St, City, ST 12345'"},
            status_code=400, headers=_CORS_HEADERS,
        )

    _STATS["analyze"] += 1
    _STATS["ips"].add(client_ip)
    payload = await anyio.to_thread.run_sync(lambda: _demo_analyze(address, noi, price))
    return JSONResponse(payload, status_code=400 if "error" in payload else 200, headers=_CORS_HEADERS)


@mcp.custom_route("/api/analyze-point", methods=["GET", "OPTIONS"])
async def api_analyze_point(request: Request) -> JSONResponse:
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS_HEADERS)

    client_ip = request.headers.get("x-forwarded-for", "")
    if not client_ip and request.client:
        client_ip = request.client.host
    client_ip = client_ip.split(",")[0].strip() or "unknown"

    if _rate_limited(client_ip):
        return JSONResponse(
            {"error": "Rate limit reached (8/min). Connect the MCP in Claude for unlimited access."},
            status_code=429, headers=_CORS_HEADERS,
        )

    try:
        lat = float(request.query_params.get("lat", ""))
        lng = float(request.query_params.get("lng", ""))
        noi = float(request.query_params.get("noi", ""))
        price = float(request.query_params.get("price", ""))
    except ValueError:
        return JSONResponse({"error": "lat, lng, noi, price must be numbers"}, status_code=400, headers=_CORS_HEADERS)

    if not (17.0 <= lat <= 72.0 and -180.0 <= lng <= -60.0):
        return JSONResponse({"error": "Click somewhere in the United States."}, status_code=400, headers=_CORS_HEADERS)
    if noi <= 0 or price <= 0 or noi >= price:
        return JSONResponse({"error": "NOI must be > 0 and price > NOI."}, status_code=400, headers=_CORS_HEADERS)

    _STATS["analyze_point"] += 1
    _STATS["ips"].add(client_ip)
    payload = await anyio.to_thread.run_sync(lambda: _demo_analyze_point(lat, lng, noi, price))
    return JSONResponse(payload, status_code=400 if "error" in payload else 200, headers=_CORS_HEADERS)


@mcp.custom_route("/api/stats", methods=["GET"])
async def api_stats(request: Request) -> JSONResponse:
    hours = (time.time() - _STATS["since"]) / 3600
    return JSONResponse({
        "since_hours_ago": round(hours, 1),
        "demo_analyses_address": _STATS["analyze"],
        "demo_analyses_map": _STATS["analyze_point"],
        "unique_visitors_api": len(_STATS["ips"]),
        "note": "Counts since last server deploy/restart (in-memory). Page views tracked separately in Vercel Analytics.",
    }, headers=_CORS_HEADERS)


# ─── Health check ─────────────────────────────────────────────────────────────

@mcp.custom_route("/health", methods=["GET", "HEAD"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "CRE Intelligence MCP"})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 0))
    if port:
        # Remote deployment (Railway, Fly.io, etc.) — HTTP transport
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        # Local Claude Desktop — stdio transport
        mcp.run()
