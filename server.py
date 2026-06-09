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
import anthropic
import requests
import json
import math
import os
import re
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
        "fed_funds_rate": "FEDFUNDS",
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

    Args:
        address: Full US property address (e.g. "1234 Main St, Charlotte, NC 28202")
    """
    # Parse address parts
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 3:
        return {"error": "Provide full address: '123 Main St, City, ST 12345'"}

    street = parts[0]
    city = parts[1]
    state_zip = parts[2]

    # Step 1: Geocode via Census Geocoder
    try:
        geo_r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/address",
            params={
                "street": street,
                "city": city,
                "state": state_zip[:2],
                "benchmark": "Public_AR_Census2020",
                "vintage": "Census2020_Census2020",
                "layers": "10",
                "format": "json"
            },
            timeout=15
        )
        geo_data = geo_r.json()
        matches = geo_data.get("result", {}).get("addressMatches", [])

        if not matches:
            return {"error": f"Could not geocode: '{address}'. Try including ZIP code."}

        match = matches[0]
        geographies = match.get("geographies", {})
        tracts  = geographies.get("Census Tracts", [])
        counties = geographies.get("Counties", [])

        if not tracts:
            return {"error": "No census tract found for this address."}

        tract   = tracts[0]
        state_fips  = tract["STATE"]
        county_fips = tract["COUNTY"]
        tract_fips  = tract["TRACT"]
        coords  = match.get("coordinates", {})

    except Exception as e:
        return {"error": f"Geocoding failed: {str(e)}"}

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
                "address_matched": match.get("matchedAddress", address),
                "coordinates": {"lat": coords.get("y"), "lng": coords.get("x")},
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
        "commercial_re_loans":   ("BUSLOANS", "Commercial & Industrial Loans Outstanding"),
        "mortgage_delinquency":  ("DRSFRMACBS", "Delinquency Rate on CRE Loans"),
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
      "months_remaining": months from June 2026 to lease end as number,
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
        "current_date": "June 2026"
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
        "memo_date":         "June 2026"
    }

    instructions = """Generate a professional CRE Investment Committee (IC) memo in markdown format.

Use ONLY the data provided — do not invent numbers or make up market statistics.
Reference actual figures from the live_rates and market_demographics sections.

Structure:
# [Property Address] — Acquisition Memo
**Date:** June 2026 | **Type:** [property_type] | **Status:** For Review

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


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
