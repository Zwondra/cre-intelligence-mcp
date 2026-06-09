# CRE Intelligence MCP

**Live market data for commercial real estate analysis — inside Claude.**

Connect this MCP to Claude Desktop and instantly access Federal Reserve interest rates, Census Bureau demographics, DCF modeling, rent roll parsing, lease abstraction, and IC memo generation — all from a single prompt.

> *"Analyze this deal: 2201 South Blvd, Charlotte NC. NOI $400k, asking $6M, retail strip."*
>
> → Pulls live SOFR from the Fed. Pulls Census demographics for that exact block. Builds a full levered 10-year DCF. Writes an institutional-quality IC memo. **30 seconds.**

---

## Why this exists

The #1 problem with AI in CRE: **66% of professionals use it daily, but only 5% trust it for actual deal decisions.**

The reason? AI guesses at rates and demographics. A DCF built on a hallucinated SOFR rate is worthless.

This MCP fixes that. Every number comes from a verified public source:
- **Interest rates** → Federal Reserve (FRED API)
- **Demographics** → US Census Bureau ACS
- **Document analysis** → Claude AI with structured output

Zero data licensing fees. Zero hallucinations on financial inputs.

---

## Tools

| Tool | What it does | Data source |
|------|-------------|-------------|
| `get_current_rates` | Live SOFR, 10yr Treasury, Fed Funds + implied cap ranges by property type | FRED |
| `get_inflation_data` | CPI, shelter inflation, rent CPI + DCF rent growth guidance | FRED |
| `get_cre_market_data` | CRE price index, C&I loan trends, delinquency rates, credit spreads | FRED |
| `get_market_demographics` | Median income, employment, vacancy, rents for any US address | Census Bureau |
| `analyze_rent_roll` | Paste PDF text → structured JSON: tenant, SF, rent, dates, expirations | Claude AI |
| `abstract_lease` | Paste lease text → term, rent schedule, TI, options, red flags | Claude AI |
| `flag_lease_risks` | Rent roll JSON → rollover risk, concentration risk, due diligence checklist | Claude AI |
| `build_dcf_model` | Full levered 10-year DCF with live rates auto-fetched from FRED | Python + FRED |
| `generate_deal_memo` | Address + NOI + price → full IC memo with live rates and demographics | Claude AI + FRED + Census |

---

## Example output

**Prompt:** *"Get me current interest rates"*

```
SOFR:           3.63%   (June 8, 2026)
SOFR 30-day:    3.59%
10yr Treasury:  4.55%
5yr Treasury:   4.29%
Fed Funds:      3.63%

Implied cap rates (spread over 10yr T):
  Core Multifamily:  5.30% – 6.05%
  Core Industrial:   5.55% – 6.30%
  Core Office:       6.05% – 7.05%
  Value-Add:         6.05% – 7.05%

Loan rate guidance:
  Floating: SOFR + 150–250bps = ~5.38%–5.88%
  Fixed:    10yr T + 150–200bps = ~6.05%–6.55%
```

**Prompt:** *"Analyze this deal: 2201 South Blvd Charlotte NC, NOI $400k, asking $6M, retail strip"*

The MCP automatically chains `get_current_rates` + `get_market_demographics` + `build_dcf_model` + `generate_deal_memo` and returns a full IC memo including:

```
Entry Cap Rate:   6.67%  (+212bps over 10yr Treasury)
Loan Rate:        5.38%  (derived from live SOFR 3.63% + 175bps)
Year 1 DSCR:      1.53x
IRR:              14.6%
Equity Multiple:  3.11x

Census Tract demographics (2023 ACS):
  Median HHI:       $141,419
  Employment rate:  97.2%
  College educated: 64.9%
  Vacancy rate:     8.6%

Recommendation: GO — subject to rent roll review and comp analysis.
```

---

## Setup

### Option A — Hosted (fastest, no API keys)

Add this to your `claude_desktop_config.json` and restart Claude Desktop:

```json
{
  "mcpServers": {
    "cre-intelligence": {
      "type": "streamable-http",
      "url": "https://cre-intelligence-mcp.onrender.com/mcp"
    }
  }
}
```

If your MCP client only supports stdio servers, use the `mcp-remote` bridge instead:

```json
{
  "mcpServers": {
    "cre-intelligence": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://cre-intelligence-mcp.onrender.com/mcp"]
    }
  }
}
```

Free during beta. All data fetching runs server-side.

### Option B — Self-hosted

#### Prerequisites
- [Claude Desktop](https://claude.ai/download)
- Python 3.11+
- Free API keys (takes ~2 minutes total):
  - **FRED:** [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
  - **Census:** [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
  - **Anthropic:** [console.anthropic.com](https://console.anthropic.com)

#### Install

```bash
git clone https://github.com/Zwondra/cre-intelligence-mcp
cd cre-intelligence-mcp
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cre-intelligence": {
      "command": "/path/to/cre-intelligence-mcp/venv/bin/python3",
      "args": ["/path/to/cre-intelligence-mcp/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "FRED_API_KEY": "your_fred_key",
        "CENSUS_API_KEY": "your_census_key"
      }
    }
  }
}
```

Restart Claude Desktop. The tools will appear automatically.

### Test it

Open Claude Desktop and say:

> *"Get me current interest rates"*

You should see it call `get_current_rates` and return live Federal Reserve data.

---

## Document analysis

The document tools (`analyze_rent_roll`, `abstract_lease`) work by pasting PDF text directly into the prompt. In Claude Desktop:

1. Open your rent roll or lease PDF
2. Copy all the text
3. Say: *"Analyze this rent roll: [paste text]"*

The tool extracts every tenant, suite, SF, rent, lease dates, and expiration into structured JSON — then `flag_lease_risks` can immediately analyze it for rollover and concentration risk.

---

## Data sources

| Source | What | Cost |
|--------|------|------|
| [FRED (Federal Reserve)](https://fred.stlouisfed.org) | SOFR, Treasury yields, Fed Funds, CPI, CRE price index | Free |
| [Census Bureau ACS](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Income, employment, housing, rents by census tract | Free |
| [Anthropic Claude](https://anthropic.com) | Document parsing, risk analysis, memo generation | Pay per use |

---

## Roadmap

- [ ] Comparable sales search (CREXI public listings)
- [ ] Multi-property portfolio analysis
- [ ] Sensitivity table generation (cap rate / NOI / LTV scenarios)
- [ ] Export to Excel / Word
- [ ] Deal history / comparison across sessions

---

## License

MIT
