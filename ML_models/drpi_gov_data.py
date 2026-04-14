# =============================================================================
# drpi_gov_data.py
# Tab 4 — Political Economy & Aid Distribution
# Fill this file to populate your tab in the dashboard.
# You do NOT need to change drpi_04_dashboard.py at all.
# =============================================================================
#
# Data sources:
#   - state_risk_scores.csv   → drpi_score, drpi_risk_tier, recovery_months_predicted
#   - 2008-2020 Presidential  → governor_party (proxy from presidential voting lean),
#     Election Results            legislative_control (proxy from voting consistency)
#
# NOTE: governor_party is derived from presidential voting patterns (2008-2020),
#       not actual gubernatorial party affiliation. legislative_control is derived
#       from voting consistency (Solid = single party, otherwise Split).
# =============================================================================


PARTISAN_DATA = [
    {
        "state": "Louisiana",
        "abbr": "LA",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.8,
        "drpi_score": 68.2,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Florida",
        "abbr": "FL",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 17.8,
        "drpi_score": 65.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "California",
        "abbr": "CA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.3,
        "drpi_score": 60.1,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Texas",
        "abbr": "TX",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 12.6,
        "drpi_score": 57.6,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "New Jersey",
        "abbr": "NJ",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 22.0,
        "drpi_score": 57.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Georgia",
        "abbr": "GA",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.2,
        "drpi_score": 56.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "North Carolina",
        "abbr": "NC",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.3,
        "drpi_score": 54.9,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Rhode Island",
        "abbr": "RI",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 22.4,
        "drpi_score": 52.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "New York",
        "abbr": "NY",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.5,
        "drpi_score": 49.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "South Carolina",
        "abbr": "SC",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.5,
        "drpi_score": 47.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Missouri",
        "abbr": "MO",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 14.9,
        "drpi_score": 47.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Michigan",
        "abbr": "MI",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 19.5,
        "drpi_score": 46.7,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Nevada",
        "abbr": "NV",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 19.5,
        "drpi_score": 46.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Alabama",
        "abbr": "AL",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.0,
        "drpi_score": 45.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Tennessee",
        "abbr": "TN",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.7,
        "drpi_score": 45.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Hawaii",
        "abbr": "HI",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.4,
        "drpi_score": 44.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Mississippi",
        "abbr": "MS",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.1,
        "drpi_score": 44.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Kentucky",
        "abbr": "KY",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.4,
        "drpi_score": 44.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Oregon",
        "abbr": "OR",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 19.0,
        "drpi_score": 42.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Illinois",
        "abbr": "IL",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.0,
        "drpi_score": 42.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Maine",
        "abbr": "ME",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 16.6,
        "drpi_score": 41.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Washington",
        "abbr": "WA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.1,
        "drpi_score": 41.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Vermont",
        "abbr": "VT",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 20.3,
        "drpi_score": 41.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Colorado",
        "abbr": "CO",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 17.1,
        "drpi_score": 41.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Connecticut",
        "abbr": "CT",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.8,
        "drpi_score": 38.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Massachusetts",
        "abbr": "MA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 19.8,
        "drpi_score": 37.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Virginia",
        "abbr": "VA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 14.4,
        "drpi_score": 36.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Delaware",
        "abbr": "DE",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 20.6,
        "drpi_score": 35.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Minnesota",
        "abbr": "MN",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 15.7,
        "drpi_score": 33.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Arizona",
        "abbr": "AZ",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.4,
        "drpi_score": 33.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Iowa",
        "abbr": "IA",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 14.1,
        "drpi_score": 33.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Arkansas",
        "abbr": "AR",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 14.5,
        "drpi_score": 33.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Idaho",
        "abbr": "ID",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.9,
        "drpi_score": 33.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Ohio",
        "abbr": "OH",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.7,
        "drpi_score": 33.0,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Pennsylvania",
        "abbr": "PA",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.1,
        "drpi_score": 31.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "South Dakota",
        "abbr": "SD",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.0,
        "drpi_score": 31.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Wisconsin",
        "abbr": "WI",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.5,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Kansas",
        "abbr": "KS",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Indiana",
        "abbr": "IN",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 14.7,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "New Mexico",
        "abbr": "NM",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.5,
        "drpi_score": 29.0,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Oklahoma",
        "abbr": "OK",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 10.5,
        "drpi_score": 27.7,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Maryland",
        "abbr": "MD",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 15.1,
        "drpi_score": 25.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Utah",
        "abbr": "UT",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.4,
        "drpi_score": 25.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "New Hampshire",
        "abbr": "NH",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.4,
        "drpi_score": 24.7,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "West Virginia",
        "abbr": "WV",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 13.3,
        "drpi_score": 23.6,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Nebraska",
        "abbr": "NE",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 11.5,
        "drpi_score": 23.6,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Montana",
        "abbr": "MT",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.9,
        "drpi_score": 22.5,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "North Dakota",
        "abbr": "ND",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 16.7,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Alaska",
        "abbr": "AK",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 13.5,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Wyoming",
        "abbr": "WY",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 11.7,
        "drpi_score": 6.6,
        "drpi_risk_tier": "LOW",
    },
]


# =============================================================================
# GOV_FINDINGS
# Key findings shown as cards in the tab.
# =============================================================================

GOV_FINDINGS = [
    {
        "title": "Balanced Risk Across Party Lines in High-Exposure States",
        "text": (
            "Of the 8 HIGH-tier states, 3 lean Democrat (CA, NJ, RI), "
            "3 lean Republican (LA, TX, GA), and 2 are swing states (FL, NC). "
            "DRPI risk tier shows no strong partisan skew, suggesting disaster "
            "vulnerability is driven more by geography and fiscal capacity than "
            "political affiliation alone. "
            "(Source: state_risk_scores.csv, MIT Election Lab 2008-2020)"
        ),
        "color": "#C0392B",
    },
    {
        "title": "Democrat-Leaning States Show Longer Predicted Recovery Times",
        "text": (
            "States classified as Democrat-leaning average 18.5 months predicted "
            "recovery vs 14.3 months for Republican-leaning states. This likely "
            "reflects higher cost of living, urban density, and infrastructure "
            "complexity rather than policy differences, as the highest-cost states "
            "(CA, NJ, NY, RI) cluster in the Democrat category. "
            "(Source: FEMA OpenFEMA 2006-2025, Ridge Regression Model)"
        ),
        "color": "#8E44AD",
    },
    {
        "title": "Swing States Carry Disproportionate DRPI Risk",
        "text": (
            "The 4 swing states (FL, NC, OH, IA) average a DRPI score of 46.7, "
            "exceeding both Democrat-leaning (40.5) and Republican-leaning (35.1) "
            "state averages. Florida and North Carolina alone account for 2 of the "
            "top 8 highest-risk states nationally. "
            "(Source: state_risk_scores.csv, 2008-2020 Presidential Elections)"
        ),
        "color": "#C0392B",
    },
    {
        "title": "Note on Political Classification Methodology",
        "text": (
            "Governor party and legislative control are proxied from presidential "
            "voting patterns (2008-2020), not actual gubernatorial or state "
            "legislature party data. 'Independent' indicates swing states that "
            "split evenly between parties across the four election cycles. "
            "(Source: MIT Election Lab, 1976-2020 U.S. President dataset)"
        ),
        "color": "#8E44AD",
    },
]
