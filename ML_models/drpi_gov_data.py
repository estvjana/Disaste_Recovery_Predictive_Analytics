# =============================================================================
# drpi_gov_data.py
# Tab 4 — Political Economy & Aid Distribution
# Fill this file to populate your tab in the dashboard.
# You do NOT need to change drpi_04_dashboard.py at all.
# =============================================================================


# HOW TO ADD YOUR DATA
# --------------------
# There are two things to fill in below:
#
#   1. PARTISAN_DATA  — one entry per state, with political + model data
#   2. GOV_FINDINGS   — your key findings, shown as cards in the tab
#
# Just follow the examples and replace the placeholder values.
# Save the file and the dashboard will update automatically.


# =============================================================================
# PARTISAN_DATA
# One entry per state. Copy the example block and fill in your values.
#
# Fields:
#   state                     full state name, e.g. "Louisiana"
#   abbr                      2-letter code, e.g. "LA"
#   governor_party            "Democrat", "Republican", or "Independent"
#   legislative_control       "Democrat", "Republican", or "Split"
#   recovery_months_predicted copy from state_risk_scores.csv
#   drpi_score                copy from state_risk_scores.csv (0-100)
#   drpi_risk_tier            "LOW", "MODERATE", "HIGH", or "CRITICAL"
#
# TIP: If you have a CSV file ready, you can load it instead:
#   import pandas as pd
#   PARTISAN_DATA = pd.read_csv("data/partisan_data.csv").to_dict("records")
#   (columns must match the field names above)
# =============================================================================

PARTISAN_DATA = [
    # Paste your state entries here, one block per state. Example:
    # {
    #     "state": "Louisiana",
    #     "abbr": "LA",
    #     "governor_party": "Democrat",
    #     "legislative_control": "Republican",
    #     "recovery_months_predicted": 21.3,
    #     "drpi_score": 68,
    #     "drpi_risk_tier": "HIGH",
    # },
]


# =============================================================================
# GOV_FINDINGS
# Your key findings shown as cards in the tab.
# Keep each finding short: one headline + one or two sentences.
#
# Fields:
#   title   short headline for the finding
#   text    1-2 sentences explaining the finding, with a data source or citation
#   color   card accent color (use crimson #C0392B or purple #8E44AD)
# =============================================================================

GOV_FINDINGS = [
    # Paste your findings here. Example:
    # {
    #     "title": "Partisan Distribution Across High-Exposure States",
    #     "text": (
    #         "Of the 15 highest-exposure states, 9 had Republican governors "
    #         "and 6 Democrat at the time of their last major disaster. "
    #         "Recovery time showed no significant difference by party alone, "
    #         "but interaction with fiscal capacity was significant (p=0.03). "
    #         "(Paper Section 4.3)"
    #     ),
    #     "color": "#C0392B",
    # },
    # {
    #     "title": "Legislative Control & FEMA Declaration Speed",
    #     "text": (
    #         "States with unified legislative control averaged 4.2 days faster "
    #         "FEMA declaration response than split-legislature states. "
    #         "(Source: FEMA OpenFEMA 2006-2024)"
    #     ),
    #     "color": "#8E44AD",
    # },
]
