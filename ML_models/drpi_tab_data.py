# =============================================================================
# drpi_tab_data.py
# Tab 7 — Policy Insights
# Fill this file to populate your tab in the dashboard.
# You do NOT need to change drpi_04_dashboard.py at all.
# =============================================================================


# HOW TO ADD YOUR DATA
# --------------------
# There are two things to fill in below:
#
#   1. POLICY_FINDINGS  — your key findings, shown as cards in the tab
#   2. POLICY_PROGRAMS  — federal/state program data (optional, for charts)
#
# Just follow the examples and replace the placeholder values.
# Save the file and the dashboard will update automatically.


# =============================================================================
# POLICY_FINDINGS
# Your key policy findings shown as color-coded cards in the tab.
# Keep each one short: one headline + one or two sentences with a citation.
#
# Fields:
#   title   short headline for the finding
#   text    1-2 sentences with data and a source or paper section
#   color   card accent color (use teal #16A085 or another from the paper)
# =============================================================================

POLICY_FINDINGS = [
    # Paste your findings here, one block per finding. Example:
    # {
    #     "title": "FEMA Declarations Accelerate Recovery",
    #     "text": (
    #         "States with 3 or more Individual Assistance declarations per year "
    #         "averaged 13.2 months to recovery, compared to 18.7 months for states "
    #         "with less than 1 per year. (Source: FEMA OpenFEMA 2006-2024)"
    #     ),
    #     "color": "#16A085",
    # },
]


# =============================================================================
# POLICY_PROGRAMS
# Optional: data on federal or state programs, used for charts or tables.
# Leave empty if you are not using it.
#
# TIP: If you have a CSV file ready, load it like this instead:
#   import pandas as pd
#   POLICY_PROGRAMS = pd.read_csv("data/policy_programs.csv").to_dict("records")
#
# Expected columns if using a CSV:
#   program_name, program_type, avg_funding_usd, states_covered, paper_finding
# =============================================================================

POLICY_PROGRAMS = [
    # Paste your program data here. Example:
    # {
    #     "program_name": "FEMA Individual Assistance",
    #     "program_type": "Federal",
    #     "avg_funding_usd": 4200,
    #     "states_covered": 50,
    #     "paper_finding": "Moderate correlation with reduced recovery time",
    # },
]
