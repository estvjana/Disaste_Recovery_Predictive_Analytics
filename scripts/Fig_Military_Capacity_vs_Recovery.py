"""
=============================================================================
Figure — Military Infrastructure and Disaster Response Capacity
         Section 4.6 — DTSC 4302
=============================================================================
Datasets:
  1. DoD Base Structure Report FY2019-FY2025 (BSR)
     Source: Office of the Assistant Secretary of Defense (EI&E)
     Path: data/MilitaryBase/DoD_BSR_Military/
  2. USDOT BTS NTAD Military Bases
     Source: Bureau of Transportation Statistics
     Path: data/MilitaryBase/NTAD_Military_Bases_*.xlsx
  3. DRPA Model Output
     Source: Authors' Disaster Recovery Prediction Algorithm
     Path: state_risk_scores.csv

Outputs:
  - Fig_Military_Capacity_vs_Recovery.png  (scatter: capacity vs recovery)
  - Fig_Military_NG_by_State.png           (bar: National Guard presence)
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSR_DIR  = os.path.join(BASE, "data", "MilitaryBase", "DoD_BSR_Military")
NTAD_F   = os.path.join(BASE, "data", "MilitaryBase",
                         "NTAD_Military_Bases_-4186673718369528560.xlsx")
RISK_F   = os.path.join(BASE, "state_risk_scores.csv")
OUT1     = os.path.join(BASE, "Fig_Military_Capacity_vs_Recovery.png")
OUT2     = os.path.join(BASE, "Fig_Military_NG_by_State.png")

# ── TOP 15 HIGH-EXPOSURE STATES ───────────────────────────────────────────────
TOP15 = {
    "Florida":        "FL", "Tennessee":     "TN", "Texas":       "TX",
    "Missouri":       "MO", "North Carolina":"NC", "Arkansas":    "AR",
    "Georgia":        "GA", "Mississippi":   "MS", "Kansas":      "KS",
    "Oklahoma":       "OK", "Kentucky":      "KY", "Nebraska":    "NE",
    "Virginia":       "VA", "Iowa":          "IA", "Louisiana":   "LA",
}
ABBREV_TO_NAME = {v: k for k, v in TOP15.items()}

# National Guard component keywords
NG_KEYWORDS = ["army guard", "air force guard", "army reserve",
                "air force reserve", "arng", "ang"]

BSR_YEARS = [2019, 2020, 2022, 2023, 2024, 2025]

# ── STEP 1: PARSE BSR FILES ───────────────────────────────────────────────────
def parse_bsr_sheet(filepath, sheet_name, year):
    """Read a BSR sheet, find header row, return clean DataFrame."""
    raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    # Find header row (contains 'Country/State')
    header_row = None
    for i in range(min(10, len(raw))):
        if str(raw.iloc[i, 0]).strip() == "Country/State":
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    df = raw.iloc[header_row:].copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df["fy"] = year
    return df

def clean_bsr_year(year):
    """Extract state-level military capacity metrics for one BSR year."""
    f = os.path.join(BSR_DIR, f"DoD_BSR_FY{year}.xlsx")
    if not os.path.exists(f):
        print(f"  Missing: {f}")
        return pd.DataFrame()

    rows = []

    for sheet in ["Federal DoD Main Report", "State Main Report"]:
        df = parse_bsr_sheet(f, sheet, year)
        if df.empty:
            continue
        # Keep only US states (exclude territories and overseas)
        state_col = "Country/State"
        comp_col  = "Component"
        prv_col   = "Plant Replacement Value ($M)"
        acres_col = "Total Acres"

        df = df[df[state_col].isin(TOP15.keys())].copy()
        df[prv_col]   = pd.to_numeric(df[prv_col],   errors="coerce").fillna(0)
        df[acres_col] = pd.to_numeric(df[acres_col], errors="coerce").fillna(0)

        # Flag National Guard / Reserve installations
        df["is_ng"] = df[comp_col].str.lower().str.strip().apply(
            lambda x: any(kw in str(x) for kw in NG_KEYWORDS)
        )

        for state in TOP15.keys():
            sub = df[df[state_col] == state]
            ng  = sub[sub["is_ng"]]
            rows.append({
                "state":          state,
                "abbr":           TOP15[state],
                "fy":             year,
                "total_sites":    len(sub),
                "ng_sites":       len(ng),
                "total_prv":      sub[prv_col].sum(),
                "ng_prv":         ng[prv_col].sum(),
                "total_acres":    sub[acres_col].sum(),
                "ng_acres":       ng[acres_col].sum(),
            })

    if not rows:
        return pd.DataFrame()

    agg = (pd.DataFrame(rows)
           .groupby(["state", "abbr", "fy"])
           .sum(numeric_only=True)
           .reset_index())
    return agg

print("Step 1 — Parsing BSR files...")
bsr_frames = [clean_bsr_year(yr) for yr in BSR_YEARS]
bsr_all    = pd.concat([f for f in bsr_frames if not f.empty], ignore_index=True)
print(f"  BSR records: {len(bsr_all)} across {bsr_all['fy'].nunique()} years")

# ── STEP 2: AVERAGE ACROSS YEARS ──────────────────────────────────────────────
print("Step 2 — Aggregating across years...")
bsr_avg = (bsr_all
           .groupby(["state", "abbr"])[
               ["total_sites", "ng_sites", "total_prv", "ng_prv",
                "total_acres", "ng_acres"]]
           .mean()
           .reset_index())

# ── STEP 3: PARSE NTAD-MB ─────────────────────────────────────────────────────
print("Step 3 — Parsing NTAD Military Bases...")
ntad = pd.read_excel(NTAD_F)

STATE_ABBREV = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR",
    "california":"CA","colorado":"CO","connecticut":"CT","delaware":"DE",
    "florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
    "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS",
    "kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD",
    "massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV",
    "new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY",
    "north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI",
    "south carolina":"SC","south dakota":"SD","tennessee":"TN","texas":"TX",
    "utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
    "west virginia":"WV","wisconsin":"WI","wyoming":"WY",
}

ntad["abbr"] = ntad["State Name Code"].str.upper().str.strip()
ntad_active  = ntad[ntad["Site Operational Status"] == "act"].copy()
ntad_active["is_ng"] = ntad_active["Site Reporting Component Code"].str.lower().apply(
    lambda x: any(kw in str(x) for kw in
                  ["armynationalguard", "airnationalguard", "usar", "afr"])
)
ntad_active["area"] = pd.to_numeric(ntad_active["Shape__Area"], errors="coerce").fillna(0)

ntad_agg = (ntad_active
            .groupby("abbr")
            .agg(
                ntad_total_bases = ("Site Name", "count"),
                ntad_ng_bases    = ("is_ng", "sum"),
                ntad_total_area  = ("area", "sum"),
                ntad_ng_area     = ("area", lambda x:
                                    ntad_active.loc[x.index[ntad_active.loc[x.index, "is_ng"]], "area"].sum()),
            )
            .reset_index())

# Keep only Top 15
ntad_top15 = ntad_agg[ntad_agg["abbr"].isin(TOP15.values())].copy()
print(f"  NTAD states matched: {len(ntad_top15)}")

# ── STEP 4: LOAD RECOVERY DATA ────────────────────────────────────────────────
print("Step 4 — Loading DRPA recovery predictions...")
risk = pd.read_csv(RISK_F)
risk_top15 = risk[risk["abbr"].isin(TOP15.values())][
    ["state", "abbr", "recovery_months_predicted",
     "unemployment_volatility_excl_covid", "drpi_score"]
].copy()
print(f"  Recovery states: {len(risk_top15)}")

# ── STEP 5: MERGE ALL ──────────────────────────────────────────────────────────
print("Step 5 — Merging datasets...")
df = (bsr_avg
      .merge(ntad_top15, on="abbr", how="left")
      .merge(risk_top15[["abbr","recovery_months_predicted",
                          "unemployment_volatility_excl_covid","drpi_score"]],
             on="abbr", how="left"))

# Military Capacity Index (0-100):
# 40% ng_sites (BSR), 30% ng_prv (BSR), 20% ntad_ng_bases, 10% ng_acres
for col in ["ng_sites","ng_prv","ntad_ng_bases","ng_acres"]:
    mx = df[col].max()
    df[f"{col}_norm"] = (df[col] / mx * 100) if mx > 0 else 0

df["military_capacity_index"] = (
    df["ng_sites_norm"]    * 0.40 +
    df["ng_prv_norm"]      * 0.30 +
    df["ntad_ng_bases_norm"] * 0.20 +
    df["ng_acres_norm"]    * 0.10
).round(2)

df = df.dropna(subset=["recovery_months_predicted","military_capacity_index"])
print(f"  Final merged states: {len(df)}")

# ── STEP 6: FIGURE 1 — SCATTER PLOT ──────────────────────────────────────────
print("Step 6 — Building scatter plot...")

slope, intercept, r, p, _ = stats.linregress(
    df["military_capacity_index"], df["recovery_months_predicted"])
r2 = r ** 2

fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

# Color by recovery duration
cmap   = plt.cm.RdYlGn_r
norm   = plt.Normalize(df["recovery_months_predicted"].min(),
                        df["recovery_months_predicted"].max())
colors = [cmap(norm(v)) for v in df["recovery_months_predicted"]]

sc = ax.scatter(df["military_capacity_index"],
                df["recovery_months_predicted"],
                c=df["recovery_months_predicted"],
                cmap=cmap, norm=norm,
                s=140, edgecolors="white", linewidths=1.2, zorder=3)

# State labels
for _, row in df.iterrows():
    ax.annotate(row["abbr"],
                (row["military_capacity_index"],
                 row["recovery_months_predicted"]),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold",
                color="#333")

# OLS trend line
x_line = np.linspace(df["military_capacity_index"].min() - 2,
                     df["military_capacity_index"].max() + 2, 200)
ax.plot(x_line, slope * x_line + intercept,
        color="#555", linewidth=1.5, linestyle="--", alpha=0.7, zorder=1,
        label="OLS Trendline")

# Reference lines
ax.axvline(df["military_capacity_index"].mean(),
           color="#bbb", linewidth=1, linestyle=":", zorder=0)
ax.axhline(df["recovery_months_predicted"].mean(),
           color="#bbb", linewidth=1, linestyle=":", zorder=0)

# OLS stats box
ax.text(0.03, 0.97,
        f"OLS Trendline (Top 15 States)\n"
        f"r\u00b2 = {r2:.2f}  |  slope = {slope:.3f}  |  p = {p:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#bbb", alpha=0.95))

cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75)
cbar.set_label("Predicted Recovery Duration (months)", fontsize=10)

ax.set_xlabel("Military Capacity Index\n"
              "(National Guard installations + PRV + base presence, normalized 0-100)",
              fontsize=11, labelpad=8)
ax.set_ylabel("Predicted Recovery Duration (months)\nDRPA Model",
              fontsize=11, labelpad=8)
ax.set_title("Military Infrastructure Capacity and Disaster Recovery Duration\n"
             "Top 15 High-Exposure U.S. States",
             fontsize=14, fontweight="bold", color="#0d1b2a", pad=10)
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.spines[["top","right"]].set_visible(False)

fig.text(0.5, -0.02,
         "Fig. X. Military Infrastructure Capacity vs. Predicted Recovery Duration "
         "(Top 15 High-Exposure States).\n"
         "Sources: DoD Base Structure Report (BSR FY2019-FY2025); "
         "USDOT BTS NTAD Military Bases (NTAD-MB); "
         "authors' DRPA model.",
         ha="center", fontsize=8.5, color="#333", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {OUT1}")

# ── STEP 7: FIGURE 2 — BAR CHART ─────────────────────────────────────────────
print("Step 7 — Building bar chart...")

df_sorted = df.sort_values("ng_sites", ascending=False)

# Color bars by recovery speed
rec_norm = plt.Normalize(df_sorted["recovery_months_predicted"].min(),
                          df_sorted["recovery_months_predicted"].max())
bar_colors = [plt.cm.RdYlGn_r(rec_norm(v))
              for v in df_sorted["recovery_months_predicted"]]

fig2, ax2 = plt.subplots(figsize=(12, 6.5))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("#fafafa")

bars = ax2.bar(df_sorted["abbr"], df_sorted["ng_sites"],
               color=bar_colors, edgecolor="white", linewidth=0.8, zorder=2)

# Recovery months label on each bar
for bar, (_, row) in zip(bars, df_sorted.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.15,
             f"{row['recovery_months_predicted']:.0f}mo",
             ha="center", va="bottom", fontsize=8, color="#333")

ax2.set_xlabel("State (Top 15 High-Exposure, sorted by National Guard installations)",
               fontsize=11, labelpad=8)
ax2.set_ylabel("Average National Guard Installations\n(BSR FY2019-FY2025)",
               fontsize=11, labelpad=8)
ax2.set_title("National Guard Presence by State and Predicted Recovery Duration\n"
              "Top 15 High-Exposure U.S. States",
              fontsize=14, fontweight="bold", color="#0d1b2a", pad=10)
ax2.tick_params(labelsize=10)
ax2.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax2.spines[["top","right"]].set_visible(False)

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=rec_norm)
sm.set_array([])
cbar2 = fig2.colorbar(sm, ax=ax2, pad=0.02, shrink=0.75)
cbar2.set_label("Predicted Recovery Duration (months)", fontsize=10)

# Reference line
ax2.axhline(df_sorted["ng_sites"].mean(),
            color="#888", linewidth=1, linestyle="--", alpha=0.6,
            label=f"Group avg = {df_sorted['ng_sites'].mean():.1f}")
ax2.legend(fontsize=9, framealpha=0.85)

fig2.text(0.5, -0.02,
          "Fig. X. National Guard Installations by State and Predicted Recovery Duration "
          "(Top 15 High-Exposure States).\n"
          "Bar labels show DRPA-predicted recovery months. "
          "Color scale from green (faster) to red (slower recovery). "
          "Sources: DoD BSR (FY2019-FY2025); authors' DRPA model.",
          ha="center", fontsize=8.5, color="#333", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {OUT2}")

# ── STEP 8: PRINT SUMMARY TABLE ──────────────────────────────────────────────
print("\nSummary Table — Military Capacity vs Recovery (Top 15 States):")
print(df[["abbr","ng_sites","ng_prv","ntad_ng_bases",
          "military_capacity_index","recovery_months_predicted"]]
      .sort_values("military_capacity_index", ascending=False)
      .to_string(index=False))
