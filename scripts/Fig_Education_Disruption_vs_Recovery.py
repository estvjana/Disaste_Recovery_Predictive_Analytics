"""
=============================================================================
FIG: Education Disruption vs. Economic Recovery Duration
      Across 15 High-Exposure U.S. States (COVID-19 Period)
=============================================================================
Scatter plot:
  X-axis : Share of school year spent in virtual learning (%)
            Source: COVID-19 School Data Hub — District-Level Learning
            Modality Dataset (2020-2023)
  Y-axis : Change in average daily attendance (2019-20 vs. 2020-21, %)
            Source: NCES Digest of Education Statistics, Table 203.80
  Color  : Predicted economic recovery time (months) from DRPA model
            Source: state_risk_scores.csv (drpi_02_predictive_models.py)

Key argument:
  States with higher virtual learning share and larger attendance drops
  also carry the longest predicted recovery times, supporting the finding
  that education disruption is a mechanism — not just a side effect —
  of slower economic recovery.

Data files required:
  - data/Education/covid_school_learning_modality_district_yearly.csv
  - data/Education/nces_avg_daily_attendance_by_state_1969_2021.xlsx
  - state_risk_scores.csv

Output: Fig_Education_Disruption_vs_Recovery.png (300 DPI)
=============================================================================
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_FILE   = os.path.join(BASE, "data", "Education",
                           "covid_school_learning_modality_district_yearly.csv")
ADA_FILE   = os.path.join(BASE, "data", "Education",
                           "nces_avg_daily_attendance_by_state_1969_2021.xlsx")
RISK_FILE  = os.path.join(BASE, "state_risk_scores.csv")
OUTPUT     = os.path.join(BASE, "Fig_Education_Disruption_vs_Recovery.png")

# ── TOP 15 HIGH-EXPOSURE STATES ───────────────────────────────────────────────
TOP15 = {
    "TX": "Texas",        "FL": "Florida",        "LA": "Louisiana",
    "OK": "Oklahoma",     "TN": "Tennessee",       "NC": "North Carolina",
    "SC": "South Carolina","VA": "Virginia",       "GA": "Georgia",
    "MO": "Missouri",     "MS": "Mississippi",     "KY": "Kentucky",
    "AR": "Arkansas",     "IA": "Iowa",            "NE": "Nebraska",
}
NAME_TO_ABBREV = {v: k for k, v in TOP15.items()}
HIGHLIGHT      = {"MS", "LA", "NC", "KY"}   # annotated states

# ── LOAD LEARNING MODALITY DATA ───────────────────────────────────────────────
print("Loading learning modality data...")
mod_raw = pd.read_csv(MOD_FILE)
mod_agg = (
    mod_raw.groupby("StateAbbrev")[["share_inperson", "share_hybrid", "share_virtual"]]
    .mean()
    .reset_index()
    .rename(columns={"StateAbbrev": "abbrev"})
)
mod_top15 = mod_agg[mod_agg["abbrev"].isin(TOP15.keys())].copy()
print(f"  States with modality data: {len(mod_top15)}")

# ── LOAD ATTENDANCE DATA ──────────────────────────────────────────────────────
print("Loading attendance data...")
xl = pd.read_excel(ADA_FILE, header=None)
ada_rows = []
for i in range(3, 66):
    sraw   = str(xl.iloc[i, 0]).strip()
    sclean = re.sub(r'[\\]+\d+[\\]+', '', sraw).strip()
    abbrev = NAME_TO_ABBREV.get(sclean)
    if abbrev:
        try:
            v19  = float(xl.iloc[i, 27])
            v20  = float(xl.iloc[i, 29])
            pct  = round(((v20 - v19) / v19) * 100, 2)
            ada_rows.append({"abbrev": abbrev, "state": sclean, "pct_change": pct})
        except:
            pass
ada_df = pd.DataFrame(ada_rows)
print(f"  States with attendance data: {len(ada_df)}")

# ── LOAD DRPA RECOVERY PREDICTIONS ───────────────────────────────────────────
print("Loading DRPA model recovery predictions...")
risk_df = pd.read_csv(RISK_FILE)
rec_by_state = (
    risk_df[risk_df["state"].isin(list(TOP15.values()))]
    .groupby("state")["recovery_months_predicted"]
    .mean()
    .reset_index()
)
rec_by_state["abbrev"] = rec_by_state["state"].map(NAME_TO_ABBREV)
print(f"  States with recovery predictions: {len(rec_by_state)}")

# ── MERGE ALL THREE DATASETS ──────────────────────────────────────────────────
df = (
    mod_top15
    .merge(ada_df[["abbrev", "pct_change"]], on="abbrev", how="inner")
    .merge(rec_by_state[["abbrev", "recovery_months_predicted"]], on="abbrev", how="left")
)
df["state_name"] = df["abbrev"].map(TOP15)
df["virtual_pct"] = (df["share_virtual"] * 100).round(1)
df = df.dropna(subset=["virtual_pct", "pct_change", "recovery_months_predicted"])
print(f"  Final merged states: {len(df)}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

# Color map: yellow (short recovery) -> dark red (long recovery)
cmap   = mcolors.LinearSegmentedColormap.from_list(
    "recovery", ["#F9E79F", "#E67E22", "#E74C3C", "#7B241C"]
)
norm   = mcolors.Normalize(
    vmin=df["recovery_months_predicted"].min(),
    vmax=df["recovery_months_predicted"].max()
)
colors = [cmap(norm(v)) for v in df["recovery_months_predicted"]]

# Scatter points
sizes = [220 if a in HIGHLIGHT else 130 for a in df["abbrev"]]
sc = ax.scatter(
    df["virtual_pct"],
    df["pct_change"],
    c=df["recovery_months_predicted"],
    cmap=cmap,
    norm=norm,
    s=sizes,
    edgecolors="white",
    linewidths=1.2,
    zorder=3,
)

# State labels
for _, row in df.iterrows():
    weight = "bold" if row["abbrev"] in HIGHLIGHT else "normal"
    fontsize = 10 if row["abbrev"] in HIGHLIGHT else 8.5
    ax.annotate(
        row["abbrev"],
        (row["virtual_pct"], row["pct_change"]),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#0d1b2a",
    )

# Median reference lines
med_x = df["virtual_pct"].median()
med_y = df["pct_change"].median()
ax.axvline(med_x, color="#bbb", linewidth=1, linestyle="--", zorder=1)
ax.axhline(med_y, color="#bbb", linewidth=1, linestyle="--", zorder=1)

# Quadrant label: worst corner
ax.text(
    df["virtual_pct"].max() - 0.5,
    df["pct_change"].min() + 0.1,
    "High virtual learning\n& large attendance loss\n(most disrupted)",
    ha="right", va="bottom",
    fontsize=8.5, color="#C0392B",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#E74C3C", alpha=0.85),
)

# Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75)
cbar.set_label("Predicted Recovery Time (months)", fontsize=10, labelpad=10)
cbar.ax.tick_params(labelsize=9)

# Axes formatting
ax.set_xlabel(
    "Share of School Year in Virtual Learning (%)\n"
    "Source: COVID-19 School Data Hub, District-Level Learning Modality Dataset (2020-2023)",
    fontsize=10, labelpad=10,
)
ax.set_ylabel(
    "Change in Avg. Daily Attendance, 2019-20 to 2020-21 (%)\n"
    "Source: NCES Digest of Education Statistics, Table 203.80",
    fontsize=10, labelpad=10,
)
ax.tick_params(labelsize=9)
ax.grid(True, color="#ebebeb", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# Title
ax.set_title(
    "Education Disruption vs. Economic Recovery Duration\n"
    "15 Highest Disaster-Exposure States — COVID-19 Period",
    fontsize=13, fontweight="bold", color="#0d1b2a", pad=14,
)

# Legend for highlighted states
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
           markersize=10, markeredgecolor="white", label="Highlighted states (MS, LA, NC, KY)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
           markersize=8, markeredgecolor="white", label="Other top-15 states"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper left",
          framealpha=0.85, edgecolor="#ccc")

# Caption
fig.text(
    0.5, -0.02,
    "Color source: DRPA Model — Predicted Recovery Months (drpi_02_predictive_models.py)  |  "
    "UNC Charlotte DTSC 4302",
    ha="center", fontsize=8, color="#888",
)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT}")
plt.show()
