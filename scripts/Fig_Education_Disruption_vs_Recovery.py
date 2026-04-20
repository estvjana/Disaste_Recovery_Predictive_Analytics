"""
=============================================================================
FIG: Education Disruption and Recovery Duration in High-Exposure States
=============================================================================
Scatter plot:
  X-axis : Virtual learning share (%)
            Source: COVID-19 School Data Hub (2020-2023)
  Y-axis : Change in average daily attendance (%)
            Source: NCES Digest of Education Statistics, Table 203.80
  Color  : Predicted recovery duration (months) from DRPA model
            Source: state_risk_scores.csv

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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_FILE  = os.path.join(BASE, "data", "Education",
                          "covid_school_learning_modality_district_yearly.csv")
ADA_FILE  = os.path.join(BASE, "data", "Education",
                          "nces_avg_daily_attendance_by_state_1969_2021.xlsx")
RISK_FILE = os.path.join(BASE, "state_risk_scores.csv")
OUTPUT    = os.path.join(BASE, "Fig_Education_Disruption_vs_Recovery.png")

# ── TOP 15 HIGH-EXPOSURE STATES ───────────────────────────────────────────────
TOP15 = {
    "TX": "Texas",          "FL": "Florida",        "LA": "Louisiana",
    "OK": "Oklahoma",       "TN": "Tennessee",       "NC": "North Carolina",
    "SC": "South Carolina", "VA": "Virginia",        "GA": "Georgia",
    "MO": "Missouri",       "MS": "Mississippi",     "KY": "Kentucky",
    "AR": "Arkansas",       "IA": "Iowa",            "NE": "Nebraska",
}
NAME_TO_ABBREV = {v: k for k, v in TOP15.items()}

# States to highlight with larger markers and bold labels
HIGHLIGHT = {"MS", "LA", "NC", "KY"}

# Manual offsets to reduce label crowding (abbrev: (x_offset, y_offset))
LABEL_OFFSET = {
    "NE": (-6, 9), "FL": (6, 9),  "AR": (0, -14),
    "GA": (6, 9),  "SC": (6, -14), "MO": (-6, 9),
    "TN": (0, -14), "VA": (6, 9),
}

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

# ── MERGE ─────────────────────────────────────────────────────────────────────
df = (
    mod_top15
    .merge(ada_df[["abbrev", "pct_change"]], on="abbrev", how="inner")
    .merge(rec_by_state[["abbrev", "recovery_months_predicted"]], on="abbrev", how="left")
)
df["state_name"] = df["abbrev"].map(TOP15)
df["virtual_pct"] = (df["share_virtual"] * 100).round(1)
df = df.dropna(subset=["virtual_pct", "pct_change", "recovery_months_predicted"])
print(f"  Final merged states: {len(df)}")

# ── FIGURE SETUP ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

cmap = mcolors.LinearSegmentedColormap.from_list(
    "recovery", ["#F9E79F", "#E67E22", "#E74C3C", "#7B241C"]
)
norm = mcolors.Normalize(
    vmin=df["recovery_months_predicted"].min(),
    vmax=df["recovery_months_predicted"].max(),
)

# ── SCATTER POINTS ────────────────────────────────────────────────────────────
sizes = [220 if a in HIGHLIGHT else 120 for a in df["abbrev"]]
sc = ax.scatter(
    df["virtual_pct"],
    df["pct_change"],
    c=df["recovery_months_predicted"],
    cmap=cmap, norm=norm,
    s=sizes,
    edgecolors="white", linewidths=1.2,
    zorder=3,
)

# ── STATE LABELS (with crowding offsets) ──────────────────────────────────────
for _, row in df.iterrows():
    abbrev   = row["abbrev"]
    offset   = LABEL_OFFSET.get(abbrev, (0, 9))
    weight   = "bold" if abbrev in HIGHLIGHT else "normal"
    fontsize = 10 if abbrev in HIGHLIGHT else 8.5
    color    = "#C0392B" if abbrev in HIGHLIGHT else "#333333"
    ax.annotate(
        abbrev,
        (row["virtual_pct"], row["pct_change"]),
        textcoords="offset points",
        xytext=offset,
        ha="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
    )

# ── REFERENCE LINES (sample means) ───────────────────────────────────────────
mean_x = df["virtual_pct"].mean()
mean_y = df["pct_change"].mean()
ax.axvline(mean_x, color="#bbb", linewidth=1, linestyle="--", zorder=1)
ax.axhline(mean_y, color="#bbb", linewidth=1, linestyle="--", zorder=1)

# ── QUADRANT ANNOTATION ───────────────────────────────────────────────────────
ax.text(
    df["virtual_pct"].max() - 0.5,
    df["pct_change"].min() + 0.1,
    "Higher virtual learning and larger\nattendance losses indicate greater\neducational disruption",
    ha="right", va="bottom",
    fontsize=8.5, color="#922B21",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5",
              edgecolor="#E74C3C", alpha=0.9),
)

# ── COLORBAR ─────────────────────────────────────────────────────────────────
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.72)
cbar.set_label("Predicted recovery duration (months)", fontsize=10, labelpad=10)
cbar.ax.tick_params(labelsize=9)

# ── AXES ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Virtual learning share (%)", fontsize=11, labelpad=8)
ax.set_ylabel("Change in average daily attendance (%)", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)
ax.grid(True, color="#ebebeb", linewidth=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# ── TITLE + SUBTITLE ─────────────────────────────────────────────────────────
ax.set_title(
    "Education Disruption and Recovery Duration in High-Exposure States",
    fontsize=14, fontweight="bold", color="#0d1b2a", pad=6,
)
fig.text(
    0.5, 0.96,
    "Top 15 U.S. states by disaster exposure during the COVID-19 period",
    ha="center", fontsize=10, color="#555", style="italic",
)

# ── LEGEND ───────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B",
           markersize=11, markeredgecolor="white", label="Highlighted states"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
           markersize=8,  markeredgecolor="white", label="Other top-15 states"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right",
          framealpha=0.85, edgecolor="#ccc",
          title="MS, LA, NC, KY highlighted", title_fontsize=8)

# ── CAPTION ──────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.03,
    "Note: Dashed lines mark the mean across the selected states. "
    "Colors show model-predicted recovery duration in months.\n"
    "Sources: COVID-19 School Data Hub; NCES Digest of Education Statistics; "
    "authors' DRPA model (UNC Charlotte DTSC 4302).",
    ha="center", fontsize=8, color="#666", linespacing=1.5,
)

plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT}")
plt.show()
