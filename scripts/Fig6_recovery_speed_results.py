"""
=============================================================================
FIG 6: Fiscal Capacity and Labor Market Recovery Speed
        Across 15 High-Exposure U.S. States  (v3 — real 2006-2024 avg)
=============================================================================
Scatter plot:
  X-axis : State fiscal capacity (per-capita STATE-only general revenue,
            2006-2024 avg, USD)  — Source: Census State Gov Finance Survey
  Y-axis : Months to return to pre-shock unemployment baseline (COVID shock)
  Dot size: Avg annual FEMA declarations (2006-2025, excl. 2020)
  Color  : U.S. region

Data used from local files:
  - laus_with_fema_disaster_exposure_2006_2025.csv  →  FEMA decl./yr
  - recovery_speed_results.csv                      →  recovery months
  - StateGovernmen Finances_ Revenue/ (all years)  →  fiscal capacity

Fiscal capacity: computed from real Census files 2006-2024.
  Values = state-government-only per-capita general revenue, 2006-2024 avg.
  Source: U.S. Census Bureau Annual Survey of State Government Finances

Output: Fig6_recovery_speed_results.png (300 DPI)
=============================================================================
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

# ── PATHS ──────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE, "data", "BLS", "laus_with_fema_disaster_exposure_2006_2025.csv")
RECOVERY_PATH = os.path.join(BASE, "recovery_speed_results.csv")
OUTPUT_PATH   = os.path.join(BASE, "visual png", "Fig6_recovery_speed_results.png")

# ── FISCAL CAPACITY (state-only per-capita general revenue, 2006-2024 avg) ─
# Source: Computed from Census Annual Survey of State Government Finances
# files 2006-2024 (wide format 2006-2016, paired ASFIN 2017-2021,
# GOVSSTATEFINTIMESERIES long format 2022-2024). Revenue in $thousands
# divided by ACS/NST annual population estimates. 2007 interpolated.
FISCAL_CAPACITY = {
    "Florida":               4425,
    "Tennessee":             4752,
    "Texas":                 4879,
    "Missouri":              5051,
    "North Carolina":        5564,
    "Arkansas":              6857,
    "Georgia":               4402,
    "Mississippi":           6569,
    "Kansas":                6251,
    "Oklahoma":              5993,
    "Kentucky":              6612,
    "Nebraska":              5739,
    "Virginia":              6050,
    "Iowa":                  6913,
    "Louisiana":             6621,
}

STATES_15 = list(FISCAL_CAPACITY.keys())

# ── REGION MAPPING ─────────────────────────────────────────────────────────
REGION = {
    "Texas":          "Gulf/South",
    "Louisiana":      "Gulf/South",
    "Mississippi":    "Gulf/South",
    "Florida":        "Southeast",
    "Tennessee":      "Southeast",
    "Georgia":        "Southeast",
    "North Carolina": "Southeast",
    "Virginia":       "Southeast",
    "Missouri":       "South Central",
    "Arkansas":       "South Central",
    "Oklahoma":       "South Central",
    "Kansas":         "South Central",
    "Kentucky":       "Midwest",
    "Nebraska":       "Midwest",
    "Iowa":           "Midwest",
}

REGION_COLORS = {
    "Gulf/South":    "#8B2500",   # dark red-brown
    "Southeast":     "#E67E22",   # orange
    "South Central": "#6C3483",   # purple
    "Midwest":       "#1565C0",   # blue
}

# ── ABBREVIATIONS ──────────────────────────────────────────────────────────
ABBR = {
    "Arkansas": "AR", "Florida": "FL", "Georgia": "GA",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Mississippi": "MS", "Missouri": "MO",
    "Nebraska": "NE", "North Carolina": "NC", "Oklahoma": "OK",
    "Tennessee": "TN", "Texas": "TX", "Virginia": "VA",
}

# ── LOAD DATA ──────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_PATH, parse_dates=["date"])
rec = pd.read_csv(RECOVERY_PATH)

# Avg annual FEMA declarations per state (excl. 2020)
df_no20 = df[df["year"] != 2020]
avg_fema = (
    df_no20.groupby(["state", "year"])["disaster_count_month"]
    .sum()
    .groupby("state")
    .mean()
    .rename("avg_fema")
)

# Filter to our 15 states and merge
rec_15 = rec[rec["state"].isin(STATES_15)].copy()
rec_15["fiscal_capacity"] = rec_15["state"].map(FISCAL_CAPACITY)
rec_15["avg_fema"]        = rec_15["state"].map(avg_fema)
rec_15["region"]          = rec_15["state"].map(REGION)
rec_15["abbr"]            = rec_15["state"].map(ABBR)
rec_15 = rec_15.dropna(subset=["fiscal_capacity", "avg_fema", "recovery_months"])
rec_15 = rec_15.sort_values("fiscal_capacity").reset_index(drop=True)

# ── OLS TREND LINE ─────────────────────────────────────────────────────────
slope, intercept, r, p, _ = stats.linregress(
    rec_15["fiscal_capacity"], rec_15["recovery_months"]
)
r2 = r ** 2
x_line = np.linspace(rec_15["fiscal_capacity"].min() - 200,
                     rec_15["fiscal_capacity"].max() + 200, 300)
y_line = slope * x_line + intercept

# ── DOT SIZES ──────────────────────────────────────────────────────────────
fema_vals   = rec_15["avg_fema"].values
fema_min    = fema_vals.min()
fema_max    = fema_vals.max()
size_scaled = ((fema_vals - fema_min) /
               (fema_max  - fema_min) * 320 + 60)

# Labels for size legend
lbl_small = f"~{int(round(fema_min / 10) * 10)} FEMA decl./yr"
lbl_large = f"~{int(round(fema_max / 10) * 10)} FEMA decl./yr"

# ── PLOT ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.13)
ax.set_facecolor("#F8F9FA")

# OLS trend line
ax.plot(x_line, y_line, color="#777", linewidth=1.5,
        linestyle="--", zorder=1)

# Scatter points + state labels
nudge_lookup = {
    "Missouri":       (-200,  0.30),
    "Tennessee":      (-180, -0.40),
    "North Carolina": (  80,  0.25),
    "Oklahoma":       (  80,  0.25),
    "Iowa":           (  80,  0.25),
    "Louisiana":      (  80,  0.25),
    "Texas":          (-180,  0.25),
    "Mississippi":    (  80, -0.40),
    "Arkansas":       (  80,  0.25),
    "Kansas":         (  80, -0.40),
    "Virginia":       (  80,  0.25),
    "Nebraska":       (  80,  0.25),
    "Kentucky":       (  80,  0.25),
    "Florida":        (-180,  0.25),
    "Georgia":        (  80, -0.40),
}

for i, row in rec_15.iterrows():
    clr  = REGION_COLORS[row["region"]]
    sz   = size_scaled[rec_15.index.get_loc(i)]
    ax.scatter(row["fiscal_capacity"], row["recovery_months"],
               s=sz, color=clr, edgecolors="white",
               linewidths=0.8, zorder=3, alpha=0.92)
    nx, ny = nudge_lookup.get(row["state"], (80, 0.25))
    ax.text(row["fiscal_capacity"] + nx,
            row["recovery_months"] + ny,
            row["abbr"], fontsize=8.5, fontweight="bold",
            color="#222", zorder=4)

# ── MEDIAN REFERENCE LINES ─────────────────────────────────────────────────
med_x = rec_15["fiscal_capacity"].median()
med_y = rec_15["recovery_months"].median()
ax.axvline(med_x, color="#aaa", linewidth=0.8, linestyle=":", zorder=0)
ax.axhline(med_y, color="#aaa", linewidth=0.8, linestyle=":", zorder=0)

# ── QUADRANT ANNOTATIONS ───────────────────────────────────────────────────
x_min = rec_15["fiscal_capacity"].min()
x_max = rec_15["fiscal_capacity"].max()
y_min = rec_15["recovery_months"].min()
y_max = rec_15["recovery_months"].max()

# Top-left: Low fiscal capacity + Slow recovery
ax.text(x_min + (med_x - x_min) * 0.08, y_max - 0.3,
        "Low fiscal capacity\nSlow recovery  △",
        fontsize=8, color="#C0392B", style="italic", va="top", ha="left")

# Top-right: High fiscal capacity + Slow recovery
ax.text((med_x + x_max) / 2, y_max - 0.3,
        "High fiscal capacity\nSlow recovery",
        fontsize=8, color="#888", style="italic", va="top", ha="center")

# Bottom-left: Low fiscal capacity + Fast recovery
ax.text(x_min + (med_x - x_min) * 0.08, y_min + 0.3,
        "Low fiscal capacity\nFast recovery",
        fontsize=8, color="#888", style="italic", va="bottom", ha="left")

# ── AXIS FORMATTING ────────────────────────────────────────────────────────
ax.set_xlabel(
    "State Fiscal Capacity  (Per-Capita General Revenue, 2006–2024 Avg, USD)",
    fontsize=10, labelpad=8)
ax.set_ylabel(
    "Months to Return to Pre-Shock Unemployment Baseline",
    fontsize=10, labelpad=8)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.25, linewidth=0.5, color="#bbb")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── TITLE & SUBTITLE ───────────────────────────────────────────────────────
fig.text(0.53, 0.97,
         "COVID-19 shock (March 2020) as reference event  |  "
         "Point size ∝ avg. annual FEMA declarations (2006–2025)",
         ha="center", va="top", fontsize=8.5, color="#555", style="italic")
fig.text(0.53, 0.91,
         "Fiscal Capacity and Labor Market Recovery Speed\n"
         "Across 15 High-Exposure U.S. States",
         ha="center", va="top", fontsize=13, fontweight="bold",
         color="#1A252F", linespacing=1.4)

# ── LEGENDS ────────────────────────────────────────────────────────────────
# Region legend
region_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=clr, markersize=9,
                  label=region, markeredgecolor="white")
    for region, clr in REGION_COLORS.items()
]
leg1 = ax.legend(handles=region_handles, title="Region",
                 loc="lower right", fontsize=8.5, title_fontsize=9,
                 framealpha=0.95, edgecolor="#ccc",
                 bbox_to_anchor=(1.0, 0.18))
ax.add_artist(leg1)

# Size + OLS legend (stacked below Region legend)
size_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor="#888", markersize=6,
                  label=lbl_small, markeredgecolor="white"),
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor="#888", markersize=12,
                  label=lbl_large, markeredgecolor="white"),
    mlines.Line2D([], [], color="#777", linewidth=1.5, linestyle="--",
                  label=f"OLS trend  (r² = {r2:.2f})"),
]
ax.legend(handles=size_handles, loc="lower right",
          fontsize=8.5, framealpha=0.95, edgecolor="#ccc")

# ── SOURCE NOTE ────────────────────────────────────────────────────────────
fig.text(0.09, 0.005,
         "Sources: BLS LAUS (recovery months); U.S. Census Bureau Annual Survey of "
         "State Government Finances 2006–2024 (fiscal capacity); FEMA Disaster Declarations Database.",
         fontsize=7.5, color="#666", style="italic")

# ── SAVE ───────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.2)
plt.show()
plt.close()

print(f"\n✓ Saved → {OUTPUT_PATH}")
print(f"  OLS slope={slope:.4f} | intercept={intercept:.2f} | r²={r2:.3f} | p={p:.4f}")
print(f"  States in chart: {len(rec_15)}")
print(f"  FEMA decl./yr range: {fema_min:.1f} – {fema_max:.1f}")
print(f"  Recovery months range: {rec_15['recovery_months'].min()} – {rec_15['recovery_months'].max()}")
