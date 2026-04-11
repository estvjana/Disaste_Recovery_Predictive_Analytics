"""
=============================================================================
Disaster-Year Volatility Premium by State
Top 15 High-Exposure States, 2006-2025 (excl. 2020)
=============================================================================
Horizontal bar chart: delta in avg. unemployment volatility between
high-disaster years (top 40% by disaster count) and low-disaster years,
for the 15 states with the highest total disaster exposure.

Data required:
  - laus_with_fema_disaster_exposure_2006_2025.csv

Output: Top15_HighExposure_RiyaNewVersion.png (300 DPI)
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- CONFIG --
DATA_PATH   = "data/BLS/laus_with_fema_disaster_exposure_2006_2025.csv"
OUTPUT_PATH = "visual png/Top15_HighExposure_RiyaNewVersion.png"

# -- LOAD & CLEAN --
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df[df["year"] != 2020].copy()

# -- TOP 15 HIGH-EXPOSURE STATES --
state_exposure = (
    df.groupby("state")["disaster_count_month"]
    .sum()
    .sort_values(ascending=False)
)
top15_states = state_exposure.head(15).index.tolist()
df_top = df[df["state"].isin(top15_states)].copy()

# -- ANNUAL VOLATILITY per state-year --
vol = (
    df_top.groupby(["state", "year"])["unemployment_rate"]
    .std()
    .reset_index()
    .rename(columns={"unemployment_rate": "volatility"})
)

# -- CLASSIFY HIGH / LOW DISASTER YEARS --
annual_dis = (
    df_top.groupby(["state", "year"])["disaster_count_month"]
    .sum()
    .reset_index()
    .rename(columns={"disaster_count_month": "annual_disasters"})
)
vol = vol.merge(annual_dis, on=["state", "year"])

def classify(group):
    threshold = group["annual_disasters"].quantile(0.60)  # top 40%
    group["high_disaster"] = group["annual_disasters"] >= threshold
    return group

vol = vol.groupby("state", group_keys=False).apply(classify)

# -- VOLATILITY PREMIUM (high minus low) --
premium = (
    vol.groupby(["state", "high_disaster"])["volatility"]
    .mean()
    .unstack("high_disaster")
    .rename(columns={False: "low_vol", True: "high_vol"})
    .assign(delta=lambda x: x["high_vol"] - x["low_vol"])
    .reset_index()
    .sort_values("delta")
)

# -- COLORS --
COLOR_POS = "#1B4F72"
COLOR_NEG = "#C0392B"
colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in premium["delta"]]

# -- PLOT --
fig, ax = plt.subplots(figsize=(11, 8))
fig.subplots_adjust(left=0.18, right=0.88, top=0.78, bottom=0.14)

bars = ax.barh(premium["state"], premium["delta"],
               color=colors, edgecolor="white", linewidth=0.4, height=0.65)

# Value labels
# Positive bars: label just outside (right) of the bar tip
# Negative bars: label just inside (right of the tip), avoiding y-axis overlap
for bar, val in zip(bars, premium["delta"]):
    sign = "+" if val >= 0 else ""
    if val >= 0:
        x_pos = bar.get_width() + 0.003
        ha = "left"
        txt_color = "#222"
    else:
        x_pos = bar.get_width() + 0.004   # just inside the bar, right of tip
        ha = "left"
        txt_color = "white"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.3f}", va="center", ha=ha,
            fontsize=8.5, fontweight="bold", color=txt_color)

# Zero line and grid
ax.axvline(0, color="#111", linewidth=1.0, zorder=3)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6, color="#aaa")
ax.set_axisbelow(True)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=10)
ax.tick_params(axis="x", labelsize=9)

# -- AXIS LABEL --
ax.set_xlabel(
    "Delta Avg. Unemployment Volatility\n"
    "(High-Disaster Years minus Low-Disaster Years, Std. Dev. of Monthly Rate)",
    fontsize=9.5, labelpad=10
)

# -- TITLE & SUBTITLE --
# Subtitle sits at y=0.97, main title at y=0.90, subplot block top=0.78 — clear separation
fig.text(0.53, 0.97,
         "Bars show the difference in avg. unemployment volatility between "
         "high-disaster years (top 40%) and low-disaster years",
         ha="center", va="top", fontsize=8.5, color="#555", style="italic")
fig.text(0.53, 0.91,
         "Disaster-Year Volatility Premium by State\n"
         "Top 15 High-Exposure States, 2006-2025 (excl. 2020)",
         ha="center", va="top", fontsize=13, fontweight="bold", linespacing=1.4)

# -- LEGEND --
legend_handles = [
    mpatches.Patch(color=COLOR_POS,
                   label="Higher volatility in disaster years (positive effect)"),
    mpatches.Patch(color=COLOR_NEG,
                   label="Lower volatility in disaster years (negative effect)"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=8.5, framealpha=0.95, edgecolor="#ccc")

# -- SOURCE NOTE --
fig.text(0.18, 0.01,
         "Source: BLS LAUS; FEMA Disaster Declarations Database.",
         fontsize=8, color="#666", style="italic")

# -- SAVE --
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.2)
plt.show()
plt.close()
print(f"Saved {OUTPUT_PATH}")
print(premium[["state", "delta"]].to_string(index=False))
