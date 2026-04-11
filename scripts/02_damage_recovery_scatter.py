import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

OUTDIR = Path("visual png")
OUTDIR.mkdir(exist_ok=True)

# ----------------------------
# LOAD FILES
# ----------------------------
recovery = pd.read_csv("recovery_speed_top15_2010_2024.csv")
fema = pd.read_csv("data/BLS/laus_with_fema_disaster_exposure_2006_2025.csv")
pop = pd.read_csv("data/Census/state_population_acs_2010_2024.csv")

# ----------------------------
# STANDARDIZE COLUMN NAMES SAFELY
# (so the script works even if your column names differ slightly)
# ----------------------------
def pick_col(df, keywords):
    cols = df.columns
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    raise ValueError(f"Could not find a column containing any of: {keywords}")

# recovery columns
rec_state = pick_col(recovery, ["state"])
rec_shock = pick_col(recovery, ["shock"])
rec_months = pick_col(recovery, ["months"])

recovery = recovery.rename(columns={
    rec_state: "state",
    rec_shock: "shock_year",
    rec_months: "months_to_recover"
})

# fema columns
fema_state = pick_col(fema, ["state"])
fema_year  = pick_col(fema, ["year"])
fema_exp   = pick_col(fema, ["exposure"])          # expects disaster_exposure_12m
fema_cnt   = pick_col(fema, ["count"])             # expects disaster_count_month

fema = fema.rename(columns={
    fema_state: "state",
    fema_year: "year",
    fema_exp: "disaster_exposure_12m",
    fema_cnt: "disaster_count_month"
})

# population columns
pop_state = pick_col(pop, ["state"])
pop_year  = pick_col(pop, ["year"])
pop_pop   = pick_col(pop, ["pop"])

pop = pop.rename(columns={
    pop_state: "state",
    pop_year: "year",
    pop_pop: "population"
})

# Standardize state strings
recovery["state"] = recovery["state"].astype(str).str.upper().str.strip()
fema["state"]     = fema["state"].astype(str).str.upper().str.strip()
pop["state"]      = pop["state"].astype(str).str.upper().str.strip()

# ----------------------------
# AGGREGATE FEMA EXPOSURE BY STATE-YEAR
# ----------------------------
fema_year = (
    fema.groupby(["state", "year"])
        .agg(
            exposure12m_mean=("disaster_exposure_12m", "mean"),
            disaster_month_sum=("disaster_count_month", "sum")
        )
        .reset_index()
)

# ----------------------------
# MERGE: recovery + FEMA exposure in the shock year
# ----------------------------
df = recovery.merge(
    fema_year,
    left_on=["state", "shock_year"],
    right_on=["state", "year"],
    how="left"
).drop(columns=["year"])

# Merge population in the same shock year
df = df.merge(
    pop,
    left_on=["state", "shock_year"],
    right_on=["state", "year"],
    how="left"
).drop(columns=["year"])

# Drop rows missing key values
df = df.dropna(subset=["exposure12m_mean", "population", "months_to_recover"]).copy()

# Per-million exposure
df["exposure12m_per_million"] = df["exposure12m_mean"] / (df["population"] / 1_000_000)

# ----------------------------
# COLORS: assign one color per state (stable order)
# ----------------------------
df = df.sort_values("state").reset_index(drop=True)
state_list = df["state"].tolist()

palette = cm.tab20(np.linspace(0, 1, len(state_list)))
state_to_color = {st: palette[i] for i, st in enumerate(state_list)}
point_colors = df["state"].map(state_to_color)

# ----------------------------
# PLOT LAYOUT (plot + legend table)
# ----------------------------
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14.2, 7.4), facecolor="white")
gs = GridSpec(1, 2, width_ratios=[3.25, 1.05], wspace=0.05)

ax = fig.add_subplot(gs[0])
ax.set_facecolor("white")

# Scatter points (colored fill + light-blue outline)
x = df["exposure12m_per_million"]
y = df["months_to_recover"]

ax.scatter(
    x, y,
    s=115,
    c=point_colors,
    alpha=0.95,
    edgecolor="#9ecae1",
    linewidth=1.4
)

# Trendline + stats (UNCHANGED conceptually)
slope, intercept, r, p, _ = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, linestyle="--", linewidth=2.2)

# Titles
fig.suptitle(
    "Recovery Speed vs FEMA Disaster Exposure (Top 15 States)",
    fontsize=22,
    fontweight="bold",
    y=0.98,
    ha="center"
)
fig.text(
    0.5, 0.93,
    "FEMA exposure is measured as average 12-month disaster exposure during the shock year (per million residents).",
    fontsize=12.5,
    color="#5a6773",
    ha="center",
    va="top"
)
ax.set_title("")

# Axis labels
ax.set_xlabel("FEMA disaster exposure (12-month avg) per 1,000,000 residents — shock year", labelpad=10)
ax.set_ylabel("Recovery time (months for unemployment to return near baseline)", labelpad=10)

# Grid + spines
ax.grid(alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# r and p box (kept, just looks clean)
ax.text(
    0.98, 0.98,
    f"r = {r:.2f}\n(p = {p:.3f})",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cfd8e3", alpha=0.98)
)

# Optional: label only a few extremes so it stays readable

# ----------------------------
# RIGHT SIDE: LEGEND TABLE
# ----------------------------
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_facecolor("white")

ax2.text(
    0.02, 0.98, "States (Top 15)",
    fontsize=13, fontweight="bold",
    ha="left", va="top", color="#1f2d3d"
)

row_h = 0.055
start_y = 0.93

for i, st in enumerate(state_list):
    y0 = start_y - i * row_h

    # Color box
    ax2.add_patch(
        mpatches.Rectangle(
            (0.02, y0 - 0.028),
            0.08, 0.032,
            facecolor=state_to_color[st],
            edgecolor="#2f3e4e",
            lw=0.6
        )
    )

    # State label
    ax2.text(
        0.13, y0 - 0.012,
        st,
        fontsize=11,
        ha="left", va="center",
        color="#1f2d3d"
    )

# Border around legend area
ax2.add_patch(
    mpatches.Rectangle(
        (0.01, 0.08), 0.94, 0.88,
        fill=False, lw=1.0, ec="#cfd8e3"
    )
)

# Save
outpath = OUTDIR / "03_recovery_speed_vs_fema_exposure_with_state_table.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {outpath}")
