import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ----------------------------
# SETTINGS
# ----------------------------
OUTDIR = Path("visual png")
OUTDIR.mkdir(exist_ok=True)

DAMAGE_FILE = "top15_states_2010_2024.csv"
RECOVERY_FILE = "recovery_speed_top15_2010_2024.csv"

OUTPNG = OUTDIR / "05_nonlinear_damage_effects_POLISHED.png"

# ----------------------------
# LOAD DATA
# ----------------------------
damage = pd.read_csv(DAMAGE_FILE)
recovery = pd.read_csv(RECOVERY_FILE)

# Standardize column names safely
damage = damage.rename(columns={damage.columns[0]: "state", damage.columns[1]: "damage_per_capita"})
recovery = recovery.rename(columns={recovery.columns[0]: "state", recovery.columns[1]: "months_to_recover"})

damage["state"] = damage["state"].astype(str).str.upper().str.strip()
recovery["state"] = recovery["state"].astype(str).str.upper().str.strip()

df = damage.merge(recovery, on="state", how="inner")
print("\n--- DEBUG: Merged DataFrame ---")
print(df)
print(f"Merged rows: {len(df)}")

# Check for missing states after merge
expected_states = set(damage["state"])
actual_states = set(df["state"])
missing_states = expected_states - actual_states
if missing_states:
    print("States missing after merge:", missing_states)
else:
    print("All states present after merge.")

# Ensure numeric + drop missing
df["damage_per_capita"] = pd.to_numeric(df["damage_per_capita"], errors="coerce")
# Robust check for months_to_recover column type
months_col = df["months_to_recover"]
if isinstance(months_col, pd.DataFrame):
    months_col = months_col.iloc[:, 0]
df["months_to_recover"] = pd.to_numeric(months_col, errors="coerce")
df = df.dropna(subset=["damage_per_capita", "months_to_recover"]).copy()

# Safety: remove duplicated columns if any
df = df.loc[:, ~df.columns.duplicated()].copy()

# ----------------------------
# BUILD X/Y (FOR LINE FITS)
# ----------------------------
x = df["damage_per_capita"].to_numpy(dtype=float)
y = df["months_to_recover"].to_numpy(dtype=float)

mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
x = x[mask]
y = y[mask]

lx = np.log(x)

# ----------------------------
# FIT 1: Linear on log(x)
# ----------------------------
slope1, intercept1, r1, p1, _ = stats.linregress(lx, y)
yhat_linear = intercept1 + slope1 * lx
r2_linear = 1 - np.sum((y - yhat_linear) ** 2) / np.sum((y - np.mean(y)) ** 2)

# ----------------------------
# FIT 2: Quadratic on log(x)  (nonlinear)
# ----------------------------
coef2 = np.polyfit(lx, y, deg=2)  # a*lx^2 + b*lx + c
yhat_quad = np.polyval(coef2, lx)
r2_quad = 1 - np.sum((y - yhat_quad) ** 2) / np.sum((y - np.mean(y)) ** 2)

# AIC helper (lower = better)
def aic(n, rss, k):
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k

n = len(y)
rss_linear = np.sum((y - yhat_linear) ** 2)
rss_quad = np.sum((y - yhat_quad) ** 2)

aic_linear = aic(n, rss_linear, k=2)  # intercept + slope
aic_quad = aic(n, rss_quad, k=3)      # a,b,c

# ----------------------------
# SMOOTH CURVES FOR PLOTTING
# ----------------------------
lx_grid = np.linspace(lx.min(), lx.max(), 300)
x_grid = np.exp(lx_grid)

y_grid_linear = intercept1 + slope1 * lx_grid
y_grid_quad = np.polyval(coef2, lx_grid)

# ----------------------------
# COLORS PER STATE + TABLE
# ----------------------------
df_plot = df.copy().sort_values("state").reset_index(drop=True)
colors = cm.tab20(np.linspace(0, 1, len(df_plot)))

state_to_color = {df_plot.loc[i, "state"]: colors[i] for i in range(len(df_plot))}

# ----------------------------
# FIGURE LAYOUT (plot + state table)
# ----------------------------
fig = plt.figure(figsize=(14.5, 7.5))
gs = GridSpec(1, 2, width_ratios=[3.4, 1.2], wspace=0.05)

ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ----------------------------
# SCATTER (NO LABELS ON POINTS)
# ----------------------------
ax.scatter(
    df_plot["damage_per_capita"],
    df_plot["months_to_recover"],
    s=110,
    alpha=0.92,
    c=[state_to_color[s] for s in df_plot["state"]],
    edgecolor="#9ecae1",      # light-blue ring like you want
    linewidth=1.4
)

# X-axis log scale
ax.set_xscale("log")

# ----------------------------
# LINES
# ----------------------------
ax.plot(
    x_grid, y_grid_linear,
    linestyle="--",
    linewidth=2.2,
    label="Linear fit (log-x)"
)

ax.plot(
    x_grid, y_grid_quad,
    linewidth=2.6,
    label="Nonlinear fit (quadratic on log-x)"
)

# ----------------------------
# TITLES
# ----------------------------
fig.suptitle(
    "Nonlinear Damage Effects: Does Higher Damage Change Recovery Behavior?",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

ax.set_title(
    "X-axis uses log scale to compare states with very different damage levels.",
    fontsize=12.5,
    color="#5a6773",
    pad=16
)

# ----------------------------
# AXES
# ----------------------------
ax.set_xlabel("Total disaster damage per capita (USD per person, log scale)", labelpad=10)
ax.set_ylabel("Recovery time (year)", labelpad=10)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Set y-axis to show integer years only
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Grid + spines
ax.grid(alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ----------------------------
# LINE LEGEND (move away from lines)
# ----------------------------
ax.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.98,
    facecolor="white",
    edgecolor="#cfd8e3"
)

# ----------------------------
# STATE COLOR TABLE (RIGHT)
# ----------------------------

ax2.text(0.02, 0.98, "States (Top 15)", fontsize=16, fontweight="bold", va="top", color="#1f2d3d")

row_h = 0.055
top = 0.94

for i, state in enumerate(df_plot["state"]):
    y0 = top - (i + 1) * row_h

    # color square
    ax2.add_patch(
        mpatches.Rectangle(
            (0.02, y0),
            0.08, row_h * 0.70,
            color=state_to_color[state],
            ec="#1f2d3d",
            lw=0.6
        )
    )
    # state text
    ax2.text(0.13, y0 + row_h * 0.35, state, fontsize=12, va="center", ha="left", color="#1f2d3d")

# border around legend/table area
ax2.add_patch(
    mpatches.Rectangle(
        (0.01, top - (len(df_plot) + 1) * row_h),
        0.95,
        (len(df_plot) + 1) * row_h,
        fill=False,
        lw=1.2,
        ec="#cfd8e3"
    )
)

# Move stats box below the state color table
stats_text = (
    f"Linear (log-x): r = {r1:.2f}, p = {p1:.3f}\n"
    f"R² linear = {r2_linear:.2f}, R² nonlinear = {r2_quad:.2f}\n"
    f"AIC linear = {aic_linear:.1f}, AIC nonlinear = {aic_quad:.1f}\n"
    f"(Lower AIC = better fit)"
)

ax2.text(
    0.02, top - (len(df_plot) + 2) * row_h,
    stats_text,
    fontsize=11,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cfd8e3", alpha=0.98)
)

# ----------------------------
# SAVE
# ----------------------------
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPNG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUTPNG}")