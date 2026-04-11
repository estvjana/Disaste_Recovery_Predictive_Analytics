"""
=============================================================================
FIG 5: COVID-19 Labor Market Recovery — Fastest vs. Slowest Among High-Exposure States
=============================================================================
Two rows of 3 subplots each:
  Top row   : 3 slowest recovering states among the top 15
  Bottom row: 3 fastest recovering states among the top 15

Data required:
  - laus_with_fema_disaster_exposure_2006_2025.csv
  - recovery_speed_results.csv

Output: Fig5_labormarkercovid19.png (300 DPI)
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ──
DATA_PATH     = "data/BLS/laus_with_fema_disaster_exposure_2006_2025.csv"
RECOVERY_PATH = "recovery_speed_results.csv"
OUTPUT_PATH   = "visual png/Fig5_labormarkercovid19.png"

TOP_15 = [
    "Texas", "Kentucky", "Missouri", "Oklahoma", "Georgia",
    "Louisiana", "Florida", "Kansas", "North Carolina", "Virginia",
    "Mississippi", "Nebraska", "Iowa", "Arkansas", "Tennessee"
]

# ── LOAD DATA ──
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
rec = pd.read_csv(RECOVERY_PATH)

# Filter to top 15 states only
df_top = df[df["state"].isin(TOP_15)].copy()
rec_top = rec[rec["state"].isin(TOP_15)].copy()

# Sort by recovery months and select slowest / fastest
rec_top_sorted = rec_top.sort_values("recovery_months", ascending=False)
slowest = rec_top_sorted.head(3)["state"].tolist()   # e.g. Texas, Florida, Virginia
fastest = rec_top_sorted.tail(3)["state"].tolist()   # e.g. Nebraska, Kentucky, Mississippi
fastest = fastest[::-1]  # show fastest first (fewest months first)

six_states = slowest + fastest

# ── BUILD TRAJECTORY per state ──
# t=0 = March 2020; range: t=-6 to t=+30
COVID_DATE = pd.Timestamp("2020-03-01")

def get_trajectory(state_name):
    s = df_top[df_top["state"] == state_name].copy()
    s = s.sort_values("date")
    s["t"] = ((s["date"].dt.year - COVID_DATE.year) * 12 +
              (s["date"].dt.month - COVID_DATE.month))
    s = s[(s["t"] >= -6) & (s["t"] <= 30)]
    return s[["t", "unemployment_rate"]].set_index("t")["unemployment_rate"]

# ── FIGURE SETUP ──
# left=0.08, right=0.97 centers the subplot block; top=0.89 gives room below the title
fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust(top=0.89, bottom=0.30, hspace=0.55, wspace=0.32,
                    left=0.08, right=0.97)

gs = gridspec.GridSpec(2, 3, figure=fig)

COLOR_SLOW = "#C0392B"
COLOR_FAST = "#1565C0"
COLOR_FILL_SLOW = "#FADBD8"
COLOR_FILL_FAST = "#D6EAF8"

row_labels = ["SLOWEST\nRECOVERY", "FASTEST\nRECOVERY"]
row_colors = [COLOR_SLOW, COLOR_FAST]

for i, state in enumerate(six_states):
    row = 0 if i < 3 else 1
    col = i % 3
    ax = fig.add_subplot(gs[row, col])

    traj = get_trajectory(state)
    rec_row = rec_top[rec_top["state"] == state].iloc[0]
    baseline = rec_row["baseline_unemployment"]
    shock    = rec_row["shock_unemployment"]
    months   = int(rec_row["recovery_months"])

    color      = COLOR_SLOW if row == 0 else COLOR_FAST
    fill_color = COLOR_FILL_SLOW if row == 0 else COLOR_FILL_FAST

    t_vals = traj.index.values
    u_vals = traj.values

    # Shaded area under curve
    ax.fill_between(t_vals, baseline, u_vals,
                    where=(u_vals >= baseline),
                    color=fill_color, alpha=0.7, zorder=1)

    # Main trajectory line
    ax.plot(t_vals, u_vals, color=color, linewidth=1.8, zorder=3)

    # Baseline dashed line
    ax.axhline(baseline, color="#888", linewidth=1.0, linestyle="--", zorder=2)

    # COVID shock vertical line
    ax.axvline(0, color=color, linewidth=1.2, zorder=2)

    # Find recovery month (first t > 0 where rate <= baseline)
    post = traj[traj.index > 0]
    recovered = post[post <= baseline]
    if len(recovered) > 0:
        rec_t = recovered.index[0]
        rec_u = recovered.iloc[0]
        ax.scatter([rec_t], [rec_u], color="#27AE60", s=70, zorder=5)
        ax.annotate(f"{rec_t} months",
                    xy=(rec_t, rec_u),
                    xytext=(rec_t - 4, rec_u + (shock - baseline) * 0.18),
                    fontsize=8, color="#27AE60", fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#27AE60",
                                   lw=1.0, connectionstyle="arc3,rad=-0.2"))

    # Peak annotation
    peak_t = t_vals[np.argmax(u_vals)]
    ax.text(peak_t + 1.0, shock + (shock - baseline) * 0.04,
            f"Peak: {shock:.1f}%", fontsize=7.5, color="#555", va="bottom")

    # Info box
    info = (f"Recovery: {months} months\n"
            f"Baseline: {baseline:.1f}%\n"
            f"Peak shock: {shock:.1f}%")
    ax.text(0.97, 0.97, info,
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#aaa", alpha=0.9))

    # Title
    ax.set_title(f"{state}\n[{months} months to recover]",
                 fontsize=10, fontweight="bold", color=color, pad=6)

    # X-axis formatting
    ax.set_xticks([-6, 0, 6, 12, 18, 24, 30])
    ax.set_xticklabels(["-6", "t=0\nCOVID", "+6", "+12", "+18", "+24", "+30"],
                       fontsize=7.5)
    ax.set_xlim(-7, 32)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # Y-axis label only for leftmost column
    if col == 0:
        ax.set_ylabel("Unemployment Rate (%)", fontsize=9)

    # X-axis label: handled by fig.supxlabel below

# Row labels (rotated, on far left)
# With top=0.89, bottom=0.30, hspace=0.55 across 2 rows:
#   each row height ≈ (0.89-0.30)/2.55 ≈ 0.231
#   top row center ≈ 0.89 - 0.231/2 ≈ 0.775
#   bottom row center ≈ 0.30 + 0.231/2 ≈ 0.415
row_y_centers = [0.775, 0.43]
for r, (label, clr, yc) in enumerate(zip(row_labels, row_colors, row_y_centers)):
    fig.text(0.005, yc, label,
             ha="left", va="center", fontsize=10, fontweight="bold",
             color=clr, rotation=90)

# ── TITLE ──
# y=0.955 sits just above top=0.89, giving a clear gap before the first row of panels
fig.suptitle("COVID-19 Labor Market Recovery: Fastest vs. Slowest Among High-Exposure States",
             fontsize=12, fontweight="bold", y=0.955)

# ── SHARED X-AXIS LABEL ──
# x=0.525 is the midpoint of left=0.08..right=0.97; y sits just below the subplot block
fig.text(0.525, 0.248, "Months from COVID shock (March 2020)",
         ha="center", va="top", fontsize=10)

# ── LEGEND (directly below x-axis label) ──
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=COLOR_SLOW, linewidth=2,
           label="Slow recovery states (18–20 months)"),
    Line2D([0], [0], color=COLOR_FAST, linewidth=2,
           label="Fast recovery states (4–12 months)"),
    Line2D([0], [0], color=COLOR_SLOW, linewidth=1.2,
           label="COVID-19 shock (March 2020)"),
    Line2D([0], [0], color="#888", linewidth=1.0, linestyle="--",
           label="Pre-COVID baseline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#27AE60",
           markersize=8, label="Month of return to baseline"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, fontsize=9, framealpha=0.95,
           edgecolor="#ddd", bbox_to_anchor=(0.5, 0.155))

# ── SOURCE NOTE (below legend) ──
fig.text(0.525, 0.075,
         "Fig. 5. Monthly unemployment rate trajectories for six high-exposure states "
         "following the COVID-19 shock (March 2020); top row shows three slowest recoverers, "
         "bottom row shows three fastest.\n"
         "All states received the same shock simultaneously; differences in recovery speed "
         "reflect structural factors including economic composition and institutional capacity. "
         "Source: BLS LAUS.",
         ha="center", va="top", fontsize=8.5, color="#555", style="italic",
         linespacing=1.6, wrap=True)

# ── SAVE ──
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.2)
plt.show()
plt.close()
print(f"✓ Saved {OUTPUT_PATH}")
print(f"  Slowest 3: {slowest}")
print(f"  Fastest 3: {fastest}")
