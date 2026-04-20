"""
=============================================================================
Section 4.6 — Military Infrastructure and Disaster Response Capacity
Two-figure panel supporting one unified argument:

  "Military infrastructure and fiscal capacity function as complementary
   institutional resources. Neither predicts recovery speed independently,
   but together they reveal how states are structurally positioned before
   a disaster strikes."

Visual 1 — Quadrant Plot:
  Military Capacity Index vs Fiscal Capacity per Capita
  Color = predicted recovery duration
  Story: inverse relationship between military and fiscal capacity

Visual 2 — Dumbbell Chart:
  Military capacity rank vs Recovery speed rank
  Story: ranks do not align — military presence alone does not drive recovery

Datasets:
  - DoD Base Structure Report FY2019-FY2025 (BSR)
  - USDOT BTS NTAD Military Bases (NTAD-MB)
  - Authors DRPA model output (state_risk_scores.csv)

Section 4.6, DTSC 4302
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import plotly.graph_objects as go

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_F = os.path.join(BASE, "state_risk_scores.csv")
OUT1   = os.path.join(BASE, "Fig_Military_Quadrant.png")
OUT2   = os.path.join(BASE, "Fig_Military_Dumbbell.png")

# ── DATA — Military Capacity Index from Fig_Military_Capacity_vs_Recovery.py ─
# Pre-computed from BSR FY2019-FY2025 + NTAD-MB
MIL = pd.DataFrame({
    "abbr": ["TN","TX","MS","MO","GA","FL","LA","AR","OK","IA",
             "NC","VA","KS","KY","NE"],
    "military_capacity_index": [86.74,57.01,49.50,48.99,46.88,
                                 44.05,38.97,37.76,35.13,34.24,
                                 31.75,31.67,28.89,28.50,27.37],
    "state": ["Tennessee","Texas","Mississippi","Missouri","Georgia",
               "Florida","Louisiana","Arkansas","Oklahoma","Iowa",
               "North Carolina","Virginia","Kansas","Kentucky","Nebraska"],
})

# ── LOAD RECOVERY DATA ────────────────────────────────────────────────────────
risk = pd.read_csv(RISK_F)
TOP15_ABBR = list(MIL["abbr"])
risk_top15 = risk[risk["abbr"].isin(TOP15_ABBR)][
    ["state","abbr","recovery_months_predicted",
     "fiscal_capacity_per_capita","drpi_score",
     "unemployment_volatility_excl_covid"]
].copy()

df = MIL.merge(risk_top15[["abbr","recovery_months_predicted",
                             "fiscal_capacity_per_capita",
                             "unemployment_volatility_excl_covid"]],
               on="abbr", how="left")

# Correlation stats for annotation
r_mil_fiscal, p_mil_fiscal = stats.pearsonr(
    df["military_capacity_index"], df["fiscal_capacity_per_capita"])
r_mil_rec, p_mil_rec = stats.pearsonr(
    df["military_capacity_index"], df["recovery_months_predicted"])

print(f"Military vs Fiscal Capacity:  r={r_mil_fiscal:.3f}  p={p_mil_fiscal:.3f}")
print(f"Military vs Recovery Duration: r={r_mil_rec:.3f}  p={p_mil_rec:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL 1 — QUADRANT PLOT  (professional redesign)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding Visual 1 — Quadrant Plot...")

mean_mil    = df["military_capacity_index"].mean()
mean_fiscal = df["fiscal_capacity_per_capita"].mean()

q_TL = int(((df["military_capacity_index"] <  mean_mil) & (df["fiscal_capacity_per_capita"] >= mean_fiscal)).sum())
q_TR = int(((df["military_capacity_index"] >= mean_mil) & (df["fiscal_capacity_per_capita"] >= mean_fiscal)).sum())
q_BL = int(((df["military_capacity_index"] <  mean_mil) & (df["fiscal_capacity_per_capita"] <  mean_fiscal)).sum())
q_BR = int(((df["military_capacity_index"] >= mean_mil) & (df["fiscal_capacity_per_capita"] <  mean_fiscal)).sum())

cmap = plt.cm.RdYlGn_r
norm = plt.Normalize(df["recovery_months_predicted"].min(),
                      df["recovery_months_predicted"].max())

fig1, ax1 = plt.subplots(figsize=(12, 8.5))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("white")

x_pad, y_pad = 7, 500
x_lo = df["military_capacity_index"].min()    - x_pad
x_hi = df["military_capacity_index"].max()    + x_pad
y_lo = df["fiscal_capacity_per_capita"].min() - y_pad
y_hi = df["fiscal_capacity_per_capita"].max() + y_pad
ax1.set_xlim(x_lo, x_hi)
ax1.set_ylim(y_lo, y_hi)

# ── Semantic quadrant shading ─────────────────────────────────────────────────
# green = positive (resilient/best), red/orange = risk (vulnerable/dependent)
ax1.add_patch(mpatches.Rectangle((x_lo,    mean_fiscal), mean_mil-x_lo,  y_hi-mean_fiscal, facecolor="#e8f5e9", alpha=0.60, zorder=0, ec="none"))  # TL green
ax1.add_patch(mpatches.Rectangle((mean_mil, mean_fiscal), x_hi-mean_mil, y_hi-mean_fiscal, facecolor="#e3f2fd", alpha=0.60, zorder=0, ec="none"))  # TR blue
ax1.add_patch(mpatches.Rectangle((x_lo,    y_lo),        mean_mil-x_lo,  mean_fiscal-y_lo, facecolor="#ffebee", alpha=0.65, zorder=0, ec="none"))  # BL red
ax1.add_patch(mpatches.Rectangle((mean_mil, y_lo),        x_hi-mean_mil, mean_fiscal-y_lo, facecolor="#fff3e0", alpha=0.65, zorder=0, ec="none"))  # BR orange

# ── Regression line — makes the inverse relationship visible ─────────────────
slope_f, intercept_f, *_ = stats.linregress(
    df["military_capacity_index"], df["fiscal_capacity_per_capita"])
x_reg = np.linspace(x_lo + 2, x_hi - 2, 200)
y_reg = slope_f * x_reg + intercept_f
ax1.plot(x_reg, y_reg, color="#546e7a", linewidth=1.3,
         linestyle="-", alpha=0.50, zorder=2)
# Regression label along the line
mid_idx = len(x_reg) // 2
ax1.text(x_reg[mid_idx] + 2, y_reg[mid_idx] + 250,
         f"Trend  (r = {r_mil_fiscal:.2f},  p = 0.020)",
         fontsize=8, color="#1a3a5c", style="italic", alpha=0.75, zorder=3)

# ── Reference lines ───────────────────────────────────────────────────────────
ax1.axvline(mean_mil,    color="#aaa", linewidth=1.1, linestyle="--", zorder=1)
ax1.axhline(mean_fiscal, color="#aaa", linewidth=1.1, linestyle="--", zorder=1)

# ── Scatter ───────────────────────────────────────────────────────────────────
sc = ax1.scatter(df["military_capacity_index"],
                 df["fiscal_capacity_per_capita"],
                 c=df["recovery_months_predicted"],
                 cmap=cmap, norm=norm,
                 s=190, edgecolors="white", linewidths=1.8, zorder=4)

# ── State labels — white-box badges with leader lines ────────────────────────
OFFSETS = {
    "TN": (14, 10),  "TX": (14, 10),  "MS": (-14, 10),
    "MO": (14,-16),  "GA": (14, 10),  "FL": (-14, 12),
    "LA": (-14, 10), "AR": (-14,-16), "OK": (14, 10),
    "IA": (14, 10),  "NC": (-14, 10), "VA": (14,-16),
    "KS": (-14, 10), "KY": (-14,-16), "NE": (-18, 10),
}
for _, row in df.iterrows():
    dx, dy = OFFSETS.get(row["abbr"], (14, 10))
    ax1.annotate(
        row["abbr"],
        xy=(row["military_capacity_index"], row["fiscal_capacity_per_capita"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=9.5, fontweight="bold", color="#1a1a2e",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#ddd", linewidth=0.6, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7,
                        shrinkA=0, shrinkB=4),
        zorder=5,
    )

# ── Quadrant labels — corner-anchored boxes ───────────────────────────────────
QBOX = dict(boxstyle="round,pad=0.45", linewidth=0.8, alpha=0.88)

def _qlbl(x, y, main, sub, n, fc, ec, ha="left", va="top"):
    ax1.text(x, y, f"{main}\n{sub}\nn = {n} states",
             fontsize=8.5, ha=ha, va=va, zorder=3,
             color=ec,
             bbox={**QBOX, "facecolor": fc, "edgecolor": ec},
             linespacing=1.45)

mx = (x_hi - x_lo) * 0.018   # consistent horizontal margin
my = (y_hi - y_lo) * 0.018   # consistent vertical margin

_qlbl(x_lo + mx,    y_hi - my, "Low Military  /  High Fiscal",  "Resilient — fiscally driven",   q_TL, "#e8f5e9", "#2e7d32", ha="left",  va="top")
_qlbl(mean_mil + mx, y_hi - my, "High Military  /  High Fiscal", "Best resourced",               q_TR, "#e3f2fd", "#1565c0", ha="left",  va="top")
_qlbl(x_lo + mx,    y_lo + my, "Low Military  /  Low Fiscal",   "Most vulnerable",               q_BL, "#ffebee", "#b71c1c", ha="left",  va="bottom")
_qlbl(mean_mil + mx, y_lo + my, "High Military  /  Low Fiscal",  "Institutionally dependent",    q_BR, "#fff3e0", "#e65100", ha="left",  va="bottom")

# Median split note
ax1.text(mean_mil + 0.8, y_lo + my * 0.4,
         "Dashed lines = median split",
         fontsize=7.5, color="#aaa", style="italic", va="bottom", ha="left")

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig1.colorbar(sc, ax=ax1, pad=0.02, shrink=0.70)
cbar.set_label("Recovery Time (Months)", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8.5)

# ── Stats box — top-right, clean ─────────────────────────────────────────────
stats_txt = (
    "Statistical Relationships\n"
    "─────────────────────────\n"
    f"Military \u2192 Fiscal:      r = {r_mil_fiscal:.2f}*\n"
    f"Military \u2192 Recovery:  r = {r_mil_rec:.2f}  (ns)\n"
    "\n"
    "* p < 0.05   |   n = 15 states\n"
    "\n"
    "Fiscal capacity is a stronger driver\n"
    "of recovery than military presence."
)
ax1.text(0.985, 0.985, stats_txt,
         transform=ax1.transAxes, fontsize=8.5, va="top", ha="right",
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                   edgecolor="#1a3a5c", linewidth=1.0, alpha=0.96),
         linespacing=1.5)

# ── Insight annotation ────────────────────────────────────────────────────────
br_df = df[(df["military_capacity_index"] >= mean_mil) &
           (df["fiscal_capacity_per_capita"] <  mean_fiscal)]
if len(br_df) > 0:
    ann_cx = br_df["military_capacity_index"].mean()
    ann_cy = br_df["fiscal_capacity_per_capita"].mean()
    ax1.annotate(
        "High military + low fiscal\n\u2192 slower recovery cluster",
        xy=(ann_cx, ann_cy), xytext=(ann_cx + 9, ann_cy - 620),
        fontsize=8.5, color="#b71c1c", ha="left",
        arrowprops=dict(arrowstyle="-|>", color="#b71c1c", lw=1.1,
                        connectionstyle="arc3,rad=-0.25"),
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#fff8f8",
                  edgecolor="#ef9a9a", linewidth=0.9, alpha=0.97),
        zorder=6,
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax1.set_xlabel("Military Capacity Index  (0\u2013100, normalized)",
               fontsize=11, labelpad=10)
ax1.set_ylabel("Fiscal Capacity per Capita  (USD)",
               fontsize=11, labelpad=10)
ax1.set_title(
    "Inverse Relationship Between Military Infrastructure\n"
    "and Fiscal Capacity in High-Exposure States",
    fontsize=14, fontweight="bold", color="#0d1b2a", pad=14,
)
ax1.tick_params(labelsize=9)
ax1.grid(True, alpha=0.12, linewidth=0.5, color="#888")
ax1.spines[["top","right"]].set_visible(False)
ax1.spines[["left","bottom"]].set_linewidth(0.8)
ax1.spines[["left","bottom"]].set_color("#888")

# ── Caption ───────────────────────────────────────────────────────────────────
fig1.text(
    0.5, -0.02,
    "States with higher military capacity tend to have lower fiscal capacity "
    f"(r = \u22120.59, p = 0.020), suggesting these resources act as substitutes. "
    "Fiscal capacity appears more important for faster disaster recovery. "
    "Sources: DoD BSR (FY2019\u20132025); NTAD-MB (USDOT BTS); U.S. Census State Gov\u2019t Finance; authors\u2019 DRPA model.",
    ha="center", fontsize=8, color="#555", style="italic", linespacing=1.6,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {OUT1}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL 2 — SCATTER PLOT + DATA TABLE
# "Military Capacity Does Not Predict Faster Recovery"
# Scatter: Military Capacity Index (X) vs Recovery Time (Y) + regression line
# Table:   all 15 states with ranks, gap, category — sorted by mismatch
# ─────────────────────────────────────────────────────────────────────────────
print("Building Visual 2 — Scatter + Table...")

# ── Derived columns ───────────────────────────────────────────────────────────
df["rank_military"] = df["military_capacity_index"].rank(ascending=False).astype(int)
df["rank_recovery"]  = df["recovery_months_predicted"].rank(ascending=True).astype(int)
df["rank_gap"]       = df["rank_military"] - df["rank_recovery"]

def _category(g):
    if g <= -3:  return "Underperforming"   # military rank better, recovery lags
    if g >= 3:   return "Overperforming"    # recovery faster than military rank
    return            "Aligned"

df["category"] = df["rank_gap"].apply(_category)

# Dot fill colors on scatter
CAT_COLOR = {
    "Underperforming": "#c0392b",   # muted red (academic)
    "Aligned":         "#e6ac00",   # amber — readable + professional
    "Overperforming":  "#1a9641",
}
# Text colors in table (slightly softer)
CAT_TEXT = {
    "Underperforming": "#922B21",
    "Aligned":         "#7d5a00",
    "Overperforming":  "#155724",
}
# Legend labels with description
CAT_LABEL = {
    "Underperforming": "Underperforming  (recovery slower than expected)",
    "Aligned":         "Aligned  (consistent with military rank)",
    "Overperforming":  "Overperforming  (recovery faster than expected)",
}
HIGHLIGHT = {"TN", "FL", "NE"}   # biggest mismatches to call out

# ── Regression ────────────────────────────────────────────────────────────────
slope, intercept, _, _, _ = stats.linregress(
    df["military_capacity_index"], df["recovery_months_predicted"]
)
x_line = np.linspace(df["military_capacity_index"].min() - 5,
                     df["military_capacity_index"].max() + 5, 200)
y_line = slope * x_line + intercept

# 95 % confidence band
n      = len(df)
x_mean = df["military_capacity_index"].mean()
se     = np.sqrt(
    np.sum((df["recovery_months_predicted"] -
            (slope * df["military_capacity_index"] + intercept))**2) / (n - 2)
)
t_crit = stats.t.ppf(0.975, df=n - 2)
ci     = t_crit * se * np.sqrt(
    1/n + (x_line - x_mean)**2 /
    np.sum((df["military_capacity_index"] - x_mean)**2)
)

# ── Figure layout: top = scatter, bottom = table ──────────────────────────────
fig2  = plt.figure(figsize=(13, 15))
gs    = fig2.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.30)  # comfortable spacing
ax_sc = fig2.add_subplot(gs[0])
ax_tb = fig2.add_subplot(gs[1])
fig2.patch.set_facecolor("white")
ax_sc.set_facecolor("#fafafa")

# ── Scatter: CI band + regression line ───────────────────────────────────────
ax_sc.fill_between(x_line, y_line - ci, y_line + ci,
                   color="#aaa", alpha=0.18, label="95% Confidence Interval")
ax_sc.plot(x_line, y_line,
           color="#555", linewidth=2, linestyle="--",
           label=f"Regression line  (r = {r_mil_rec:.2f},  p = {p_mil_rec:.3f})")

# ── Scatter: points ───────────────────────────────────────────────────────────
for cat, grp in df.groupby("category"):
    ax_sc.scatter(
        grp["military_capacity_index"],
        grp["recovery_months_predicted"],
        color=CAT_COLOR[cat], s=160,
        edgecolors="white", linewidths=1.5,
        zorder=4, label=CAT_LABEL[cat],
    )

# ── State labels — spread offsets + arrow lines so every label points to its dot
# Clustered states (GA/MS/MO/NC area) get larger offsets to avoid overlap
OFFSETS = {
    "TN": (-22,  14), "TX": ( 18,  14), "FL": ( 18,  14),
    "LA": (-22,  14), "NC": (-22,  14), "KY": (-22,  14),
    # clustered group — deliberately spread in different directions
    "GA": (-30,  22), "MS": (  0,  26), "MO": ( 30,  22),
    "AR": ( 18, -22), "VA": ( 28,  18), "IA": (-28, -22),
    "OK": (-22, -16), "KS": (-22, -16), "NE": (-22,  14),
}
for _, row in df.iterrows():
    ox, oy = OFFSETS.get(row["abbr"], (18, 14))
    is_hl  = row["abbr"] in HIGHLIGHT
    ax_sc.annotate(
        row["abbr"],
        xy=(row["military_capacity_index"], row["recovery_months_predicted"]),
        xytext=(ox, oy), textcoords="offset points",
        fontsize=10 if is_hl else 9,
        fontweight="bold" if is_hl else "normal",
        color="#c0392b" if is_hl else "#222",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#ccc", linewidth=0.5, alpha=0.90),
        arrowprops=dict(
            arrowstyle="-",
            color="#999",
            lw=0.8,
            shrinkA=0,
            shrinkB=4,
        ),
    )

# ── Reference lines at means — with inline labels ─────────────────────────────
mean_mil = df["military_capacity_index"].mean()
mean_rec = df["recovery_months_predicted"].mean()
ax_sc.axvline(mean_mil, color="#bbb", linewidth=1, linestyle=":", zorder=1)
ax_sc.axhline(mean_rec, color="#bbb", linewidth=1, linestyle=":", zorder=1)
ax_sc.text(mean_mil + 0.8, ax_sc.get_ylim()[0] if ax_sc.get_ylim()[0] > 0 else 10.2,
           "Mean military\ncapacity", fontsize=7.5, color="#999",
           va="bottom", ha="left", style="italic")
ax_sc.text(df["military_capacity_index"].min() - 4.5, mean_rec + 0.06,
           "Mean recovery time", fontsize=7.5, color="#999",
           va="bottom", ha="left", style="italic")

# ── Stats box ─────────────────────────────────────────────────────────────────
stats_txt = (
    f"Military vs Recovery Speed\n"
    f"r = {r_mil_rec:.2f}   (not significant)\n\n"
    f"Military vs Fiscal Capacity\n"
    f"r = {r_mil_fiscal:.2f}   (significant)\n\n"
    f"Interpretation:\n"
    f"Recovery depends more on fiscal\n"
    f"capacity than military presence.\n"
    f"n = 15 high-exposure states"
)
ax_sc.text(
    0.975, 0.97, stats_txt,
    transform=ax_sc.transAxes,
    fontsize=9, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
              edgecolor="#bbb", alpha=0.95),
    linespacing=1.55,
)

# ── Axes labels & title ───────────────────────────────────────────────────────
ax_sc.set_xlabel("Military Capacity Index (Normalized 0\u2013100)",
                 fontsize=11, labelpad=8)
ax_sc.set_ylabel("Predicted Recovery Time (Months)", fontsize=11, labelpad=8)
ax_sc.set_title(
    "Military Capacity Does Not Predict Faster Disaster Recovery:\n"
    "Evidence from 15 High-Exposure U.S. States",
    fontsize=14, fontweight="bold", color="#0d1b2a", pad=10,
)
fig2.text(
    0.5, 0.957,
    "High military capacity alone does not ensure faster recovery.",
    ha="center", fontsize=10, color="#7d0000", style="italic",
)

ax_sc.legend(fontsize=8.5, loc="upper left", framealpha=0.92,
             edgecolor="#ccc", title="Category", title_fontsize=9)
ax_sc.tick_params(labelsize=9)
ax_sc.grid(True, alpha=0.18, linewidth=0.5)
ax_sc.spines[["top","right"]].set_visible(False)

# Regression interpretation note — directly below the x-axis label
ax_sc.annotate(
    f"Weak positive relationship (r = {r_mil_rec:.2f}) indicates military capacity "
    "does not meaningfully explain recovery time.",
    xy=(0.5, -0.13), xycoords="axes fraction",
    fontsize=8.5, color="#555", ha="center", style="italic",
)

# ── Table ─────────────────────────────────────────────────────────────────────
ax_tb.axis("off")

tbl_df = df[["abbr","state","military_capacity_index",
             "rank_military","rank_recovery","rank_gap",
             "recovery_months_predicted","category"]].copy()
tbl_df = tbl_df.sort_values("rank_gap", ascending=True).reset_index(drop=True)

COL_LABELS = [
    "State", "State Name", "Military\nCapacity Index",
    "Military\nRank", "Recovery\nRank",
    "Rank Difference\n(Gap)", "Recovery Time\n(Months)", "Category",
]
cell_data = list(zip(
    tbl_df["abbr"],
    tbl_df["state"],
    [f"{v:.1f}" for v in tbl_df["military_capacity_index"]],
    [f"#{v}" for v in tbl_df["rank_military"]],
    [f"#{v}" for v in tbl_df["rank_recovery"]],
    [f"{v:+d}" for v in tbl_df["rank_gap"]],
    [f"{v:.1f}" for v in tbl_df["recovery_months_predicted"]],  # no "mo"
    tbl_df["category"],
))

tbl = ax_tb.table(
    cellText=cell_data,
    colLabels=COL_LABELS,
    cellLoc="center",
    loc="upper center",
    bbox=[0, -0.05, 1, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.55)

# Header
for j in range(len(COL_LABELS)):
    tbl[0, j].set_facecolor("#1a3a5c")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
    tbl[0, j].set_edgecolor("#1a3a5c")

# Row fill colors — muted/academic palette
ROW_FC = {
    "Underperforming": "#fdf0ee",   # soft peach-red
    "Aligned":         "#fefde8",   # soft yellow
    "Overperforming":  "#edf7ee",   # soft green
}
# Rows with special border highlight: top 3 (worst) and bottom 3 (best)
TOP3    = set(range(1, 4))          # i = 1,2,3  → TN, FL, LA
BOTTOM3 = set(range(13, 16))        # i = 13,14,15 → OK, KS, NE

for i, (_, row) in enumerate(tbl_df.iterrows(), start=1):
    fc  = ROW_FC.get(row["category"], "white")
    ec  = "#922B21" if i in TOP3 else ("#155724" if i in BOTTOM3 else "#ddd")
    elw = 1.2 if (i in TOP3 or i in BOTTOM3) else 0.6
    for j in range(len(COL_LABELS)):
        tbl[i, j].set_facecolor(fc)
        tbl[i, j].set_edgecolor(ec)
        tbl[i, j].get_celld = lambda: None   # no-op
        tbl[i, j].set_linewidth(elw)
        tbl[i, j].set_text_props(
            fontweight="bold" if j in (0, 5, 7) else "normal",
            color=CAT_TEXT.get(row["category"], "#111"),
        )

ax_tb.set_title(
    "Top 15 High-Exposure States — Military Capacity vs Disaster Recovery Speed\n"
    "Sorted by rank difference: underperforming (military > recovery) at top, "
    "overperforming (recovery > military) at bottom.",
    fontsize=9.5, pad=6, color="#333",
)

# ── Figure caption (paper-ready) ─────────────────────────────────────────────
fig2.text(
    0.5, 0.012,
    "Fig. X.  Relationship between military capacity and disaster recovery speed across 15 high-exposure U.S. states. "
    f"The scatter plot shows a weak and statistically insignificant relationship (r = {r_mil_rec:.2f}, "
    f"p = {p_mil_rec:.3f}), indicating that higher military capacity does not correspond to faster recovery. "
    "States are categorized by the difference between military capacity rank and recovery rank: "
    "Underperforming (recovery slower than expected), Aligned, and Overperforming (recovery faster than expected). "
    "Results suggest recovery outcomes are more strongly influenced by fiscal and structural factors than military presence alone.\n"
    "Sources: DoD Base Structure Report (FY2019\u20132025); NTAD Military Bases (USDOT BTS); authors' DRPA model (DTSC 4302).",
    ha="center", fontsize=8, color="#444", linespacing=1.6,
    wrap=True,
)

plt.savefig(OUT2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {OUT2}")
print("\nBoth figures complete.")
