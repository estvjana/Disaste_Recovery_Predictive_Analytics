"""
Fig_Aid_Recovery_Slope.py
Section 5.3 — Aid-to-Recovery Rank Gap
Clean diverging bar — all elements inside, zero overlaps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────
data = [
    ('LA', 1,   7,  16),
    ('FL', 2,  14,  19),
    ('TX', 3,  15,  20),
    ('MS', 4,   5,  12),
    ('NC', 5,   9,  17),
    ('GA', 6,   3,  12),
    ('KY', 7,   2,   6),
    ('TN', 8,  11,  17),
    ('VA', 9,  13,  18),
    ('OK',10,  10,  17),
    ('MO',11,   8,  17),
    ('AR',12,   6,  16),
    ('IA',13,  12,  17),
    ('KS',14,   4,  12),
    ('NE',15,   1,   4),
]

rows    = sorted(data, key=lambda x: (x[1] - x[2]))
abbrs   = [r[0] for r in rows]
aid_rks = [r[1] for r in rows]
rec_rks = [r[2] for r in rows]
gaps    = [r[1] - r[2] for r in rows]
y       = np.arange(len(rows))

r_val, p_val = stats.pearsonr(aid_rks, rec_rks)

def bar_color(g):
    if g >  2: return '#2e7d32'
    if g < -2: return '#c62828'
    return '#90a4ae'

colors = [bar_color(g) for g in gaps]

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
plt.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.14)

# ── Background shading ────────────────────────────────────────────────
ax.axvspan(-16.5, 0,   alpha=0.035, color='#c62828', zorder=0)
ax.axvspan(0,  16.5,   alpha=0.035, color='#2e7d32', zorder=0)

# ── Bars ──────────────────────────────────────────────────────────────
ax.barh(y, gaps, color=colors, height=0.60,
        edgecolor='white', linewidth=0.5, zorder=3)

# ── Zero line ─────────────────────────────────────────────────────────
ax.axvline(0, color='#37474f', linewidth=1.8, zorder=4)

# ── Gap value labels ──────────────────────────────────────────────────
for i, g in enumerate(gaps):
    if g == 0:
        continue
    lbl  = f'+{g}' if g > 0 else str(g)
    xpos = g + (0.55 if g > 0 else -0.55)
    ha   = 'left' if g > 0 else 'right'
    ax.text(xpos, i, lbl,
            ha=ha, va='center', fontsize=8.5,
            fontweight='bold', color=colors[i])

# ── Y axis — state abbreviations only ────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(abbrs, fontsize=11, fontweight='bold', color='#1a1a2e')
ax.tick_params(axis='y', length=0, pad=6)

# ── Axis limits ────────────────────────────────────────────────────────
ax.set_xlim(-16.5, 16.5)
ax.set_ylim(-0.6, len(rows) - 0.2)

# ── Stats box — RIGHT side inside empty area (FL/TX rows go left, right is empty)
# FL is y=0, TX is y=1, LA is y=2 — all have negative bars so x>0 is empty
p_str     = f'p = {p_val:.3f}'
stats_txt = (
    f'Aid vs Recovery:  r = {r_val:+.2f}  ({p_str})\n'
    f'Aid rank does not explain recovery speed\n'
    f'n = 15 high-exposure states'
)
ax.text(2.5, 1.0, stats_txt,
        fontsize=8.5, va='center', ha='left',
        family='monospace', color='#1a3a5c',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='white',
                  edgecolor='#1a3a5c', linewidth=1.1, alpha=0.97),
        zorder=6)

# ── 2 simple inline annotations ───────────────────────────────────────
# NE — text just above the NE bar (y=14, top of chart)
ne_i = abbrs.index('NE')
ax.text(7.5, ne_i - 0.55,
        'Least aid received — fastest recovery (4 mo)',
        ha='left', va='center', fontsize=8,
        color='#1b5e20', style='italic',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f1f8f1',
                  edgecolor='#a5d6a7', lw=0.8, alpha=0.95))

# TX — label in empty right space of TX row (y=1)
tx_i = abbrs.index('TX')
ax.text(2.5, tx_i + 0.8,
        'Top 3 aid recipients — among slowest recoveries',
        ha='left', va='center', fontsize=8,
        color='#c62828', style='italic')

# ── Cluster labels inside shaded areas — below bars ──────────────────
ax.text(-8.2, -0.45, '\u25c4  High aid  /  slower recovery',
        ha='center', va='center', fontsize=8.5,
        color='#b71c1c', fontweight='bold', style='italic')

ax.text(8.2, -0.45, 'Low aid  /  faster recovery  \u25ba',
        ha='center', va='center', fontsize=8.5,
        color='#1b5e20', fontweight='bold', style='italic')

# ── Legend — horizontal, below x-axis label ──────────────────────────
legend_patches = [
    mpatches.Patch(color='#c62828', label='Underperformer  —  more aid, slower recovery'),
    mpatches.Patch(color='#90a4ae', label='Neutral'),
    mpatches.Patch(color='#2e7d32', label='Overperformer  —  less aid, faster recovery'),
]
ax.legend(handles=legend_patches,
          loc='lower center', bbox_to_anchor=(0.5, -0.13),
          fontsize=8.5, frameon=True,
          edgecolor='#cccccc', fancybox=True, ncol=3,
          framealpha=0.97)

# ── X axis ────────────────────────────────────────────────────────────
ax.set_xlabel('Aid-to-Recovery Rank Gap  (Aid Rank \u2212 Recovery Rank)',
              fontsize=9, color='#455a64', labelpad=9)
ax.xaxis.grid(True, color='#eceff1', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#cfd8dc')
ax.tick_params(axis='x', colors='#546e7a', labelsize=8.5)

# ── Title + subtitle — subtitle ABOVE title, clearly separated ────────
# 1. Subtitle — small, gray, italic, at top
fig.text(0.53, 0.965,
    'Positive values indicate faster-than-expected recovery given aid received '
    '(15 high-exposure U.S. states)',
    ha='center', va='top', fontsize=11.5,
    color='#6b7280', style='italic')

# 2. Main title — bold, dark, below subtitle with clear gap
fig.text(0.53, 0.918,
    'States Receiving More Aid Often Recover More Slowly',
    ha='center', va='top', fontsize=18,
    fontweight='bold', color='#0d1b2a')

# ── Caption ───────────────────────────────────────────────────────────
fig.text(0.53, 0.005,
    'Gap = Aid Rank minus Recovery Rank. States sorted worst (top) to best (bottom). '
    'Sources: FEMA Disaster Dollar Database (Carnegie Endowment, 2003\u20132025); '
    'BLS LAUS; Authors\u2019 DRPA model.',
    ha='center', va='bottom', fontsize=7.5, color='#aaa', style='italic')

plt.savefig('Fig_Aid_Recovery_Slope.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Done: Fig_Aid_Recovery_Slope.png")
