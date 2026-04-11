"""
Fig 7 - Section 3.4: Service-Based Economic Structure and Recovery Time
Across High-Exposure U.S. States

Data Sources:
  - BEA State GDP by Industry (SAGDP2), 2019-2023 average
  - BLS LAUS COVID-19 recovery months (recovery_speed_results.csv)

Author: DTSC 4302 - The Bounce-Back
"""

import os
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # commented out so plt.show() opens a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDP_FOLDER   = os.path.join(BASE_DIR, "data", "BEA", "BEA_StateGDP_1997_2024")
RECOVERY_CSV = os.path.join(BASE_DIR, "recovery_speed_results.csv")
OUTPUT       = os.path.join(BASE_DIR, "visual png", "Fig7_EconomicStructure_Scatter.png")

# ── STATES ─────────────────────────────────────────────────────────────────────
STATES = {
    'TX': 'Texas',        'KY': 'Kentucky',      'MO': 'Missouri',
    'OK': 'Oklahoma',     'GA': 'Georgia',        'LA': 'Louisiana',
    'FL': 'Florida',      'KS': 'Kansas',         'NC': 'North Carolina',
    'VA': 'Virginia',     'MS': 'Mississippi',    'NE': 'Nebraska',
    'IA': 'Iowa',         'AR': 'Arkansas',       'TN': 'Tennessee'
}

YEARS = ['2019', '2020', '2021', '2022', '2023']

# ── COLOR PALETTE (consistent with Fig_EconomicStructure stacked bar) ──────────
PALETTE = {
    'Natural Resources': '#C0392B',   # red
    'Industry':          '#2980B9',   # blue
    'Services':          '#27AE60',   # green
    'Government':        '#8E44AD',   # purple
}

# ── DATA EXTRACTION ────────────────────────────────────────────────────────────
def get_sectors(abbr):
    """Extract sector GDP shares (2019-2023 avg) for a given state abbreviation."""
    fpath = os.path.join(GDP_FOLDER, f"SAGDP2_{abbr}_1997_2024.csv")
    if not os.path.exists(fpath):
        print(f"  Warning: {fpath} not found")
        return None
    df = pd.read_csv(fpath, encoding='latin1', low_memory=False)
    df['Description'] = df['Description'].str.strip()

    def gv(desc):
        row = df[df['Description'] == desc]
        if row.empty:
            return 0
        vals = []
        for y in YEARS:
            try:
                v = float(str(row[y].values[0]).replace(',', ''))
                if not np.isnan(v):
                    vals.append(v)
            except Exception:
                pass
        return np.mean(vals) if vals else 0

    total = gv('All industry total')
    if total == 0:
        return None

    return {
        'abbr': abbr,
        'NatRes': 100 * (
            gv('Agriculture, forestry, fishing and hunting') +
            gv('Mining, quarrying, and oil and gas extraction')
        ) / total,
        'Industry': 100 * (
            gv('Manufacturing') +
            gv('Construction') +
            gv('Utilities')
        ) / total,
        'Services': 100 * (
            gv('Finance, insurance, real estate, rental, and leasing') +
            gv('Professional and business services') +
            gv('Educational services, health care, and social assistance') +
            gv('Retail trade') +
            gv('Information') +
            gv('Wholesale trade') +
            gv('Other services (except government and government enterprises)')
        ) / total,
        'Tourism': 100 * gv(
            'Arts, entertainment, recreation, accommodation, and food services'
        ) / total,
        'Govt': 100 * gv('Government and government enterprises') / total,
    }


def assign_profile(row):
    """Classify dominant economic profile for color coding."""
    if row['NatRes'] >= 4.0:
        return 'Natural Resources'
    elif row['Industry'] >= 20.0:
        return 'Industry'
    elif row['Govt'] >= 15.0:
        return 'Government'
    else:
        return 'Services'


# ── BUILD DATASET ──────────────────────────────────────────────────────────────
records = [get_sectors(a) for a in STATES]
sdf = pd.DataFrame([r for r in records if r])

rec = pd.read_csv(RECOVERY_CSV)
rec = rec[rec['abbr'].isin(STATES.keys())][['abbr', 'recovery_months']]

merged = sdf.merge(rec, on='abbr')
merged['profile'] = merged.apply(assign_profile, axis=1)
merged['color']   = merged['profile'].map(PALETTE)

# ── OLS TRENDLINE ──────────────────────────────────────────────────────────────
x = merged['Services'].values
y = merged['recovery_months'].values
m, b = np.polyfit(x, y, 1)
r    = np.corrcoef(x, y)[0, 1]
r2   = r ** 2

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 7.5))

# Scatter dots
for _, row in merged.iterrows():
    ax.scatter(row['Services'], row['recovery_months'],
               color=row['color'], s=140, zorder=5,
               edgecolors='white', linewidths=0.9, alpha=0.93)

# State labels (hand-tuned offsets to avoid overlap)
label_offsets = {
    'FL': ( 0.4,  0.5),  'NE': ( 0.4, -1.4),  'KY': (-2.8,  0.5),
    'TX': ( 0.4,  0.5),  'GA': ( 0.4, -1.4),  'MS': (-2.8,  0.5),
    'VA': ( 0.4,  0.5),  'NC': ( 0.4,  0.5),  'MO': (-2.8,  0.5),
    'TN': (-2.8, -1.4),  'LA': (-2.8,  0.5),  'KS': ( 0.4, -1.4),
    'AR': ( 0.4,  0.5),  'IA': (-2.8,  0.5),  'OK': (-2.8, -1.4),
}
for _, row in merged.iterrows():
    dx, dy = label_offsets.get(row['abbr'], (0.3, 0.4))
    ax.annotate(row['abbr'],
                xy=(row['Services'], row['recovery_months']),
                xytext=(row['Services'] + dx, row['recovery_months'] + dy),
                fontsize=9.5, fontweight='bold', color='#1a1a1a',
                va='bottom', ha='left' if dx >= 0 else 'right')

# OLS trendline
x_line = np.linspace(44, 73, 300)
ax.plot(x_line, m * x_line + b, color='#444444', linewidth=1.7,
        linestyle='--', zorder=3, alpha=0.7)

# OLS stats box — top-left
ax.text(0.035, 0.97,
        f'OLS Trendline\n$r^2$ = {r2:.2f}  |  slope = {m:+.2f}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', ha='left', style='italic', zorder=10,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f5f5f5',
                  edgecolor='#bbbbbb', alpha=0.95))

# Legend
legend_items = [
    mpatches.Patch(color=PALETTE['Natural Resources'], label='Resource-Intensive (NR > 4%)'),
    mpatches.Patch(color=PALETTE['Industry'],          label='Industry-Intensive (Mfg > 20%)'),
    mpatches.Patch(color=PALETTE['Government'],        label='Government-Intensive (Govt > 15%)'),
    mpatches.Patch(color=PALETTE['Services'],          label='Service-Dominated (Mixed)'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=9.5, framealpha=0.9, edgecolor='#cccccc',
          title='Dominant Economic Profile', title_fontsize=9.5,
          borderpad=0.8, labelspacing=0.5)

# Axis labels & title
ax.set_xlabel('Service Sector Share of State GDP (%)', fontsize=11.5, labelpad=10)
ax.set_ylabel('COVID-19 Recovery Time\n(Months to Return to Baseline Unemployment)',
              fontsize=11, labelpad=10)
ax.set_title('Service-Based Economic Structure and Recovery Time\n'
             'Across High-Exposure U.S. States',
             fontsize=13, fontweight='bold', pad=14)

ax.set_xlim(44, 73)
ax.set_ylim(2, 24)
ax.grid(True, linestyle='--', alpha=0.3, color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Interpretation note
fig.text(0.5, -0.01,
         'Note: Weak positive relationship (r\u00b2\u2009=\u20090.08) suggests '
         'economic structure alone does not fully explain recovery differences.',
         ha='center', fontsize=9, color='#555555', style='italic')

# Source
ax.text(0.99, 0.01,
        'Source: BEA SAGDP2; BLS LAUS. 2019\u20132023 average.',
        transform=ax.transAxes, fontsize=7.8, color='#888888',
        ha='right', va='bottom', style='italic')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()

print(f"\n\u2713 Saved: {OUTPUT}")
print(f"  OLS: slope={m:+.3f} | r\u00b2={r2:.3f}")
print(f"  States: {len(merged)}")
print(f"  Service share range: {merged['Services'].min():.1f}% \u2013 {merged['Services'].max():.1f}%")
