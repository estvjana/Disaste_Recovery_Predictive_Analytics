"""
=============================================================================
Fig 10 — Section 4.2: Recovery Time Variation and the Role of Industry
          Composition in High-Exposure U.S. States
=============================================================================
Two-panel chart:
  Left  : Horizontal bar chart — recovery months per state, colored by
           dominant economic profile
  Right : Horizontal bar chart — average recovery months by profile group

Data:
  - BEA SAGDP2 (2019-2023 avg) → economic profile classification
  - recovery_speed_results.csv  → COVID recovery months

Colors consistent with Fig 7 palette.
=============================================================================
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDP_FOLDER   = os.path.join(BASE_DIR, "data", "BEA", "BEA_StateGDP_1997_2024")
RECOVERY_CSV = os.path.join(BASE_DIR, "recovery_speed_results.csv")
OUTPUT       = os.path.join(BASE_DIR, "visual png", "Fig10_recovery_by_industry.png")

# ── STATES ─────────────────────────────────────────────────────────────────────
STATES = {
    'TX': 'Texas',        'KY': 'Kentucky',      'MO': 'Missouri',
    'OK': 'Oklahoma',     'GA': 'Georgia',        'LA': 'Louisiana',
    'FL': 'Florida',      'KS': 'Kansas',         'NC': 'North Carolina',
    'VA': 'Virginia',     'MS': 'Mississippi',    'NE': 'Nebraska',
    'IA': 'Iowa',         'AR': 'Arkansas',       'TN': 'Tennessee',
}

YEARS = ['2019', '2020', '2021', '2022', '2023']

# ── COLOR PALETTE (consistent with Fig 7) ──────────────────────────────────────
PALETTE = {
    'Natural Resources': '#E67E22',   # orange
    'Industry':          '#2980B9',   # blue
    'Services':          '#27AE60',   # green
    'Government':        '#8E44AD',   # purple
}

# ── GDP SECTOR EXTRACTION ──────────────────────────────────────────────────────
def get_sectors(abbr):
    fpath = os.path.join(GDP_FOLDER, f"SAGDP2_{abbr}_1997_2024.csv")
    if not os.path.exists(fpath):
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
        'Govt': 100 * gv('Government and government enterprises') / total,
    }


def assign_profile(row):
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
rec = rec[rec['abbr'].isin(STATES.keys())][['abbr', 'state', 'recovery_months']]

merged = sdf.merge(rec, on='abbr')
merged['profile'] = merged.apply(assign_profile, axis=1)
merged['color']   = merged['profile'].map(PALETTE)

# Sort by recovery months descending
merged = merged.sort_values('recovery_months', ascending=True).reset_index(drop=True)

# ── PROFILE AVERAGES ──────────────────────────────────────────────────────────
profile_avg = (
    merged.groupby('profile')
    .agg(avg_months=('recovery_months', 'mean'),
         count=('recovery_months', 'size'))
    .reset_index()
)
profile_avg['color'] = profile_avg['profile'].map(PALETTE)

# Sort fastest → slowest top to bottom (Industry at top = highest index in barh)
profile_order = ['Government', 'Services', 'Natural Resources', 'Industry']
profile_avg['sort_key'] = profile_avg['profile'].map(
    {p: i for i, p in enumerate(profile_order)}
)
profile_avg = profile_avg.sort_values('sort_key').reset_index(drop=True)

# ── PLOT (two panels) ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                gridspec_kw={'width_ratios': [2.2, 1]})
fig.patch.set_facecolor('#F8F9FA')
fig.subplots_adjust(left=0.12, right=0.95, top=0.70, bottom=0.18,
                    wspace=0.50)

# ── LEFT PANEL: Recovery time per state ───────────────────────────────────────
bars1 = ax1.barh(range(len(merged)), merged['recovery_months'],
                 color=merged['color'], edgecolor='white', linewidth=0.5,
                 height=0.72, zorder=3)

ax1.set_yticks(range(len(merged)))
ax1.set_yticklabels(merged['state'], fontsize=9.5, fontweight='bold')
ax1.set_facecolor('#F8F9FA')
ax1.set_xlabel('Recovery Time (Months)', fontsize=10, labelpad=8, color='#6C757D')
ax1.set_title('Recovery Time per State', fontsize=12, fontweight='bold',
              pad=22, color='#2C3E50')
ax1.set_xlim(0, 28)
ax1.grid(True, axis='x', color='#E0E0E0', linewidth=0.8, linestyle='--')
ax1.tick_params(colors='#6C757D')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#E0E0E0')
ax1.spines['bottom'].set_color('#E0E0E0')

# Value labels — consistent offset, skip Nebraska & Texas (handled by annotations)
fastest = merged.iloc[0]
slowest = merged.iloc[-1]
LABEL_GAP = 0.4

for i, (_, row) in enumerate(merged.iterrows()):
    if i == 0 or i == len(merged) - 1:
        continue  # annotations replace these
    ax1.text(row['recovery_months'] + LABEL_GAP, i,
             f"{int(row['recovery_months'])} mo",
             va='center', fontsize=8.5, fontweight='bold', color='#2C3E50')

# Nebraska — right of bar, vertically centered, gold highlight
ax1.text(fastest['recovery_months'] + LABEL_GAP, 0,
         f"Nebraska \u2014 {int(fastest['recovery_months'])} mo (fastest)",
         va='center', ha='left', fontsize=8.5, fontweight='bold',
         color='#D97706', style='italic')

# Texas — right of bar end, vertically centered (same style as other labels)
ax1.text(slowest['recovery_months'] + LABEL_GAP, len(merged) - 1,
         f"Texas \u2014 {int(slowest['recovery_months'])} mo (longest)",
         va='center', ha='left', fontsize=8.5, fontweight='bold',
         color='#D97706', style='italic')

# Grouping dividers (Fast ≤12, Medium 13–17, Slow ≥18)
tier_boundaries = []
for i in range(1, len(merged)):
    prev = merged.iloc[i - 1]['recovery_months']
    curr = merged.iloc[i]['recovery_months']
    if prev <= 12 < curr:
        tier_boundaries.append(('fast_med', i - 0.5))
    elif prev <= 17 < curr:
        tier_boundaries.append(('med_slow', i - 0.5))

for kind, y_pos in tier_boundaries:
    ax1.axhline(y_pos, color='#bbb', linestyle=':', linewidth=0.9, alpha=0.8)

fast_med_y = next((y for k, y in tier_boundaries if k == 'fast_med'), None)
med_slow_y = next((y for k, y in tier_boundaries if k == 'med_slow'), None)

zone_ranges = []
if fast_med_y is not None and med_slow_y is not None:
    zone_ranges = [
        ('Fast',   -0.5,       fast_med_y, '#EEF2F7'),   # light cool gray-blue
        ('Medium', fast_med_y, med_slow_y, '#F8F9FA'),   # same as background (neutral)
        ('Slow',   med_slow_y, len(merged) - 0.5, '#FDECEA'),  # light pink-red
    ]
for label, y0, y1, bg in zone_ranges:
    ax1.axhspan(y0, y1, facecolor=bg, alpha=0.45, zorder=0)
    mid = (y0 + y1) / 2
    ax1.text(27.5, mid, label, va='center', ha='right', fontsize=8,
             color='#999', fontstyle='italic')

# Average reference line
avg_all = merged['recovery_months'].mean()
ax1.axvline(avg_all, color='#999', linestyle='--', linewidth=1.0,
            alpha=0.7, zorder=2)
ax1.text(avg_all + 0.3, len(merged) - 0.1,
         f"Avg: {avg_all:.0f} mo",
         fontsize=8, color='#777', va='bottom', ha='left',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))

# ── RIGHT PANEL: Average by profile ──────────────────────────────────────────
bars2 = ax2.barh(range(len(profile_avg)), profile_avg['avg_months'],
                 color=profile_avg['color'], edgecolor='white',
                 linewidth=0.5, height=0.6, zorder=3)

ax2.set_yticks(range(len(profile_avg)))
ax2.set_yticklabels(profile_avg['profile'], fontsize=9.5, fontweight='bold')
ax2.set_facecolor('#F8F9FA')
ax2.set_xlabel('Avg Recovery (Months)', fontsize=10, labelpad=8, color='#6C757D')
ax2.set_title('Average Recovery Time\nby Economic Profile',
              fontsize=11, fontweight='bold', pad=22, color='#2C3E50')
ax2.set_xlim(0, 25)
ax2.grid(True, axis='x', color='#E0E0E0', linewidth=0.8, linestyle='--')
ax2.tick_params(colors='#6C757D')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#E0E0E0')
ax2.spines['bottom'].set_color('#E0E0E0')

# Value labels — consistent offset, centered, same style
R_LABEL_GAP = 0.4
for i, (_, row) in enumerate(profile_avg.iterrows()):
    ax2.text(row['avg_months'] + R_LABEL_GAP, i,
             f"{row['avg_months']:.1f} mo",
             va='center', ha='left', fontsize=9, fontweight='bold', color='#2C3E50')

# ── SUPTITLE & SUBTITLE ──────────────────────────────────────────────────────
fig.text(0.53, 0.97,
         'How Industry Composition Shapes Recovery Time Across U.S. States',
         ha='center', va='top', fontsize=14, fontweight='bold',
         color='#2C3E50')

fig.text(0.53, 0.91,
         'States with stronger industrial bases recover faster, while service- and '
         'government-heavy economies take longer to return to pre-shock conditions.',
         ha='center', va='top', fontsize=9.5, color='#6C757D', style='italic')

fig.text(0.53, 0.86,
         'Recovery speed is strongly linked to each state\u2019s economic structure.',
         ha='center', va='top', fontsize=9, color='#6C757D')

# ── LEGEND ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=PALETTE['Natural Resources'], label='Natural Resources'),
    mpatches.Patch(color=PALETTE['Industry'],          label='Industry'),
    mpatches.Patch(color=PALETTE['Services'],          label='Services'),
    mpatches.Patch(color=PALETTE['Government'],        label='Government'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=9.5, framealpha=0.95, edgecolor='#ccc',
           title='Economic Profile', title_fontsize=10,
           bbox_to_anchor=(0.53, 0.01))

# ── SAVE & SHOW ──────────────────────────────────────────────────────────────
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='#F8F9FA',
            pad_inches=0.4)
plt.show()
plt.close()

# ── PRINT SUMMARY ────────────────────────────────────────────────────────────
print(f"\n✓ Saved: {OUTPUT}")
print(f"\n  Recovery by state:")
for _, row in merged.sort_values('recovery_months', ascending=False).iterrows():
    print(f"    {row['state']:<20} {int(row['recovery_months']):>3} mo  "
          f"({row['profile']})")
print(f"\n  Average by profile:")
for _, row in profile_avg.iterrows():
    print(f"    {row['profile']:<22} {row['avg_months']:.1f} mo  "
          f"(n={int(row['count'])})")
