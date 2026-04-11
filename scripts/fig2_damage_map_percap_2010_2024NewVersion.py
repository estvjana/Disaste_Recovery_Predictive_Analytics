import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# FINAL PAPER FIGURE: MAP ON TOP + RANKED PANEL BELOW
# =========================================================

# -----------------------------
# Settings
# -----------------------------
FILE = "state_year_damage_per_capita.csv"
START_YEAR = 2010
END_YEAR = 2024
TOP_N = 15

TITLE    = f"Disaster Damage Per Capita by State ({START_YEAR}–{END_YEAR})"
SUBTITLE = "Cumulative per-resident burden from reported property and crop losses, scaled by state population."
FOOTER   = (
    f"Sources: NOAA Storm Events Database (property + crop damage) · "
    f"U.S. Census Bureau American Community Survey (ACS) population estimates · "
    f"Analysis window: {START_YEAR}–{END_YEAR}."
)
CAPTION = (
    f"Fig. 2. Cumulative per-capita disaster damage by U.S. state, {START_YEAR}–{END_YEAR}. "
    "Values represent the sum of reported property and crop losses from the NOAA Storm Events Database "
    "divided by state population estimates from the U.S. Census Bureau ACS. "
    "States are shaded by damage tier (see legend); callouts identify the five highest-burden states; "
    "the ranked panel lists the top 15 states by total per-capita damage."
)

OUTFILE_HTML = "visual html/fig2_map_with_bottom_ranked_panel.html"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(FILE)

state_col = "state"
year_col = "year"
percap_col = "damage_per_capita"

df = df[(df[year_col] >= START_YEAR) & (df[year_col] <= END_YEAR)].copy()

state_totals = (
    df.groupby(state_col, as_index=False)[percap_col]
      .sum()
      .sort_values(percap_col, ascending=False)
)

# -----------------------------
# State names -> abbreviations
# -----------------------------
state_to_abbrev = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC"
}

state_totals["state_clean"] = state_totals[state_col].astype(str).str.strip().str.upper()

if state_totals["state_clean"].str.len().max() <= 2:
    state_totals["abbr"] = state_totals["state_clean"]
else:
    state_totals["abbr"] = state_totals["state_clean"].map(state_to_abbrev)

state_totals = state_totals.dropna(subset=["abbr"]).copy()

# -----------------------------
# Bins for legend
# -----------------------------
bins = [0, 50, 150, 300, 1000, float(state_totals[percap_col].max()) + 1]
bin_labels = [
    "$0 to $50",
    "$51 to $150",
    "$151 to $300",
    "$301 to $1,000",
    "More than $1,000"
]

state_totals["damage_bin"] = pd.cut(
    state_totals[percap_col],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

bin_code_map = {
    "$0 to $50": 1,
    "$51 to $150": 2,
    "$151 to $300": 3,
    "$301 to $1,000": 4,
    "More than $1,000": 5
}
state_totals["bin_code"] = state_totals["damage_bin"].astype(str).map(bin_code_map)

# -----------------------------
# Publication-style blue scale
# -----------------------------
blue_steps = [
    [0.00, "#eff4fb"], [0.20, "#eff4fb"],
    [0.20, "#d6e3f3"], [0.40, "#d6e3f3"],
    [0.40, "#a9c5e5"], [0.60, "#a9c5e5"],
    [0.60, "#6f9fd2"], [0.80, "#6f9fd2"],
    [0.80, "#2b66ad"], [1.00, "#2b66ad"],
]

# -----------------------------
# Build map
# -----------------------------
fig = px.choropleth(
    state_totals,
    locations="abbr",
    locationmode="USA-states",
    color="bin_code",
    scope="usa",
    color_continuous_scale=blue_steps,
    range_color=(1, 5),
    hover_name="state_clean",
    hover_data={
        percap_col: ":,.0f",
        "abbr": False,
        "damage_bin": True,
        "bin_code": False
    }
)

fig.update_traces(marker_line_color="white", marker_line_width=1.1)

# Improve map background / ocean / frame
fig.update_geos(
    bgcolor="white",
    showlakes=True,
    lakecolor="#ddeeff",
    showocean=False,
    framecolor="#c5d3e0",
    framewidth=1,
)

# -----------------------------
# State abbreviations on the map
# -----------------------------
fig.add_trace(go.Scattergeo(
    locations=state_totals["abbr"],
    locationmode="USA-states",
    text=state_totals["abbr"],
    mode="text",
    textfont=dict(size=10, color="#23364d"),
    showlegend=False,
    hoverinfo="skip"
))

# -----------------------------
# Outside callouts for key states
# -----------------------------
state_centers = {
    "LA": (-91.5, 31.0),
    "HI": (-157.0, 20.5),
    "FL": (-81.6, 27.8),
    "OR": (-120.5, 44.0),
    "SD": (-100.0, 44.5)
}

label_positions = {
    "LA": (-87.5, 28.5),
    "HI": (-154.5, 24.8),   # above Hawaii islands, inside map viewport
    "FL": (-75.5, 26.5),
    "OR": (-127.5, 46.5),
    "SD": (-94.5, 46.8)
}

# Geo callouts for mainland states (HI handled separately via paper coords)
top_callouts = state_totals[state_totals["abbr"].isin(["LA", "FL", "OR", "SD"])].copy()

for _, row in top_callouts.iterrows():
    abbr = row["abbr"]
    value = row[percap_col]

    lon0, lat0 = state_centers[abbr]
    lon1, lat1 = label_positions[abbr]

    fig.add_trace(go.Scattergeo(
        lon=[lon0, lon1],
        lat=[lat0, lat1],
        mode="lines",
        line=dict(color="#6b7280", width=1),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scattergeo(
        lon=[lon1],
        lat=[lat1],
        mode="text",
        text=[f"${value:,.0f}"],
        textfont=dict(size=10, color="#111827"),
        showlegend=False,
        hoverinfo="skip"
    ))

# -----------------------------
# Hawaii callout — geo coordinates kept inside inset viewport
# Inset visible range: lat [18–23], lon [-162 to -153]
# Line goes right from main island → label in open ocean east of islands
# -----------------------------
hi_val = state_totals.loc[state_totals["abbr"] == "HI", percap_col].values[0]

# Hawaii callout
# Line: geo trace within the inset viewport (no clipping issue for lines)
fig.add_trace(go.Scattergeo(
    lon=[-155.5, -152.8],
    lat=[20.8, 20.8],
    mode="lines",
    line=dict(color="#6b7280", width=1),
    showlegend=False,
    hoverinfo="skip"
))

# Value label: locationmode="USA-states" = screen-space rendering, never clipped.
# "bottom right" places text in the clear white ocean space outside the dark island.
fig.add_trace(go.Scattergeo(
    locations=["HI"],
    locationmode="USA-states",
    text=[f"    ${hi_val:,.0f}"],   # leading spaces shift label to the right
    mode="text",
    textposition="bottom right",
    textfont=dict(size=10, color="#111827"),
    showlegend=False,
    hoverinfo="skip"
))

# -----------------------------
# Prepare bottom ranked panel (HTML-based)
# -----------------------------
top15 = state_totals.head(TOP_N).copy()

# Colors for each bin level (1=lightest → 5=darkest)
bin_colors = {
    5: "#163d73",  # Extreme burden  (>$1,000)
    4: "#2b66ad",  # Very high burden ($301–$1,000)
    3: "#6f9fd2",  # High burden      ($151–$300)
    2: "#a9c5e5",  # Moderate burden  ($51–$150)
    1: "#d6e3f3",  # Elevated burden  ($0–$50)
}
burden_labels = {
    5: "Extreme burden",
    4: "Very high burden",
    3: "High burden",
    2: "Moderate burden",
    1: "Elevated burden",
}

def make_row_html(rank, abbr, state_name, value, bin_code, highlight=False):
    color = bin_colors.get(int(bin_code), "#d6e3f3")
    bar = (
        f"<span style='display:inline-block;width:5px;height:18px;"
        f"background:{color};border-radius:2px;vertical-align:middle;"
        f"margin-right:8px;flex-shrink:0;'></span>"
    )
    rank_abbr = f"<b>{rank}. {abbr}</b>"
    val_fmt = f"${value:,.0f}"
    if highlight:
        rank_abbr = f"<b style='color:#163d73;'>{rank}. {abbr}</b>"
        val_fmt   = f"<b style='color:#163d73;'>{val_fmt}</b>"
    return (
        f"<div style='display:flex;align-items:center;margin-bottom:7px;font-size:13px;color:#1f2d3d;'>"
        f"{bar}{rank_abbr}&nbsp;{state_name}&nbsp;&nbsp;{val_fmt}"
        f"</div>"
    )

cols = [[], [], []]
for i, (_, row) in enumerate(top15.iterrows(), start=1):
    html_row = make_row_html(
        i, row["abbr"], row["state_clean"].title(),
        row[percap_col], row["bin_code"], highlight=(i <= 3)
    )
    if i <= 5:
        cols[0].append(html_row)
    elif i <= 10:
        cols[1].append(html_row)
    else:
        cols[2].append(html_row)

col_header_style = (
    "color:#4a6fa5;font-size:12px;font-weight:600;"
    "margin-bottom:10px;letter-spacing:0.03em;"
)

def make_col_html(header, rows):
    return (
        f"<div style='flex:1;padding-right:24px;'>"
        f"<div style='{col_header_style}'>{header}</div>"
        + "".join(rows) +
        f"</div>"
    )

legend_items_html = "".join(
    f"<div style='display:flex;align-items:center;margin-bottom:8px;'>"
    f"<span style='display:inline-block;width:16px;height:16px;background:{bin_colors[code]};"
    f"border-radius:2px;margin-right:10px;flex-shrink:0;'></span>"
    f"<span style='font-size:12px;color:#1f2d3d;'>{burden_labels[code]}</span>"
    f"</div>"
    for code in [5, 4, 3, 2, 1]
)

ranked_panel_html = f"""
<div style="font-family:Arial,sans-serif;border:1px solid #c8d6e6;border-radius:6px;
            padding:18px 24px;background:#f8fafc;display:flex;align-items:flex-start;
            max-width:960px;margin:6px auto 10px auto;box-sizing:border-box;">
  <div style="flex:1;">
    <div style="font-weight:bold;font-size:15px;color:#1f2d3d;margin-bottom:4px;">Top-ranked states</div>
    <div style="display:flex;margin-top:8px;">
      {make_col_html("Ranks 1–5",  cols[0])}
      {make_col_html("Ranks 6–10", cols[1])}
      {make_col_html("Ranks 11–15",cols[2])}
    </div>
  </div>
  <div style="margin-left:28px;border:1px solid #c8d6e6;padding:14px 18px;
              background:white;border-radius:4px;flex-shrink:0;min-width:160px;">
    <div style="font-weight:bold;font-size:13px;color:#1f2d3d;margin-bottom:10px;">Burden group</div>
    {legend_items_html}
  </div>
</div>
"""

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    title=dict(text="", pad=dict(t=0)),   # fully cleared — title via annotation
    paper_bgcolor="white",
    plot_bgcolor="white",
    width=1100,
    height=640,
    margin=dict(l=10, r=160, t=145, b=20),
    coloraxis_colorbar=dict(
        title="Disaster damage<br>per capita<br>(USD/person)",
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=bin_labels,
        len=0.50,
        x=0.97,
        xanchor="left",
        y=0.80,
        yanchor="top",
        thickness=20,
        tickfont=dict(size=11),
        title_font=dict(size=11),
    ),
    annotations=[
        # Row 1: bold main title — centered over the map
        dict(
            x=0.5, y=1.20,
            xref="paper", yref="paper",
            text=f"<b>Disaster Damage Per Capita by State (2010–2024)</b>",
            showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(size=21, color="#1f2d3d"),
            align="center"
        ),
        # Row 2: small gray subtitle centered below title
        dict(
            x=0.5, y=1.115,
            xref="paper", yref="paper",
            text=SUBTITLE,
            showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(size=12, color="#6b7280"),
            align="center"
        ),
    ],
)

# -----------------------------
# Save and open (map + HTML panel combined)
# -----------------------------
map_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="choropleth-map")

footer_html = (
    f"<p style='text-align:center;font-size:11px;color:gray;margin:4px auto 2px;'>{FOOTER}</p>"
    f"<p style='text-align:center;font-size:11px;color:#1f2d3d;margin:2px auto 12px;max-width:880px;'>{CAPTION}</p>"
)

full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{TITLE}</title>
  <style>
    body {{ margin: 0; padding: 0; background: white; font-family: Arial, sans-serif; }}
    #choropleth-map {{ width: 100%; }}
  </style>
</head>
<body>
{map_div}
{ranked_panel_html}
{footer_html}
</body>
</html>"""

with open(OUTFILE_HTML, "w", encoding="utf-8") as f:
    f.write(full_html)

import webbrowser, os
webbrowser.open("file://" + os.path.abspath(OUTFILE_HTML))
print(f"Saved HTML: {OUTFILE_HTML}")