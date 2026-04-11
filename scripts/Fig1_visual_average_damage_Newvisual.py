import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# FIG 1 — CHOROPLETH: AVG YEARLY FEMA DISASTER DECLARATIONS
# =========================================================

# -----------------------------
# Settings
# -----------------------------
FILE       = "data/BLS/laus_with_fema_disaster_exposure_2006_2025.csv"
START_YEAR = 2006
END_YEAR   = 2025
TOP_N      = 15
TITLE      = f"FEMA Disaster Declaration Frequency by State ({START_YEAR}–{END_YEAR})"
SUBTITLE   = "Average number of FEMA disaster declarations issued per year, by state."
FOOTER     = "Sources: FEMA Disaster Declarations Database · Analysis window: 2006–2025."
CAPTION    = (
    "Fig. 1. Choropleth map of average yearly FEMA disaster declarations by U.S. state, 2006–2025. "
    "States are shaded by declaration frequency tier (see legend); callouts identify the five "
    "highest-frequency states; the ranked panel lists the top 15 states by average annual declarations."
)
OUTFILE = "visual html/fig1_fema_declarations_map.html"

# -----------------------------
# Build state totals
# -----------------------------
df = pd.read_csv(FILE)
df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]

yearly = (
    df.groupby(["state", "year"])["disaster_count_month"]
      .sum()
      .reset_index()
)
yearly.columns = ["state", "year", "yearly_declarations"]

state_totals = (
    yearly.groupby("state")["yearly_declarations"]
          .mean()
          .reset_index()
)
state_totals.columns = ["state", "avg_declarations"]
state_totals = state_totals.sort_values("avg_declarations", ascending=False).reset_index(drop=True)

# -----------------------------
# State name → abbreviation
# -----------------------------
state_to_abbrev = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS",
    "Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
    "Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT",
    "Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM",
    "New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}
state_totals["abbr"] = state_totals["state"].map(state_to_abbrev)
state_totals = state_totals.dropna(subset=["abbr"]).copy()

# -----------------------------
# Bins
# -----------------------------
max_val = state_totals["avg_declarations"].max() + 1
bins   = [0, 20, 40, 60, 90, max_val]
labels = ["0 to 20", "21 to 40", "41 to 60", "61 to 90", "More than 90"]
bin_code_map = {l: i+1 for i, l in enumerate(labels)}

state_totals["decl_bin"]  = pd.cut(state_totals["avg_declarations"], bins=bins,
                                    labels=labels, include_lowest=True)
state_totals["bin_code"]  = state_totals["decl_bin"].astype(str).map(bin_code_map)

# -----------------------------
# Orange/amber colour scale (distinct from Fig 2 blue)
# -----------------------------
blue_steps = [
    [0.00, "#fef3e2"], [0.20, "#fef3e2"],
    [0.20, "#fcd9a0"], [0.40, "#fcd9a0"],
    [0.40, "#f9b055"], [0.60, "#f9b055"],
    [0.60, "#e07d20"], [0.80, "#e07d20"],
    [0.80, "#7a3a06"], [1.00, "#7a3a06"],
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
    hover_name="state",
    hover_data={"avg_declarations": ":,.1f", "abbr": False,
                "decl_bin": True, "bin_code": False}
)
fig.update_traces(marker_line_color="white", marker_line_width=1.1)

fig.update_geos(
    bgcolor="white", showlakes=True, lakecolor="#ddeeff",
    showocean=False, framecolor="#c5d3e0", framewidth=1,
)

# State abbreviation labels
fig.add_trace(go.Scattergeo(
    locations=state_totals["abbr"], locationmode="USA-states",
    text=state_totals["abbr"], mode="text",
    textfont=dict(size=10, color="#23364d"),
    showlegend=False, hoverinfo="skip"
))

# -----------------------------
# Callouts for top 5 states
# -----------------------------
state_centers = {
    "TX": (-100.0, 31.5), "KY": (-84.5, 37.5),
    "MO": (-92.5, 38.5),  "OK": (-97.0, 35.5),
    "GA": (-83.5, 32.5)
}
label_positions = {
    "TX": (-107.0, 27.5), "KY": (-79.0, 40.5),
    "MO": (-87.5, 41.5),  "OK": (-91.0, 31.5),
    "GA": (-75.5, 30.5)
}

top_callouts = state_totals[state_totals["abbr"].isin(state_centers.keys())].copy()

for _, row in top_callouts.iterrows():
    abbr  = row["abbr"]
    value = row["avg_declarations"]
    lon0, lat0 = state_centers[abbr]
    lon1, lat1 = label_positions[abbr]

    fig.add_trace(go.Scattergeo(
        lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
        line=dict(color="#6b7280", width=1),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scattergeo(
        lon=[lon1], lat=[lat1], mode="text",
        text=[f"{value:,.1f}"],
        textfont=dict(size=10, color="#111827"),
        showlegend=False, hoverinfo="skip"
    ))

# Hawaii value label
fig.add_trace(go.Scattergeo(
    locations=["HI"], locationmode="USA-states",
    text=[f"{state_totals.loc[state_totals['abbr']=='HI','avg_declarations'].values[0]:,.1f}"],
    mode="text", textposition="bottom right",
    textfont=dict(size=10, color="#111827"),
    showlegend=False, hoverinfo="skip"
))

# -----------------------------
# Bottom ranked panel
# -----------------------------
top15 = state_totals.head(TOP_N).copy()

bin_colors = {"0 to 20":"#fef3e2","21 to 40":"#fcd9a0","41 to 60":"#f9b055",
              "61 to 90":"#e07d20","More than 90":"#7a3a06"}

def make_row(rank, abbr, state, value, bin_label, bold=False):
    color  = bin_colors.get(bin_label, "#a9c5e5")
    weight = "700" if bold else "400"
    val_w  = "700" if bold else "400"
    return (
        f"<span style='display:inline-block;width:5px;height:16px;"
        f"background:{color};border-radius:2px;margin-right:7px;vertical-align:middle;'></span>"
        f"<span style='font-weight:700'>{rank}. {abbr}</span> "
        f"<span style='font-weight:{weight}'>{state.title()}</span> "
        f"<span style='font-weight:{val_w};color:#163d73'>&nbsp;{value:,.1f}</span>"
    )

col1, col2, col3 = [], [], []
for i, (_, row) in enumerate(top15.iterrows(), start=1):
    line = make_row(i, row["abbr"], row["state"], row["avg_declarations"],
                    str(row["decl_bin"]), bold=(i <= 3))
    if   i <= 5:  col1.append(line)
    elif i <= 10: col2.append(line)
    else:         col3.append(line)

col1_txt = "<br>".join(col1)
col2_txt = "<br>".join(col2)
col3_txt = "<br>".join(col3)

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    title=dict(
        text=f"<b>{TITLE}</b>",
        x=0.5, xanchor="center",
        font=dict(size=22, color="#1f2d3d")
    ),
    paper_bgcolor="white", plot_bgcolor="white",
    width=1100, height=600,
    margin=dict(l=10, r=160, t=110, b=20),
    coloraxis_colorbar=dict(
        title="Avg yearly<br>declarations",
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=labels,
        len=0.50, x=0.97, xanchor="left",
        y=0.80, yanchor="top",
        thickness=20,
        tickfont=dict(size=11),
        title_font=dict(size=11),
    ),
    annotations=[
        # subtitle only — panel is rendered in HTML below the map
        dict(x=0.5, y=1.07, xref="paper", yref="paper",
             text=SUBTITLE, showarrow=False, xanchor="center",
             font=dict(size=12.5, color="#5b6775"), align="center"),
    ],
    shapes=[]
)

# -----------------------------
# Build legend box (same style as Fig 2)
# -----------------------------
burden_labels = [
    ("More than 90",  "#7a3a06", "Extreme frequency"),
    ("61 to 90",      "#e07d20", "Very high frequency"),
    ("41 to 60",      "#f9b055", "High frequency"),
    ("21 to 40",      "#fcd9a0", "Moderate frequency"),
    ("0 to 20",       "#fef3e2", "Low frequency"),
]

legend_rows = "".join([
    f"<tr><td style='padding:3px 8px 3px 0'>"
    f"<div style='width:18px;height:18px;background:{c};border:1px solid #ccc;border-radius:2px'></div></td>"
    f"<td style='font-size:12px;color:#1f2d3d;padding:3px 0'>{lab}</td></tr>"
    for _, c, lab in burden_labels
])
legend_html = (
    f"<div style='border:1px solid #d0d8e4;border-radius:6px;padding:12px 16px;"
    f"background:white;display:inline-block'>"
    f"<div style='font-weight:700;font-size:13px;margin-bottom:8px;color:#1f2d3d'>Frequency group</div>"
    f"<table style='border-collapse:collapse'>{legend_rows}</table></div>"
)

map_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="fig1-map")

# Bottom ranked panel HTML
def ranked_row_html(rank, abbr, state, value, bin_label, bold=False):
    color  = bin_colors.get(bin_label, "#a9c5e5")
    fw     = "700" if bold else "400"
    val_fw = "700" if bold else "400"
    val_col= "#163d73" if bold else "#1f2d3d"
    return (
        f"<div style='display:flex;align-items:center;margin-bottom:7px'>"
        f"<div style='width:5px;height:20px;background:{color};border-radius:2px;"
        f"flex-shrink:0;margin-right:8px'></div>"
        f"<span style='font-weight:700;color:#1f2d3d'>{rank}. {abbr}</span>&nbsp;"
        f"<span style='font-weight:{fw};color:#1f2d3d'>{state.title()}</span>&nbsp;&nbsp;"
        f"<span style='font-weight:{val_fw};color:{val_col}'>{value:,.1f}</span></div>"
    )

c1_html = c2_html = c3_html = ""
for i, (_, row) in enumerate(top15.iterrows(), start=1):
    h = ranked_row_html(i, row["abbr"], row["state"],
                        row["avg_declarations"], str(row["decl_bin"]), bold=(i<=3))
    if   i <= 5:  c1_html += h
    elif i <= 10: c2_html += h
    else:         c3_html += h

panel_html = f"""
<div style="border:1px solid #d8e0ea;border-radius:6px;padding:16px 20px 14px;
            background:#f8fafc;width:fit-content;margin:0 auto 8px;
            display:flex;align-items:flex-start;font-family:Arial,sans-serif;gap:20px">
  <div style="flex:0 0 auto">
    <div style="font-weight:700;font-size:15px;margin-bottom:4px;color:#1f2d3d">
      Top-ranked states</div>
    <div style="display:flex;gap:36px">
      <div>
        <div style="color:#4a6fa5;font-size:12px;font-weight:600;margin-bottom:8px">
          Ranks 1–5</div>{c1_html}
      </div>
      <div>
        <div style="color:#4a6fa5;font-size:12px;font-weight:600;margin-bottom:8px">
          Ranks 6–10</div>{c2_html}
      </div>
      <div>
        <div style="color:#4a6fa5;font-size:12px;font-weight:600;margin-bottom:8px">
          Ranks 11–15</div>{c3_html}
      </div>
    </div>
  </div>
  <div style="align-self:flex-start;margin-top:0">{legend_html}</div>
</div>"""

footer_html = (
    f"<p style='text-align:center;font-size:11px;color:gray;margin:4px auto 2px'>{FOOTER}</p>"
    f"<p style='text-align:center;font-size:11px;color:#1f2d3d;margin:2px auto 12px;"
    f"max-width:960px'>{CAPTION}</p>"
)

full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Fig 1 – FEMA Declaration Frequency Map</title>
<style>body{{margin:0;padding:10px;background:white;font-family:Arial,sans-serif}}</style>
</head><body>
{map_div}
{panel_html}
{footer_html}
</body></html>"""

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"Saved: {OUTFILE}")

import webbrowser, os
webbrowser.open("file://" + os.path.abspath(OUTFILE))
