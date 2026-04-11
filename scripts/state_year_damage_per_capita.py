import pandas as pd
import plotly.express as px

# -------------------------
# SETTINGS
# -------------------------
FILE = "state_year_damage_per_capita.csv"
START_YEAR = 2010
END_YEAR = 2024
TOP_N = 15

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(FILE)

print("Columns found:", df.columns.tolist())

# Detect columns automatically
state_col = [c for c in df.columns if "state" in c.lower()][0]
year_col = [c for c in df.columns if "year" in c.lower()][0]
percap_col = [c for c in df.columns if "per" in c.lower()][0]

# Filter years
df = df[(df[year_col] >= START_YEAR) & (df[year_col] <= END_YEAR)]

# Aggregate total damage per capita
ranked = (
    df.groupby(state_col)[percap_col]
    .sum()
    .reset_index()
)

ranked = ranked.sort_values(percap_col, ascending=False)

# Keep Top 15
top15 = ranked.head(TOP_N)

print("\nTop 15 states:")
print(top15)

# -------------------------
# PLOT
# -------------------------
fig = px.bar(
    top15.sort_values(percap_col),
    x=percap_col,
    y=state_col,
    orientation="h",
    text=percap_col,
    title="Top 15 States by Disaster Damage Per Capita (2010–2024)"
)

fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")

fig.update_layout(
    xaxis_title="Total Damage Per Capita ($ per person)",
    yaxis_title="State",
    height=700
)

fig.write_html("visual html/top15_damage_per_capita.html")
fig.show()