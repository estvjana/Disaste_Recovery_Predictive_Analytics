# =============================================================================
# drpi_gov_data.py
# Tab 4 — Political Economy & Aid Distribution
# =============================================================================
#
# Data sources (canonical):
#   - Supabase table `drpi_risk_scores`  → all DRPI + partisan fields
#     (loaded at import time via drpi_config credentials)
#
# Data sources (fallback, used only when Supabase is unreachable):
#   - Hardcoded snapshot derived from:
#       drpi_risk_scores.csv                (DRPI fields)
#       MIT Election Lab 2008-2020           (partisan proxy fields)
#
# Expected Supabase schema for `drpi_risk_scores`:
#   state                     text   (e.g. "Louisiana")
#   abbr                      text   (e.g. "LA")
#   governor_party            text   ("Democrat" | "Republican" | "Independent")
#   legislative_control       text   ("Democrat" | "Republican" | "Split")
#   recovery_months_predicted numeric
#   drpi_score                numeric
#   drpi_risk_tier            text   ("HIGH" | "MODERATE" | "LOW")
#
# Override the table name by setting the env var DRPI_PARTISAN_TABLE.
# =============================================================================

import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    from drpi_config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    # drpi_config isn't importable — carry on with fallback data only.
    SUPABASE_URL = ""
    SUPABASE_KEY = ""


# Name of the Supabase table holding the partisan + DRPI data. Override via env.
_PARTISAN_TABLE = os.environ.get("DRPI_PARTISAN_TABLE", "drpi_risk_scores")

# Required fields every row must expose. If Supabase returns rows missing any
# of these, we reject the response and fall back to the hardcoded snapshot.
_REQUIRED_FIELDS = {
    "state",
    "abbr",
    "governor_party",
    "legislative_control",
    "recovery_months_predicted",
    "drpi_score",
    "drpi_risk_tier",
}


def _load_partisan_from_supabase():
    """Fetch partisan/DRPI state data from Supabase.

    Returns a list of row dicts on success, or ``None`` if credentials are
    missing, the network is unreachable, the table is empty, or the response
    doesn't match the expected schema. Never raises.
    """
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None

    # Prefer supabase-py if installed, else hit the PostgREST endpoint directly.
    try:
        from supabase import create_client  # type: ignore
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            resp = client.table(_PARTISAN_TABLE).select("*").execute()
            rows = getattr(resp, "data", None)
        except Exception as exc:  # pragma: no cover - network / auth errors
            print(f"  ⚠  Supabase query failed ({exc}); using fallback data.")
            return None
    except ImportError:
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{_PARTISAN_TABLE}?select=*"
        req = Request(url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Accept": "application/json",
        })
        try:
            with urlopen(req, timeout=10) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as exc:
            print(f"  ⚠  Supabase REST call failed ({exc}); using fallback data.")
            return None

    if not rows or not isinstance(rows, list):
        return None

    missing = _REQUIRED_FIELDS - set(rows[0].keys())
    if missing:
        print(
            f"  ⚠  Supabase table '{_PARTISAN_TABLE}' is missing fields "
            f"{sorted(missing)}; using fallback data."
        )
        return None

    # Coerce numeric fields in case PostgREST returns them as strings.
    for row in rows:
        for field in ("drpi_score", "recovery_months_predicted"):
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError):
                pass

    return rows


# =============================================================================
# _FALLBACK_PARTISAN_DATA
# Used only when Supabase is unreachable. Keep in sync with production schema.
# =============================================================================

_FALLBACK_PARTISAN_DATA = [
    {
        "state": "Louisiana",
        "abbr": "LA",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.8,
        "drpi_score": 68.2,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Florida",
        "abbr": "FL",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 17.8,
        "drpi_score": 65.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "California",
        "abbr": "CA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.3,
        "drpi_score": 60.1,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Texas",
        "abbr": "TX",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 12.6,
        "drpi_score": 57.6,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "New Jersey",
        "abbr": "NJ",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 22.0,
        "drpi_score": 57.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Georgia",
        "abbr": "GA",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.2,
        "drpi_score": 56.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "North Carolina",
        "abbr": "NC",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.3,
        "drpi_score": 54.9,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "Rhode Island",
        "abbr": "RI",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 22.4,
        "drpi_score": 52.4,
        "drpi_risk_tier": "HIGH",
    },
    {
        "state": "New York",
        "abbr": "NY",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.5,
        "drpi_score": 49.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "South Carolina",
        "abbr": "SC",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.5,
        "drpi_score": 47.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Missouri",
        "abbr": "MO",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 14.9,
        "drpi_score": 47.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Michigan",
        "abbr": "MI",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 19.5,
        "drpi_score": 46.7,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Nevada",
        "abbr": "NV",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 19.5,
        "drpi_score": 46.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Alabama",
        "abbr": "AL",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 16.0,
        "drpi_score": 45.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Tennessee",
        "abbr": "TN",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.7,
        "drpi_score": 45.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Hawaii",
        "abbr": "HI",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 21.4,
        "drpi_score": 44.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Mississippi",
        "abbr": "MS",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.1,
        "drpi_score": 44.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Kentucky",
        "abbr": "KY",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.4,
        "drpi_score": 44.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Oregon",
        "abbr": "OR",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 19.0,
        "drpi_score": 42.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Illinois",
        "abbr": "IL",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.0,
        "drpi_score": 42.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Maine",
        "abbr": "ME",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 16.6,
        "drpi_score": 41.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Washington",
        "abbr": "WA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.1,
        "drpi_score": 41.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Vermont",
        "abbr": "VT",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 20.3,
        "drpi_score": 41.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Colorado",
        "abbr": "CO",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 17.1,
        "drpi_score": 41.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Connecticut",
        "abbr": "CT",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.8,
        "drpi_score": 38.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Massachusetts",
        "abbr": "MA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 19.8,
        "drpi_score": 37.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Virginia",
        "abbr": "VA",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 14.4,
        "drpi_score": 36.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Delaware",
        "abbr": "DE",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 20.6,
        "drpi_score": 35.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Minnesota",
        "abbr": "MN",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 15.7,
        "drpi_score": 33.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Arizona",
        "abbr": "AZ",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.4,
        "drpi_score": 33.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Iowa",
        "abbr": "IA",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 14.1,
        "drpi_score": 33.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Arkansas",
        "abbr": "AR",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 14.5,
        "drpi_score": 33.3,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Idaho",
        "abbr": "ID",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.9,
        "drpi_score": 33.1,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Ohio",
        "abbr": "OH",
        "governor_party": "Independent",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.7,
        "drpi_score": 33.0,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Pennsylvania",
        "abbr": "PA",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 16.1,
        "drpi_score": 31.6,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "South Dakota",
        "abbr": "SD",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.0,
        "drpi_score": 31.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Wisconsin",
        "abbr": "WI",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.5,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Kansas",
        "abbr": "KS",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Indiana",
        "abbr": "IN",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 14.7,
        "drpi_score": 30.2,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "New Mexico",
        "abbr": "NM",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 18.5,
        "drpi_score": 29.0,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Oklahoma",
        "abbr": "OK",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 10.5,
        "drpi_score": 27.7,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Maryland",
        "abbr": "MD",
        "governor_party": "Democrat",
        "legislative_control": "Democrat",
        "recovery_months_predicted": 15.1,
        "drpi_score": 25.9,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "Utah",
        "abbr": "UT",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 15.4,
        "drpi_score": 25.5,
        "drpi_risk_tier": "MODERATE",
    },
    {
        "state": "New Hampshire",
        "abbr": "NH",
        "governor_party": "Democrat",
        "legislative_control": "Split",
        "recovery_months_predicted": 15.4,
        "drpi_score": 24.7,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "West Virginia",
        "abbr": "WV",
        "governor_party": "Republican",
        "legislative_control": "Split",
        "recovery_months_predicted": 13.3,
        "drpi_score": 23.6,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Nebraska",
        "abbr": "NE",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 11.5,
        "drpi_score": 23.6,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Montana",
        "abbr": "MT",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.9,
        "drpi_score": 22.5,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "North Dakota",
        "abbr": "ND",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 16.7,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Alaska",
        "abbr": "AK",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 13.4,
        "drpi_score": 13.5,
        "drpi_risk_tier": "LOW",
    },
    {
        "state": "Wyoming",
        "abbr": "WY",
        "governor_party": "Republican",
        "legislative_control": "Republican",
        "recovery_months_predicted": 11.7,
        "drpi_score": 6.6,
        "drpi_risk_tier": "LOW",
    },
]


# =============================================================================
# PARTISAN_DATA — resolved at import time
# Prefer Supabase; fall back to the hardcoded snapshot if unreachable.
# =============================================================================

_supabase_rows = _load_partisan_from_supabase()
if _supabase_rows:
    PARTISAN_DATA = _supabase_rows
    PARTISAN_DATA_SOURCE = f"supabase:{_PARTISAN_TABLE}"
    print(f"  ✓ Loaded {len(PARTISAN_DATA)} rows from {PARTISAN_DATA_SOURCE}")
else:
    PARTISAN_DATA = _FALLBACK_PARTISAN_DATA
    PARTISAN_DATA_SOURCE = "fallback:hardcoded"


# =============================================================================
# GOV_FINDINGS
# Key findings shown as cards in the tab.
# =============================================================================

GOV_FINDINGS = [
    {
        "title": "Balanced Risk Across Party Lines in High-Exposure States",
        "text": (
            "Of the 8 HIGH-tier states, 3 lean Democrat (CA, NJ, RI), "
            "3 lean Republican (LA, TX, GA), and 2 are swing states (FL, NC). "
            "DRPI risk tier shows no strong partisan skew, suggesting disaster "
            "vulnerability is driven more by geography and fiscal capacity than "
            "political affiliation alone. "
            "(Source: drpi_risk_scores.csv, MIT Election Lab 2008-2020)"
        ),
        "color": "#C0392B",
    },
    {
        "title": "Democrat-Leaning States Show Longer Predicted Recovery Times",
        "text": (
            "States classified as Democrat-leaning average 18.5 months predicted "
            "recovery vs 14.3 months for Republican-leaning states. This likely "
            "reflects higher cost of living, urban density, and infrastructure "
            "complexity rather than policy differences, as the highest-cost states "
            "(CA, NJ, NY, RI) cluster in the Democrat category. "
            "(Source: FEMA OpenFEMA 2006-2025, Ridge Regression Model)"
        ),
        "color": "#8E44AD",
    },
    {
        "title": "Swing States Carry Disproportionate DRPI Risk",
        "text": (
            "The 4 swing states (FL, NC, OH, IA) average a DRPI score of 46.7, "
            "exceeding both Democrat-leaning (40.5) and Republican-leaning (35.1) "
            "state averages. Florida and North Carolina alone account for 2 of the "
            "top 8 highest-risk states nationally. "
            "(Source: drpi_risk_scores.csv, 2008-2020 Presidential Elections)"
        ),
        "color": "#C0392B",
    },
    {
        "title": "Note on Political Classification Methodology",
        "text": (
            "Governor party and legislative control are proxied from presidential "
            "voting patterns (2008-2020), not actual gubernatorial or state "
            "legislature party data. 'Independent' indicates swing states that "
            "split evenly between parties across the four election cycles. "
            "(Source: MIT Election Lab, 1976-2020 U.S. President dataset)"
        ),
        "color": "#8E44AD",
    },
]


# =============================================================================
# FOLIUM MAP BUILDER
# Produces an interactive Leaflet/Folium choropleth of the PARTISAN_DATA.
# The dashboard file can import and embed this without any other changes.
# =============================================================================

# Public US-states GeoJSON (from the standard Folium example datasets).
# Pulled once at module load and cached so repeated map builds are cheap.
_STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/python-visualization/folium/main/"
    "examples/data/us-states.json"
)
_STATE_GEOJSON_CACHE = None


def _get_state_geojson():
    """Fetch and cache the US-states GeoJSON. Returns None if offline."""
    global _STATE_GEOJSON_CACHE
    if _STATE_GEOJSON_CACHE is not None:
        return _STATE_GEOJSON_CACHE
    try:
        import requests
        resp = requests.get(_STATE_GEOJSON_URL, timeout=10)
        resp.raise_for_status()
        _STATE_GEOJSON_CACHE = resp.json()
    except Exception:
        _STATE_GEOJSON_CACHE = None
    return _STATE_GEOJSON_CACHE


# Color palettes / tier mappings used by the map.
_RISK_TIER_COLORS = {
    "HIGH": "#C0392B",
    "MODERATE": "#E67E22",
    "LOW": "#27AE60",
}

_PARTY_COLORS = {
    "Democrat": "#2E86DE",
    "Republican": "#C0392B",
    "Independent": "#8E44AD",
}


def _score_to_color(score, vmin=0.0, vmax=70.0):
    """Map a DRPI score to a hex color along a white -> deep-red ramp."""
    import branca.colormap as cm
    ramp = cm.LinearColormap(
        colors=["#FFF5F0", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"],
        vmin=vmin,
        vmax=vmax,
    )
    return ramp(score)


def _recovery_to_color(months, vmin=10.0, vmax=23.0):
    """Map recovery months to a viridis-like ramp."""
    import branca.colormap as cm
    ramp = cm.LinearColormap(
        colors=["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"],
        vmin=vmin,
        vmax=vmax,
    )
    return ramp(months)


def build_drpi_folium_map(color_by: str = "drpi_score"):
    """
    Build an interactive Folium map of the PARTISAN_DATA.

    Parameters
    ----------
    color_by : str
        One of:
          - 'drpi_score'                (continuous red ramp)
          - 'recovery_months_predicted' (continuous viridis ramp)
          - 'drpi_risk_tier'            (HIGH/MODERATE/LOW discrete colors)
          - 'governor_party'            (Dem/Rep/Ind discrete colors)

    Returns
    -------
    folium.Map
        A ready-to-render map. Call `.get_root().render()` for an HTML
        string, `._repr_html_()` for an iframe-safe snippet, or `.save(path)`
        to write to disk.
    """
    import folium
    import branca.colormap as cm

    valid = {
        "drpi_score",
        "recovery_months_predicted",
        "drpi_risk_tier",
        "governor_party",
    }
    if color_by not in valid:
        raise ValueError(
            f"color_by must be one of {sorted(valid)}, got {color_by!r}"
        )

    # Index data by state abbreviation for fast lookup during styling.
    data_by_abbr = {row["abbr"]: row for row in PARTISAN_DATA}
    data_by_name = {row["state"]: row for row in PARTISAN_DATA}

    # Base map centered on the contiguous US.
    fmap = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="cartodbpositron",
        control_scale=True,
    )

    geojson = _get_state_geojson()
    if geojson is None:
        # Graceful fallback: no shapes available, drop labeled markers instead.
        _add_marker_fallback(fmap, color_by, data_by_abbr)
        return fmap

    # Build a colormap / style function appropriate to the chosen metric.
    legend = None
    if color_by == "drpi_score":
        legend = cm.LinearColormap(
            colors=["#FFF5F0", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"],
            vmin=0, vmax=70, caption="DRPI Risk Score",
        )
    elif color_by == "recovery_months_predicted":
        legend = cm.LinearColormap(
            colors=["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"],
            vmin=10, vmax=23, caption="Predicted Recovery (months)",
        )

    def style_function(feature):
        state_name = feature["properties"].get("name")
        row = data_by_name.get(state_name)
        default = {
            "fillColor": "#DDDDDD",
            "color": "#888888",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }
        if row is None:
            return default

        if color_by == "drpi_score":
            fill = _score_to_color(row["drpi_score"])
        elif color_by == "recovery_months_predicted":
            fill = _recovery_to_color(row["recovery_months_predicted"])
        elif color_by == "drpi_risk_tier":
            fill = _RISK_TIER_COLORS.get(row["drpi_risk_tier"], "#DDDDDD")
        else:  # governor_party
            fill = _PARTY_COLORS.get(row["governor_party"], "#DDDDDD")

        return {
            "fillColor": fill,
            "color": "#444444",
            "weight": 0.7,
            "fillOpacity": 0.82,
        }

    def highlight_function(_feature):
        return {"weight": 2.5, "color": "#000000", "fillOpacity": 0.92}

    # Inject DRPI fields onto the GeoJSON features so tooltips can read them.
    for feature in geojson["features"]:
        name = feature["properties"].get("name")
        row = data_by_name.get(name, {})
        feature["properties"]["drpi_score"] = row.get("drpi_score", "N/A")
        feature["properties"]["drpi_risk_tier"] = row.get("drpi_risk_tier", "N/A")
        feature["properties"]["recovery_months_predicted"] = row.get(
            "recovery_months_predicted", "N/A"
        )
        feature["properties"]["governor_party"] = row.get("governor_party", "N/A")
        feature["properties"]["legislative_control"] = row.get(
            "legislative_control", "N/A"
        )

    tooltip = folium.GeoJsonTooltip(
        fields=[
            "name",
            "drpi_score",
            "drpi_risk_tier",
            "recovery_months_predicted",
            "governor_party",
            "legislative_control",
        ],
        aliases=[
            "State:",
            "DRPI Score:",
            "Risk Tier:",
            "Recovery (months):",
            "Governor Lean:",
            "Legislature:",
        ],
        localize=True,
        sticky=False,
        labels=True,
        style=(
            "background-color: white; color: #333; font-family: sans-serif; "
            "font-size: 12px; padding: 6px; border-radius: 4px;"
        ),
    )

    folium.GeoJson(
        geojson,
        name="States",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
    ).add_to(fmap)

    # Add legends.
    if legend is not None:
        legend.add_to(fmap)
    else:
        _add_categorical_legend(fmap, color_by)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def _add_marker_fallback(fmap, color_by, data_by_abbr):
    """If GeoJSON can't be fetched, place circle markers on state centroids."""
    import folium
    # Rough state-centroid coords (lat, lon). Keeps the map usable offline.
    centroids = {
        "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
        "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
        "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
        "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
        "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
        "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
        "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
        "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
        "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
        "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
        "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
        "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
        "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
        "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
        "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
        "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
        "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
        "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
        "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
        "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
        "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
        "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
        "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
        "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954570),
        "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
    }

    for abbr, row in data_by_abbr.items():
        if abbr not in centroids:
            continue
        lat, lon = centroids[abbr]
        if color_by == "drpi_score":
            color = _score_to_color(row["drpi_score"])
        elif color_by == "recovery_months_predicted":
            color = _recovery_to_color(row["recovery_months_predicted"])
        elif color_by == "drpi_risk_tier":
            color = _RISK_TIER_COLORS.get(row["drpi_risk_tier"], "#888")
        else:
            color = _PARTY_COLORS.get(row["governor_party"], "#888")

        popup_html = (
            f"<b>{row['state']}</b><br>"
            f"DRPI: {row['drpi_score']} ({row['drpi_risk_tier']})<br>"
            f"Recovery: {row['recovery_months_predicted']} mo<br>"
            f"Gov lean: {row['governor_party']}"
        )
        folium.CircleMarker(
            location=(lat, lon),
            radius=8 + row["drpi_score"] / 12,
            color="#333",
            weight=0.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=row["state"],
        ).add_to(fmap)


def _add_categorical_legend(fmap, color_by):
    """Inject a small HTML legend for categorical color-by modes."""
    import folium
    if color_by == "drpi_risk_tier":
        title = "Risk Tier"
        items = [("HIGH", "#C0392B"), ("MODERATE", "#E67E22"), ("LOW", "#27AE60")]
    elif color_by == "governor_party":
        title = "Partisan Lean"
        items = [
            ("Democrat", "#2E86DE"),
            ("Republican", "#C0392B"),
            ("Independent", "#8E44AD"),
        ]
    else:
        return

    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{color};margin-right:6px;border:1px solid #555;"></span>'
        f"{label}</div>"
        for label, color in items
    )
    html = f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 8px 12px;
        border: 1px solid #888; border-radius: 4px;
        font-family: sans-serif; font-size: 12px;">
      <div style="font-weight:600;margin-bottom:4px;">{title}</div>
      {rows}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def build_drpi_map_html(color_by: str = "drpi_score") -> str:
    """
    Convenience wrapper: return the map as a standalone HTML string.
    Useful if the dashboard embeds via components.html(...) or an iframe.
    """
    fmap = build_drpi_folium_map(color_by=color_by)
    return fmap.get_root().render()


if __name__ == "__main__":
    # Running the file directly writes a standalone HTML preview.
    out_path = "drpi_map_preview.html"
    build_drpi_folium_map("drpi_score").save(out_path)
    print(f"Wrote {out_path}")