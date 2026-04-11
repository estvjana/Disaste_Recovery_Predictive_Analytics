#!/usr/local/bin/python3.12
"""
=============================================================================
DRPA — Disaster Recovery Predictive Analytics
Step 04: Streamlit Dashboard  (v2 — analytics-grade)
=============================================================================
NEW in v2:
  - SHAP waterfall chart per state (explainability)
  - Compound Vulnerability Index (CVI) radar chart per state
  - Economic profile comparison bar chart (paper Fig 10 equivalent)
  - Recovery time distribution by economic profile
  - Fiscal capacity vs recovery scatter (paper Fig 6 equivalent)
  - Service sector share vs recovery scatter (paper Fig 7 equivalent)
  - Volatility timeline with major event annotations (paper Fig 4)
  - Research-aligned color palette and labels

Run: /usr/local/bin/python3.12 -m streamlit run ML_models/drpi_04_dashboard.py
=============================================================================
"""
import os, sys, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# Suppress Streamlit ScriptRunContext warnings that appear during startup
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

st.set_page_config(
    page_title="DRPA | UNC Charlotte DTSC 4302",
    page_icon=None, layout="wide",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load secrets ──────────────────────────────────────────────────────────────
def _load_env():
    env_file = Path(ROOT) / "config" / ".env"
    if not env_file.exists(): return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            key, _, val = line.partition("=")
            if key.strip() not in os.environ:
                os.environ[key.strip()] = val.strip()
_load_env()

# ── Supabase client (optional) ────────────────────────────────────────────────
def _get_supabase():
    url = os.environ.get("SUPABASE_URL","")
    key = os.environ.get("SUPABASE_KEY","")
    if not url or not key: return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except: return None

@st.cache_data(ttl=300)
def load_from_supabase():
    sb = _get_supabase()
    if not sb: return None
    try:
        res = sb.table("drpi_risk_scores").select("*").order("drpi_score", desc=True).execute()
        if res.data:
            return pd.DataFrame(res.data)
    except: pass
    return None

@st.cache_data
def load_data():
    scores = pd.read_csv(os.path.join(ROOT,"state_narratives.csv") if
                         os.path.exists(os.path.join(ROOT,"state_narratives.csv"))
                         else os.path.join(ROOT,"state_risk_scores.csv"))
    shap_file = os.path.join(ROOT,"state_shap_values.csv")
    shap_df = pd.read_csv(shap_file) if os.path.exists(shap_file) else pd.DataFrame()
    bls_file = os.path.join(ROOT,"data","BLS","laus_with_fema_disaster_exposure_2006_2025.csv")
    bls = pd.read_csv(bls_file, parse_dates=["date"]) if os.path.exists(bls_file) else pd.DataFrame()
    return scores, shap_df, bls

df, shap_df, bls = load_data()

TIER_COLORS = {"CRITICAL":"#C0392B","HIGH":"#E67E22","MODERATE":"#F4D03F","LOW":"#27AE60"}
PROFILE_COLORS = {
    "Industry-Intensive":"#27AE60",   # green  — fastest recovery
    "Service-Dominated":"#2980B9",    # blue
    "Resource-Intensive":"#E67E22",   # orange
    "Government-Intensive":"#8E44AD", # purple — slowest recovery
}
PROFILE_SHORT = {
    "Industry-Intensive":  "Industry",
    "Service-Dominated":   "Services",
    "Resource-Intensive":  "Resources",
    "Government-Intensive":"Government",
}
PAPER_PROFILE_MONTHS = {
    "Industry-Intensive":11.5,"Service-Dominated":16.2,
    "Resource-Intensive":14.4,"Government-Intensive":18.0
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Branding — no emoji, paper-accurate title ─────────────────────────────
    st.markdown("""
    <div style='padding: 8px 0 6px 0;'>
        <p style='font-size:28px; font-weight:800; color:#0d1b2a;
                  letter-spacing:1px; margin:0 0 6px 0;'>DRPA</p>
        <p style='font-size:14px; color:#444; margin:0 0 14px 0; line-height:1.6;'>
            Disaster Recovery<br>Predictive Analytics
        </p>
        <p style='font-size:10px; font-weight:700; color:#555; text-transform:uppercase;
                  letter-spacing:1.2px; margin:0 0 8px 0;'>Research Team</p>
        <p style='font-size:13px; color:#333; line-height:1.9; margin:0 0 10px 0;'>
            Ana Julia Abreu Estevez<br>
            Wendy Ceja-Huerta<br>
            Maria Eduarda C. F. de Resende Silva<br>
            Jake Fabrizio<br>
            Carolina Rangel Lara<br>
            Riya Vadadoria
        </p>
        <p style='font-size:11px; color:#888; margin:0;'>UNC Charlotte · DTSC 4302</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "Overview Dashboard",
        "Risk Map",
        "Research Analytics",
        "State Deep-Dive",
    ])
    st.divider()

    # ── Data source display only ──────────────────────────────────────────────
    data_source = "Local CSV"

    st.divider()
    st.markdown("<p style='font-size:11px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0'>DRPA Score Weights</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:12px;color:#444;line-height:1.8;margin:0'>
    30% &nbsp; Recovery time (BLS)<br>
    25% &nbsp; FEMA exposure (FEMA)<br>
    20% &nbsp; Compound vulnerability (CVI)<br>
    15% &nbsp; Damage per capita (FEMA/Census)<br>
    10% &nbsp; Unemployment volatility (BLS)
    </p>
    """, unsafe_allow_html=True)

# ── Data always loads from Local CSV ─────────────────────────────────────────
# filtered is set inside each page that needs it
filtered = df.copy()  # default fallback

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview Dashboard":
    st.markdown("""
    <div style='padding: 8px 0 4px 0;'>
        <p style='font-size:11px; font-weight:600; letter-spacing:2px; color:#888;
                  text-transform:uppercase; margin:0 0 6px 0;'>
            UNC Charlotte · DTSC 4302
        </p>
        <h1 style='color:#0d1b2a; font-size:28px; font-weight:700;
                   letter-spacing:-0.5px; margin:0 0 6px 0; line-height:1.2;'>
            Disaster Recovery Predictive Analytics
        </h1>
        <p style='color:#555; font-size:14px; margin:0; line-height:1.5;'>
            If a major disaster struck your state today, how long would recovery take
            across labor markets, fiscal systems, and economic sectors?
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Inline filters ────────────────────────────────────────────────────────
    _fcol1, _fcol2 = st.columns([1, 1])
    with _fcol1:
        selected_tiers = st.multiselect(
            "Filter by Risk Tier",
            ["CRITICAL", "HIGH", "MODERATE", "LOW"],
            default=["CRITICAL", "HIGH", "MODERATE", "LOW"],
        )
    with _fcol2:
        all_states = sorted(df["state"].dropna().unique().tolist())
        selected_states = st.multiselect(
            "Focus on States",
            all_states,
            default=[],
            placeholder="All states shown. Select to filter.",
        )

    # Apply filters
    filtered = df[df["drpi_risk_tier"].isin(selected_tiers)].copy()
    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]

    st.markdown("""
    <div style='background:#0d1b2a;border-radius:8px;padding:12px 20px;margin:8px 0 4px 0;'>
        <p style='color:#f0f0f0;font-size:14px;margin:0;line-height:1.5;'>
            <span style='color:#F4D03F;font-weight:700;'>How to use this tool:</span>
            &nbsp; The <b style='color:#fff'>Overview Dashboard</b> shows the national risk landscape across all 50 states.
            Use the <b style='color:#fff'>Risk Map</b> to explore individual indicators like fiscal capacity or disaster exposure.
            Go to <b style='color:#fff'>Research Analytics</b> to see the paper's findings visualized.
            Then use <b style='color:#fff'>State Deep-Dive</b> to simulate recovery for any state.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── KPI Row — reacts to filters ───────────────────────────────────────────
    _is_filtered = bool(selected_states) or set(selected_tiers) != {"CRITICAL","HIGH","MODERATE","LOW"}
    _scope_label = f"among {len(filtered)} selected" if _is_filtered else "national"

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        _nat_avg    = df["drpi_score"].mean()          # national average, always from full dataset
        _n_pool     = len(filtered)
        n_above_avg = len(filtered[filtered["drpi_score"] > _nat_avg])
        _c1_label   = "States Above Avg" if not _is_filtered else "Selected Above Avg"
        _c1_sub     = f"of {_n_pool} states · national avg: {_nat_avg:.0f}/100"
        st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:1px;color:#888;text-transform:uppercase;margin:0'>{_c1_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:32px;font-weight:700;color:#E67E22;margin:2px 0 0 0'>{n_above_avg}</p>", unsafe_allow_html=True)
        st.caption(_c1_sub)
    with c2:
        n_high = len(filtered[filtered["drpi_risk_tier"]=="HIGH"])
        st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:1px;color:#888;text-transform:uppercase;margin:0'>High Risk States</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:32px;font-weight:700;color:#E67E22;margin:2px 0 0 0'>{n_high}</p>", unsafe_allow_html=True)
        st.caption(_scope_label)
    with c3:
        avg_rec = filtered['recovery_months_predicted'].mean() if len(filtered) > 0 else 0
        st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:1px;color:#888;text-transform:uppercase;margin:0'>Avg Recovery Time</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:32px;font-weight:700;color:#0d1b2a;margin:2px 0 0 0'>{avg_rec:.1f} <span style='font-size:16px;font-weight:400;color:#888'>months</span></p>", unsafe_allow_html=True)
        st.caption(_scope_label)
    with c4:
        if len(filtered) > 0:
            top_state = filtered.loc[filtered["drpi_score"].idxmax(),"state"]
            top_score = filtered["drpi_score"].max()
        else:
            top_state, top_score = "—", 0
        _kpi_label = "Selected Highest" if _is_filtered else "Highest Risk State"
        st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:1px;color:#888;text-transform:uppercase;margin:0'>{_kpi_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px;font-weight:700;color:#C0392B;margin:2px 0 0 0'>{top_state}</p><p style='font-size:12px;color:#888;margin:0'>DRPA {top_score:.0f}/100</p>", unsafe_allow_html=True)
    with c5:
        if "fiscal_capacity_per_capita" in filtered.columns and len(filtered) > 0:
            fc = filtered["fiscal_capacity_per_capita"].mean()
            st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:1px;color:#888;text-transform:uppercase;margin:0'>Avg Fiscal Capacity</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:28px;font-weight:700;color:#0d1b2a;margin:2px 0 0 0'>${fc:,.0f}</p><p style='font-size:12px;color:#888;margin:0'>per capita</p>", unsafe_allow_html=True)
            st.caption(_scope_label)
    st.divider()

    col1, col2 = st.columns([1.5, 1])

    with col1:
        _map_title = "Disaster Recovery Risk: Selected States" if _is_filtered else "How Vulnerable Is Each State? Recovery Risk by State"
        st.markdown(f"<p style='font-size:13px;font-weight:600;color:#333;margin:0 0 8px 0;'>{_map_title}</p>", unsafe_allow_html=True)
        # Build clean hover template
        fig_map = px.choropleth(
            filtered, locations="abbr", locationmode="USA-states",
            color="drpi_score", hover_name="state",
            custom_data=["abbr","drpi_score","drpi_risk_tier",
                         "recovery_months_predicted","instability_tier","economic_profile"],
            color_continuous_scale=[[0,"#27AE60"],[0.33,"#F4D03F"],[0.66,"#E67E22"],[1,"#C0392B"]],
            range_color=(0,100), scope="usa",
        )
        fig_map.update_traces(
            hovertemplate=(
                "<b>%{hovertext}  (%{customdata[0]})</b><br>"
                "──────────────────────<br>"
                "DRPA Score:       <b>%{customdata[1]:.1f} / 100</b><br>"
                "Risk Tier:        <b>%{customdata[2]}</b><br>"
                "Recovery Time:    <b>%{customdata[3]:.0f} months</b><br>"
                "Labor Instability:<b>%{customdata[4]}</b><br>"
                "Economic Profile: <b>%{customdata[5]}</b>"
                "<extra></extra>"
            )
        )
        fig_map.update_layout(
            height=430, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
            coloraxis_colorbar=dict(
                title=dict(text="DRPA Score", font=dict(size=11)),
                tickvals=[0,25,50,75,100],
                ticktext=["0  LOW","25","50  HIGH","75","100  CRITICAL"],
                thickness=12, len=0.75,
            ),
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_map, key="overview_map")

    with col2:
        _bar_title = f"Selected States: Risk Ranking ({len(filtered)} states)" if _is_filtered else "Which States Face the Longest Recovery? Top 15 Ranked"
        st.markdown(f"<p style='font-size:13px;font-weight:600;color:#333;margin:0 0 8px 0;'>{_bar_title}</p>", unsafe_allow_html=True)
        top15 = filtered.nlargest(15,"drpi_score")
        fig_bar = px.bar(
            top15.sort_values("drpi_score"),
            x="drpi_score", y="abbr", orientation="h",
            color="drpi_risk_tier", color_discrete_map=TIER_COLORS,
            text="drpi_score",
            custom_data=["state","drpi_risk_tier","recovery_months_predicted"],
            labels={"drpi_score":"DRPA Score","abbr":"","drpi_risk_tier":"Risk Tier"},
        )
        fig_bar.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside",
            textfont=dict(size=11, color="#333"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "DRPA Score: <b>%{x:.1f}</b><br>"
                "Risk Tier: <b>%{customdata[1]}</b><br>"
                "Recovery: <b>%{customdata[2]:.0f} months</b>"
                "<extra></extra>"
            ),
        )
        fig_bar.update_layout(
            height=430, margin=dict(l=0,r=50,t=0,b=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0,112], showgrid=True,
                       gridcolor="#f0f0f0", zeroline=False,
                       title=dict(text="DRPA Score", font=dict(size=11))),
            yaxis=dict(tickfont=dict(size=11, color="#333")),
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_bar, key="overview_bar")

    # ── Recovery × Exposure scatter ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p style='font-size:15px;font-weight:600;color:#0d1b2a;margin:0 0 4px 0;'>Recovery Time vs. FEMA Disaster Exposure</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px;color:#888;margin:0 0 12px 0;'>Bubble size = avg damage per capita &nbsp;·&nbsp; Color = risk tier &nbsp;·&nbsp; Labels = state abbreviation &nbsp;·&nbsp; 2010–2024</p>", unsafe_allow_html=True)
    fig_sc = px.scatter(
        filtered, x="avg_disaster_exposure_12m", y="recovery_months_predicted",
        color="drpi_risk_tier", size="avg_damage_per_capita", size_max=35,
        text="abbr", hover_name="state",
        color_discrete_map=TIER_COLORS,
        custom_data=["state","drpi_score","drpi_risk_tier",
                     "recovery_months_predicted","instability_tier","economic_profile"],
        labels={"avg_disaster_exposure_12m":"Avg FEMA Exposure (declarations / yr)",
                "recovery_months_predicted":"Predicted Recovery Time (months)",
                "drpi_risk_tier":"Risk Tier"},
    )
    fig_sc.update_traces(
        textposition="top center",
        textfont=dict(size=9, color="#333"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "──────────────────────<br>"
            "DRPA Score:      <b>%{customdata[1]:.1f} / 100</b>  (%{customdata[2]})<br>"
            "Recovery Time:   <b>%{customdata[3]:.0f} months</b><br>"
            "Labor:           <b>%{customdata[4]}</b><br>"
            "Profile:         <b>%{customdata[5]}</b><br>"
            "FEMA Exposure:   <b>%{x:.0f} decl/yr</b>"
            "<extra></extra>"
        ),
    )
    fig_sc.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=dict(text="Risk Tier", font=dict(size=11)),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#ddd", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_sc, key="overview_scatter")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK MAP (detailed)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Map":

    # ── Metric metadata ────────────────────────────────────────────────────────
    _metric_meta = {
        "Composite Risk": {
            "col":   "drpi_score",
            "title": "DRPA Risk Map: Composite Risk",
            "desc":  "Overall disaster recovery risk score (0–100) combining disaster exposure, economic structure, fiscal capacity, and labor market volatility. Higher = higher risk.",
            "scale": [[0,"#27AE60"],[0.33,"#F4D03F"],[0.66,"#E67E22"],[1,"#C0392B"]],
            "direction": "Higher = Higher Risk",
            "insight": "Southern states tend to score the highest: Louisiana, Florida, and Texas are consistently at the top, mostly due to frequent hurricanes and floods paired with limited state funding for recovery.",
        },
        "Recovery Time (Months)": {
            "col":   "recovery_months_predicted",
            "title": "DRPA Risk Map: Recovery Time",
            "desc":  "Predicted months for a state to return to its pre-disaster unemployment baseline, based on economic structure, fiscal capacity, and disaster exposure. Higher = longer recovery.",
            "scale": [[0,"#27AE60"],[0.5,"#F4D03F"],[1,"#C0392B"]],
            "direction": "Higher = Longer Recovery (Worse)",
            "insight": "States like Louisiana and New Mexico, which rely heavily on government spending, tend to take 18+ months to recover, compared to about 11.5 months for industry heavy states.",
        },
        "Vulnerability Index (CVI)": {
            "col":   "compound_vulnerability_index",
            "title": "DRPA Risk Map: Compound Vulnerability Index",
            "desc":  "Composite vulnerability score (0–1) built from five dimensions: labor volatility, disaster exposure, damage burden, fiscal weakness, and structural risk. Higher = more vulnerable.",
            "scale": [[0,"#27AE60"],[0.5,"#F4D03F"],[1,"#C0392B"]],
            "direction": "Higher = More Vulnerable (Worse)",
            "insight": "Gulf Coast and Southeast states rank highest on vulnerability. Meanwhile, Midwest and Northeast states tend to score lower, not because they avoid disasters, but because they have more resources to absorb them.",
        },
        "Fiscal Capacity": {
            "col":   "fiscal_capacity_per_capita",
            "title": "DRPA Risk Map: Fiscal Capacity",
            "desc":  "State government revenue per capita (USD). Higher fiscal capacity means more resources available for emergency response, relief programs, and infrastructure repair. Higher = stronger capacity.",
            "scale": [[0,"#C0392B"],[0.5,"#F4D03F"],[1,"#27AE60"]],
            "direction": "Higher = Stronger Capacity (Better)",
            "insight": "Connecticut, New Jersey, and Massachusetts have the most fiscal room, averaging over $14,000 per capita. That buffer makes a real difference when it comes to funding recovery after a disaster.",
        },
        "Disaster Exposure": {
            "col":   "avg_disaster_exposure_12m",
            "title": "DRPA Risk Map: Disaster Exposure",
            "desc":  "Average number of FEMA major-disaster declarations per 12-month rolling window (2006–2024). States with more frequent declarations face recurring and compounding recovery burdens.",
            "scale": [[0,"#27AE60"],[0.5,"#F4D03F"],[1,"#C0392B"]],
            "direction": "Higher = More Exposure (Worse)",
            "insight": "Texas, Louisiana, and California get hit the most, with each averaging over 60 FEMA declarations per year, which is about 3 times more than a typical state.",
        },
        "Damage per Capita": {
            "col":   "avg_damage_per_capita",
            "title": "DRPA Risk Map: Damage per Capita",
            "desc":  "Average per-capita property and crop damage from federally declared disasters (USD). Measures the economic magnitude of disaster events relative to population size.",
            "scale": [[0,"#27AE60"],[0.5,"#F4D03F"],[1,"#C0392B"]],
            "direction": "Higher = Greater Damage Burden (Worse)",
            "insight": "Small states like Vermont and North Dakota don't make headlines, but their damage per person is surprisingly high, a reminder that total damage numbers alone can be misleading.",
        },
    }

    _spacer, _drop_col = st.columns([1.5, 1])
    with _drop_col:
        _metric_label = st.selectbox(
            "Select indicator:",
            list(_metric_meta.keys()),
            index=0,
        )
    st.title(f"Risk Map: {_metric_label}")
    meta       = _metric_meta[_metric_label]
    metric_opt = meta["col"]
    st.markdown(
        f"<div style='background:#f0f4f8;border-left:4px solid #2980B9;border-radius:4px;"
        f"padding:10px 16px;margin:4px 0 10px 0;'>"
        f"<span style='font-size:13px;color:#1a1a2e;line-height:1.5;'>{meta['desc']}</span>"
        f"<hr style='border:none;border-top:1px solid #d0d8e4;margin:8px 0;'>"
        f"<span style='font-size:12px;color:#2980B9;font-weight:600;'>Key Insight &nbsp;</span>"
        f"<span style='font-size:12px;color:#333;'>{meta['insight']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Color direction label
    direction_color = "#C0392B" if "Worse" in meta["direction"] or "Risk" in meta["direction"] else "#27AE60"
    st.markdown(
        f"<p style='font-size:11px;font-weight:700;color:{direction_color};"
        f"text-transform:uppercase;letter-spacing:0.8px;margin:0 0 6px 0;'>"
        f"▲ {meta['direction']}</p>",
        unsafe_allow_html=True,
    )

    # ── Generate per-state key insight sentence ───────────────────────────────
    _tier_emoji  = {"CRITICAL":"🔴","HIGH":"🟠","MODERATE":"🟡","LOW":"🟢"}
    _profile_msg = {
        "Industry-Intensive":   "Industrial economy supports faster stabilization",
        "Service-Dominated":    "Service based structure tends to slow labor recovery",
        "Resource-Intensive":   "Resource dependent economy adds structural fragility",
        "Government-Intensive": "Government heavy economy extends recovery timelines",
    }
    _fiscal_med = df["fiscal_capacity_per_capita"].median() if "fiscal_capacity_per_capita" in df.columns else 10000

    def _state_insight(row):
        profile = str(row.get("economic_profile", ""))
        tier    = str(row.get("drpi_risk_tier", "MODERATE"))
        fiscal  = float(row.get("fiscal_capacity_per_capita", _fiscal_med))
        base    = _profile_msg.get(profile, "Mixed economic structure affects recovery speed")
        if fiscal < _fiscal_med * 0.8:
            fiscal_note = "limited fiscal buffers reduce response capacity"
        elif fiscal > _fiscal_med * 1.25:
            fiscal_note = "strong fiscal capacity supports faster emergency response"
        else:
            fiscal_note = "moderate fiscal capacity available for recovery"
        return f"{base}; {fiscal_note}."

    if "_insight" not in df.columns:
        df["_insight"] = df.apply(_state_insight, axis=1)

    def _tier_label(t):
        return {"CRITICAL":"Critical Risk","HIGH":"High Risk",
                "MODERATE":"Moderate Risk","LOW":"Low Risk"}.get(str(t), str(t))

    df["_tier_label"]  = df["drpi_risk_tier"].apply(_tier_label)
    df["_tier_emoji"]  = df["drpi_risk_tier"].map(_tier_emoji).fillna("⚪")

    _profile_dot_map = {
        "Industry-Intensive":  "🟢",
        "Resource-Intensive":  "🟠",
        "Service-Dominated":   "🔵",
        "Government-Intensive":"🟣",
    }
    df["_profile_dot"] = df["economic_profile"].map(_profile_dot_map).fillna("⚪")

    # ── Dynamic per-indicator hover template ─────────────────────────────────
    # Fixed custom_data indices (same for all maps):
    #   [0] drpi_score            [1] drpi_risk_tier
    #   [2] recovery_months_pred  [3] _tier_label
    #   [4] _tier_emoji           [5] economic_profile
    #   [6] _insight              [7] compound_vulnerability_index
    #   [8] fiscal_capacity_pc    [9] avg_disaster_exposure_12m
    #   [10] avg_damage_per_capita [11] _profile_dot

    # Primary line changes per selected indicator
    _primary_lines = {
        "Composite Risk": (
            "Composite Score: %{customdata[0]:.1f} / 100<br>"
            "Recovery: ~%{customdata[2]:.0f} months<br>"
            "Economy: %{customdata[11]} %{customdata[5]}<br>"
        ),
        "Recovery Time (Months)": (
            "Recovery Time: ~%{customdata[2]:.0f} months<br>"
            "DRPA Score: %{customdata[0]:.1f} / 100<br>"
            "Economy: %{customdata[11]} %{customdata[5]}<br>"
        ),
        "Vulnerability Index (CVI)": (
            "CVI Score: %{customdata[7]:.3f} (0–1 scale)<br>"
            "DRPA Score: %{customdata[0]:.1f} / 100<br>"
            "Recovery: ~%{customdata[2]:.0f} months<br>"
        ),
        "Fiscal Capacity": (
            "Fiscal Capacity: $%{customdata[8]:,.0f} / capita<br>"
            "DRPA Score: %{customdata[0]:.1f} / 100<br>"
            "Recovery: ~%{customdata[2]:.0f} months<br>"
        ),
        "Disaster Exposure": (
            "FEMA Events: %{customdata[9]:.0f} / year<br>"
            "DRPA Score: %{customdata[0]:.1f} / 100<br>"
            "Recovery: ~%{customdata[2]:.0f} months<br>"
        ),
        "Damage per Capita": (
            "Avg Damage: $%{customdata[10]:,.0f} / capita<br>"
            "DRPA Score: %{customdata[0]:.1f} / 100<br>"
            "Recovery: ~%{customdata[2]:.0f} months<br>"
        ),
    }

    primary = _primary_lines.get(_metric_label, _primary_lines["Composite Risk"])
    hover_template = (
        "<b>%{hovertext}</b><br>"
        "──────────────────<br>"
        + primary +
        "<extra></extra>"
    )

    # Safe fallback for missing columns
    def _safe_col(col):
        return col if col in df.columns else "drpi_score"

    fig = px.choropleth(
        df, locations="abbr", locationmode="USA-states",
        color=metric_opt, hover_name="state",
        color_continuous_scale=meta["scale"],
        scope="usa",
        labels={metric_opt: _metric_label},
        custom_data=[
            "drpi_score",                              # [0]
            "drpi_risk_tier",                          # [1]
            "recovery_months_predicted",               # [2]
            "_tier_label",                             # [3]
            "_tier_emoji",                             # [4]
            "economic_profile",                        # [5]
            "_insight",                                # [6]
            _safe_col("compound_vulnerability_index"), # [7]
            _safe_col("fiscal_capacity_per_capita"),   # [8]
            _safe_col("avg_disaster_exposure_12m"),    # [9]
            _safe_col("avg_damage_per_capita"),        # [10]
            "_profile_dot",                            # [11]
        ],
    )
    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(
        height=520, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        coloraxis_colorbar=dict(title=_metric_label, thickness=12, len=0.6),
    )
    st.plotly_chart(fig, key="detail_map", width="stretch")


    # ── Economic Profile + Disaster & Economic Drivers — side by side ────────
    _exp_med   = df["avg_disaster_exposure_12m"].median()   if "avg_disaster_exposure_12m"    in df.columns else 20
    _fisc_med  = df["fiscal_capacity_per_capita"].median()  if "fiscal_capacity_per_capita"   in df.columns else 10000
    _dmg_med   = df["avg_damage_per_capita"].median()       if "avg_damage_per_capita"        in df.columns else 500
    _cvi_med   = df["compound_vulnerability_index"].median()if "compound_vulnerability_index" in df.columns else 0.5
    _rec_med   = df["recovery_months_predicted"].median()   if "recovery_months_predicted"    in df.columns else 14

    def _drivers_text(label, data):
        lines = []
        exp   = data["avg_disaster_exposure_12m"].mean()   if "avg_disaster_exposure_12m"    in data.columns else _exp_med
        fisc  = data["fiscal_capacity_per_capita"].mean()  if "fiscal_capacity_per_capita"   in data.columns else _fisc_med
        dmg   = data["avg_damage_per_capita"].mean()       if "avg_damage_per_capita"        in data.columns else _dmg_med
        cvi   = data["compound_vulnerability_index"].mean()if "compound_vulnerability_index" in data.columns else _cvi_med
        rec   = data["recovery_months_predicted"].mean()   if "recovery_months_predicted"    in data.columns else _rec_med

        if label == "Composite Risk":
            if exp > _exp_med * 1.2:
                lines.append("States with more FEMA declarations (FEMA OpenFEMA) tend to carry higher baseline risk, as repeated events strain recovery systems before full stabilization is reached.")
            if fisc < _fisc_med * 0.9:
                lines.append("States with below average revenue per capita (U.S. Census) take longer to deploy recovery funding, which directly raises their composite score.")
            if rec > _rec_med:
                lines.append("Predicted recovery times above the national median (BLS LAUS) signal that both economic structure and fiscal constraints are slowing the process.")
            lines.append("States that score high on both disaster exposure and fiscal weakness consistently rank in the top risk tier across all indicators.")
        elif label == "Recovery Time (Months)":
            if rec > _rec_med:
                lines.append("States with government heavy or service dominated economies (BEA State GDP) show longer recovery timelines because these sectors are slower to stabilize after economic disruptions.")
            if fisc < _fisc_med * 0.9:
                lines.append("Limited state revenue (U.S. Census) reduces the speed at which disaster relief programs and multi-sector recovery interventions can be activated.")
            if exp > _exp_med * 1.2:
                lines.append("When FEMA declarations occur frequently (FEMA OpenFEMA), states rarely complete recovery before the next event begins, extending predicted timelines.")
            lines.append("Among all variables in the model, economic structure is the strongest predictor of how many months a state needs to return to pre-disaster economic conditions.")
        elif label == "Vulnerability Index (CVI)":
            if exp > _exp_med:
                lines.append("High disaster frequency (FEMA OpenFEMA) keeps states in a persistent recovery cycle. Before one event resolves, the next one arrives, preventing a return to baseline.")
            if fisc < _fisc_med:
                lines.append("States with low fiscal capacity (U.S. Census) have less room to absorb shocks. When revenue is limited, the cost of disasters falls more heavily on households and local economies.")
            if dmg > _dmg_med:
                lines.append("Above average per capita damage (FEMA damage records) signals that disasters are economically significant relative to the size of the state population, compounding long term vulnerability.")
            lines.append("The CVI is not only about how often disasters happen. It measures whether a state has the economic and fiscal conditions to recover when they do.")
        elif label == "Fiscal Capacity":
            if fisc > _fisc_med * 1.2:
                lines.append("States with strong fiscal capacity (U.S. Census) can fund disaster response from their own revenue, reducing wait times associated with federal reimbursement processes.")
            else:
                lines.append("States below the national fiscal median (U.S. Census) depend more heavily on FEMA reimbursements, which introduces delays in the recovery timeline.")
            lines.append("Fiscal capacity does not reduce disaster risk directly. It determines how quickly a state can act once a disaster occurs.")
        elif label == "Disaster Exposure":
            if exp > _exp_med * 1.5:
                lines.append("The most exposed states (FEMA OpenFEMA) face overlapping recovery burdens. Each new declaration begins before the economic effects of the previous one have fully resolved.")
            lines.append("Gulf Coast and Southern states account for a large share of total FEMA major-disaster declarations between 2006 and 2024.")
            lines.append("High exposure alone does not place a state in the top risk tier. When combined with low fiscal capacity, the compounding effect is what drives vulnerability scores up.")
        elif label == "Damage per Capita":
            if dmg > _dmg_med:
                lines.append("Per-capita damage (FEMA damage records, U.S. Census) tends to be higher in smaller or rural states, where the same dollar amount of damage is spread across fewer people.")
            lines.append("States with high per-capita damage and limited fiscal reserves face the steepest recovery challenges, as the economic burden falls on a smaller tax base.")
            lines.append("Looking at damage per capita rather than total damage reveals patterns that aggregate figures tend to obscure, particularly for less populous states.")
        return lines

    _driver_lines = _drivers_text(_metric_label, df)
    _show_profile = _metric_label in ("Recovery Time (Months)", "Composite Risk")

    if _show_profile:
        _col_left, _col_right = st.columns([1, 1], gap="medium")
    else:
        _col_left, _col_right = None, st.columns([1])[0]

    if _show_profile and _col_left is not None:
        with _col_left:
            st.markdown("""
<div style='background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;
            padding:10px 16px;margin:10px 0 6px 0;height:100%;'>
    <p style='font-size:11px;font-weight:700;color:#555;text-transform:uppercase;
              letter-spacing:0.8px;margin:0 0 8px 0;'>Economic Profile: What It Means for Recovery</p>
    <div style='display:flex;gap:20px;flex-wrap:wrap;'>
        <span style='font-size:12px;color:#27AE60;font-weight:600;'>🟢 Industry-Intensive</span>
        <span style='font-size:12px;color:#333;'>Manufacturing &amp; construction → <b>fastest recovery ~11.5 mo</b></span>
    </div>
    <div style='display:flex;gap:20px;flex-wrap:wrap;margin-top:4px;'>
        <span style='font-size:12px;color:#E67E22;font-weight:600;'>🟠 Resource-Intensive</span>
        <span style='font-size:12px;color:#333;'>Mining, agriculture &amp; energy → <b>~14.4 mo</b></span>
    </div>
    <div style='display:flex;gap:20px;flex-wrap:wrap;margin-top:4px;'>
        <span style='font-size:12px;color:#2980B9;font-weight:600;'>🔵 Service-Dominated</span>
        <span style='font-size:12px;color:#333;'>Retail, tourism &amp; finance → <b>~16.2 mo</b></span>
    </div>
    <div style='display:flex;gap:20px;flex-wrap:wrap;margin-top:4px;'>
        <span style='font-size:12px;color:#8E44AD;font-weight:600;'>🟣 Government-Intensive</span>
        <span style='font-size:12px;color:#333;'>Federal &amp; military employment → <b>slowest recovery ~18 mo</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

    if _driver_lines:
        _bullets = "".join(
            f"<div style='display:flex;gap:8px;margin-bottom:5px;'>"
            f"<span style='color:#E67E22;font-weight:700;margin-top:1px;'>▸</span>"
            f"<span style='font-size:12px;color:#2c2c2c;line-height:1.5;'>{ln}</span>"
            f"</div>"
            for ln in _driver_lines
        )
        with _col_right:
            st.markdown(
                f"<div style='background:#fffaf4;border-left:4px solid #E67E22;border-radius:4px;"
                f"padding:10px 16px;margin:10px 0 6px 0;'>"
                f"<span style='font-size:11px;font-weight:700;color:#E67E22;text-transform:uppercase;"
                f"letter-spacing:0.8px;'>Disaster &amp; Economic Drivers</span>"
                f"<div style='margin-top:8px;'>{_bullets}</div>"
                + (
                f"<hr style='border:none;border-top:1px solid #f0d9b5;margin:10px 0;'>"
                f"<span style='font-size:11px;font-weight:700;color:#E67E22;text-transform:uppercase;"
                f"letter-spacing:0.8px;'>How the CVI Score is Built</span>"
                f"<p style='font-size:12px;color:#444;margin:8px 0 8px 0;line-height:1.5;'>"
                f"Five factors drawn from BLS, FEMA, U.S. Census, and BEA data, each weighted by how much it contributes to recovery outcomes:</p>"
                f"<table style='width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px;'>"
                f"<thead>"
                f"<tr style='background:#f0d9b5;'>"
                f"<th style='text-align:left;padding:6px 10px;color:#7d4e00;font-weight:700;border:1px solid #e8c88a;'>Dimension</th>"
                f"<th style='text-align:center;padding:6px 10px;color:#7d4e00;font-weight:700;border:1px solid #e8c88a;'>Weight</th>"
                f"<th style='text-align:left;padding:6px 10px;color:#7d4e00;font-weight:700;border:1px solid #e8c88a;'>What it measures</th>"
                f"</tr>"
                f"</thead>"
                f"<tbody>"
                f"<tr style='background:#fff;'>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#2c2c2c;'>Labor Volatility</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;text-align:center;font-weight:700;color:#E67E22;'>25%</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#555;'>How unstable unemployment gets after a disaster</td>"
                f"</tr>"
                f"<tr style='background:#fffaf4;'>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#2c2c2c;'>Disaster Exposure</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;text-align:center;font-weight:700;color:#E67E22;'>25%</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#555;'>How often FEMA declares major disasters per year</td>"
                f"</tr>"
                f"<tr style='background:#fff;'>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#2c2c2c;'>Damage Burden</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;text-align:center;font-weight:700;color:#E67E22;'>20%</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#555;'>Average disaster damage per person (in USD)</td>"
                f"</tr>"
                f"<tr style='background:#fffaf4;'>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#2c2c2c;'>Fiscal Weakness</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;text-align:center;font-weight:700;color:#E67E22;'>15%</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#555;'>How limited state funding is for emergency recovery</td>"
                f"</tr>"
                f"<tr style='background:#fff;'>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#2c2c2c;'>Structural Risk</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;text-align:center;font-weight:700;color:#E67E22;'>15%</td>"
                f"<td style='padding:5px 10px;border:1px solid #edd9a3;color:#555;'>How service heavy the economy is. Service sectors tend to recover more slowly after disruptions.</td>"
                f"</tr>"
                f"</tbody>"
                f"</table>"
                f"<p style='font-family:\"Caveat\",\"Comic Sans MS\",cursive;font-size:16px;color:#7d4e00;"
                f"margin:6px 0 4px 0;line-height:1.6;'>"
                f"CVI = (0.25 × labor) + (0.25 × exposure) + (0.20 × damage) + (0.15 × fiscal) + (0.15 × structure)"
                f"</p>"
                f"<p style='font-size:11px;color:#888;margin:4px 0 0 0;font-style:italic;'>"
                f"Each factor is normalized to a 0–1 scale before combining. Higher score = more vulnerable.</p>"
                if _metric_label == "Vulnerability Index (CVI)" else ""
                )
                + f"</div>",
                unsafe_allow_html=True,
            )

    # ── CVI Dimension Maps ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Disaster Recovery Vulnerability Across U.S. States (CVI)")
    st.markdown(
        "The **Compound Vulnerability Index (CVI)** combines five independent risk dimensions "
        "into a single composite score (0–1) for each state. Each dimension is normalized "
        "so that **darker red = higher vulnerability**. Together they explain *why* a state "
        "scores high or low on the overall DRPA. A state may be exposed to many disasters "
        "but recover quickly if it has high fiscal capacity and a stable labor market.",
        unsafe_allow_html=False,
    )

    # Dimension metadata: (column, short title, weight, what it measures, data source)
    cvi_meta = [
        ("cvi_labor",    "Labor Volatility",   "25% of CVI",
         "Standard deviation of monthly unemployment rates (2010–2024). "
         "High volatility means the workforce is more sensitive to economic shocks following a disaster.",
         "Source: BLS LAUS"),
        ("cvi_disaster", "Disaster Exposure",  "25% of CVI",
         "Average number of FEMA major-disaster declarations per 12-month window (2006–2024). "
         "States with more frequent declarations face recurring recovery burdens.",
         "Source: FEMA OpenFEMA"),
        ("cvi_damage",   "Damage Burden",      "20% of CVI",
         "Average per-capita property and crop damage from federally declared disasters. "
         "Measures the economic magnitude of disaster events relative to population size.",
         "Source: FEMA / U.S. Census"),
        ("cvi_fiscal",   "Fiscal Weakness",    "15% of CVI",
         "Inverse of state revenue per capita. States with lower fiscal capacity have fewer "
         "resources to fund emergency response, relief programs, and infrastructure repair.",
         "Source: U.S. Census State Government Finances"),
        ("cvi_structure","Structural Risk",    "15% of CVI",
         "Service-sector share of state GDP. Service-dominant economies tend to recover more "
         "slowly because service jobs are harder to resume remotely during infrastructure disruptions.",
         "Source: BEA State GDP"),
    ]

    cols = st.columns(5)
    for i, (dim, lbl, weight, description, source) in enumerate(cvi_meta):
        if dim not in df.columns:
            continue
        with cols[i]:
            cvi_hover = (
                f"<b>%{{hovertext}}</b><br>"
                f"{lbl}: %{{customdata[0]:.3f}}<br>"
                f"Risk Level: %{{customdata[2]}} %{{customdata[1]}}<br>"
                f"<i>Higher score = more vulnerable</i>"
                f"<extra></extra>"
            )
            fig_mini = px.choropleth(
                df, locations="abbr", locationmode="USA-states",
                color=dim, hover_name="state", color_continuous_scale="OrRd",
                range_color=(0, 1), scope="usa", labels={dim: lbl},
                custom_data=[dim, "_tier_label", "_tier_emoji"],
            )
            fig_mini.update_traces(hovertemplate=cvi_hover)
            fig_mini.update_layout(
                height=200, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text=f"<b>{lbl}</b>", x=0.5, font_size=11),
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(bgcolor="rgba(0,0,0,0)"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_mini, key=f"cvi_{dim}")
            st.markdown(
                f"<p style='font-size:10px;font-weight:700;color:#C0392B;"
                f"letter-spacing:0.5px;margin:0 0 3px 0;text-transform:uppercase;'>"
                f"{weight}</p>"
                f"<p style='font-size:11px;color:#333;margin:0 0 4px 0;line-height:1.4;'>"
                f"{description}</p>"
                f"<p style='font-size:10px;color:#888;margin:0;font-style:italic;'>{source}</p>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESEARCH ANALYTICS (paper figures replicated + extended)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Research Analytics":
    st.subheader("Research Analytics: Dataset Findings Visualized")
    st.caption("Findings derived from FEMA · BLS · NOAA · BEA · U.S. Census datasets | UNC Charlotte DTSC 4302")

    # ── Custom tab styling ────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Tab list container */
    div[data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e8e8e8 !important;
        padding-bottom: 0px;
    }
    /* All tabs base style */
    div[data-baseweb="tab-list"] button[role="tab"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border-radius: 6px 6px 0 0 !important;
        border: 1px solid #ddd !important;
        border-bottom: none !important;
        background: #f5f5f5 !important;
        color: #666 !important;
        letter-spacing: 0.3px;
        transition: all 0.15s ease;
    }
    /* Tab 1 — Economic Structure — green accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(1) {
        border-top: 3px solid #27AE60 !important;
        color: #27AE60 !important;
    }
    /* Tab 2 — Fiscal Capacity & Recovery — blue accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(2) {
        border-top: 3px solid #2980B9 !important;
        color: #2980B9 !important;
    }
    /* Tab 3 — Service Economy & Recovery — orange accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(3) {
        border-top: 3px solid #E67E22 !important;
        color: #E67E22 !important;
    }
    /* Tab 4 — Political Economy & Aid — crimson accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(4) {
        border-top: 3px solid #C0392B !important;
        color: #C0392B !important;
    }
    /* Tab 5 — Education Disruption — amber gold accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(5) {
        border-top: 3px solid #D4AC0D !important;
        color: #D4AC0D !important;
    }
    /* Tab 6 — Recovery Predictors — purple accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(6) {
        border-top: 3px solid #8E44AD !important;
        color: #8E44AD !important;
    }
    /* Tab 7 — Policy Insights — teal accent */
    div[data-baseweb="tab-list"] button[role="tab"]:nth-child(7) {
        border-top: 3px solid #16A085 !important;
        color: #16A085 !important;
    }
    /* Active tab — white background, bold, lifted */
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        background: #ffffff !important;
        font-weight: 800 !important;
        font-size: 13px !important;
        color: #0d1b2a !important;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.06) !important;
    }
    /* Hover effect */
    div[data-baseweb="tab-list"] button[role="tab"]:hover {
        background: #efefef !important;
        color: #0d1b2a !important;
    }
    /* Hide the default red underline highlight */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "  Economic Structure  ",
        "  Fiscal Capacity & Recovery  ",
        "  Service Economy & Recovery  ",
        "  Political Economy & Aid  ",
        "  Education Disruption  ",
        "  Recovery Predictors  ",
        "  Policy Insights  ",
    ])

    # ── TAB 1: Recovery by Economic Profile (Paper Fig 10) ───────────────────
    with tab1:
        # ── Clear main message header ─────────────────────────────────────────
        st.markdown(
            "<div style='margin:0 0 14px 0;'>"
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
            "Recovery Outcomes and Industry Composition Across High-Exposure States</p>"
            "<p style='font-size:13px;color:#555;margin:0;'>"
            "<span style='color:#27AE60;'>Industry:</span> <b style='color:#27AE60;'>11.5 mo</b> &nbsp;·&nbsp; "
            "<span style='color:#E67E22;'>Resources:</span> <b style='color:#E67E22;'>14.4 mo</b> &nbsp;·&nbsp; "
            "<span style='color:#2980B9;'>Services:</span> <b style='color:#2980B9;'>16.2 mo</b> &nbsp;·&nbsp; "
            "<span style='color:#8E44AD;'>Government:</span> <b style='color:#8E44AD;'>18.0 mo</b> "
            "&nbsp;&nbsp;<span style='color:#aaa;font-size:12px;'>| Dataset averages by economic profile</span>"
            "</p></div>",
            unsafe_allow_html=True,
        )


        if "economic_profile" in df.columns:
            prof_df = df.dropna(subset=["economic_profile","recovery_months_predicted"])
            prof_avg = (
                prof_df.groupby("economic_profile")
                .agg(avg_predicted=("recovery_months_predicted","mean"),
                     count=("state","count"),
                     states=("state", lambda x:", ".join(sorted(x))))
                .reset_index()
            )
            prof_avg["paper_benchmark"] = prof_avg["economic_profile"].map(PAPER_PROFILE_MONTHS)
            prof_avg["label"] = prof_avg["economic_profile"].map(PROFILE_SHORT).fillna(prof_avg["economic_profile"])
            # Sort fastest → slowest by paper benchmark
            prof_avg = prof_avg.sort_values("paper_benchmark").reset_index(drop=True)

            # ── Single clean bar chart ────────────────────────────────────────
            fig_prof = go.Figure()
            fig_prof.add_bar(
                x=prof_avg["label"],
                y=prof_avg["avg_predicted"],
                name="DRPA Predicted",
                marker_color=[PROFILE_COLORS.get(p, "#888") for p in prof_avg["economic_profile"]],
                text=[f"{v:.1f} mo" for v in prof_avg["avg_predicted"]],
                textposition="outside",
                textfont=dict(size=14, color="#0d1b2a", family="Arial Black"),
                width=0.4,
                customdata=list(zip(
                    prof_avg["economic_profile"],
                    prof_avg["avg_predicted"],
                    prof_avg["count"],
                    prof_avg["paper_benchmark"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "──────────────────<br>"
                    "DRPA Predicted Recovery: <b>%{customdata[1]:.1f} months</b><br>"
                    "Dataset Benchmark: <b>%{customdata[3]:.1f} months</b><br>"
                    "States in this group: <b>%{customdata[2]}</b><br>"
                    "<i style='font-size:11px;color:#888;'>Based on BLS LAUS labor market data (2010–2024)</i>"
                    "<extra></extra>"
                ),
            )
            fig_prof.add_bar(
                x=prof_avg["label"],
                y=prof_avg["paper_benchmark"],
                name="Dataset Benchmark",
                marker_color="rgba(0,0,0,0.10)",
                marker_line=dict(color="#555", width=2),
                text=[f"{v} mo" for v in prof_avg["paper_benchmark"]],
                textposition="outside",
                textfont=dict(size=12, color="#888"),
                width=0.4,
                customdata=list(zip(
                    prof_avg["economic_profile"],
                    prof_avg["paper_benchmark"],
                    prof_avg["avg_predicted"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}: Benchmark</b><br>"
                    "──────────────────<br>"
                    "Dataset Benchmark: <b>%{customdata[1]:.1f} months</b><br>"
                    "DRPA Predicted: <b>%{customdata[2]:.1f} months</b><br>"
                    "<i style='font-size:11px;color:#888;'>Reference value from research dataset</i>"
                    "<extra></extra>"
                ),
            )
            fig_prof.update_layout(
                barmode="group",
                height=430,
                xaxis_title="",
                yaxis_title="Months to Recover",
                yaxis=dict(range=[0, 25], gridcolor="#f0f0f0", zeroline=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                            font=dict(size=12)),
                font=dict(size=13),
                bargap=0.35,
                margin=dict(t=20, b=60),
            )
            st.plotly_chart(fig_prof, key="prof_bar")

            # ── Key Findings box (Tab 1) ──────────────────────────────────────
            st.markdown(
                "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:8px 0 20px 0;'>"
                "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
                "letter-spacing:1.2px;margin:0 0 4px 0;'>Key Findings</p>"
                "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:1.9;'>"
                "<li>Industry-based economies recover the fastest, averaging <b style='color:#F4D03F;'>~11.5 months</b>.</li>"
                "<li>Service economies take around <b style='color:#F4D03F;'>16.2 months</b>, roughly 40% longer.</li>"
                "<li>Government-dependent economies are the slowest at <b style='color:#F4D03F;'>~18.0 months</b>, which is 1.6× slower than industry states.</li>"
                "<li>Economic structure is a stronger predictor of recovery speed than fiscal revenue alone. "
                "The colored bars show model predictions; gray bars show historical averages. "
                "When the colored bar exceeds gray, current conditions signal a longer recovery ahead.</li>"
                "</ul>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── State breakdown ───────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:12px;font-weight:700;color:#888;"
                "text-transform:uppercase;letter-spacing:0.8px;margin:0 0 8px 0;'>"
                "States by Economic Profile</p>",
                unsafe_allow_html=True,
            )
            for _, row in prof_avg.iterrows():
                color = PROFILE_COLORS.get(row["economic_profile"], "#888")
                short = PROFILE_SHORT.get(row["economic_profile"], row["economic_profile"])
                st.markdown(
                    f"<div style='background:{color}15;border-left:4px solid {color};"
                    f"padding:7px 14px;border-radius:0 6px 6px 0;margin:4px 0'>"
                    f"<span style='font-weight:700;color:{color};font-size:13px;'>{short}</span>"
                    f"<span style='color:#777;font-size:12px;'>"
                    f" &nbsp;·&nbsp; {row['count']} states &nbsp;·&nbsp; avg {row['avg_predicted']:.1f} mo"
                    f" &nbsp;·&nbsp; dataset benchmark {row['paper_benchmark']} mo</span>"
                    f"<br><span style='font-size:12px;color:#444;'>{row['states']}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── TAB 2: Fiscal Capacity vs Recovery (Paper Fig 6) ─────────────────────
    with tab2:
        if "fiscal_capacity_per_capita" in df.columns:
            fc_df = df.dropna(subset=["fiscal_capacity_per_capita","recovery_months_predicted"]).copy()

            # OLS trendline stats
            from numpy.polynomial import polynomial as P
            x = fc_df["fiscal_capacity_per_capita"].values
            y = fc_df["recovery_months_predicted"].values
            coeffs = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - np.polyval(coeffs,x))**2) / np.sum((y-y.mean())**2)

            # ── Pre-compute tooltip context fields ────────────────────────────
            _avg_recovery  = fc_df["recovery_months_predicted"].mean()
            _med_exposure  = fc_df["avg_disaster_exposure_12m"].median()
            _med_fiscal    = fc_df["fiscal_capacity_per_capita"].median()

            def _fc_vs_avg(v):
                diff = v - _avg_recovery
                if diff > 1.5:   return f"Above average (+{diff:.1f} mo)"
                elif diff < -1.5: return f"Below average ({diff:.1f} mo)"
                else:             return "Near national average"

            def _fc_insight(row):
                hi_exp  = row["avg_disaster_exposure_12m"] >= _med_exposure
                hi_fisc = row["fiscal_capacity_per_capita"] >= _med_fiscal
                if hi_exp and not hi_fisc:
                    return "High disaster exposure with limited fiscal resources increases recovery vulnerability."
                elif hi_exp and hi_fisc:
                    return "High exposure despite strong fiscal resources points to structural economic risk."
                elif not hi_exp and not hi_fisc:
                    return "Limited fiscal capacity may slow recovery when disasters occur."
                else:
                    return "Strong fiscal resources and lower exposure support faster recovery."

            fc_df["_vs_avg"]   = fc_df["recovery_months_predicted"].apply(_fc_vs_avg)
            fc_df["_insight"]  = fc_df.apply(_fc_insight, axis=1)
            _profile_col = fc_df["economic_profile"] if "economic_profile" in fc_df.columns else pd.Series(["—"]*len(fc_df))

            # ── Title + Blue insight box (top) ───────────────────────────────
            st.markdown(
                "<div style='margin:0 0 14px 0;'>"
                "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
                "Fiscal Capacity Is a Weak Predictor of Recovery</p>"
                "<p style='font-size:13px;color:#555;margin:0;'>"
                "Economic structure explains recovery speed better than available state revenue."
                "</p></div>",
                unsafe_allow_html=True,
            )
            # ── Row 1: Quadrant scatter ───────────────────────────────────────
            import plotly.graph_objects as go

            _med_rec   = fc_df["recovery_months_predicted"].median()
            _med_fisc2 = fc_df["fiscal_capacity_per_capita"].median()

            # Label all states
            fc_df["_label"] = fc_df["abbr"]

            # Scale bubble size from avg FEMA declarations — matches paper Fig 6
            _exp_vals = fc_df["avg_disaster_exposure_12m"].fillna(1)
            _sizeref  = 2.0 * _exp_vals.max() / (28 ** 2)   # normalise so max bubble ≈ 28px

            fig_fc = go.Figure()

            # One trace per risk tier for clean legend
            for tier in ["CRITICAL","HIGH","MODERATE","LOW"]:
                _sub = fc_df[fc_df["drpi_risk_tier"] == tier]
                if _sub.empty:
                    continue
                fig_fc.add_trace(go.Scatter(
                    x=_sub["fiscal_capacity_per_capita"],
                    y=_sub["recovery_months_predicted"],
                    mode="markers+text",
                    name=tier,
                    marker=dict(
                        color=TIER_COLORS.get(tier, "#aaa"),
                        size=_exp_vals.reindex(_sub.index).values,
                        sizemode="area",
                        sizeref=_sizeref,
                        sizemin=6,
                        line=dict(width=1, color="white"),
                        opacity=0.88,
                    ),
                    text=_sub["_label"],
                    textposition="top center",
                    textfont=dict(size=9, color="#222", family="Arial Black, Arial"),
                    customdata=np.stack((
                        _sub["state"],
                        _sub["fiscal_capacity_per_capita"],
                        _sub["recovery_months_predicted"],
                        _sub["drpi_risk_tier"],
                        _sub["avg_disaster_exposure_12m"],
                        _profile_col.reindex(_sub.index).values,
                        _sub["_vs_avg"].values,
                        _sub["_insight"].values,
                    ), axis=-1),
                    hovertemplate=(
                        "<b style='font-size:14px;'>%{customdata[0]}</b><br>"
                        "<span style='font-size:11px;color:#888;'>Economy type: %{customdata[5]}</span><br>"
                        "<span style='color:#e0e0e0;'>────────────────────────────</span><br>"
                        "<span style='font-size:11px;color:#444;'>Predicted recovery: </span>"
                        "<b style='font-size:12px;color:#222;'>%{customdata[2]:.1f} months</b><br>"
                        "<span style='font-size:10px;color:#999;padding-left:4px;'>%{customdata[6]}</span><br>"
                        "<span style='font-size:11px;color:#444;'>State revenue per person: </span>"
                        "<b style='font-size:11px;color:#222;'>$%{customdata[1]:,.0f}</b><br>"
                        "<span style='font-size:11px;color:#444;'>Risk classification: </span>"
                        "<b style='font-size:11px;color:#222;'>%{customdata[3]}</b><br>"
                        "<span style='font-size:11px;color:#444;'>Avg. disaster declarations/yr: </span>"
                        "<b style='font-size:11px;color:#222;'>%{customdata[4]:.0f}</b><br>"
                        "<span style='color:#e0e0e0;'>────────────────────────────</span><br>"
                        "<span style='font-size:11px;color:#444;font-weight:600;'>What this means: </span><br>"
                        "<i style='font-size:11px;color:#555;'>%{customdata[7]}</i>"
                        "<extra></extra>"
                    ),
                ))

            # Quadrant lines at medians
            fig_fc.add_vline(x=_med_fisc2, line_dash="dash", line_color="#bbb", line_width=1)
            fig_fc.add_hline(y=_med_rec,   line_dash="dash", line_color="#bbb", line_width=1)

            # Quadrant labels — pinned to corners (xref/yref paper keeps them outside data)
            _pad_x = 0.01  # fraction of paper coords
            for _txt, _px, _py, _xanchor, _yanchor in [
                ("Low fiscal · Slow recovery",  0.01, 0.99, "left",  "top"),
                ("High fiscal · Slow recovery", 0.99, 0.99, "right", "top"),
                ("Low fiscal · Fast recovery",  0.01, 0.01, "left",  "bottom"),
                ("High fiscal · Fast recovery", 0.99, 0.01, "right", "bottom"),
            ]:
                fig_fc.add_annotation(
                    xref="paper", yref="paper",
                    x=_px, y=_py,
                    text=f"<b>{_txt}</b>",
                    showarrow=False,
                    font=dict(size=10, color="#c0c0c0"),
                    xanchor=_xanchor,
                    yanchor=_yanchor,
                    align="center",
                    bgcolor="rgba(255,255,255,0.6)",
                    borderpad=3,
                )

            fig_fc.update_layout(
                title=dict(text=f"Fiscal Capacity vs Recovery Time  |  R² = {r2:.2f}  (median lines shown)", font_size=13),
                xaxis=dict(title="Fiscal Capacity (state revenue $ / person)",
                           tickprefix="$", showgrid=True, gridcolor="#f0f0f0", zeroline=False),
                yaxis=dict(title="Predicted Recovery Time (months)",
                           showgrid=True, gridcolor="#f0f0f0", zeroline=False),
                legend=dict(title="Risk Level", orientation="v"),
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=60, r=40, t=50, b=60),
                font=dict(size=11),
            )
            st.plotly_chart(fig_fc, key="fiscal_scatter", width="stretch")

            # ── Blue insight box — below chart ────────────────────────────────
            st.markdown(
                "<div style='background:#eaf4fb;border-left:4px solid #2980B9;border-radius:4px;"
                "padding:12px 16px;margin:12px 0 12px 0;'>"
                "<p style='font-size:13px;font-weight:700;color:#1a5276;margin:0 0 4px 0;'>"
                "Fiscal capacity alone is a weak predictor of recovery speed. Economic structure plays a much larger role.</p>"
                "<p style='font-size:12px;color:#2c3e50;margin:0;line-height:1.6;'>"
                "Recovery time does not strongly improve with higher fiscal capacity. While states with more resources "
                f"may recover slightly faster, the relationship is weak (R² = {r2:.2f}). "
                "What matters more is how a state's economy is structured, including its dependence on services, industry, or government sectors."
                "</p></div>",
                unsafe_allow_html=True,
            )

            # ── Key Findings box (Tab 2) ──────────────────────────────────────
            st.markdown(
                "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:10px 0 10px 0;'>"
                "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
                "letter-spacing:1.2px;margin:0 0 4px 0;'>Key Findings</p>"
                "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:1.9;'>"
                "<li>Fiscal capacity alone does not determine recovery speed. Economic structure is a stronger and more consistent predictor.</li>"
                f"<li>OLS regression shows a slope of <b style='color:#F4D03F;'>{coeffs[0]:.4f}</b> with R² = <b style='color:#F4D03F;'>{r2:.3f}</b>, "
                "indicating only a modest relationship between fiscal capacity and recovery time.</li>"
                "<li>States with high fiscal capacity but service-heavy economies still show long recovery times, "
                "while industry-based states recover faster regardless of revenue levels.</li>"
                "</ul>"
                "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
                "Source: U.S. Census / BLS LAUS (2010–2024)</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Row 3: Gray "What the Data Shows" full width ─────────────────
            st.markdown(
                "<div style='background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;"
                "padding:12px 18px;margin:0 0 0 0;'>"
                "<p style='font-size:13px;font-weight:800;color:#333;text-transform:uppercase;"
                "letter-spacing:1px;margin:0 0 10px 0;'>What the Data Shows</p>"
                "<ul style='font-size:12px;color:#444;line-height:2;margin:0;padding-left:18px;'>"
                "<li><b>Each bubble</b> represents one U.S. state</li>"
                "<li><b>X-axis:</b> Fiscal capacity measured as state revenue per person (U.S. Census)</li>"
                "<li><b>Y-axis:</b> Predicted recovery time in months (BLS LAUS model output)</li>"
                "<li><b>Bubble size:</b> Average annual FEMA disaster declarations, with larger bubbles meaning more frequent disasters (matches paper Fig. 6)</li>"
                "<li><b>Colors:</b> Risk tier assigned by the DRPA model (Low, Moderate, High, Critical)</li>"
                "<li><b>Dashed lines:</b> Median values for fiscal capacity and recovery time, dividing the chart into four quadrants</li>"
                "<li><b>Quadrant labels:</b> Describe the combination of fiscal capacity and recovery speed for states in that region</li>"
                "</ul></div>",
                unsafe_allow_html=True,
            )

    # ── TAB 3: Service Sector vs Recovery (Paper Fig 7) ───────────────────────
    with tab3:

        # ── Title + subtitle ──────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:16px 0 12px 0;margin:0 0 12px 0;'>"
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 5px 0;'>"
            "States with Service Economies Recover More Slowly</p>"
            "<p style='font-size:13px;color:#555;margin:0;line-height:1.6;'>"
            "States with higher service sector dependence tend to experience longer recovery times after disasters."
            "</p></div>",
            unsafe_allow_html=True,
        )


        svc_df = df.dropna(subset=["service_share_pct","recovery_months_predicted"])
        x = svc_df["service_share_pct"].values
        y = svc_df["recovery_months_predicted"].values
        coeffs_s = np.polyfit(x, y, 1)
        r2_s = 1 - np.sum((y-np.polyval(coeffs_s,x))**2)/np.sum((y-y.mean())**2)

        fig_svc = px.scatter(
            svc_df, x="service_share_pct", y="recovery_months_predicted",
            color="economic_profile", size="avg_damage_per_capita", size_max=28,
            text="abbr", hover_name="state", trendline="ols",
            color_discrete_map=PROFILE_COLORS,
            labels={"service_share_pct":"Service Sector Share of GDP (%)",
                    "recovery_months_predicted":"Predicted Recovery (months)",
                    "economic_profile":"Economic Structure Type"},
            title="Relationship Between Service Share and Recovery Time",
        )
        fig_svc.update_traces(
            marker=dict(line=dict(width=1, color="white")),
            textposition="top center",
            textfont=dict(size=8, color="#222", family="Arial Black, Arial"),
            selector=dict(mode="markers+text"),
        )
        fig_svc.update_traces(
            customdata=np.stack((
                svc_df["state"],
                svc_df["service_share_pct"],
                svc_df["recovery_months_predicted"],
                svc_df["economic_profile"],
                svc_df["avg_damage_per_capita"],
            ), axis=-1),
            hovertemplate=(
                "<b style='font-size:13px;'>%{customdata[0]}</b><br>"
                "<span style='font-size:11px;color:#888;'>Economic structure: %{customdata[3]}</span><br>"
                "<span style='color:#e0e0e0;'>────────────────────────</span><br>"
                "<span style='font-size:11px;color:#444;'>Predicted recovery: </span>"
                "<b style='font-size:12px;'>%{customdata[2]:.1f} months</b><br>"
                "<span style='font-size:11px;color:#444;'>Service sector share: </span>"
                "<b>%{customdata[1]:.1f}%</b> of GDP<br>"
                "<span style='font-size:11px;color:#444;'>Avg damage per capita: </span>"
                "<b>$%{customdata[4]:,.0f}</b><br>"
                "<span style='color:#e0e0e0;'>────────────────────────</span><br>"
                "<i style='font-size:10px;color:#aaa;'>Source: BEA State GDP · BLS LAUS 2010–2024</i>"
                "<extra></extra>"
            ),
            selector=dict(mode="markers+text"),
        )

        # Annotations
        _annot_states = {"NJ": "High service share → slower recovery",
                         "WY": "Low service share → faster recovery"}
        for _abbr, _note in _annot_states.items():
            _row = svc_df[svc_df["abbr"] == _abbr]
            if not _row.empty:
                fig_svc.add_annotation(
                    x=float(_row["service_share_pct"].values[0]),
                    y=float(_row["recovery_months_predicted"].values[0]),
                    text=_note, showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor="#888", ax=40, ay=-30,
                    font=dict(size=10, color="#555"),
                    bgcolor="rgba(255,255,255,0.85)", borderpad=3,
                )

        fig_svc.update_layout(
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            legend_title_text="Economic Structure Type",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_svc, key="svc_scatter", width="stretch")

        # ── Orange insight box — below chart ──────────────────────────────────
        st.markdown(
            "<div style='background:#fff7f0;border-left:4px solid #E67E22;border-radius:4px;"
            "padding:10px 16px;margin:12px 0 10px 0;'>"
            "<p style='font-size:13px;font-weight:700;color:#a04000;margin:0 0 6px 0;'>"
            "States with larger service sectors recover more slowly.</p>"
            "<p style='font-size:12px;color:#2c3e50;margin:0;line-height:1.8;'>"
            "As service share increases, recovery time rises.<br>"
            "States where services dominate the economy average <b>16.2 months</b> to recover, "
            "compared to <b>11.5 months</b> for states with industry economies, a gap of about 4.7 months."
            "</p></div>",
            unsafe_allow_html=True,
        )

        # Technical note under chart
        st.markdown(
            f"<p style='font-size:11px;color:#aaa;margin:0 0 14px 0;text-align:right;'>"
            f"Trend: positive relationship (R² = {r2_s:.2f}) · "
            f"Slope: {coeffs_s[0]:+.2f} · Source: BEA State GDP + BLS LAUS 2010–2024</p>",
            unsafe_allow_html=True,
        )

        # ── What this means ───────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;"
            "padding:12px 16px;margin:0 0 12px 0;'>"
            "<p style='font-size:12px;font-weight:700;color:#333;text-transform:uppercase;"
            "letter-spacing:0.8px;margin:0 0 6px 0;'>What This Means</p>"
            "<p style='font-size:12px;color:#444;margin:0 0 8px 0;line-height:1.8;'>"
            "States where services make up a large share of the economy rely on demand-driven sectors "
            "like tourism, retail, and hospitality. These sectors are more sensitive to disruptions "
            "and take longer to return to normal activity after a disaster."
            "</p>"
            "<ul style='font-size:12px;color:#444;line-height:1.9;margin:0;padding-left:16px;'>"
            "<li>Service sectors depend on consumer activity, which drops sharply after disasters</li>"
            "<li>Physical infrastructure recovers faster than service networks and supply chains</li>"
            "<li>States with more industry can resume production with fewer demand-side barriers</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

        # ── Key Findings box (Tab 3) ──────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 4px 0;'>Key Findings</p>"
            "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:1.9;'>"
            "<li>Economic structure plays a meaningful role in recovery speed, more so than fiscal revenue.</li>"
            "<li>States with <b style='color:#F4D03F;'>service-heavy economies</b> consistently show longer recovery times compared to industry-based states.</li>"
            "<li>States like New Jersey and New York, with service shares above 70%, show some of the <b style='color:#F4D03F;'>longest predicted recovery times</b> in the dataset.</li>"
            "<li>Service sector concentration creates demand-side fragility during disasters. When consumer spending drops, service-dependent states have fewer buffers to sustain employment.</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 6: Recovery Predictors ────────────────────────────────────────────
    with tab6:

        # ── Page title ────────────────────────────────────────────────────────
        st.markdown(
            "<div style='margin:0 0 16px 0;'>"
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
            "Recovery Predictors: What the Model Found</p>"
            "<p style='font-size:13px;color:#555;margin:0;'>"
            "Recovery outcomes are shaped by a combination of hazard exposure, economic structure, and fiscal conditions, not just one factor alone."
            "</p></div>",
            unsafe_allow_html=True,
        )


        if not shap_df.empty:
            # ── Clean feature name mapping ────────────────────────────────────
            _feat_label_map = {
                "avg_disaster_exposure_12m":    "Disaster Exposure",
                "fiscal_capacity_per_capita":   "Fiscal Capacity per Capita",
                "service_share_pct":            "Service Sector Share",
                "compound_vulnerability_index": "Composite Vulnerability Index",
                "unemployment_shock_magnitude": "Unemployment Shock",
                "avg_damage_per_capita":        "Average Damage per Capita",
            }

            shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
            feat_names_raw = [c.replace("shap_","") for c in shap_cols]
            feat_names_clean = [_feat_label_map.get(f, f.replace("_"," ").title()) for f in feat_names_raw]
            mean_shap = shap_df[shap_cols].abs().mean()

            fig_shap = px.bar(
                x=mean_shap.values, y=feat_names_clean,
                orientation="h",
                color=mean_shap.values,
                color_continuous_scale="Purples",
                labels={"x":"Mean Absolute SHAP Value","y":""},
                title="Which Factors Matter Most Overall?",
                text=[f"{v:.3f}" for v in mean_shap.values],
            )
            fig_shap.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "──────────────────<br>"
                    "Overall influence: <b>%{x:.4f}</b><br>"
                    "<i style='font-size:11px;color:#888;'>Higher = more influence on predicted recovery time</i>"
                    "<extra></extra>"
                ),
            )
            fig_shap.update_layout(
                height=380, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="Mean absolute SHAP value across all states",
                margin=dict(l=10, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_shap, key="shap_global")

            # ── Purple insight box — below first chart ────────────────────────
            st.markdown(
                "<div style='background:#ede6f7;border-left:4px solid #8E44AD;border-radius:4px;"
                "padding:12px 16px;margin:12px 0 16px 0;'>"
                "<p style='font-size:13px;font-weight:700;color:#6c3483;margin:0 0 4px 0;'>"
                "Disaster exposure, composite vulnerability, and fiscal capacity are the strongest drivers of predicted recovery time.</p>"
                "<p style='font-size:12px;color:#2c3e50;margin:0;line-height:1.6;'>"
                "This section shows which variables carry the most weight in the model's predictions. "
                "The first chart ranks factors by overall influence across all states. "
                "The second shows which factor most often acts as the single top driver for individual states."
                "</p></div>",
                unsafe_allow_html=True,
            )

            # ── Between-charts explanation ────────────────────────────────────
            st.markdown(
                "<div style='background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;"
                "padding:10px 16px;margin:4px 0 14px 0;'>"
                "<p style='font-size:12px;font-weight:700;color:#333;text-transform:uppercase;"
                "letter-spacing:0.8px;margin:0 0 6px 0;'>What the Data Means</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.8;'>"
                "<b>How to read this page:</b> The chart above shows which variables matter most "
                "on average across all 50 states. The chart below shows which single factor is the "
                "top driver most often when looking at states one by one. Together they give both a "
                "general and a state-level view of what shapes multi-sector recovery, spanning "
                "labor markets, fiscal systems, economic structure, and disaster exposure."
                "</p></div>",
                unsafe_allow_html=True,
            )

            # Top driver distribution
            if "top_driver" in shap_df.columns:
                top_drv = shap_df["top_driver"].value_counts().reset_index()
                top_drv.columns = ["Feature","# States"]
                top_drv["Feature"] = top_drv["Feature"].map(
                    lambda f: _feat_label_map.get(f, f.replace("_"," ").title())
                )
                fig_drv = px.bar(
                    top_drv, x="Feature", y="# States",
                    color="# States", color_continuous_scale="Blues",
                    title="Which Factor Most Often Dominates by State?",
                    text="# States",
                )
                fig_drv.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "──────────────────<br>"
                        "Top driver in <b>%{y} states</b><br>"
                        "<i style='font-size:11px;color:#888;'>Number of states where this is the #1 driver</i>"
                        "<extra></extra>"
                    ),
                )
                fig_drv.update_layout(
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    xaxis_title="",
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig_drv, key="shap_driver")

                # ── Interpretation below second chart ─────────────────────────
                st.markdown(
                    "<div style='background:#fff8e1;border-left:3px solid #F4D03F;border-radius:4px;"
                    "padding:10px 16px;margin:4px 0 16px 0;'>"
                    "<p style='font-size:12px;font-weight:700;color:#7d6608;margin:0 0 4px 0;'>Interpretation</p>"
                    "<p style='font-size:12px;color:#5d4e00;margin:0;line-height:1.7;'>"
                    "Composite vulnerability and fiscal capacity are the most frequent top drivers across states, "
                    "while service sector share also plays a major role in many cases. "
                    "This pattern shows that recovery outcomes are not explained by a single variable. "
                    "They reflect a combination of structural economic conditions and hazard exposure."
                    "</p></div>",
                    unsafe_allow_html=True,
                )

        # ── Technical expander ────────────────────────────────────────────────
        with st.expander("Show technical details: Ridge Regression model"):
            st.markdown(
                "<div style='background:#ede6f7;border-left:4px solid #8E44AD;border-radius:4px;"
                "padding:14px 18px;margin:0 0 6px 0;'>"
                "<p style='font-size:11px;font-weight:700;color:#8E44AD;text-transform:uppercase;"
                "letter-spacing:0.8px;margin:0 0 10px 0;'>Ridge Regression: General Formula</p>"
                "<p style='font-family:Georgia,serif;font-size:15px;font-style:italic;color:#5b2d8e;"
                "margin:0 0 12px 0;line-height:1.9;'>"
                "Recovery Months = &beta;<sub>0</sub> + &beta;<sub>1</sub>(exposure) "
                "+ &beta;<sub>2</sub>(CVI) + &beta;<sub>3</sub>(damage) "
                "+ &beta;<sub>4</sub>(unemployment) + &beta;<sub>5</sub>(fiscal) + &lambda;&#8214;&beta;&#8214;&sup2;"
                "</p>"
                "<div style='font-size:12px;color:#444;line-height:2.1;'>"
                "<b>&beta;<sub>0</sub></b> &nbsp; Base recovery time before any variable is applied (intercept)<br>"
                "<b>&beta;<sub>1</sub> to &beta;<sub>5</sub></b> &nbsp; Learned coefficients: how much each input shifts the predicted months<br>"
                "<b>&lambda;&#8214;&beta;&#8214;&sup2;</b> &nbsp; Ridge penalty: keeps the model balanced when inputs are correlated with each other"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Coefficients Table ────────────────────────────────────────────
            st.markdown(
            "<div style='background:#f4f0f8;border-left:4px solid #8E44AD;border-radius:4px;"
            "padding:12px 18px;margin:6px 0 10px 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#8E44AD;text-transform:uppercase;"
            "letter-spacing:0.8px;margin:0 0 6px 0;'>Actual Coefficients From Your Data</p>"
            "<p style='font-size:12px;color:#444;margin:0 0 10px 0;line-height:1.6;'>"
            "Each variable is standardized before entering the model (z-scores), so the coefficients show relative influence rather than raw units. "
            "A positive coefficient means that as the variable increases, recovery takes longer. A negative one means the opposite.</p>"
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr style='background:#e8dff5;'>"
            "<th style='text-align:left;padding:6px 10px;color:#5b2d8e;border:1px solid #d0bbf0;'>Variable</th>"
            "<th style='text-align:left;padding:6px 10px;color:#5b2d8e;border:1px solid #d0bbf0;'>Source</th>"
            "<th style='text-align:center;padding:6px 10px;color:#5b2d8e;border:1px solid #d0bbf0;'>Coefficient</th>"
            "<th style='text-align:left;padding:6px 10px;color:#5b2d8e;border:1px solid #d0bbf0;'>What it means</th>"
            "</tr></thead><tbody>"
            "<tr style='background:#fff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Compound Vulnerability (CVI)</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>BLS, FEMA, Census, BEA</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#C0392B;'>+2.96</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>Strongest driver. Higher vulnerability leads to significantly longer recovery.</td>"
            "</tr>"
            "<tr style='background:#f9f5ff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Fiscal Capacity per Capita</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>U.S. Census</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#C0392B;'>+2.52</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>High-revenue states tend to have longer recovery, correlated with larger and more complex economies.</td>"
            "</tr>"
            "<tr style='background:#fff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Disaster Exposure</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>FEMA OpenFEMA</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#27AE60;'>-1.86</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>Negative. States with frequent disasters have built more resilient recovery systems over time.</td>"
            "</tr>"
            "<tr style='background:#f9f5ff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Service Sector Share</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>BEA State GDP</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#E67E22;'>+0.78</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>More service heavy economies take longer to stabilize after labor market disruptions.</td>"
            "</tr>"
            "<tr style='background:#fff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Unemployment Shock</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>BLS LAUS</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#E67E22;'>+0.57</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>Larger unemployment spikes after a disaster are associated with longer recovery timelines.</td>"
            "</tr>"
            "<tr style='background:#f9f5ff;'>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;'>Damage per Capita</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#888;font-size:11px;'>FEMA / U.S. Census</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;text-align:center;font-weight:700;color:#888;'>+0.13</td>"
            "<td style='padding:5px 10px;border:1px solid #e0d0f5;color:#555;'>Weakest individual effect. Per-capita damage has limited predictive power on its own.</td>"
            "</tr>"
            "</tbody></table>"
            "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
            "Intercept = 16.40 months &nbsp;·&nbsp; Lambda = 0.1 &nbsp;·&nbsp; "
            "LOO R² = 0.139 &nbsp;·&nbsp; LOO MAE = 3.40 months &nbsp;·&nbsp; n = 50 states"
            "</p></div>",
            unsafe_allow_html=True,
        )

        # ── Key Findings box (Tab 6) ──────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:20px 0 0 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 4px 0;'>Key Findings</p>"
            "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:1.9;'>"
            "<li><b style='color:#F4D03F;'>Compound Vulnerability (CVI)</b> is the strongest driver of longer recovery "
            "(coefficient +2.96), meaning states with overlapping hazard, fiscal, and structural weaknesses take significantly longer to stabilize.</li>"
            "<li><b style='color:#F4D03F;'>Fiscal Capacity</b> (coefficient +2.52) shows a counterintuitive positive effect. "
            "High-revenue states tend to have larger, more complex economies that take longer to fully recover.</li>"
            "<li><b style='color:#F4D03F;'>Disaster Exposure</b> has a negative coefficient (−1.86), meaning states with frequent disasters "
            "have built more resilient systems and tend to recover faster than less-exposed states.</li>"
            "<li>Service sector share and unemployment shock both add recovery time, but their individual effects are smaller. "
            "It is the combination of factors captured by the CVI that matters most.</li>"
            "</ul>"
            "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
            "Sources: BLS LAUS · FEMA OpenFEMA · U.S. Census · BEA State GDP · Ridge Regression (λ=0.1, n=50)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 4: Political Economy & Aid ───────────────────────────────────────
    with tab4:

        from drpi_gov_data import PARTISAN_DATA, GOV_FINDINGS

        st.markdown(
            "<div style='margin:0 0 14px 0;'>"
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
            "Political Economy, Aid Distribution, and Disaster Recovery</p>"
            "<p style='font-size:13px;color:#555;margin:0;'>"
            "Partisan distribution across U.S. states, disaster aid allocation by party affiliation, "
            "and how political and institutional factors shape recovery outcomes, "
            "derived from FEMA · Census · BLS · MIT Election datasets."
            "</p></div>",
            unsafe_allow_html=True,
        )

        if PARTISAN_DATA or GOV_FINDINGS:
            for item in GOV_FINDINGS:
                color = item.get("color", "#C0392B")
                st.markdown(
                    f"<div style='background:{color}12;border-left:4px solid {color};"
                    f"padding:10px 16px;border-radius:0 6px 6px 0;margin:6px 0'>"
                    f"<p style='font-weight:700;color:{color};font-size:13px;margin:0 0 3px 0'>"
                    f"{item.get('title','')}</p>"
                    f"<p style='font-size:13px;color:#444;margin:0;'>{item.get('text','')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("""
            <div style='background:#f5f5f5;border:2px dashed #C0392B;border-radius:8px;
                        padding:40px;text-align:center;margin:20px 0'>
                <p style='font-size:16px;font-weight:700;color:#C0392B;margin:0 0 8px 0;'>
                    Content Coming Soon
                </p>
                <p style='font-size:13px;color:#888;margin:0;'>
                    Open <code>ML_models/drpi_tab_data.py</code> and fill in
                    <code>PARTISAN_DATA</code> and <code>GOV_FINDINGS</code>
                    to populate this tab.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 5: Education Disruption ───────────────────────────────────────────
    with tab5:

        # ── TITLE (visual hierarchy: biggest element) ─────────────────────────
        st.markdown(
            "<p style='font-size:28px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
            "Education Disruption as a Recovery Indicator</p>",
            unsafe_allow_html=True,
        )

        # ── HEADLINE INSIGHT (second most prominent) ──────────────────────────
        st.markdown(
            "<div style='background:#fff8e1;border-left:4px solid #D4AC0D;border-radius:4px;"
            "padding:10px 16px;margin:0 0 14px 0;'>"
            "<p style='font-size:14px;font-weight:700;color:#7d5a00;margin:0;'>"
            "States with prolonged remote and hybrid learning experienced slower economic recovery patterns."
            "</p>"
            "<p style='font-size:12px;color:#666;margin:4px 0 0 0;'>"
            "Virtual learning share and attendance loss are used as input features in the DRPA predictive model."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Subtitle ──────────────────────────────────────────────────────────
        st.caption(
            "15 highest disaster-exposure states · COVID School Data Hub · NCES Table 203.80 · DRPA Model"
        )

        # ── Load education datasets ───────────────────────────────────────────
        import re as _re
        _EDU_ROOT = os.path.join(ROOT, "data", "education")

        _TOP15_FULL = {
            "TX":"Texas","FL":"Florida","LA":"Louisiana","OK":"Oklahoma",
            "TN":"Tennessee","NC":"North Carolina","SC":"South Carolina",
            "VA":"Virginia","GA":"Georgia","MO":"Missouri","MS":"Mississippi",
            "KY":"Kentucky","AR":"Arkansas","IA":"Iowa","NE":"Nebraska",
        }
        _TOP15_ABBREV = list(_TOP15_FULL.keys())
        _NAME_TO_ABBREV = {v:k for k,v in _TOP15_FULL.items()}

        _mod_file = os.path.join(_EDU_ROOT, "covid_school_learning_modality_district_yearly.csv")
        _ada_file = os.path.join(_EDU_ROOT, "nces_avg_daily_attendance_by_state_1969_2021.xlsx")

        if os.path.exists(_mod_file) and os.path.exists(_ada_file):

            # ── Load modality data ────────────────────────────────────────────
            _mod_raw = pd.read_csv(_mod_file)
            _mod_agg = (
                _mod_raw.groupby("StateAbbrev")[["share_inperson","share_hybrid","share_virtual"]]
                .mean()
                .reset_index()
            )

            # ── Load ADA attendance data ──────────────────────────────────────
            _xl = pd.read_excel(_ada_file, header=None)
            _ada_rows = []
            for _i in range(3, 66):
                _sraw   = str(_xl.iloc[_i, 0]).strip()
                _sclean = _re.sub(r'[\\]+\d+[\\]+', '', _sraw).strip()
                _abbrev = _NAME_TO_ABBREV.get(_sclean)
                if _abbrev:
                    try:
                        _v19 = float(_xl.iloc[_i, 27])
                        _v20 = float(_xl.iloc[_i, 29])
                        _pct = round(((_v20 - _v19) / _v19) * 100, 2)
                        _ada_rows.append({
                            "abbrev": _abbrev, "state": _sclean,
                            "ada_2019": _v19, "ada_2020": _v20, "pct_change": _pct,
                        })
                    except:
                        pass
            _ada_df = pd.DataFrame(_ada_rows)

            # ── Build shared datasets ─────────────────────────────────────────
            _mod_top15 = (
                _mod_agg[_mod_agg["StateAbbrev"].isin(_TOP15_ABBREV)]
                .copy()
                .rename(columns={"StateAbbrev": "abbrev"})
            )
            _rec_by_state = (
                df[df["state"].isin(list(_TOP15_FULL.values()))]
                .groupby("state")["recovery_months_predicted"]
                .mean()
                .reset_index()
                .rename(columns={"state": "full_name",
                                 "recovery_months_predicted": "avg_recovery"})
            )
            _rec_by_state["abbrev"] = _rec_by_state["full_name"].map(_NAME_TO_ABBREV)

            _scatter_df = (
                _mod_top15
                .merge(_ada_df[["abbrev", "pct_change"]], on="abbrev", how="inner")
                .merge(_rec_by_state[["abbrev", "avg_recovery", "full_name"]],
                       on="abbrev", how="left")
            )

            # ─────────────────────────────────────────────────────────────────
            # VISUAL 1 — Learning Mode Stacked Bar
            # ─────────────────────────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:16px;font-weight:700;color:#0d1b2a;margin:20px 0 2px 0;'>"
                "Visual 1: Learning Mode by State</p>"
                "<p style='font-size:12px;color:#888;margin:0 0 8px 0;'>"
                "🟢 In-Person = stability &nbsp;|&nbsp; 🟡 Hybrid = transition &nbsp;|&nbsp; "
                "🔴 Virtual = disruption &nbsp;&nbsp;"
                "<span style='font-size:11px;background:#eaf4ff;color:#1a5276;"
                "border-radius:3px;padding:1px 6px;font-weight:600;'>"
                "Used as input feature in DRPA predictive model</span></p>",
                unsafe_allow_html=True,
            )

            _mod_bar = _mod_top15.copy()
            _mod_bar = _mod_bar.sort_values("share_virtual", ascending=True)
            _HIGHLIGHT = {"NC", "KY", "TX", "MS"}

            _bar_colors_ip  = ["#1a7a3a" if a in _HIGHLIGHT else "#27AE60" for a in _mod_bar["abbrev"]]
            _bar_colors_hyb = ["#b8860b" if a in _HIGHLIGHT else "#F4D03F" for a in _mod_bar["abbrev"]]
            _bar_colors_vir = ["#922b21" if a in _HIGHLIGHT else "#E74C3C" for a in _mod_bar["abbrev"]]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="In-Person", y=_mod_bar["abbrev"],
                x=(_mod_bar["share_inperson"] * 100).round(1),
                orientation="h", marker_color=_bar_colors_ip,
                hovertemplate="<b>%{y}</b> | In-Person: %{x:.1f}%<extra></extra>",
            ))
            fig_bar.add_trace(go.Bar(
                name="Hybrid", y=_mod_bar["abbrev"],
                x=(_mod_bar["share_hybrid"] * 100).round(1),
                orientation="h", marker_color=_bar_colors_hyb,
                hovertemplate="<b>%{y}</b> | Hybrid: %{x:.1f}%<extra></extra>",
            ))
            fig_bar.add_trace(go.Bar(
                name="Virtual", y=_mod_bar["abbrev"],
                x=(_mod_bar["share_virtual"] * 100).round(1),
                orientation="h", marker_color=_bar_colors_vir,
                hovertemplate="<b>%{y}</b> | Virtual: %{x:.1f}%<extra></extra>",
            ))
            # Annotation on highest virtual share state
            _ky_idx = list(_mod_bar["abbrev"]).index("KY") if "KY" in list(_mod_bar["abbrev"]) else None
            if _ky_idx is not None:
                fig_bar.add_annotation(
                    y="KY", x=100,
                    text="  Highest virtual share",
                    showarrow=True, arrowhead=2, arrowcolor="#C0392B",
                    ax=50, ay=0,
                    font=dict(size=10, color="#C0392B"), xanchor="left",
                )
            fig_bar.update_layout(
                barmode="stack",
                title=dict(text="Learning Mode Distribution: Top 15 Disaster-Exposure States",
                           font=dict(size=13, color="#0d1b2a")),
                xaxis=dict(title="Share of School Year (%)", ticksuffix="%",
                           showgrid=False, range=[0, 100]),
                yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                legend=dict(title="Learning Mode Impact on Stability",
                            orientation="h", y=-0.18,
                            font=dict(size=11)),
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=80, l=50, r=30),
                font=dict(size=12),
            )
            st.plotly_chart(fig_bar, key="edu_bar", width="stretch")

            # ─────────────────────────────────────────────────────────────────
            # VISUAL 2 — Attendance Drop Bar
            # ─────────────────────────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:16px;font-weight:700;color:#0d1b2a;margin:24px 0 2px 0;'>"
                "Visual 2: Avg. Daily Attendance Change (2019–20 to 2020–21)</p>"
                "<p style='font-size:12px;color:#888;margin:0 0 8px 0;'>"
                "Larger drops = greater education disruption during the COVID period</p>",
                unsafe_allow_html=True,
            )

            _ada_top15 = _ada_df[_ada_df["abbrev"].isin(_TOP15_ABBREV)].copy()
            _ada_top15 = _ada_top15.sort_values("pct_change", ascending=True)  # descending impact

            # Color intensity: stronger red = bigger drop
            def _ada_color(v):
                if v < -5:  return "#922B21"
                elif v < -3: return "#E74C3C"
                elif v < -1: return "#E67E22"
                elif v < 0:  return "#F4D03F"
                else:        return "#27AE60"

            _ada_colors = [_ada_color(v) for v in _ada_top15["pct_change"]]
            _worst_state = _ada_top15.iloc[0]

            fig_ada = go.Figure()
            fig_ada.add_trace(go.Bar(
                y=_ada_top15["abbrev"],
                x=_ada_top15["pct_change"].round(2),
                orientation="h",
                marker_color=_ada_colors,
                text=[f"{v:+.1f}%" for v in _ada_top15["pct_change"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b>: %{x:+.2f}%<extra></extra>",
            ))
            # Zero reference line
            fig_ada.add_vline(x=0, line_color="#333", line_width=1.5)
            # Annotation on worst state
            fig_ada.add_annotation(
                y=_worst_state["abbrev"],
                x=float(_worst_state["pct_change"]) - 0.3,
                text=f"  Largest drop: {_worst_state['state']} ({_worst_state['pct_change']:+.1f}%)",
                showarrow=True, arrowhead=2, arrowcolor="#922B21",
                ax=-60, ay=0,
                font=dict(size=10, color="#922B21"), xanchor="right",
            )
            fig_ada.update_layout(
                title=dict(text="Attendance Change by State | 0% = No Change",
                           font=dict(size=13, color="#0d1b2a")),
                xaxis=dict(title="% Change in Avg. Daily Attendance",
                           ticksuffix="%", showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=50, l=50, r=80),
                font=dict(size=12),
                showlegend=False,
            )
            st.plotly_chart(fig_ada, key="edu_ada", width="stretch")

            # ─────────────────────────────────────────────────────────────────
            # VISUAL 3 — Main Relationship Scatter (model connection)
            # ─────────────────────────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:16px;font-weight:700;color:#0d1b2a;margin:24px 0 2px 0;'>"
                "Visual 3: Education Disruption vs. Recovery Duration</p>"
                "<p style='font-size:12px;color:#888;margin:0 0 8px 0;'>"
                "This is the key relationship: states with more disruption take longer to recover economically."
                "</p>",
                unsafe_allow_html=True,
            )

            if len(_scatter_df) >= 4:
                _sc_x = (_scatter_df["share_virtual"] * 100).round(1)
                _sc_y = _scatter_df["pct_change"].round(2)
                _sc_r = _scatter_df["avg_recovery"].fillna(_scatter_df["avg_recovery"].mean())
                _sc_names = _scatter_df["full_name"].fillna(_scatter_df["abbrev"])
                _sc_abbrev = _scatter_df["abbrev"]

                # Highlight key states with larger markers
                _SCATTER_HL = {"NC", "KY", "TX", "MS"}
                _marker_sizes = [22 if a in _SCATTER_HL else 14 for a in _sc_abbrev]
                _text_weights = [a if a in _SCATTER_HL else a for a in _sc_abbrev]

                _scatter_fig = go.Figure()
                _scatter_fig.add_trace(go.Scatter(
                    x=_sc_x, y=_sc_y,
                    mode="markers+text",
                    text=_sc_abbrev,
                    textposition="top center",
                    textfont=dict(size=11, color="#0d1b2a", family="Arial Black"),
                    marker=dict(
                        size=_marker_sizes,
                        color=_sc_r,
                        colorscale=[
                            [0.0, "#F9E79F"],
                            [0.4, "#E67E22"],
                            [0.7, "#E74C3C"],
                            [1.0, "#7B241C"],
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Predicted Recovery<br>(months)",
                                       font=dict(size=11, color="#555")),
                            thickness=14, len=0.65,
                            tickfont=dict(size=10), outlinewidth=0,
                        ),
                        line=dict(width=2, color="#fff"),
                        cmin=_sc_r.min(), cmax=_sc_r.max(),
                    ),
                    customdata=list(zip(_sc_names, _sc_x, _sc_y, _sc_r.round(1))),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Virtual share: <b>%{customdata[1]:.1f}%</b><br>"
                        "Attendance drop: <b>%{customdata[2]:.1f}%</b><br>"
                        "Avg predicted recovery: <b>%{customdata[3]:.1f} months</b>"
                        "<extra></extra>"
                    ),
                ))

                _mid_x = float(_sc_x.median())
                _mid_y = float(_sc_y.median())
                _scatter_fig.add_vline(x=_mid_x, line_dash="dot", line_color="#ccc", line_width=1.2)
                _scatter_fig.add_hline(y=_mid_y, line_dash="dot", line_color="#ccc", line_width=1.2)

                _scatter_fig.add_annotation(
                    x=float(_sc_x.max()), y=float(_sc_y.min()),
                    text="High virtual + large attendance loss<br><b>Most compounded disruption</b>",
                    showarrow=False, font=dict(size=10, color="#C0392B"),
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(255,245,245,0.9)", borderpad=6,
                )

                _scatter_fig.update_layout(
                    title=dict(
                        text="Education Disruption vs. Recovery Duration | High-Exposure States",
                        font=dict(size=14, color="#0d1b2a", family="Arial Black"),
                    ),
                    xaxis=dict(
                        title="% of School Year in Virtual Learning  ·  source: COVID School Data Hub",
                        ticksuffix="%", showgrid=True, gridcolor="#f5f5f5", zeroline=False,
                    ),
                    yaxis=dict(
                        title="Attendance Change 2019→2021 (%)  ·  source: NCES Table 203.80",
                        ticksuffix="%", showgrid=True, gridcolor="#f5f5f5", zeroline=False,
                    ),
                    height=500,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=60, b=60, l=75, r=110),
                    font=dict(size=12),
                    showlegend=False,
                )
                st.plotly_chart(_scatter_fig, key="edu_scatter", width="stretch")

                # ── What This Means (compact, modern) ─────────────────────────
                st.markdown(
                    "<div style='background:#f9f9f9;border:1px solid #e0e0e0;"
                    "border-radius:6px;padding:12px 18px;margin:12px 0 0 0;'>"
                    "<p style='font-size:13px;font-weight:700;color:#333;margin:0 0 4px 0;'>"
                    "What This Means</p>"
                    "<p style='font-size:12px;color:#555;margin:0;line-height:1.7;'>"
                    "Each point is one of the <b>15 highest disaster-exposure states</b>. "
                    "States in the bottom-right corner, with high virtual learning share <i>and</i> large "
                    "attendance drops, show the most compounded disruption and tend to carry the "
                    "darkest color (longer predicted recovery). "
                    "NBER evidence links school closures to a 3.8 pp drop in full-time parental employment."
                    "</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        else:
            st.info("Education dataset files not found in data/education/ folder.")

        # ── Key Findings box (Tab 5) ──────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:20px 0 0 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 4px 0;'>Key Findings</p>"
            "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:1.9;'>"
            "<li><b style='color:#F4D03F;'>Mississippi, Louisiana, and North Carolina</b> sit in the worst-hit corner, "
            "with high virtual learning share and large attendance drops, carrying some of the longest predicted recovery times.</li>"
            "<li><b style='color:#F4D03F;'>Kentucky</b> is a notable outlier: highest virtual share (~36%) but near-zero attendance drop, "
            "suggesting remote infrastructure or state policy buffered the attendance impact.</li>"
            "<li>States that lost the most attendance also face the longest economic recovery. "
            "Education disruption and economic burden overlap in the same high-exposure states.</li>"
            "<li>Education disruption is not a side effect of disasters; it is a mechanism that weakens "
            "a state's long-run recovery capacity by reducing workforce quality and delaying stable employment.</li>"
            "</ul>"
            "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
            "Sources: COVID School Data Hub (learning modality, ref. 30) · "
            "NCES Digest of Education Statistics Table 203.80 (attendance, ref. 31) · "
            "Burbio K-12 School Opening Tracker (ref. 27) · "
            "Collins et al. (2021) via NBER, employment impact of school closures (ref. 32) · "
            "Lai et al. (2019), school recovery trajectories (ref. 7) · "
            "Boustan et al. (2017), long-run human capital losses (ref. 1)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 7: Policy Insights ────────────────────────────────────────────────
    with tab7:
        from drpi_tab_data import POLICY_FINDINGS, POLICY_PROGRAMS


        st.markdown(
            "<div style='margin:0 0 14px 0;'>"
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
            "Policy Insights</p>"
            "<p style='font-size:13px;color:#555;margin:0;'>"
            "Historical disaster policy, FEMA reform, and what data-driven oversight means for resource planning "
            "before and after disasters occur, derived from FEMA · BLS · NOAA · BEA · U.S. Census datasets."
            "</p></div>",
            unsafe_allow_html=True,
        )

        if POLICY_FINDINGS:
            for item in POLICY_FINDINGS:
                color = item.get("color", "#16A085")
                st.markdown(
                    f"<div style='background:{color}12;border-left:4px solid {color};"
                    f"padding:10px 16px;border-radius:0 6px 6px 0;margin:6px 0'>"
                    f"<p style='font-weight:700;color:{color};font-size:13px;margin:0 0 3px 0'>"
                    f"{item.get('title','')}</p>"
                    f"<p style='font-size:13px;color:#444;margin:0;'>{item.get('text','')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("""
            <div style='background:#f5f5f5;border:2px dashed #16A085;border-radius:8px;
                        padding:40px;text-align:center;margin:20px 0'>
                <p style='font-size:16px;font-weight:700;color:#16A085;margin:0 0 8px 0;'>
                    Content Coming Soon
                </p>
                <p style='font-size:13px;color:#888;margin:0;'>
                    Open <code>ML_models/drpi_tab_data.py</code> and fill in
                    <code>POLICY_FINDINGS</code> and <code>POLICY_PROGRAMS</code>
                    to populate this tab.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — STATE DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "State Deep-Dive":
    # ── Purpose statement ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin-bottom:14px;'>
        <p style='color:#F4D03F;font-size:14px;font-weight:700;margin:0 0 6px 0;letter-spacing:0.3px;'>
            This tool predicts how quickly a state can recover after a disaster and how unstable
            its labor market may become, helping identify high-risk regions and inform policy needs.
        </p>
        <p style='color:#aaa;font-size:12px;margin:0;line-height:1.6;'>
            We combine three models: a <b style='color:#fff;'>Ridge Regression</b> for recovery time prediction,
            a <b style='color:#fff;'>Random Forest</b> for labor instability classification,
            and <b style='color:#fff;'>SHAP values</b> to explain what drives each state's result.
            All models are validated using Leave-One-Out Cross Validation (LOO-CV).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 1: State selector ────────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>Step 1: Select a State</p>", unsafe_allow_html=True)
    selected = st.selectbox("Select a state", sorted(df["state"].dropna().tolist()), label_visibility="collapsed")
    row = df[df["state"]==selected].iloc[0]
    tier_color = TIER_COLORS.get(row["drpi_risk_tier"],"#888")

    # ── STEP 2: Main result — BIG ─────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px 0;'>Step 2: Risk Result</p>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1.2, 1.2, 1.6])
    with r1:
        st.markdown(f"""
        <div style='background:{tier_color}18;border:2px solid {tier_color};
                    border-radius:10px;padding:16px 20px;text-align:center;'>
            <p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;
                      letter-spacing:1px;margin:0 0 4px 0;'>Risk Level</p>
            <p style='font-size:32px;font-weight:900;color:{tier_color};margin:0;
                      letter-spacing:1px;'>{row['drpi_risk_tier']}</p>
            <p style='font-size:12px;color:#888;margin:4px 0 0 0;'>DRPA Score: {row['drpi_score']:.0f} / 100</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        low = row.get('recovery_months_low', 0)
        high = row.get('recovery_months_high', 0)
        st.markdown(f"""
        <div style='background:#f8f9fa;border:2px solid #0d1b2a;
                    border-radius:10px;padding:16px 20px;text-align:center;'>
            <p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;
                      letter-spacing:1px;margin:0 0 4px 0;'>Predicted Recovery Time</p>
            <p style='font-size:32px;font-weight:900;color:#0d1b2a;margin:0;'>
                {row['recovery_months_predicted']:.0f}
                <span style='font-size:16px;font-weight:400;color:#888;'>months</span>
            </p>
            <p style='font-size:12px;color:#888;margin:4px 0 0 0;'>90% range: {low:.0f}–{high:.0f} mo</p>
        </div>""", unsafe_allow_html=True)
    with r3:
        instability_color = {"CRITICAL":"#C0392B","HIGH":"#E67E22","MODERATE":"#F4D03F","LOW":"#27AE60"}.get(row["instability_tier"],"#888")
        st.markdown(f"""
        <div style='background:{instability_color}12;border:2px solid {instability_color};
                    border-radius:10px;padding:16px 20px;text-align:center;'>
            <p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;
                      letter-spacing:1px;margin:0 0 4px 0;'>Labor Market Instability</p>
            <p style='font-size:32px;font-weight:900;color:{instability_color};margin:0;'>
                {row['instability_tier']}
            </p>
            <p style='font-size:12px;color:#888;margin:4px 0 0 0;'>
                +{row['unemployment_shock_magnitude']:.1f} pp shock · {row['economic_profile']}
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:12px;color:#666;margin:10px 0 0 0;line-height:1.6;'>"
        "This result combines a <b>regression prediction</b> (recovery time) and a "
        "<b>classification model</b> (labor instability tier), explained using SHAP values below."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── STEP 3: Explanation — the wow factor ──────────────────────────────────
    st.markdown("<p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px 0;'>Step 3: Why This State?</p>", unsafe_allow_html=True)

    if "narrative" in row and pd.notna(row.get("narrative","")):
        # Shorten to first 2 sentences for demo clarity
        full = str(row["narrative"])
        sentences = full.split(". ")
        short = ". ".join(sentences[:3]) + ("." if len(sentences) > 3 else "")
        st.info(short)
        if "keywords" in row and pd.notna(row.get("keywords","")):
            tags = [t.strip() for t in str(row["keywords"]).split(",")]
            st.markdown(" ".join([f"`{t}`" for t in tags if t]))

    # ── What This Means — dynamic narrative box ───────────────────────────────
    _rec   = row["recovery_months_predicted"]
    _tier  = row["drpi_risk_tier"]
    _inst  = row["instability_tier"]
    _prof  = row.get("economic_profile", "mixed")
    _fisc  = row.get("fiscal_capacity_per_capita", 0)
    _svc   = row.get("service_share_pct", 0)
    _natl_avg_rec = df["recovery_months_predicted"].mean()

    _fisc_note = (
        "its strong fiscal capacity helps cushion the impact"
        if _fisc > df["fiscal_capacity_per_capita"].median()
        else "limited fiscal capacity reduces its ability to fund a fast response"
    )
    _svc_note = (
        f"a service-heavy economy ({_svc:.0f}% service share) slows recovery as consumer-dependent sectors take longer to rebound"
        if _svc > 60
        else f"its economic structure ({_prof}) provides relatively more recovery stability"
    )
    _vs_avg = (
        f"{abs(_rec - _natl_avg_rec):.0f} months {'longer' if _rec > _natl_avg_rec else 'shorter'} than the national average"
    )

    st.markdown(
        f"<div style='background:#f0f4ff;border-left:4px solid #2980B9;border-radius:6px;"
        f"padding:12px 16px;margin:12px 0 0 0;'>"
        f"<p style='font-size:12px;font-weight:700;color:#1a4a7a;text-transform:uppercase;"
        f"letter-spacing:0.8px;margin:0 0 6px 0;'>What This Means</p>"
        f"<p style='font-size:13px;color:#1a3a5c;margin:0;line-height:1.8;'>"
        f"<b>{selected}</b> is expected to take approximately <b>{_rec:.0f} months</b> to recover "
        f"after a major disaster — <b>{_vs_avg}</b>. "
        f"Its labor market instability is classified as <b>{_inst}</b>. "
        f"At the state level, {_fisc_note}, and {_svc_note}. "
        f"Overall risk tier: <b style='color:{tier_color};'>{_tier}</b>."
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:11px;color:#999;margin:6px 0 0 0;'>"
        "Model validated using Leave-One-Out Cross Validation (LOO-CV) across all 50 states."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── STEP 4: Supporting data ───────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px 0;'>Step 4: Supporting Data</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("FEMA Exposure", f"{row['avg_disaster_exposure_12m']:.0f} declarations/yr")
    with m2: st.metric("Avg Damage per Capita", f"${row.get('avg_damage_per_capita',0):,.0f}")
    with m3: st.metric("Fiscal Capacity", f"${row['fiscal_capacity_per_capita']:,.0f}" if pd.notna(row.get("fiscal_capacity_per_capita")) else "—")
    with m4: st.metric("CVI Score", f"{row.get('compound_vulnerability_index',0):.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    # ── CVI Radar Chart ───────────────────────────────────────────────────────
    with col1:
        st.markdown("**Compound Vulnerability Index: 5 Dimensions**")
        st.caption("Higher CVI = higher vulnerability and slower recovery. Combines exposure, fiscal strength, and structural risk.")
        cvi_dims = ["cvi_labor","cvi_disaster","cvi_damage","cvi_fiscal","cvi_structure"]
        cvi_labels = ["Labor Volatility","Disaster Exposure","Damage Burden","Fiscal Weakness","Structural Risk"]
        cvi_vals = [float(row.get(d, 0)) if pd.notna(row.get(d,0)) else 0 for d in cvi_dims]
        cvi_vals_closed = cvi_vals + [cvi_vals[0]]
        cvi_labels_closed = cvi_labels + [cvi_labels[0]]

        # Convert hex color to rgba for Plotly compatibility
        def hex_to_rgba(hex_color, alpha=0.2):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=cvi_vals_closed, theta=cvi_labels_closed,
            fill="toself",
            fillcolor=hex_to_rgba(tier_color, 0.2),
            line=dict(color=tier_color, width=2),
            name=row["state"],
        ))
        # National average
        avg_vals = [float(df[d].mean()) if d in df.columns else 0.5 for d in cvi_dims]
        avg_vals_closed = avg_vals + [avg_vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals_closed, theta=cvi_labels_closed,
            fill="toself",
            fillcolor="rgba(100,100,100,0.1)",
            line=dict(color="gray", width=1.5, dash="dot"),
            name="National Avg",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0,1], tickfont_size=9)),
            height=360, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_radar, key="cvi_radar")

    # ── SHAP Waterfall per state ──────────────────────────────────────────────
    with col2:
        st.markdown("**SHAP Explanation: What Drives This State's Recovery Prediction?**")
        if not shap_df.empty and "state" in shap_df.columns:
            s_row = shap_df[shap_df["state"]==selected]
            if not s_row.empty:
                s_row = s_row.iloc[0]
                shap_cols  = [c for c in shap_df.columns if c.startswith("shap_")]
                feat_names = [c.replace("shap_","") for c in shap_cols]
                shap_vals  = [float(s_row[c]) for c in shap_cols]
                base_val   = float(s_row.get("base_value", df["recovery_months_predicted"].mean()))

                # Sort by absolute impact
                order = np.argsort(np.abs(shap_vals))[::-1]
                sorted_feats  = [feat_names[i] for i in order]
                sorted_shap   = [shap_vals[i]  for i in order]
                colors = ["#C0392B" if v>0 else "#27AE60" for v in sorted_shap]

                fig_wf = go.Figure(go.Bar(
                    x=sorted_shap, y=sorted_feats,
                    orientation="h", marker_color=colors,
                    text=[f"{v:+.2f}" for v in sorted_shap],
                    textposition="outside",
                ))
                fig_wf.add_vline(x=0, line_width=1.5, line_color="black")
                fig_wf.update_layout(
                    title=f"SHAP values (base = {base_val:.1f} mo)",
                    xaxis_title="Impact on predicted recovery (months)",
                    height=360, paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(zeroline=True),
                    yaxis=dict(categoryorder="total ascending"),
                )
                st.plotly_chart(fig_wf, key="shap_waterfall")
                st.caption("🔴 Red = increases recovery time (worse)  |  🟢 Green = decreases (better)")
                st.markdown(
                    "<p style='font-size:12px;color:#555;margin:6px 0 0 0;line-height:1.6;'>"
                    "Higher service sector share and disaster exposure tend to increase recovery time, "
                    "while stronger fiscal capacity and lower vulnerability reduce it. "
                    "Each bar shows how much that factor shifts the prediction for this specific state."
                    "</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("SHAP data not available for this state.")
        else:
            st.info("Run drpi_02_predictive_models.py to generate SHAP values.")

    # ── Full detail table ──────────────────────────────────────────────────────
    with st.expander("📂 Full Feature & Score Detail"):
        detail = {
            "DRPA Score":                 f"{row['drpi_score']}/100",
            "Risk Tier":                  row["drpi_risk_tier"],
            "Predicted Recovery":         f"{row['recovery_months_predicted']:.0f} months",
            "Recovery 90% Interval":      f"{row.get('recovery_months_low',0):.0f}–{row.get('recovery_months_high',0):.0f} months",
            "Labor Instability Tier":     row["instability_tier"],
            "Economic Profile":           row.get("economic_profile","—"),
            "Profile Benchmark":          f"{row.get('profile_benchmark_months','—')} months (paper avg)",
            "─────────────────────────────":"──────────",
            "Avg FEMA Exposure":          f"{row['avg_disaster_exposure_12m']:.1f} decl/yr",
            "Avg Damage per Capita":      f"${row['avg_damage_per_capita']:,.0f}",
            "Avg Unemployment Rate":      f"{row['avg_unemployment_rate']:.2f}%",
            "Unemployment Volatility":    f"{row['unemployment_volatility_excl_covid']:.3f}",
            "COVID Shock Magnitude":      f"+{row['unemployment_shock_magnitude']:.1f} pp",
            "Fiscal Capacity":            f"${row.get('fiscal_capacity_per_capita',0):,.0f}/person",
            "Service Sector Share":       f"{row['service_share_pct']:.1f}%",
            "Industry Share":             f"{row['industry_share_pct']:.1f}%",
            "Natural Resources Share":    f"{row['natres_share_pct']:.1f}%",
            "──────────────────────────": "──────────",
            "CVI: Labor Volatility":     f"{row.get('cvi_labor',0):.3f}",
            "CVI: Disaster Exposure":    f"{row.get('cvi_disaster',0):.3f}",
            "CVI: Damage Burden":        f"{row.get('cvi_damage',0):.3f}",
            "CVI: Fiscal Weakness":      f"{row.get('cvi_fiscal',0):.3f}",
            "CVI: Structural Risk":      f"{row.get('cvi_structure',0):.3f}",
            "CVI: Composite":            f"{row.get('compound_vulnerability_index',0):.3f}",
            "P(LOW)":                     f"{row.get('prob_LOW',0):.0%}",
            "P(MODERATE)":                f"{row.get('prob_MODERATE',0):.0%}",
            "P(HIGH)":                    f"{row.get('prob_HIGH',0):.0%}",
            "P(CRITICAL)":                f"{row.get('prob_CRITICAL',0):.0%}",
        }
        st.table(pd.DataFrame({"Metric":detail.keys(),"Value":detail.values()}))

    # Download
    st.download_button("⬇️ Download Full Dataset (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="drpi_state_risk_scores.csv", mime="text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "DRPA v2.0: Disaster Recovery Predictive Analytics | "
    "UNC Charlotte · DTSC 4302 | "
    "Data: FEMA · BLS LAUS · NOAA Storm Events · BEA SAGDP2 · U.S. Census | "
    "Models: Ridge Regression (LOO-CV) + Random Forest · SHAP Explainability"
)
