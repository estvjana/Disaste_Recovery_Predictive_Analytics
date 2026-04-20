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
    "Industry-Intensive":  "#4ECDB4",  # teal   — fastest recovery (11.5 mo)
    "Resource-Intensive":  "#F5A623",  # orange — mid-fast         (14.4 mo)
    "Service-Dominated":   "#2E5BE3",  # blue   — mid-slow         (16.2 mo)
    "Government-Intensive":"#D4496A",  # pink   — slowest recovery (18.0 mo)
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
    <p style='font-size:10px;color:#888;margin:8px 0 0 0;line-height:1.6;'>
    <b>CVI includes 6 dimensions:</b><br>
    Labor · Disaster · Damage · Fiscal · Structure · Education
    </p>
    <p style='font-size:10px;color:#aaa;margin:6px 0 0 0;font-style:italic;'>
    Military Capacity tested — excluded<br>(non-significant, collinear w/ fiscal)
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
                f"CVI = (0.25 × labor) + (0.20 × exposure) + (0.20 × damage) + (0.15 × fiscal) + (0.10 × structure) + (0.10 × education)"
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
        "The **Compound Vulnerability Index (CVI)** combines six independent risk dimensions "
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
        ("cvi_structure","Structural Risk",    "10% of CVI",
         "Service-sector share of state GDP. Service-dominant economies tend to recover more "
         "slowly because service jobs are harder to resume remotely during infrastructure disruptions.",
         "Source: BEA State GDP"),
        ("cvi_education","Education Disruption","10% of CVI",
         "Average virtual learning share during COVID-19 (2020–2023). Higher virtual enrollment "
         "signals deeper school closures, which compound labor market disruption and slow recovery.",
         "Source: CSDH 2020–2023; NCES-ADA Table 203.80"),
    ]

    cols = st.columns(6)
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
    /* Tab 4 — Governance, Aid & Recovery — crimson accent */
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "  Economic Structure  ",
        "  Fiscal Capacity & Recovery  ",
        "  Service Economy & Recovery  ",
        "  Governance, Aid & Recovery  ",
        "  Education Disruption  ",
        "  Military Capacity  ",
        "  Recovery Predictors  ",
        "  Policy Insights  ",
    ])

    # ── TAB 1: Recovery by Economic Profile (Paper Fig 14) ───────────────────
    with tab1:

        # ── Title ─────────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:27px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
            "How Industry Composition Shapes Recovery Time Across U.S. States</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 4px 0;font-style:italic;'>"
            "States with stronger industrial bases recover faster; service- and government-heavy "
            "economies take longer to return to pre-shock conditions (Section 5.2)</p>",
            unsafe_allow_html=True,
        )

        # ── KPI strip — 2 insight cards ───────────────────────────────────────
        _t1k1, _t1k2 = st.columns(2)
        with _t1k1:
            st.markdown(
                "<div style='background:#eaf4fb;border-left:4px solid #27AE60;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#1e8449;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "How much does economic profile affect recovery speed?</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "11.5 → 18.0 months</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Industry-intensive states (KY, TN, IN) recover in <b>11.5 months</b> on average. "
                "Government-intensive states (VA, MD, HI) take <b>18.0 months</b>. "
                "That is a <b>6.5-month gap</b> driven entirely by economic composition, "
                "not by disaster severity or fiscal resources (Section 5.2).</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _t1k2:
            st.markdown(
                "<div style='background:#fff8e1;border-left:4px solid #E67E22;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#7d4700;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Even within the same profile, wide variation exists</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "4 vs 20 months</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Nebraska and Texas are <b>both Resource-Intensive</b>, yet Nebraska recovered in "
                "<b>4 months</b> while Texas took <b>20 months</b>. "
                "A 16-month gap within the same profile shows that economic structure sets "
                "the baseline, but disaster exposure and fiscal capacity determine the outcome.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        if "economic_profile" in df.columns:
            # Top 15 high-exposure states — matching paper Fig 14
            _T15 = ["Texas","Florida","Louisiana","Oklahoma","Tennessee","North Carolina",
                    "Georgia","Missouri","Mississippi","Virginia","Arkansas","Iowa",
                    "Kansas","Nebraska","Kentucky"]
            _t1_states = df[df["state"].isin(_T15)].dropna(
                subset=["recovery_months","economic_profile"]).copy()
            _t1_states = _t1_states.sort_values("recovery_months", ascending=True).reset_index(drop=True)
            _t1_colors = [PROFILE_COLORS.get(p, "#888") for p in _t1_states["economic_profile"]]

            # Group averages (paper Fig 14 right panel)
            _prof_avg = (
                df.dropna(subset=["economic_profile","recovery_months"])
                .groupby("economic_profile")
                .agg(avg_actual=("recovery_months","mean"),
                     avg_predicted=("recovery_months_predicted","mean"),
                     count=("state","count"),
                     states=("state", lambda x:", ".join(sorted(x))))
                .reset_index()
            )
            _BENCH = {"Industry-Intensive":11.5,"Resource-Intensive":14.4,
                      "Service-Dominated":16.2,"Government-Intensive":18.0}
            _prof_avg["benchmark"] = _prof_avg["economic_profile"].map(_BENCH)
            _prof_avg = _prof_avg.sort_values("benchmark").reset_index(drop=True)

            # ── Chart 1: Recovery per State (full width) ─────────────────────
            st.markdown(
                "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:12px 0 2px 0;'>"
                "Recovery Time per State "
                "<span style='font-size:11px;font-weight:400;color:#888;'>"
                "Top 15 High-Exposure States · Months to return to pre-shock unemployment</span></p>",
                unsafe_allow_html=True,
            )
            _avg_line = _t1_states["recovery_months"].mean()
            fig_states = go.Figure()
            fig_states.add_trace(go.Bar(
                x=_t1_states["recovery_months"],
                y=_t1_states["state"],
                orientation="h",
                marker_color=_t1_colors,
                marker_line=dict(color="white", width=0.8),
                text=[f"{int(v)} mo" for v in _t1_states["recovery_months"]],
                textposition="outside",
                textfont=dict(size=11, color="#333"),
                customdata=np.stack((
                    _t1_states["economic_profile"],
                    _t1_states["recovery_months"],
                    _t1_states["recovery_months_predicted"],
                    _t1_states["abbr"],
                    _t1_states.get("profile_benchmark_months",
                        _t1_states["economic_profile"].map(
                            {"Industry-Intensive":11.5,"Resource-Intensive":14.4,
                             "Service-Dominated":16.2,"Government-Intensive":18.0}
                        ).fillna(16.2)),
                    (_t1_states["recovery_months"] - _t1_states["recovery_months_predicted"]).round(1),
                ), axis=-1),
                hovertemplate=(
                    "<b>%{y}  (%{customdata[3]})</b><br>"
                    "Economic profile: %{customdata[0]}<br>"
                    "Group benchmark: %{customdata[4]:.1f} months<br>"
                    "Actual recovery: <b>%{customdata[1]:.0f} months</b><br>"
                    "Model predicted: %{customdata[2]:.1f} months<br>"
                    "vs model: <b>%{customdata[5]:+.1f} months</b><br>"
                    "<extra></extra>"
                ),
            ))
            fig_states.add_vline(
                x=_avg_line, line_dash="dot", line_color="#aaa", line_width=1.5,
                annotation_text=f"Avg: {_avg_line:.0f} mo",
                annotation_position="top right",
                annotation_font=dict(size=9, color="#888"),
            )
            for _an_st, _an_txt, _ax in [
                ("Texas",    "TX — 20 mo (longest)",  40),
                ("Nebraska", "NE — 4 mo (fastest)",   40),
            ]:
                _ar = _t1_states[_t1_states["state"] == _an_st]
                if not _ar.empty:
                    fig_states.add_annotation(
                        x=float(_ar["recovery_months"].values[0]),
                        y=_an_st,
                        text=_an_txt, showarrow=True,
                        arrowhead=2, ax=_ax, ay=0,
                        font=dict(size=9, color="#333"),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#ccc", borderwidth=1, borderpad=3,
                    )
            fig_states.update_layout(
                height=500, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Recovery Time (Months)", gridcolor="#f0f0f0",
                           zeroline=False, range=[0, 28], tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11, color="#222")),
                margin=dict(t=10, b=50, l=10, r=70),
                font=dict(size=11), showlegend=False,
            )
            st.plotly_chart(fig_states, key="t1_states", use_container_width=True)

            # ── Color legend ──────────────────────────────────────────────────
            st.markdown(
                "<div style='display:flex;align-items:center;gap:6px;"
                "background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;"
                "padding:10px 18px;margin:0 0 4px 0;flex-wrap:wrap;'>"
                "<span style='font-size:11px;font-weight:700;color:#444;"
                "text-transform:uppercase;letter-spacing:0.8px;margin-right:8px;'>Economic Profile:</span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;margin-right:16px;'>"
                "<span style='width:14px;height:14px;border-radius:3px;background:#4ECDB4;"
                "display:inline-block;'></span>"
                "<span style='font-size:12px;color:#222;'><b>Industry</b> — fastest (11.5 mo)</span></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;margin-right:16px;'>"
                "<span style='width:14px;height:14px;border-radius:3px;background:#F5A623;"
                "display:inline-block;'></span>"
                "<span style='font-size:12px;color:#222;'><b>Resource</b> — 14.4 mo</span></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;margin-right:16px;'>"
                "<span style='width:14px;height:14px;border-radius:3px;background:#2E5BE3;"
                "display:inline-block;'></span>"
                "<span style='font-size:12px;color:#222;'><b>Service</b> — 16.2 mo</span></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;'>"
                "<span style='width:14px;height:14px;border-radius:3px;background:#D4496A;"
                "display:inline-block;'></span>"
                "<span style='font-size:12px;color:#222;'><b>Government</b> — slowest (18.0 mo)</span></span>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Chart 2: Average by Profile (full width) ─────────────────────
            st.markdown(
                "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:18px 0 2px 0;'>"
                "Average Recovery Time by Economic Profile "
                "<span style='font-size:11px;font-weight:400;color:#888;'>"
                "Paper benchmark averages · All 50 states</span></p>",
                unsafe_allow_html=True,
            )
            fig_grp = go.Figure(go.Bar(
                x=_prof_avg["benchmark"],
                y=_prof_avg["economic_profile"].map(PROFILE_SHORT).fillna(_prof_avg["economic_profile"]),
                orientation="h",
                marker_color=[PROFILE_COLORS.get(p,"#888") for p in _prof_avg["economic_profile"]],
                text=[f"{v:.1f} mo" for v in _prof_avg["benchmark"]],
                textposition="outside",
                textfont=dict(size=12, color="#333"),
                customdata=np.stack((
                    _prof_avg["economic_profile"],
                    _prof_avg["benchmark"],
                    _prof_avg["count"],
                    _prof_avg["states"],
                    (_prof_avg["benchmark"] - 11.5).round(1),
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Benchmark: <b>%{customdata[1]:.1f} months</b><br>"
                    "States (%{customdata[2]:.0f}): %{customdata[3]}<br>"
                    "Extra vs fastest group: <b>+%{customdata[4]:.1f} months</b><br>"
                    "<extra></extra>"
                ),
            ))
            fig_grp.update_layout(
                height=260, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Avg Recovery (Months)", gridcolor="#f0f0f0",
                           zeroline=False, range=[0, 24], tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=12, color="#222")),
                margin=dict(t=10, b=50, l=10, r=60),
                font=dict(size=11), showlegend=False,
            )
            st.plotly_chart(fig_grp, key="t1_grp", use_container_width=True)

            # ── Source note ───────────────────────────────────────────────────
            st.caption(
                "Bar colors = economic profile (orange: Resource · blue: Service · "
                "green: Industry · purple: Government). "
                "Recovery = months to return to pre-COVID unemployment baseline (BLS LAUS). "
                "Sources: BEA SAGDP2 · BLS LAUS 2010–2024 · FEMA Declarations Database."
            )

            # ── Key narrative callout ─────────────────────────────────────────
            st.markdown(
                "<div style='background:#fff8e1;border-left:4px solid #E67E22;border-radius:4px;"
                "padding:12px 18px;margin:0 0 14px 0;'>"
                "<p style='font-size:13px;font-weight:700;color:#7d4700;margin:0 0 6px 0;'>"
                "Within-profile variation is as important as the group difference.</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.8;'>"
                "Nebraska and Texas are <b>both classified as Resource-Intensive</b>, yet Nebraska recovered "
                "in <b>4 months</b> while Texas took <b>20 months</b>, a 16-month gap within the same profile. "
                "This confirms that economic structure contributes to recovery outcomes, but does not fully "
                "determine them. Disaster exposure, fiscal capacity, and structural conditions each explain "
                "part of the variation that no single variable can capture alone (Sections 3.2–3.4)."
                "</p></div>",
                unsafe_allow_html=True,
            )

            # ── Key findings ──────────────────────────────────────────────────
            st.markdown(
                "<div style='background:#0d1b2a;border-radius:8px;padding:16px 22px;margin:0;'>"
                "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
                "letter-spacing:1.2px;margin:0 0 8px 0;'>Key Findings — Section 5.2</p>"
                "<ul style='font-size:13px;color:#ffffff;margin:0;padding-left:18px;line-height:2.0;'>"
                "<li><b style='color:#F4D03F;'>Industry-intensive states</b> (Kentucky, Tennessee) averaged "
                "<b style='color:#F4D03F;'>11.5 months</b> to recover, the fastest of any group. "
                "Durable goods manufacturing responds predictably to federal stimulus and supply chain recovery.</li>"
                "<li><b style='color:#F4D03F;'>Service-dominated states</b> (Florida, Missouri, Georgia) averaged "
                "<b style='color:#F4D03F;'>16.2 months</b>, reflecting the direct impact of pandemic restrictions "
                "on hospitality, retail, and personal services.</li>"
                "<li><b style='color:#F4D03F;'>Resource-intensive states</b> showed the widest variation: Nebraska "
                "recovered in 4 months while oil-dependent states (Louisiana, Oklahoma, Texas) took much longer "
                "due to compounded commodity price shocks.</li>"
                "<li><b style='color:#F4D03F;'>Government-intensive Virginia</b> required 18 months. Public-sector "
                "employment adjustment moves more gradually than private-sector rebound.</li>"
                "</ul>"
                "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
                "Sources: BEA SAGDP2 · BLS LAUS 2010–2024 · FEMA Disaster Declarations Database</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── State breakdown (collapsible) ─────────────────────────────────
            with st.expander("Full State Classification by Economic Profile"):
                _prof_all = (
                    df.dropna(subset=["economic_profile","recovery_months_predicted"])
                    .groupby("economic_profile")
                    .agg(avg_predicted=("recovery_months_predicted","mean"),
                         count=("state","count"),
                         states=("state", lambda x:", ".join(sorted(x))))
                    .reset_index()
                )
                _prof_all["benchmark"] = _prof_all["economic_profile"].map(_BENCH)
                _prof_all = _prof_all.sort_values("benchmark").reset_index(drop=True)
                for _, _row in _prof_all.iterrows():
                    _color = PROFILE_COLORS.get(_row["economic_profile"], "#888")
                    _short = PROFILE_SHORT.get(_row["economic_profile"], _row["economic_profile"])
                    st.markdown(
                        f"<div style='background:{_color}15;border-left:4px solid {_color};"
                        f"padding:7px 14px;border-radius:0 6px 6px 0;margin:4px 0'>"
                        f"<span style='font-weight:700;color:{_color};font-size:13px;'>{_short}</span>"
                        f"<span style='color:#777;font-size:12px;'>"
                        f" &nbsp;·&nbsp; {_row['count']} states &nbsp;·&nbsp; "
                        f"benchmark {_row['benchmark']} mo</span>"
                        f"<br><span style='font-size:12px;color:#444;'>{_row['states']}</span></div>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 2: Fiscal Capacity vs Recovery (Paper Fig 6 + Section 5.3) ──────────
    with tab2:
        if "fiscal_capacity_per_capita" in df.columns:
            # ── Cohort: 15 highest-exposure states ───────────────────────────
            _TOP15 = {"LA","FL","TX","MS","NC","GA","KY","TN","VA","OK","MO","AR","IA","KS","NE"}
            _REGION = {
                "LA":"Gulf",         "FL":"Gulf",         "TX":"Gulf",         "MS":"Gulf",
                "NC":"Southeast",    "VA":"Southeast",    "GA":"Southeast",
                "TN":"Southeast",    "KY":"Southeast",
                "AR":"South Central","OK":"South Central","MO":"South Central",
                "NE":"Midwest",      "KS":"Midwest",      "IA":"Midwest",
            }
            _REG_COLORS = {
                "Gulf":          "#E67E22",
                "Southeast":     "#8E44AD",
                "South Central": "#795548",
                "Midwest":       "#00838F",
            }

            _y_col = "recovery_months" if "recovery_months" in df.columns else "recovery_months_predicted"
            fc_df  = df[df["abbr"].isin(_TOP15)].dropna(
                subset=["fiscal_capacity_per_capita", _y_col]).copy()
            fc_df["region"] = fc_df["abbr"].map(_REGION).fillna("Other")

            # OLS stats
            from scipy import stats as _sfc
            _xfc  = fc_df["fiscal_capacity_per_capita"].values.astype(float)
            _yfc  = fc_df[_y_col].values.astype(float)
            _nfc  = len(fc_df)
            _cfc  = np.polyfit(_xfc, _yfc, 1)
            _yhat = np.polyval(_cfc, _xfc)
            _r2_fc    = 1 - np.sum((_yfc - _yhat)**2) / np.sum((_yfc - _yfc.mean())**2)
            _slope_fc = _cfc[0]

            # 90 % CI band
            _xfc_line  = np.linspace(_xfc.min() - 300, _xfc.max() + 300, 300)
            _yfc_line  = np.polyval(_cfc, _xfc_line)
            _resid_fc  = _yfc - _yhat
            _se_fc     = np.std(_resid_fc, ddof=2)
            _xfc_mean  = _xfc.mean()
            _sxx_fc    = np.sum((_xfc - _xfc_mean)**2)
            _t_fc      = _sfc.t.ppf(0.95, df=max(_nfc - 2, 1))
            _margin_fc = _t_fc * _se_fc * np.sqrt(1/_nfc + (_xfc_line - _xfc_mean)**2 / _sxx_fc)

            # ── Title ─────────────────────────────────────────────────────────
            st.markdown(
                "<div style='padding:16px 0 6px 0;'>"
                "<p style='font-size:27px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
                "Fiscal Capacity and Disaster Recovery Speed</p>"
                "<p style='font-size:14px;color:#555;margin:0 0 4px 0;font-style:italic;'>"
                "State revenue per capita explains only ~12% of variance in recovery time — "
                "economic structure and disaster exposure matter more (Sections 4.2 &amp; 5.3)</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── KPI strip — 2 cards ───────────────────────────────────────────
            _fk1, _fk2 = st.columns(2)
            with _fk1:
                st.markdown(
                    "<div style='background:#f0f4ff;border-left:4px solid #2980B9;"
                    "border-radius:6px;padding:14px 20px;'>"
                    "<p style='font-size:11px;font-weight:600;color:#555;"
                    "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                    "Fiscal Capacity Explains Only</p>"
                    f"<p style='font-size:38px;font-weight:900;color:#0d1b2a;margin:0;'>"
                    f"{_r2_fc*100:.0f}%</p>"
                    "<p style='font-size:12px;color:#444;margin:4px 0 0 0;'>"
                    "of the variance in recovery time across the 15 highest-exposure states "
                    "(OLS R² = " + f"{_r2_fc:.3f}" + "). "
                    "Economic structure and disaster exposure explain the rest.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with _fk2:
                st.markdown(
                    "<div style='background:#fff8e1;border-left:4px solid #E67E22;"
                    "border-radius:6px;padding:14px 20px;'>"
                    "<p style='font-size:11px;font-weight:600;color:#555;"
                    "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                    "Nebraska vs Texas — Same Aid Level, 16-Month Gap</p>"
                    "<p style='font-size:38px;font-weight:900;color:#0d1b2a;margin:0;'>"
                    "4 vs 20 months</p>"
                    "<p style='font-size:12px;color:#444;margin:4px 0 0 0;'>"
                    "Nebraska (lowest fiscal capacity) recovered in <b>4 months</b>. "
                    "Texas (high fiscal, high exposure) took <b>20 months</b>. "
                    "More state revenue did not mean faster recovery.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

            # ── Scatter: Fiscal Capacity vs Actual Recovery ────────────────────
            _fig_fc2 = go.Figure()

            # 90 % CI band
            _fig_fc2.add_trace(go.Scatter(
                x=np.concatenate([_xfc_line, _xfc_line[::-1]]),
                y=np.concatenate([_yfc_line + _margin_fc,
                                  (_yfc_line - _margin_fc)[::-1]]),
                fill="toself",
                fillcolor="rgba(44,130,201,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% Confidence Band",
            ))

            # OLS line
            _fig_fc2.add_trace(go.Scatter(
                x=_xfc_line, y=_yfc_line,
                mode="lines",
                line=dict(color="#2c82c9", width=2),
                name=f"OLS Trend  (R²={_r2_fc:.3f},  slope={_slope_fc:.5f})",
                hoverinfo="skip",
            ))

            # Points by region
            for _reg, _grp in fc_df.groupby("region"):
                _rc   = _REG_COLORS.get(_reg, "#888")
                _econ = (_grp["economic_profile"].fillna("N/A")
                         if "economic_profile" in _grp.columns
                         else pd.Series(["N/A"]*len(_grp), index=_grp.index))
                _exp  = (_grp["avg_disaster_exposure_12m"].fillna(0)
                         if "avg_disaster_exposure_12m" in _grp.columns
                         else pd.Series([0]*len(_grp), index=_grp.index))
                _cvi  = (_grp["compound_vulnerability_index"].fillna(
                             _grp["compound_vulnerability_index"].median())
                         if "compound_vulnerability_index" in _grp.columns
                         else pd.Series([0.5]*len(_grp), index=_grp.index))
                # OLS residual: positive = recovered SLOWER than fiscal level predicts
                _resid = (_grp[_y_col].values
                          - np.polyval(_cfc, _grp["fiscal_capacity_per_capita"].values))
                _fig_fc2.add_trace(go.Scatter(
                    x=_grp["fiscal_capacity_per_capita"],
                    y=_grp[_y_col],
                    mode="markers+text",
                    name=_reg,
                    text=_grp["abbr"],
                    textposition="top center",
                    textfont=dict(size=9.5, color="#222", family="Arial Black"),
                    marker=dict(
                        color=_rc,
                        size=12,
                        line=dict(width=1.5, color="white"),
                        opacity=0.90,
                    ),
                    customdata=np.stack((
                        _grp["state"].values,
                        _grp["fiscal_capacity_per_capita"].values,
                        _grp[_y_col].values,
                        _grp["region"].values,
                        _econ.values,
                        _exp.values,
                        _resid.round(1),
                        _cvi.values,
                    ), axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>  ·  %{customdata[3]} region<br>"
                        "Economic profile: %{customdata[4]}<br>"
                        "Fiscal capacity: <b>$%{customdata[1]:,.0f}</b> / person"
                        "  (state revenue per capita)<br>"
                        "Actual recovery: <b>%{customdata[2]:.0f} months</b><br>"
                        "Avg disaster declarations / yr: %{customdata[5]:.0f}<br>"
                        "Deviation from trend: <b>%{customdata[6]:+.1f} months</b>"
                        "  (+ = slower than fiscal level predicts)<br>"
                        "Compound vulnerability index: %{customdata[7]:.2f}"
                        "<extra></extra>"
                    ),
                ))

            # Key outlier annotations
            _annots = {
                "LA": ("⚠ High exposure + strong fiscal<br>Gulf history still slows recovery", 55, -30),
                "NE": ("Lowest fiscal → fastest recovery<br>4 months — Midwest efficiency", 55, 20),
                "TX": ("Top-3 aid recipient<br>Slowest recovery: 20 months", -60, -25),
            }
            for _aa, (_note, _axo, _ayo) in _annots.items():
                _row = fc_df[fc_df["abbr"] == _aa]
                if not _row.empty:
                    _fig_fc2.add_annotation(
                        x=float(_row["fiscal_capacity_per_capita"].iloc[0]),
                        y=float(_row[_y_col].iloc[0]),
                        text=_note,
                        showarrow=True, arrowhead=2,
                        arrowsize=0.8, arrowwidth=1.2,
                        arrowcolor="#555",
                        ax=_axo, ay=_ayo,
                        font=dict(size=9.5, color="#333"),
                        bgcolor="rgba(255,255,255,0.90)",
                        bordercolor="#bbb",
                        borderwidth=1, borderpad=4,
                    )

            _y_label = ("Actual Recovery Time (months to return to pre-shock baseline)"
                        if _y_col == "recovery_months"
                        else "Predicted Recovery Time (months)")
            _fig_fc2.update_layout(
                title=dict(
                    text=(f"Fiscal Capacity vs Recovery Time  |  "
                          f"15 High-Exposure States  |  R² = {_r2_fc:.3f}"),
                    font_size=13,
                ),
                xaxis=dict(
                    title="State Revenue per Capita ($)",
                    tickprefix="$",
                    showgrid=True, gridcolor="#f0f0f0", zeroline=False,
                ),
                yaxis=dict(
                    title=_y_label,
                    showgrid=True, gridcolor="#f0f0f0", zeroline=False,
                ),
                legend=dict(
                    title="Region",
                    orientation="v",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1,
                ),
                height=500,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=60, r=40, t=50, b=60),
                font=dict(size=11),
            )
            st.plotly_chart(_fig_fc2, key="fiscal_scatter2", use_container_width=True)

            # ── Method note ───────────────────────────────────────────────────
            st.markdown(
                "<div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"
                "padding:10px 18px;margin:4px 0 14px 0;'>"
                "<p style='font-size:11px;color:#555;margin:0;line-height:1.9;'>"
                "<b>Notes:</b> Scatter shows the 15 highest-disaster-exposure U.S. states. "
                "Y-axis is actual months to return to pre-COVID baseline unemployment "
                "(BLS LAUS 2020–2024). X-axis is fiscal capacity measured as total state revenue "
                "divided by population (U.S. Census State Finance, 2019). "
                "Shaded region = 90% OLS prediction interval. "
                "Colors identify regional groupings. OLS slope and R² computed via ordinary "
                "least squares on n = 15. "
                "<br><b>Sources:</b> U.S. Census Bureau State &amp; Local Finance · "
                "BLS LAUS 2010–2024 · FEMA Disaster Declarations Database.</p></div>",
                unsafe_allow_html=True,
            )

            # ── Key findings — Section 4.2 ────────────────────────────────────
            st.markdown(
                "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:10px 0 22px 0;'>"
                "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
                "letter-spacing:1.2px;margin:0 0 6px 0;'>Key Findings — Section 4.2  ·  Fiscal Capacity</p>"
                "<ul style='font-size:13px;color:#ffffff;margin:0;padding-left:18px;line-height:2.0;'>"
                "<li>Fiscal capacity explains only <b style='color:#F4D03F;'>~12% of variance</b> in recovery "
                f"time (R² = {_r2_fc:.3f}), confirming the paper's finding that state revenue is a "
                "weak predictor of recovery speed.</li>"
                "<li><b style='color:#F4D03F;'>Nebraska</b> — the lowest-fiscal state in the cohort — "
                "recovered in just <b style='color:#F4D03F;'>4 months</b>. Texas, with far greater "
                "revenue, took <b style='color:#F4D03F;'>20 months</b>. The inverse relationship is "
                "driven by economic structure and disaster exposure, not state wealth.</li>"
                "<li>Gulf states (Louisiana, Florida, Texas, Mississippi) cluster in the "
                "<b style='color:#F4D03F;'>high-exposure, slow-recovery</b> zone regardless of fiscal "
                "level — pointing to structural and geographic risk that money cannot quickly offset.</li>"
                "</ul>"
                "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
                "Sources: U.S. Census State Finance · BLS LAUS 2010–2024 · "
                "FEMA Disaster Declarations Database</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Section 5.3 divider + title ───────────────────────────────────
            st.markdown(
                "<hr style='border:none;border-top:1px solid #dee2e6;margin:4px 0 16px 0;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='padding:0 0 10px 0;'>"
                "<p style='font-size:22px;font-weight:800;color:#0d1b2a;margin:0 0 4px 0;'>"
                "States Receiving More Aid Often Recover More Slowly</p>"
                "<p style='font-size:13px;color:#555;margin:0;font-style:italic;'>"
                "Aid volume does not explain recovery speed — "
                "Section 5.3, Aid-to-Recovery Rank Gap Analysis</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Fig 15: Aid-Recovery Rank Gap (Carnegie Endowment data) ──────
            _AID_DATA = [
                ('LA', 1,  7, 16), ('FL', 2, 14, 19), ('TX', 3, 15, 20),
                ('MS', 4,  5, 12), ('NC', 5,  9, 17), ('GA', 6,  3, 12),
                ('KY', 7,  2,  6), ('TN', 8, 11, 17), ('VA', 9, 13, 18),
                ('OK',10, 10, 17), ('MO',11,  8, 17), ('AR',12,  6, 16),
                ('IA',13, 12, 17), ('KS',14,  4, 12), ('NE',15,  1,  4),
            ]  # (abbr, aid_rank, recovery_rank, recovery_months)
            _aid_sorted = sorted(_AID_DATA, key=lambda r: r[1] - r[2])
            _aid_abbrs  = [r[0] for r in _aid_sorted]
            _aid_gaps   = [r[1] - r[2] for r in _aid_sorted]
            _aid_rec_mo = [r[3] for r in _aid_sorted]
            _aid_aid_rk = [r[1] for r in _aid_sorted]
            _aid_rec_rk = [r[2] for r in _aid_sorted]

            def _gap_col(g):
                return '#c62828' if g < -2 else '#2e7d32' if g > 2 else '#90a4ae'
            _aid_colors = [_gap_col(g) for g in _aid_gaps]

            from scipy.stats import pearsonr as _pr
            _r_aid, _p_aid = _pr(
                [r[1] for r in _AID_DATA],
                [r[2] for r in _AID_DATA],
            )
            _p_str = f"p = {_p_aid:.3f}"

            _fig_aid = go.Figure()

            # Background shading
            _fig_aid.add_vrect(x0=-16.5, x1=0,
                               fillcolor="#c62828", opacity=0.04,
                               layer="below", line_width=0)
            _fig_aid.add_vrect(x0=0, x1=16.5,
                               fillcolor="#2e7d32", opacity=0.04,
                               layer="below", line_width=0)

            # Three bar groups for legend clarity
            for _lbl_a, _test_a, _lc_a in [
                ("Underperformer — more aid, slower recovery",
                 lambda g: g < -2, "#c62828"),
                ("Neutral",
                 lambda g: -2 <= g <= 2, "#90a4ae"),
                ("Overperformer — less aid, faster recovery",
                 lambda g: g > 2, "#2e7d32"),
            ]:
                _idx_a = [i for i, g in enumerate(_aid_gaps) if _test_a(g)]
                if not _idx_a:
                    continue
                _g_grp  = [_aid_gaps[i]   for i in _idx_a]
                _a_grp  = [_aid_abbrs[i]  for i in _idx_a]
                _rm_grp = [_aid_rec_mo[i] for i in _idx_a]
                _ar_grp = [_aid_aid_rk[i] for i in _idx_a]
                _rr_grp = [_aid_rec_rk[i] for i in _idx_a]
                _fig_aid.add_trace(go.Bar(
                    x=_g_grp,
                    y=_a_grp,
                    orientation="h",
                    name=_lbl_a,
                    marker=dict(color=_lc_a, line=dict(color="white", width=0.5)),
                    width=0.60,
                    customdata=list(zip(_a_grp, _ar_grp, _rr_grp, _rm_grp, _g_grp)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Aid rank: %{customdata[1]}  ·  Recovery rank: %{customdata[2]}<br>"
                        "Actual recovery: <b>%{customdata[3]} months</b><br>"
                        "Gap: <b>%{customdata[4]:+d}</b>"
                        "<extra></extra>"
                    ),
                ))

            # Zero line
            _fig_aid.add_vline(x=0, line_color="#37474f", line_width=1.8)

            # Gap value annotations (outside bars)
            for _abbr_a, _gap_a in zip(_aid_abbrs, _aid_gaps):
                if _gap_a == 0:
                    continue
                _lbl_v = f"+{_gap_a}" if _gap_a > 0 else str(_gap_a)
                _xpos  = _gap_a + (0.8 if _gap_a > 0 else -0.8)
                _fig_aid.add_annotation(
                    x=_xpos, y=_abbr_a,
                    text=f"<b>{_lbl_v}</b>",
                    showarrow=False,
                    font=dict(size=8.5, color=_gap_col(_gap_a)),
                    xanchor="left" if _gap_a > 0 else "right",
                    yanchor="middle",
                )

            # Stats box
            _fig_aid.add_annotation(
                xref="paper", yref="paper",
                x=0.76, y=0.08,
                text=(
                    f"Aid vs Recovery Rank:  r = {_r_aid:+.2f}  ({_p_str})<br>"
                    "Aid rank does not explain recovery speed<br>"
                    "n = 15 high-exposure states"
                ),
                showarrow=False,
                font=dict(size=9, color="#1a3a5c", family="monospace"),
                bgcolor="white",
                bordercolor="#1a3a5c",
                borderwidth=1,
                borderpad=6,
                align="left",
            )

            _fig_aid.update_layout(
                title=dict(
                    text="Aid-to-Recovery Rank Gap  (Aid Rank − Recovery Rank)",
                    font_size=13,
                ),
                xaxis=dict(
                    title="Aid Rank − Recovery Rank  "
                          "(positive = faster recovery than aid rank implies)",
                    range=[-16.5, 16.5],
                    showgrid=True, gridcolor="#eceff1",
                    zeroline=False,
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color="#1a1a2e"),
                    categoryorder="array",
                    categoryarray=_aid_abbrs,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.24,
                    xanchor="center", x=0.5,
                    font=dict(size=10),
                ),
                height=490,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=50, r=40, t=50, b=110),
                bargap=0.30,
                font=dict(size=11),
                barmode="overlay",
            )
            st.plotly_chart(_fig_aid, key="aid_gap_chart", use_container_width=True)

            # ── Method note — Fig 15 ──────────────────────────────────────────
            st.markdown(
                "<div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"
                "padding:10px 18px;margin:4px 0 14px 0;'>"
                "<p style='font-size:11px;color:#555;margin:0;line-height:1.9;'>"
                "<b>Notes:</b> Bars show the difference between a state's <em>aid rank</em> "
                "(1 = most FEMA disaster aid received) and its <em>recovery rank</em> "
                "(1 = fastest recovery) among the 15 highest-exposure states. "
                "Positive values = faster-than-expected recovery relative to aid received "
                "(overperformers). Negative values = slower recovery despite high aid "
                "(underperformers). "
                f"Pearson r between aid rank and recovery rank: r = {_r_aid:+.2f}, "
                f"{_p_str} — not statistically significant. "
                "States sorted from worst (top) to best (bottom) gap. "
                "<br><b>Sources:</b> FEMA Disaster Dollar Database (Carnegie Endowment for "
                "International Peace, 2003–2025); BLS LAUS; Authors' DRPA model.</p></div>",
                unsafe_allow_html=True,
            )

            # ── Key findings — Section 5.3 ────────────────────────────────────
            st.markdown(
                "<div style='background:#0d1b2a;border-radius:8px;padding:14px 20px;margin:0 0 10px 0;'>"
                "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
                "letter-spacing:1.2px;margin:0 0 6px 0;'>Key Findings — Section 5.3  ·  Aid Allocation</p>"
                "<ul style='font-size:13px;color:#ffffff;margin:0;padding-left:18px;line-height:2.0;'>"
                "<li>The Pearson correlation between aid rank and recovery rank is "
                f"<b style='color:#F4D03F;'>r = {_r_aid:+.2f}</b> ({_p_str}) — "
                "not statistically significant. Aid volume alone does not explain how quickly "
                "states recover.</li>"
                "<li><b style='color:#F4D03F;'>Louisiana, Florida, and Texas</b> — the top-3 aid "
                "recipients — are among the <b style='color:#F4D03F;'>slowest recoverers</b> "
                "(16–20 months), confirming that disbursement does not substitute for "
                "structural resilience.</li>"
                "<li><b style='color:#F4D03F;'>Nebraska</b> received the least aid yet recovered "
                "fastest (4 months), suggesting that <b style='color:#F4D03F;'>economic composition "
                "and institutional efficiency</b> matter more than raw transfer amounts.</li>"
                "<li>The result is consistent with the paper's core argument: recovery speed is "
                "shaped primarily by economic structure, disaster exposure, and institutional "
                "capacity — not by federal aid volume.</li>"
                "</ul>"
                "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
                "Sources: FEMA Disaster Dollar Database (Carnegie Endowment) · "
                "BLS LAUS · FEMA Disaster Declarations Database</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── TAB 3: Service Sector vs Recovery (Paper Fig 7) ───────────────────────
    with tab3:

        # ── Regression stats ──────────────────────────────────────────────────
        svc_df = df.dropna(subset=["service_share_pct","recovery_months_predicted",
                                   "avg_damage_per_capita","economic_profile"]).copy()
        _x3 = svc_df["service_share_pct"].values
        _y3 = svc_df["recovery_months_predicted"].values
        _n3 = len(svc_df)
        _coeffs3 = np.polyfit(_x3, _y3, 1)
        _slope3   = _coeffs3[0]
        _r2_3    = 1 - np.sum((_y3 - np.polyval(_coeffs3, _x3))**2) / np.sum((_y3 - _y3.mean())**2)

        # 90 % confidence band around OLS line
        from scipy import stats as _stats3
        _x3_line  = np.linspace(_x3.min() - 1, _x3.max() + 1, 200)
        _y3_line  = np.polyval(_coeffs3, _x3_line)
        _resid3   = _y3 - np.polyval(_coeffs3, _x3)
        _se3      = np.std(_resid3, ddof=2)
        _x3_mean  = _x3.mean()
        _sxx3     = np.sum((_x3 - _x3_mean)**2)
        _t3       = _stats3.t.ppf(0.95, df=_n3 - 2)   # 90 % CI
        _margin3  = _t3 * _se3 * np.sqrt(1/_n3 + (_x3_line - _x3_mean)**2 / _sxx3)

        # Label only top-5 slowest + top-5 fastest — all others = no text
        _slow5 = set(svc_df.nlargest(5,  "recovery_months_predicted")["abbr"])
        _fast5 = set(svc_df.nsmallest(5, "recovery_months_predicted")["abbr"])
        _label_set = _slow5 | _fast5
        svc_df["_label"] = svc_df["abbr"].apply(lambda a: a if a in _label_set else "")

        # ── Title + subtitle ──────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:16px 0 6px 0;'>"
            "<p style='font-size:27px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
            "Service Sector Dependence and Disaster Recovery Time</p>"
            "<p style='font-size:14px;color:#555;margin:0 0 4px 0;font-style:italic;'>"
            "Higher service share is associated with slower labor market recovery (Section 4.4)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── KPI strip — 2 insight cards ──────────────────────────────────────
        _sk1, _sk2 = st.columns(2)
        with _sk1:
            st.markdown(
                "<div style='background:#eaf4fb;border-left:4px solid #2980B9;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#1a5276;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "How much does service share explain recovery time?</p>"
                f"<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                f"R² = {_r2_3:.3f}</p>"
                f"<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                f"Service sector share explains <b>{_r2_3*100:.1f}% of the variance</b> in "
                f"predicted recovery time across all 50 states. "
                f"Each additional 1 percentage point of service share is associated with "
                f"<b>{_slope3:+.2f} extra months</b> of recovery (OLS slope, Section 4.4).</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _sk2:
            st.markdown(
                "<div style='background:#f0faf0;border-left:4px solid #27AE60;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#1e8449;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Industry vs Service states — how big is the gap?</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "4.7 months longer</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Service-dominated states (NY, NJ, RI, CA) average <b>16.2 months</b> to recover. "
                "Industry-intensive states (KY, TN, IN) average <b>11.5 months</b>. "
                "Government-intensive states take the longest: <b>18.0 months</b>.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        # Profile order and benchmarks
        _PROF_ORDER = ["Industry-Intensive","Resource-Intensive",
                       "Service-Dominated","Government-Intensive"]
        _BENCH3 = {"Industry-Intensive":11.5,"Resource-Intensive":14.4,
                   "Service-Dominated":16.2,"Government-Intensive":18.0}

        # ── Two-panel layout ──────────────────────────────────────────────────
        # ── Chart 1: Box plot (full width) ───────────────────────────────────
        st.markdown(
                "<p style='font-size:14px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
                "Distribution of Recovery Time by Economic Profile</p>"
                "<p style='font-size:11px;color:#888;margin:0 0 6px 0;'>"
                "Each dot = one state (hover for details). "
                "Box = middle 50% of states. Line = median. Diamond = benchmark average.</p>",
                unsafe_allow_html=True,
            )

        _fig3 = go.Figure()

        # Subtle alternating row shading
        for _pi, _prof in enumerate(_PROF_ORDER):
            _fig3.add_hrect(
                y0=_pi - 0.48, y1=_pi + 0.48,
                fillcolor="rgba(248,249,250,0.8)" if _pi % 2 == 0 else "rgba(255,255,255,0)",
                layer="below", line_width=0,
            )

        # Vertical reference line at overall cohort average
        _overall_avg3 = float(svc_df["recovery_months_predicted"].mean())
        _fig3.add_vline(
        x=_overall_avg3,
        line_dash="dot", line_color="#ccc", line_width=1.2,
        )
        _fig3.add_annotation(
        xref="x", yref="paper",
        x=_overall_avg3, y=1.01,
        text=f"Overall avg {_overall_avg3:.1f} mo",
        showarrow=False,
        font=dict(size=9, color="#aaa"),
        xanchor="center",
        )

        def _hex_rgba(hexcol, alpha=0.18):
            """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)' for Plotly shapes."""
            h = hexcol.lstrip("#")
            if len(h) == 3:
                h = h[0]*2 + h[1]*2 + h[2]*2
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        for _pi, _prof in enumerate(_PROF_ORDER):
            _grp3  = svc_df[svc_df["economic_profile"] == _prof].copy()
            _col3  = PROFILE_COLORS.get(_prof, "#888")
            _avg3  = _BENCH3[_prof]
            _cnt3  = len(_grp3)
            _vals3 = _grp3["recovery_months_predicted"].values
            _short = PROFILE_SHORT.get(_prof, _prof)

            # Box: manual IQR bar (Q1–Q3) drawn as a thin filled shape
            _q1 = float(np.percentile(_vals3, 25))
            _q3 = float(np.percentile(_vals3, 75))
            _med = float(np.median(_vals3))

            # IQR bar
            _fig3.add_shape(
                type="rect",
                x0=_q1, x1=_q3, y0=_pi - 0.25, y1=_pi + 0.25,
                fillcolor=_hex_rgba(_col3, 0.18),
                line=dict(color=_col3, width=1.5),
                layer="above",
            )
            # Median line inside box
            _fig3.add_shape(
                type="line",
                x0=_med, x1=_med, y0=_pi - 0.25, y1=_pi + 0.25,
                line=dict(color=_col3, width=2.5),
                layer="above",
            )
            # Whiskers
            _wlo = float(_vals3.min())
            _whi = float(_vals3.max())
            _fig3.add_shape(
                type="line",
                x0=_wlo, x1=_q1, y0=_pi, y1=_pi,
                line=dict(color=_col3, width=1.2, dash="dot"),
                layer="above",
            )
            _fig3.add_shape(
                type="line",
                x0=_q3, x1=_whi, y0=_pi, y1=_pi,
                line=dict(color=_col3, width=1.2, dash="dot"),
                layer="above",
            )

            # Individual state dots (no text labels — hover only)
            _fisc3 = (_grp3["fiscal_capacity_per_capita"].fillna(
                          _grp3["fiscal_capacity_per_capita"].median())
                      if "fiscal_capacity_per_capita" in _grp3.columns
                      else pd.Series([6500]*_cnt3, index=_grp3.index))
            _abbr3 = (_grp3["abbr"].values
                      if "abbr" in _grp3.columns
                      else _grp3["state"].str[:2].str.upper().values)
            # Rank within profile group (1 = fastest)
            _rank3 = _grp3["recovery_months_predicted"].rank(method="min").astype(int).values
            _fig3.add_trace(go.Scatter(
                x=_vals3,
                y=[_pi] * _cnt3,
                mode="markers",
                name=f"{_short}  ({_cnt3} states)",
                marker=dict(
                    color=_col3,
                    size=9,
                    opacity=0.70,
                    line=dict(width=1, color="white"),
                    symbol="circle",
                ),
                customdata=np.stack((
                    _grp3["state"].values,
                    _grp3["recovery_months_predicted"].values,
                    _grp3["service_share_pct"].values,
                    _grp3["economic_profile"].values,
                    _fisc3.values,
                    _abbr3,
                    _rank3,
                    np.full(_cnt3, _avg3),
                    np.full(_cnt3, _cnt3),
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}  (%{customdata[5]})</b><br>"
                    "Profile: %{customdata[3]}<br>"
                    "Rank within group: #%{customdata[6]} of %{customdata[8]:.0f} states"
                    "  (1 = fastest)<br>"
                    "<br>"
                    "Predicted recovery: <b>%{customdata[1]:.1f} months</b><br>"
                    "Group benchmark: %{customdata[7]:.1f} months"
                    "  (%{customdata[1]:.1f} vs %{customdata[7]:.1f})<br>"
                    "Service sector share: %{customdata[2]:.1f}% of GDP<br>"
                    "Fiscal capacity: $%{customdata[4]:,.0f} / person<br>"
                    "<extra></extra>"
                ),
                legendgroup=_prof,
            ))

            # Benchmark diamond with label
            _fig3.add_trace(go.Scatter(
                x=[_avg3], y=[_pi],
                mode="markers+text",
                name=f"Benchmark avg",
                text=[f"  {_avg3} mo"],
                textposition="middle right",
                textfont=dict(size=10, color=_col3, family="Arial Black"),
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=_col3,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    f"<b>{_prof}</b><br>"
                    f"Paper benchmark: <b>{_avg3} months</b><br>"
                    f"n = {_cnt3} states"
                    "<extra></extra>"
                ),
                legendgroup=_prof,
                showlegend=False,
            ))

        _fig3.update_layout(
        height=360,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(
            title="Predicted Recovery Time (months)",
            showgrid=True, gridcolor="#f0f0f0",
            zeroline=False, range=[8, 28],
            tickfont=dict(size=10),
            dtick=2,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(_PROF_ORDER))),
            ticktext=[
                "<b>" + PROFILE_SHORT.get(p, p) + "</b>"
                for p in _PROF_ORDER
            ],
            tickfont=dict(size=12),
            showgrid=False,
            range=[-0.6, len(_PROF_ORDER) - 0.4],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.26,
            xanchor="center", x=0.5,
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=0,
        ),
        margin=dict(t=28, b=70, l=10, r=70),
        font=dict(size=11),
        )
        st.plotly_chart(_fig3, key="svc_box", use_container_width=True)

        # ── Chart 2: Average by Profile bars (full width) ────────────────────
        st.markdown(
            "<p style='font-size:14px;font-weight:700;color:#0d1b2a;margin:18px 0 2px 0;'>"
            "Average Recovery Time by Profile</p>"
            "<p style='font-size:11px;color:#888;margin:0 0 6px 0;'>"
            "Benchmark months · dashed line = Industry baseline (11.5 mo)</p>",
            unsafe_allow_html=True,
        )
        _base3  = 11.5
        _fig3r  = go.Figure()

        for _prof in _PROF_ORDER:
            _col3  = PROFILE_COLORS.get(_prof, "#888")
            _avg3  = _BENCH3[_prof]
            _gap3  = round(_avg3 - _base3, 1)
            _short = PROFILE_SHORT.get(_prof, _prof)
            _gap_lbl = f"  +{_gap3} mo vs baseline" if _gap3 > 0 else "  baseline"

            # States in this profile group
            _s_grp   = svc_df[svc_df["economic_profile"] == _prof]
            _s_cnt   = len(_s_grp)
            _s_names = ", ".join(sorted(_s_grp["abbr"].dropna().tolist())) if _s_cnt else "N/A"
            _gap_txt = (f"+{_gap3} mo longer than Industry-Intensive states"
                        if _gap3 > 0 else "Baseline group (fastest recovery)")
            _profile_defs = {
                "Industry-Intensive":   "≥20% GDP from manufacturing and construction",
                "Resource-Intensive":   "≥4% GDP from agriculture and mining",
                "Government-Intensive": "≥15% GDP from government",
                "Service-Dominated":    "All other states — high services, finance, retail",
            }
            _def = _profile_defs.get(_prof, "")

            # Full-length bar showing total benchmark months
            _fig3r.add_trace(go.Bar(
                x=[_avg3],
                y=[_short],
                orientation="h",
                marker=dict(
                    color=_hex_rgba(_col3, 0.82),
                    line=dict(color=_col3, width=1.5),
                ),
                text=[f"<b>{_avg3} mo</b>{_gap_lbl}"],
                textposition="outside",
                textfont=dict(size=10, color=_col3),
                hovertemplate=(
                    f"<b>{_prof}</b><br>"
                    f"Definition: {_def}<br>"
                    f"States in group ({_s_cnt}): {_s_names}<br>"
                    f"<br>"
                    f"Benchmark average: <b>{_avg3} months</b><br>"
                    f"vs Industry baseline (11.5 mo): <b>+{_gap3} months</b><br>"
                    f"Meaning: {_gap_txt}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Industry baseline reference line
        _fig3r.add_vline(
        x=_base3,
        line_dash="dash", line_color="#27AE60", line_width=1.8,
        )
        _fig3r.add_annotation(
        xref="x", yref="paper",
        x=_base3, y=1.04,
        text=f"<b>Industry baseline<br>{_base3} mo</b>",
        showarrow=False,
        font=dict(size=9, color="#27AE60"),
        xanchor="center",
        )

        _fig3r.update_layout(
        height=360,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(
            title="Average recovery time (months)",
            showgrid=True, gridcolor="#f0f0f0",
            zeroline=False, range=[0, 23],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#222"),
            categoryorder="array",
            categoryarray=[PROFILE_SHORT.get(p, p) for p in _PROF_ORDER[::-1]],
            showgrid=False,
        ),
        margin=dict(t=34, b=50, l=10, r=10),
        font=dict(size=11),
        )
        st.plotly_chart(_fig3r, key="svc_gap_bar", use_container_width=True)

        # ── Method note (journal-style) ───────────────────────────────────────
        st.markdown(
            "<div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"
            "padding:10px 18px;margin:4px 0 16px 0;'>"
            "<p style='font-size:11px;color:#555;margin:0;line-height:1.9;'>"
            "<b>Notes:</b> "
            "Left panel shows all 50 U.S. states as individual dots, grouped by economic profile "
            "(BEA SAGDP2, 2019–2023 average) and positioned along the recovery time axis. "
            "Diamond markers show the paper's group benchmark averages. "
            "Slight vertical jitter applied for readability. "
            "Right panel shows extra recovery months relative to the fastest group "
            "(Industry-Intensive, 11.5-month baseline). "
            "Profiles classified as: Industry-Intensive (≥20% GDP from manufacturing/construction), "
            "Resource-Intensive (≥4% agriculture/mining), Government-Intensive (≥15% government), "
            "Service-Dominated (all others). "
            "<br><b>Sources:</b> BEA State GDP (SAGDP2) · BLS LAUS 2010–2024 · "
            "Ridge Regression (λ=0.1, LOO-CV, n=50 states)."
            "</p></div>",
            unsafe_allow_html=True,
        )

        # ── Interactive Choropleth map ────────────────────────────────────────
        st.markdown(
            "<p style='font-size:19px;font-weight:800;color:#0d1b2a;margin:8px 0 2px 0;'>"
            "Geographic Distribution — Explore Recovery Drivers by State</p>"
            "<p style='font-size:12px;color:#666;margin:0 0 8px 0;'>"
            "Switch between views to compare how recovery time, service share, and economic "
            "profile cluster geographically. Hover over any state for full statistics.</p>",
            unsafe_allow_html=True,
        )

        _map_view = st.radio(
            "Map view:",
            ["Recovery Time", "Service Sector Share", "Economic Profile"],
            horizontal=True,
            key="t3_map_view",
        )

        # US state centroid coordinates for abbreviation labels
        _STATE_CENTS = {
            "AL":(32.81,-86.79),"AK":(64.20,-153.40),"AZ":(34.17,-111.09),
            "AR":(34.75,-92.13),"CA":(37.25,-119.75),"CO":(38.99,-105.55),
            "CT":(41.60,-72.69),"DE":(38.99,-75.51),"FL":(28.63,-82.45),
            "GA":(32.68,-83.22),"HI":(20.89,-157.00),"ID":(44.07,-114.74),
            "IL":(40.00,-89.00),"IN":(39.77,-86.15),"IA":(42.08,-93.50),
            "KS":(38.53,-98.38),"KY":(37.64,-84.67),"LA":(30.98,-91.96),
            "ME":(45.25,-69.00),"MD":(39.05,-76.64),"MA":(42.26,-71.81),
            "MI":(44.18,-84.46),"MN":(46.39,-94.64),"MS":(32.74,-89.67),
            "MO":(38.46,-92.29),"MT":(47.03,-110.45),"NE":(41.49,-99.90),
            "NV":(39.35,-116.64),"NH":(43.45,-71.56),"NJ":(40.11,-74.40),
            "NM":(34.84,-106.25),"NY":(42.95,-75.53),"NC":(35.63,-79.81),
            "ND":(47.44,-100.47),"OH":(40.19,-82.67),"OK":(35.57,-96.93),
            "OR":(43.94,-120.56),"PA":(40.59,-77.21),"RI":(41.68,-71.51),
            "SC":(33.84,-80.90),"SD":(44.37,-100.35),"TN":(35.86,-86.66),
            "TX":(31.47,-99.33),"UT":(39.32,-111.09),"VT":(44.07,-72.67),
            "VA":(37.43,-78.66),"WA":(47.38,-120.44),"WV":(38.64,-80.62),
            "WI":(44.27,-89.62),"WY":(42.96,-107.55),
        }

        # Enrich map_df with extra columns from main df if missing
        _map_df = svc_df.copy()
        for _ec in ["fiscal_capacity_per_capita","avg_disaster_exposure_12m","recovery_months"]:
            if _ec not in _map_df.columns and _ec in df.columns:
                _map_df = _map_df.merge(
                    df[["abbr",_ec]].drop_duplicates("abbr"),
                    on="abbr", how="left",
                )

        _fiscal_m = (_map_df["fiscal_capacity_per_capita"].fillna(0)
                     if "fiscal_capacity_per_capita" in _map_df.columns
                     else pd.Series([0]*len(_map_df), index=_map_df.index))
        _disexp_m = (_map_df["avg_disaster_exposure_12m"].fillna(0)
                     if "avg_disaster_exposure_12m" in _map_df.columns
                     else pd.Series([0]*len(_map_df), index=_map_df.index))
        _rec_act_m = (_map_df["recovery_months"].fillna(0)
                      if "recovery_months" in _map_df.columns
                      else _map_df["recovery_months_predicted"].fillna(0))

        # ── Configure by view ─────────────────────────────────────────────
        if _map_view == "Recovery Time":
            _z_m      = _map_df["recovery_months_predicted"]
            _cscale_m = [[0,"#dceefb"],[0.35,"#5ba4cf"],[0.65,"#2980B9"],[1,"#0d2a4a"]]
            _cbar_m   = "Predicted<br>Recovery (mo)"
            _zmin_m, _zmax_m = 10, 23
            _map_title_m = "Longer Predicted Recovery Clusters in Coastal & Government-Heavy States"

        elif _map_view == "Service Sector Share":
            _z_m      = _map_df["service_share_pct"]
            _cscale_m = [[0,"#fff3e0"],[0.35,"#ffb74d"],[0.65,"#e65100"],[1,"#4e1a00"]]
            _cbar_m   = "Service Share<br>(% of GDP)"
            _zmin_m   = float(_map_df["service_share_pct"].min()) - 1
            _zmax_m   = float(_map_df["service_share_pct"].max()) + 1
            _map_title_m = "Service-Sector Dominance Is Highest in Coastal and Urban States"

        else:  # Economic Profile
            _prof_enc = {"Industry-Intensive":1,"Resource-Intensive":2,
                         "Service-Dominated":3,"Government-Intensive":4}
            _z_m = _map_df["economic_profile"].map(_prof_enc).fillna(3)
            _cscale_m = [
                [0.00,"#27AE60"],[0.249,"#27AE60"],
                [0.250,"#E67E22"],[0.499,"#E67E22"],
                [0.500,"#2980B9"],[0.749,"#2980B9"],
                [0.750,"#8E44AD"],[1.000,"#8E44AD"],
            ]
            _cbar_m   = "Profile"
            _zmin_m, _zmax_m = 1, 4
            _map_title_m = "Economic Profile Classification by State (BEA SAGDP2, 2019–2023)"

        _fig_map3 = go.Figure()

        # Choropleth layer
        _fig_map3.add_trace(go.Choropleth(
            locations=_map_df["abbr"],
            z=_z_m,
            locationmode="USA-states",
            colorscale=_cscale_m,
            zmin=_zmin_m, zmax=_zmax_m,
            colorbar=dict(
                title=dict(text=_cbar_m, font=dict(size=10)),
                thickness=14, len=0.55,
                tickfont=dict(size=9), outlinewidth=0,
                x=1.01,
            ),
            customdata=np.stack((
                _map_df["state"].values,
                _map_df["recovery_months_predicted"].values,
                _map_df["service_share_pct"].values,
                _map_df["economic_profile"].values,
                _fiscal_m.values,
                _disexp_m.values,
                _rec_act_m.values,
            ), axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "────────────────────────<br>"
                "Predicted recovery: <b>%{customdata[1]:.1f} months</b><br>"
                "Actual recovery: %{customdata[6]:.0f} months<br>"
                "Service share: %{customdata[2]:.1f}% of GDP<br>"
                "Economic profile: <b>%{customdata[3]}</b><br>"
                "Fiscal capacity: $%{customdata[4]:,.0f} / person<br>"
                "Avg disaster declarations / yr: %{customdata[5]:.0f}"
                "<extra></extra>"
            ),
            marker_line_color="white",
            marker_line_width=0.7,
        ))

        # State abbreviation label overlay (Scattergeo)
        _lbl_abbrs  = [a for a in _map_df["abbr"] if a in _STATE_CENTS]
        _lbl_lats   = [_STATE_CENTS[a][0] for a in _lbl_abbrs]
        _lbl_lons   = [_STATE_CENTS[a][1] for a in _lbl_abbrs]
        _fig_map3.add_trace(go.Scattergeo(
            lat=_lbl_lats, lon=_lbl_lons,
            text=_lbl_abbrs,
            mode="text",
            textfont=dict(size=7.5, color="rgba(0,0,0,0.50)", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        ))

        _fig_map3.update_layout(
            title=dict(
                text=_map_title_m,
                font=dict(size=11, color="#555"),
                x=0.5, xanchor="center",
                y=0.97,
            ),
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showlakes=True,  lakecolor="#e8f4f8",
                showland=True,   landcolor="#fafafa",
                showframe=False,
                bgcolor="white",
                showcoastlines=False,
            ),
            height=470,
            paper_bgcolor="white",
            margin=dict(t=35, b=5, l=5, r=5),
            font=dict(size=11),
        )
        st.plotly_chart(_fig_map3, key="svc_map3_v2", use_container_width=True)

        # Economic Profile legend strip
        if _map_view == "Economic Profile":
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            for _pci, (_pname, _pcol) in enumerate(PROFILE_COLORS.items()):
                _short = PROFILE_SHORT.get(_pname, _pname)
                [_pc1, _pc2, _pc3, _pc4][_pci].markdown(
                    f"<div style='background:{_pcol}18;border-left:3px solid {_pcol};"
                    f"padding:5px 10px;border-radius:3px;font-size:11px;"
                    f"color:{_pcol};font-weight:700;'>{_short}</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Source: BLS LAUS · BEA SAGDP2 · Ridge Regression model (n = 50 states). "
            "Recovery time = months for state unemployment to return to pre-disaster baseline. "
            "Hover over any state for full details."
        )

        # ── Econometric interpretation ────────────────────────────────────────
        st.markdown(
            "<div style='background:#fff7f0;border-left:4px solid #E67E22;border-radius:4px;"
            "padding:12px 18px;margin:16px 0 10px 0;'>"
            "<p style='font-size:13px;font-weight:700;color:#a04000;margin:0 0 6px 0;'>"
            "States with higher service-sector shares exhibit longer predicted recovery times, "
            "holding other structural factors constant.</p>"
            "<p style='font-size:12px;color:#2c3e50;margin:0 0 8px 0;line-height:1.8;'>"
            "Each 1 percentage-point increase in service share is associated with an estimated "
            f"<b>{_slope3:+.2f} additional months</b> of recovery time (slope = {_slope3:+.3f}, "
            f"R² = {_r2_3:.3f}). "
            "Service-dominated states average <b>16.2 months</b> to recover, compared to "
            "<b>11.5 months</b> for industry-intensive states, a gap of approximately 4.7 months."
            "</p>"
            "<p style='font-size:12px;color:#2c3e50;margin:0;line-height:1.8;'>"
            "While the relationship is positive, the dispersion around the trend line indicates that "
            "industry composition interacts with additional factors, including fiscal capacity and "
            "disaster severity, reinforcing the need for a multi-variable model rather than a "
            "single-predictor approach."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── What this means ───────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;"
            "padding:14px 18px;margin:0 0 14px 0;'>"
            "<p style='font-size:12px;font-weight:700;color:#333;text-transform:uppercase;"
            "letter-spacing:0.8px;margin:0 0 8px 0;'>What This Means</p>"
            "<p style='font-size:12px;color:#444;margin:0 0 8px 0;line-height:1.8;'>"
            "States where services dominate the economy rely on demand-driven sectors such as "
            "tourism, retail, and hospitality. These sectors are more sensitive to disruptions "
            "and require consumer confidence to recover, which takes longer to rebuild than "
            "physical infrastructure."
            "</p>"
            "<ul style='font-size:12px;color:#444;line-height:2.0;margin:0;padding-left:16px;'>"
            "<li>Service sector employment depends on consumer spending, which contracts sharply after disasters</li>"
            "<li>Industrial and resource-based economies can resume output with fewer demand-side barriers</li>"
            "<li>The service-recovery gap persists even after controlling for fiscal capacity and damage severity</li>"
            "<li>This finding supports targeted policy differentiation: service-heavy states may require "
            "longer aid commitment windows and demand-stimulus mechanisms</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

        # ── Key Findings ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:16px 22px;margin:0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 8px 0;'>Key Findings — Section 4.4</p>"
            "<ul style='font-size:13px;color:#ffffff;margin:0;padding-left:18px;line-height:2.0;'>"
            "<li>Economic structure explains <b style='color:#F4D03F;'>"
            f"{_r2_3*100:.1f}% of the variance</b> in predicted recovery time across all 50 states.</li>"
            "<li><b style='color:#F4D03F;'>Service-dominated states</b> (NY, NJ, RI, CA) average 16.2 months "
            "to recover, which is 4.7 months longer than industry-intensive states.</li>"
            "<li>Resource-intensive states (WY, OK, ND, TX, NE) consistently show the fastest recovery, "
            "driven by capital-heavy production that is less dependent on consumer demand.</li>"
            "<li>Service sector concentration creates <b style='color:#F4D03F;'>demand-side fragility</b>: "
            "when consumer spending drops after a disaster, service-dependent states have fewer "
            "structural buffers to sustain employment.</li>"
            "</ul>"
            "<p style='font-size:11px;color:#888;margin:10px 0 0 0;font-style:italic;'>"
            "Sources: BEA State GDP (SAGDP2) · BLS LAUS 2010–2024 · "
            "Ridge Regression (λ = 0.1, LOO-CV, n = 50 states)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 7: Recovery Predictors ────────────────────────────────────────────
    with tab7:

        # Color palette — red = slower recovery  |  green = faster recovery
        _C_PURPLE  = "#c0392b"   # red  — slows recovery
        _C_BLUE    = "#1e8449"   # green — speeds recovery
        _C_LPURPLE = "#fde8e8"   # light red background
        _C_LBLUE   = "#e8f5e9"   # light green background

        # ── Header ────────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 2px 0;'>"
            "Recovery Predictors: What the Model Found</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 16px 0;'>"
            "Ridge Regression + SHAP · 6 features · 50 states · Leave-One-Out Cross-Validation</p>",
            unsafe_allow_html=True,
        )

        # ── Plain-language intro ──────────────────────────────────────────────
        st.markdown(
            "<div style='background:#f0f4ff;border-left:4px solid #2980B9;"
            "border-radius:4px;padding:12px 18px;margin:0 0 18px 0;'>"
            "<p style='font-size:13px;color:#1a3a5c;margin:0;line-height:1.8;'>"
            "<b>What this tab shows:</b> Our model tested 6 factors to predict how long a state "
            "takes to recover after a disaster. The charts below show <b>which factors matter most</b> "
            "(left chart) and <b>why a specific state recovers fast or slow</b> (right chart, pick any state)."
            "</p></div>",
            unsafe_allow_html=True,
        )

        # ── KPI row (3 HTML insight cards) ───────────────────────────────────
        # Green = good news  |  Red = warning / bad outcome
        _k1, _k2, _k3 = st.columns(3)
        with _k1:
            # Green: model is validated and intentionally conservative — good news
            st.markdown(
                "<div style='border:1px solid #d5e8d4;border-top:3px solid #1e8449;"
                "border-radius:6px;padding:14px 18px;background:#f6fff6;'>"
                "<p style='font-size:10px;font-weight:700;color:#1e8449;text-transform:uppercase;"
                "letter-spacing:1px;margin:0 0 4px 0;'>Model Accuracy (R²)  ·  Validated</p>"
                "<p style='font-size:34px;font-weight:900;color:#0d1b2a;margin:0;'>0.139</p>"
                "<p style='font-size:12px;color:#444;margin:4px 0 0 0;line-height:1.6;'>"
                "Explains <b>13.9% of variance</b> in recovery time. Intentionally modest: "
                "with 50 states and 6 features, the model avoids overfitting rather than "
                "chasing a high R². Validated with LOO-CV across all 50 states.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _k2:
            # Red: prediction error is an undesirable outcome — the model still carries uncertainty
            st.markdown(
                "<div style='border:1px solid #f5c6cb;border-top:3px solid #c0392b;"
                "border-radius:6px;padding:14px 18px;background:#fff8f8;'>"
                "<p style='font-size:10px;font-weight:700;color:#c0392b;text-transform:uppercase;"
                "letter-spacing:1px;margin:0 0 4px 0;'>Avg Prediction Error (MAE)  ·  Caution</p>"
                "<p style='font-size:34px;font-weight:900;color:#0d1b2a;margin:0;'>3.4 months</p>"
                "<p style='font-size:12px;color:#444;margin:4px 0 0 0;line-height:1.6;'>"
                "On average, predictions are off by <b>3.4 months</b>. Given that recovery "
                "times range from 4 to 20 months (a 16-month spread), this represents "
                "roughly a 21% margin of error.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _k3:
            # Red: CVI dominating means vulnerability risk is the biggest driver — warning signal
            st.markdown(
                "<div style='border:1px solid #f5c6cb;border-top:3px solid #c0392b;"
                "border-radius:6px;padding:14px 18px;background:#fff8f8;'>"
                "<p style='font-size:10px;font-weight:700;color:#c0392b;text-transform:uppercase;"
                "letter-spacing:1px;margin:0 0 4px 0;'>Strongest Predictor (SHAP)  ·  Warning</p>"
                "<p style='font-size:34px;font-weight:900;color:#0d1b2a;margin:0;'>CVI +2.96 mo</p>"
                "<p style='font-size:12px;color:#444;margin:4px 0 0 0;line-height:1.6;'>"
                "The <b>Compound Vulnerability Index</b> shifts predictions by "
                "<b>+2.96 months</b> on average, more than any other factor. "
                "States with overlapping risks across labor, fiscal, and disaster dimensions "
                "consistently take the longest to recover.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        if not shap_df.empty:
            _feat_label_map = {
                "avg_disaster_exposure_12m":    "Disaster Exposure",
                "fiscal_capacity_per_capita":   "Fiscal Capacity",
                "service_share_pct":            "Service Sector Share",
                "compound_vulnerability_index": "Compound Vulnerability (CVI)",
                "unemployment_shock_magnitude": "Unemployment Shock",
                "avg_damage_per_capita":        "Damage per Capita",
            }
            shap_cols      = [c for c in shap_df.columns if c.startswith("shap_")]
            feat_names_raw = [c.replace("shap_","") for c in shap_cols]
            feat_names_lbl = [_feat_label_map.get(f, f.replace("_"," ").title()) for f in feat_names_raw]
            mean_shap        = shap_df[shap_cols].abs().mean()
            mean_shap_signed = shap_df[shap_cols].mean()   # signed: + = slows, - = speeds

            # ── Two-column layout ─────────────────────────────────────────────
            _lc, _rc = st.columns([1, 1.2])

            # ── LEFT: global importance colored by direction ───────────────────
            with _lc:
                st.markdown(
                    "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
                    "Which factor matters most for ALL states?</p>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Bar length = strength of influence on recovery time across all 50 states. "
                    "Red = tends to slow recovery. Green = tends to speed recovery."
                )

                _sorted_idx = mean_shap.argsort()
                _sv  = mean_shap.values[_sorted_idx]
                _sl  = [feat_names_lbl[i] for i in _sorted_idx]

                # Color by net direction: purple = net slower, blue = net faster
                _raw_sorted = [feat_names_raw[i] for i in _sorted_idx]
                _bar_colors = [
                    _C_PURPLE if float(mean_shap_signed[f"shap_{r}"]) >= 0 else _C_BLUE
                    for r in _raw_sorted
                ]
                _txt_colors = [
                    "#c0392b" if float(mean_shap_signed[f"shap_{r}"]) >= 0 else "#1e8449"
                    for r in _raw_sorted
                ]

                fig_shap = go.Figure(go.Bar(
                    x=_sv, y=_sl, orientation="h",
                    marker_color=_bar_colors,
                    marker_line=dict(color="white", width=0.5),
                    text=[f"{v:.2f} mo" for v in _sv],
                    textposition="outside",
                    textfont=dict(size=10, color=_txt_colors, family="Arial Black"),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Avg SHAP impact: <b>%{x:.3f} months</b><br>"
                        "Red = this factor tends to make recovery SLOWER across states.<br>"
                        "Green = this factor tends to make recovery FASTER across states.<br>"
                        "<i>Larger bar = stronger overall influence on the prediction.</i>"
                        "<extra></extra>"
                    ),
                ))
                fig_shap.update_layout(
                    height=390, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(
                        title="Avg SHAP value (months): how much each factor shifts the recovery prediction",
                        gridcolor="#eeeeee", tickfont=dict(size=9),
                        range=[0, float(_sv.max()) * 1.40],
                    ),
                    yaxis=dict(tickfont=dict(size=11, color="#222")),
                    margin=dict(t=10, b=55, l=10, r=20),
                    font=dict(size=11),
                    showlegend=False,
                )
                st.plotly_chart(fig_shap, key="shap_global", use_container_width=True)

            # ── RIGHT: per-state SHAP — purple = slows down, blue = speeds up ──
            with _rc:
                st.markdown(
                    "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
                    "Why does THIS state recover fast or slow?</p>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Pick a state below. "
                    "Red bars = that factor is making recovery SLOWER. "
                    "Green bars = that factor is making recovery FASTER."
                )
                if "state" in shap_df.columns:
                    _avail_states = sorted(shap_df["state"].dropna().unique().tolist())
                    _sel_state = st.selectbox(
                        "Select a state to inspect:", _avail_states,
                        index=_avail_states.index("Louisiana")
                              if "Louisiana" in _avail_states else 0,
                        key="shap_state_sel",
                    )
                    _srow = shap_df[shap_df["state"] == _sel_state]
                    if not _srow.empty:
                        _sv2 = [float(_srow[c].values[0]) for c in shap_cols]
                        # Purple = positive (makes recovery longer), Blue = negative (faster)
                        _sw_colors = [_C_PURPLE if v > 0 else _C_BLUE for v in _sv2]
                        _sorted2 = sorted(zip(_sv2, feat_names_lbl, _sw_colors),
                                          key=lambda x: x[0])
                        _sv2_s, _sl2_s, _sc2_s = zip(*_sorted2)

                        _rec_pred = df[df["state"] == _sel_state]["recovery_months_predicted"].values
                        _rec_str  = f"{_rec_pred[0]:.1f} months" if len(_rec_pred) else "—"

                        # Text labels: clean "+X.XX mo" / "-X.XX mo" colored by direction
                        _sw_txt = [
                            (f"+{abs(v):.2f} mo  SLOWER" if v > 0 else f"-{abs(v):.2f} mo  FASTER")
                            for v in _sv2_s
                        ]
                        _sw_txt_colors = list(_sc2_s)   # purple=slower, blue=faster

                        # x-axis range with padding for outside labels
                        _x_max = max(abs(v) for v in _sv2_s) if _sv2_s else 1
                        _x_pad = _x_max * 0.55

                        fig_sw = go.Figure(go.Bar(
                            x=list(_sv2_s), y=list(_sl2_s),
                            orientation="h",
                            marker_color=list(_sc2_s),
                            marker_line=dict(color="white", width=0.5),
                            text=_sw_txt,
                            textposition="outside",
                            textfont=dict(size=10, color=_sw_txt_colors),
                            cliponaxis=False,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Effect on recovery: <b>%{x:+.2f} months</b><br>"
                                "Positive = this factor is making recovery <b>SLOWER</b><br>"
                                "Negative = this factor is making recovery <b>FASTER</b><br>"
                                "<i>SHAP value: how much this factor shifts the prediction "
                                "for this specific state relative to the model baseline.</i>"
                                "<extra></extra>"
                            ),
                        ))
                        fig_sw.add_vline(x=0, line_width=2, line_color="#bbb")
                        fig_sw.add_annotation(
                            xref="x", yref="paper",
                            x=-_x_max * 0.4, y=1.04,
                            text="Faster recovery",
                            showarrow=False,
                            font=dict(size=9, color=_C_BLUE, family="Arial Black"),
                        )
                        fig_sw.add_annotation(
                            xref="x", yref="paper",
                            x=_x_max * 0.4, y=1.04,
                            text="Slower recovery",
                            showarrow=False,
                            font=dict(size=9, color=_C_PURPLE, family="Arial Black"),
                        )
                        fig_sw.update_layout(
                            height=420, plot_bgcolor="white", paper_bgcolor="white",
                            title=dict(
                                text=f"{_sel_state}  ·  Model predicts: {_rec_str}",
                                font=dict(size=13, color="#0d1b2a"),
                            ),
                            xaxis=dict(
                                title="Effect on recovery time (months)",
                                gridcolor="#eeeeee", zeroline=False,
                                tickfont=dict(size=9),
                                range=[-(_x_max + _x_pad), _x_max + _x_pad],
                            ),
                            yaxis=dict(tickfont=dict(size=11, color="#222")),
                            margin=dict(t=50, b=55, l=10, r=10),
                            font=dict(size=10), showlegend=False,
                        )
                        st.plotly_chart(fig_sw, key="shap_waterfall", use_container_width=True)

            # ── Legend strip ─────────────────────────────────────────────────
            st.markdown(
                "<div style='display:flex;gap:24px;align-items:center;"
                "background:#fafafa;border:1px solid #ddd;border-radius:6px;"
                "padding:10px 18px;margin:4px 0 20px 0;'>"
                "<span style='font-size:13px;color:#c0392b;font-weight:700;'>Red bar</span>"
                "<span style='font-size:12px;color:#555;'>= this factor makes recovery <b>SLOWER</b></span>"
                "<span style='font-size:20px;color:#ccc;'>|</span>"
                "<span style='font-size:13px;color:#1e8449;font-weight:700;'>Green bar</span>"
                "<span style='font-size:12px;color:#555;'>= this factor makes recovery <b>FASTER</b></span>"
                "<span style='font-size:20px;color:#ccc;'>|</span>"
                "<span style='font-size:12px;color:#555;'>"
                "Longer bar = stronger effect. Zero line = no effect.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.divider()

        # ── Technical expander ────────────────────────────────────────────────
        with st.expander("Technical Details: Model Formula and Full Coefficient Table"):
            st.markdown(
                "<p style='font-family:Georgia,serif;font-size:14px;font-style:italic;"
                "color:#0d1b2a;margin:0 0 10px 0;line-height:2.2;'>"
                "Recovery Months = &beta;<sub>0</sub> + &beta;<sub>1</sub>(exposure) "
                "+ &beta;<sub>2</sub>(CVI) + &beta;<sub>3</sub>(damage) "
                "+ &beta;<sub>4</sub>(unemployment) + &beta;<sub>5</sub>(fiscal) + &lambda;&#8214;&beta;&#8214;&sup2;"
                "</p>"
                "<p style='font-size:12px;color:#555;margin:0 0 12px 0;line-height:1.7;'>"
                "All features are standardized before entering the model. "
                "A positive coefficient means that factor <b>increases</b> predicted recovery time. "
                "A negative coefficient means it <b>decreases</b> it (faster recovery). "
                "&lambda; = 0.1 selected via Leave-One-Out CV.</p>",
                unsafe_allow_html=True,
            )
            _full_coef = pd.DataFrame([
                {"Variable": "Compound Vulnerability (CVI)", "Source": "BLS · FEMA · Census · BEA",
                 "Coeff": "+2.96", "Direction": "Slower",
                 "Plain English": "States with more overlapping risks take much longer to recover"},
                {"Variable": "Fiscal Capacity per Capita",   "Source": "U.S. Census",
                 "Coeff": "+2.52", "Direction": "Slower",
                 "Plain English": "Wealthier states have larger economies, more to restabilize"},
                {"Variable": "Disaster Exposure",            "Source": "FEMA OpenFEMA",
                 "Coeff": "-1.86", "Direction": "Faster",
                 "Plain English": "States hit more often build better response systems over time"},
                {"Variable": "Service Sector Share",         "Source": "BEA State GDP",
                 "Coeff": "+0.78", "Direction": "Slower",
                 "Plain English": "Service jobs (retail, hospitality) take longer to bounce back"},
                {"Variable": "Unemployment Shock",           "Source": "BLS LAUS",
                 "Coeff": "+0.57", "Direction": "Slower",
                 "Plain English": "Bigger job losses mean a longer road back to normal"},
                {"Variable": "Damage per Capita",            "Source": "FEMA / U.S. Census",
                 "Coeff": "+0.13", "Direction": "Slower",
                 "Plain English": "Physical damage matters, but its signal is already captured in CVI"},
            ])

            def _color_direction(val):
                if val == "Slower":
                    return "color: #c0392b; font-weight: 700;"   # red = bad outcome
                if val == "Faster":
                    return "color: #1e8449; font-weight: 700;"   # green = good outcome
                return ""

            def _color_coeff(val):
                try:
                    v = float(str(val).replace("+",""))
                    if v > 0: return "color: #c0392b; font-weight: 600;"   # red = slows recovery
                    if v < 0: return "color: #1e8449; font-weight: 600;"   # green = speeds recovery
                except: pass
                return ""

            st.dataframe(
                _full_coef.style
                    .applymap(_color_direction, subset=["Direction"])
                    .applymap(_color_coeff,     subset=["Coeff"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Baseline prediction = 16.40 months  ·  λ = 0.1  ·  "
                "Model fit (R²) = 0.139  ·  Avg error = 3.40 months  ·  50 states"
            )

        # ── Key findings ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:18px 24px;margin:20px 0 0 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 10px 0;'>Key Findings: Section 6.1</p>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:12px 16px;'>"
            "<p style='font-size:13px;font-weight:700;color:#F4D03F;margin:0 0 6px 0;'>"
            "Vulnerability Beats Everything Else</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.7;'>"
            "The Compound Vulnerability Index scores +2.96, the highest of any factor. "
            "States with weak finances, high disaster exposure, and service-heavy economies "
            "consistently take the longest to recover. No single variable tells the full story.</p></div>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:12px 16px;'>"
            "<p style='font-size:13px;font-weight:700;color:#F4D03F;margin:0 0 6px 0;'>"
            "Being Hit Often Actually Helps</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.7;'>"
            "Disaster Exposure is the only factor with a negative score (−1.86), "
            "meaning states hit by disasters more frequently tend to recover <i>faster</i>. "
            "Repeated exposure forces states to build better emergency systems over time.</p></div>"
            "</div>"
            "<p style='font-size:11px;color:#888;margin:12px 0 0 0;font-style:italic;'>"
            "Sources: BLS LAUS · FEMA OpenFEMA · U.S. Census · BEA State GDP · "
            "Ridge Regression (λ = 0.1, LOO-CV, n = 50)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 4: Governance, Aid & Recovery ────────────────────────────────────
    with tab4:

        try:
            from drpi_gov_data import PARTISAN_DATA, GOV_FINDINGS, build_drpi_map_html as _build_gov_map
            _gov_data_ok = True
        except Exception as _gov_err:
            PARTISAN_DATA, GOV_FINDINGS, _build_gov_map = [], [], None
            _gov_data_ok = False

        import streamlit.components.v1 as _components_tab4

        # ── TITLE — matches other tabs exactly ───────────────────────────────
        st.markdown(
            "<p style='font-size:27px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
            "Governance, Aid Allocation, and Recovery Outcomes</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 14px 0;font-style:italic;'>"
            "How state-level political structure, fiscal capacity, and federal aid distribution "
            "shape disaster recovery speed across the 50 U.S. states "
            "(FEMA declarations · Census State Finances · BLS LAUS · MIT Election Lab).</p>",
            unsafe_allow_html=True,
        )

        # ── KPI CARDS — 2-column layout matching other tabs ──────────────────
        _g4c1, _g4c2 = st.columns(2)
        with _g4c1:
            st.markdown(
                "<div style='background:#fdecea;border-left:4px solid #C0392B;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#922b21;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Risk is bipartisan — disaster exposure transcends party lines</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "3 · 3 · 2 split</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Of the <b>8 HIGH-tier states</b>, 3 lean Democrat (CA, NJ, RI), "
                "3 lean Republican (LA, TX, GA), and 2 are swing states (FL, NC). "
                "DRPI risk shows no strong partisan skew: geography and fiscal capacity "
                "matter more than political affiliation. "
                "(Source: DRPI Risk Scores · MIT Election Lab 2008-2020)</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _g4c2:
            st.markdown(
                "<div style='background:#f5eef8;border-left:4px solid #8E44AD;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#6c3483;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Democrat-leaning states average longer predicted recovery times</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "18.5 vs 14.3 months</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Democrat-leaning states average <b>18.5 months</b> predicted recovery vs "
                "<b>14.3 months</b> for Republican-leaning states — a 4.2-month gap. "
                "This reflects higher urban density and infrastructure complexity, not policy: "
                "CA, NJ, NY, and RI anchor the Democrat average upward. "
                "(Source: FEMA OpenFEMA 2006-2025 · Ridge Regression Model)</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        _g4c3, _g4c4 = st.columns(2)
        with _g4c3:
            st.markdown(
                "<div style='background:#fdecea;border-left:4px solid #C0392B;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#922b21;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Swing states carry the highest composite disaster risk score</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "DRPI 46.7 avg</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "The 4 swing states (FL, NC, OH, IA) average a DRPI score of <b>46.7</b>, "
                "exceeding Democrat-leaning (<b>40.5</b>) and Republican-leaning (<b>35.1</b>) averages. "
                "Florida and North Carolina alone represent 2 of the top 8 highest-risk states nationally. "
                "(Source: DRPI Risk Scores · 2008-2020 Presidential Elections)</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _g4c4:
            st.markdown(
                "<div style='background:#f5eef8;border-left:4px solid #8E44AD;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#6c3483;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Methodology: partisan classification approach</p>"
                "<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                "4 election cycles</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "Party affiliation is proxied from presidential voting patterns across "
                "<b>4 cycles (2008-2020)</b>, not actual gubernatorial or state legislature data. "
                "'Independent' indicates swing states that split evenly between parties. "
                "(Source: MIT Election Lab, 1976-2020 U.S. President dataset)</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:16px 0 4px 0;'></div>", unsafe_allow_html=True)

        # ── MAP SECTION TITLE ─────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:20px;font-weight:800;color:#0d1b2a;margin:0 0 2px 0;'>"
            "Interactive State-Level Recovery Risk Map</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 10px 0;font-style:italic;'>"
            "Select a variable to color all 50 states. Hover each state for full details.</p>",
            unsafe_allow_html=True,
        )

        _map_options = {
            "DRPI Risk Score (composite)":       "drpi_score",
            "Predicted Recovery Time (months)":  "recovery_months_predicted",
            "Risk Tier (High / Moderate / Low)": "drpi_risk_tier",
            "Governor Party Affiliation":        "governor_party",
        }
        _selected_label = st.selectbox(
            "Color map by:",
            options=list(_map_options.keys()),
            index=0,
            key="tab4_map_selector",
        )
        _selected_color_by = _map_options[_selected_label]

        # ── EMBED MAP ─────────────────────────────────────────────────────────
        if _gov_data_ok and _build_gov_map is not None:
            try:
                _gov_map_html = _build_gov_map(color_by=_selected_color_by)
                _components_tab4.html(_gov_map_html, height=540, scrolling=False)
                st.caption(
                    "Sources: FEMA Disaster Declarations Database · Census State Finance Time Series · "
                    "BLS LAUS 2006-2025 · MIT Election Data Lab 2008-2020. "
                    "DRPI Score is a composite of labor volatility, disaster exposure, fiscal capacity, "
                    "and economic structure (see Section 3 of the paper)."
                )
            except Exception as _map_exc:
                st.warning(
                    f"Map render error: {_map_exc}. "
                    "Check that folium and branca are installed (`pip install folium branca`)."
                )
        else:
            st.info(
                "Map unavailable. Ensure `drpi_gov_data.py` is in the same folder as this dashboard "
                "and that `folium` and `branca` are installed."
            )

        # ── PARTISAN DATA TABLE ───────────────────────────────────────────────
        if PARTISAN_DATA:
            st.markdown(
                "<p style='font-size:18px;font-weight:800;color:#0d1b2a;margin:28px 0 8px 0;'>"
                "State-Level Recovery and Governance Data</p>",
                unsafe_allow_html=True,
            )
            _pd_df = pd.DataFrame(PARTISAN_DATA)

            # Color-code risk tier column
            def _tier_style(val):
                colors = {"HIGH": "#fde8e8", "MODERATE": "#fff8e1", "LOW": "#e8f5e9"}
                text   = {"HIGH": "#b03a2e", "MODERATE": "#7d5a00", "LOW": "#117a65"}
                bg = colors.get(str(val).upper(), "#f9f9f9")
                fg = text.get(str(val).upper(), "#333")
                return f"background-color:{bg};color:{fg};font-weight:700;"

            _display_cols = [c for c in [
                "state", "abbr", "governor_party", "legislative_control",
                "drpi_risk_tier", "drpi_score", "recovery_months_predicted"
            ] if c in _pd_df.columns]

            _styled_df = (
                _pd_df[_display_cols]
                .sort_values("recovery_months_predicted", ascending=False)
                .rename(columns={
                    "state": "State", "abbr": "Abbr",
                    "governor_party": "Governor Party",
                    "legislative_control": "Legislature",
                    "drpi_risk_tier": "Risk Tier",
                    "drpi_score": "DRPI Score",
                    "recovery_months_predicted": "Recovery (mo)",
                })
                .reset_index(drop=True)
            )

            if "Risk Tier" in _styled_df.columns:
                st.dataframe(
                    _styled_df.style.applymap(_tier_style, subset=["Risk Tier"]),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.dataframe(_styled_df, use_container_width=True, height=400)

        elif not _gov_data_ok:
            st.markdown(
                "<div style='background:#fff8e1;border-left:4px solid #D4AC0D;"
                "border-radius:4px;padding:12px 18px;margin:16px 0;'>"
                "<p style='font-size:13px;font-weight:700;color:#7d5a00;margin:0 0 4px 0;'>"
                "Data module not loaded</p>"
                "<p style='font-size:12px;color:#555;margin:0;'>"
                "Ensure <code>drpi_gov_data.py</code> is in the <code>ML_models/</code> folder "
                "alongside this dashboard and that all dependencies are installed.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── TAB 5: Education Disruption ───────────────────────────────────────────
    with tab5:

        # ── TITLE ─────────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:30px;font-weight:900;color:#0d1b2a;margin:0 0 2px 0;'>"
            "High-Exposure States Show Slower Educational Recovery</p>"
            "<p style='font-size:14px;color:#555;margin:0 0 14px 0;'>"
            "School closures introduced an additional layer of economic strain on top of labor market disruption. "
            "In high-exposure states, education disruption compounded existing unemployment volatility "
            "(Section 4.5, CSDH + NCES-ADA + ACS S1501).</p>",
            unsafe_allow_html=True,
        )

        # ── HEADLINE INSIGHT ──────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#fff8e1;border-left:4px solid #D4AC0D;border-radius:4px;"
            "padding:12px 18px;margin:0 0 18px 0;'>"
            "<p style='font-size:15px;font-weight:700;color:#7d5a00;margin:0 0 4px 0;'>"
            "States with prolonged remote learning experienced slower economic recovery."
            "</p>"
            "<p style='font-size:12px;color:#666;margin:0;'>"
            "Virtual learning share (CSDH) and average daily attendance loss (NCES-ADA Table 203.80) "
            "serve as contextual indicators of recovery capacity. While school closure duration is not "
            "a core variable in the DRPA model, it captures recovery dimensions beyond unemployment data alone."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── MARKER LEGEND INLINE ──────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:12px;color:#444;margin:0 0 6px 0;'>"
            "<span style='color:#e02424;font-size:14px;'>&#9679;</span> Top 15 High-Exposure States (FEMA Declarations Database) &nbsp;|&nbsp; "
            "<span style='color:#1a56db;font-size:14px;'>&#9679;</span> Lower Exposure States &nbsp;|&nbsp; "
            "Use layer control (top-right of map) to switch disaster event years.</p>",
            unsafe_allow_html=True,
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
            # MAIN VISUAL — Education Disruption vs. Recovery Duration
            # ─────────────────────────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:12px;color:#888;margin:0 0 8px 0;'>"
                "States with more education disruption during COVID take longer to recover economically. "
                "Color reflects predicted recovery time from the DRPA model."
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
                    customdata=list(zip(
                        _sc_names,                                          # 0
                        _sc_x,                                              # 1 virtual %
                        _sc_y,                                              # 2 attendance change
                        _sc_r.round(1),                                     # 3 recovery months
                        (_scatter_df["share_inperson"] * 100).round(1),     # 4 in-person %
                        (_scatter_df["share_hybrid"]   * 100).round(1),     # 5 hybrid %
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>  (Top 15 High-Exposure State)<br>"
                        "<br>"
                        "Learning modality during COVID:<br>"
                        "  Virtual: <b>%{customdata[1]:.1f}%</b>  |  "
                        "In-person: %{customdata[4]:.1f}%  |  "
                        "Hybrid: %{customdata[5]:.1f}%<br>"
                        "Avg daily attendance change (2019 to 2021): <b>%{customdata[2]:.1f}%</b>"
                        "  (negative = fewer students in class)<br>"
                        "<br>"
                        "Predicted economic recovery: <b>%{customdata[3]:.1f} months</b><br>"
                        "  States with more virtual learning tend to recover more slowly."
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

        # ── Educational Attainment Map (ACS S1501) ────────────────────────────
        st.markdown(
            "<p style='font-size:20px;font-weight:800;color:#0d1b2a;margin:28px 0 2px 0;'>"
            "Change in Bachelor's Degree Attainment by State</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 10px 0;'>"
            "Default view shows educational growth by state. Switch layers to compare disaster event years "
            "(2011 Tornado Outbreak · 2017 Harvey/Irma · 2019 Midwest Floods · 2020 COVID). "
            "High-exposure states in red.</p>",
            unsafe_allow_html=True,
        )

        _grad_files = {
            2011: os.path.join(_EDU_ROOT, "2011gradpercent.csv"),
            2017: os.path.join(_EDU_ROOT, "2017gradpercent.csv"),
            2018: os.path.join(_EDU_ROOT, "2018gradpercent.csv"),
            2019: os.path.join(_EDU_ROOT, "2019gradpercent.csv"),
            2020: os.path.join(_EDU_ROOT, "2020gradpercent.csv"),
            2024: os.path.join(_EDU_ROOT, "2024gradpercent.csv"),
        }

        _map_html_path = os.path.join(ROOT, "education_map.html")

        if all(os.path.exists(p) for p in _grad_files.values()):
            import subprocess as _sp
            _edu_script = os.path.join(ROOT, "education_wendy.py")
            if not os.path.exists(_map_html_path):
                _sp.run(["python", _edu_script], cwd=ROOT, capture_output=True)

        if os.path.exists(_map_html_path):
            import streamlit.components.v1 as _components
            with open(_map_html_path, "r", encoding="utf-8") as _f:
                _map_html = _f.read()
            _components.html(_map_html, height=550, scrolling=False)
            st.caption(
                "Source: U.S. Census Bureau ACS Educational Attainment, Table S1501 "
                "(CSDH; 2011, 2017, 2018, 2019, 2020, 2024). "
                "Red markers = Top 15 high-exposure states. "
                "Note: 2020 uses ACS 5-Year Estimates due to COVID-19 data collection suspension."
            )
        else:
            st.info(
                "Educational attainment map not available. "
                "Run education_wendy.py to generate education_map.html, "
                "then place it in the project root folder."
            )

        # ── Kentucky outlier callout ──────────────────────────────────────────
        st.markdown(
            "<div style='background:#eaf4fb;border-left:4px solid #2980b9;border-radius:4px;"
            "padding:10px 16px;margin:12px 0 0 0;'>"
            "<p style='font-size:13px;font-weight:700;color:#1a5276;margin:0 0 2px 0;'>"
            "Notable Pattern: Kentucky (KY)</p>"
            "<p style='font-size:12px;color:#444;margin:0;'>"
            "Kentucky recorded the highest virtual learning share (~36%) among the top 15 states but a "
            "near-zero attendance drop, suggesting remote infrastructure or state policy buffered the "
            "attendance impact (CSDH, 2020-2023). Despite this, Kentucky remains associated with longer "
            "predicted recovery periods alongside MS, LA, and NC, indicating structural factors beyond "
            "attendance loss shape its recovery trajectory (Section 4.5).</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Key Findings box (Tab 5) ──────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:16px 22px;margin:16px 0 0 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 8px 0;'>Key Findings</p>"
            "<ul style='font-size:14px;color:#ffffff;margin:0;padding-left:18px;line-height:2;'>"
            "<li><b style='color:#F4D03F;'>Mississippi, Louisiana, and North Carolina</b> show the longest "
            "predicted recovery periods, driven by high virtual learning share and large attendance losses "
            "(CSDH 2020-2023; NCES-ADA Table 203.80).</li>"
            "<li><b style='color:#F4D03F;'>Kentucky</b> recorded the highest virtual share but near-zero "
            "attendance drop. It remains among the slowest-recovering states, indicating other structural "
            "factors contribute beyond attendance loss alone.</li>"
            "<li>States with the greatest attendance loss also face the longest economic recovery periods. "
            "Education disruption and disaster burden overlap in the same high-exposure states "
            "(BLS LAUS; FEMA Disaster Declarations).</li>"
            "<li>Education disruption is not only a side effect of disasters. Research from the NBER "
            "links school closures to a 3.8 percentage point decline in full-time employment among mothers, "
            "reducing long-run workforce capacity in high-exposure states (Collins et al., 2021).</li>"
            "</ul>"
            "<p style='font-size:12px;color:#F4D03F;font-weight:600;margin:10px 0 4px 0;'>"
            "While school closure duration is not a core variable in the DRPA model, these patterns serve "
            "as contextual indicators that recovery extends beyond what unemployment data alone can capture "
            "(Section 4.5).</p>"
            "<p style='font-size:11px;color:#888;margin:0;font-style:italic;'>"
            "Sources: CSDH (ref. 30) · NCES-ADA Table 203.80 (ref. 31) · ACS S1501 (ref. 34) · "
            "Burbio K-12 Tracker (ref. 27) · Collins et al. 2021 via NBER (ref. 32)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 6: Military Capacity ─────────────────────────────────────────────
    with tab6:

        # ── Build dataset ────────────────────────────────────────────────────
        MIL_INDEX = {
            "Tennessee":86.74,"Texas":57.01,"Mississippi":49.50,"Missouri":48.99,
            "Georgia":46.88,"Florida":44.05,"Louisiana":38.97,"Arkansas":37.76,
            "Oklahoma":35.13,"Iowa":34.24,"North Carolina":31.75,"Virginia":31.67,
            "Kansas":28.89,"Kentucky":28.50,"Nebraska":27.37,
        }
        _mil_df = pd.DataFrame(list(MIL_INDEX.items()),
                               columns=["state","military_capacity_index"])
        _top15  = df[df["state"].isin(MIL_INDEX.keys())][
            ["state","abbr","recovery_months_predicted","fiscal_capacity_per_capita"]
        ].copy()
        _mp = _mil_df.merge(_top15, on="state", how="left").dropna(
            subset=["recovery_months_predicted","fiscal_capacity_per_capita"])

        # Ranks
        _mp["rank_mil"] = _mp["military_capacity_index"].rank(ascending=False).astype(int)
        _mp["rank_rec"] = _mp["recovery_months_predicted"].rank(ascending=True).astype(int)
        _mp["rank_gap"] = _mp["rank_mil"] - _mp["rank_rec"]

        def _cat(g):
            if g <= -3: return "Underperforming"
            if g >=  3: return "Overperforming"
            return "Aligned"
        _mp["category"] = _mp["rank_gap"].apply(_cat)

        CAT_COLOR = {"Underperforming":"#c0392b","Aligned":"#e6ac00","Overperforming":"#27ae60"}

        from scipy.stats import pearsonr as _pr
        _r_rec,  _p_rec  = _pr(_mp["military_capacity_index"], _mp["recovery_months_predicted"])
        _r_fisc, _p_fisc = _pr(_mp["military_capacity_index"], _mp["fiscal_capacity_per_capita"])

        # ── Page header ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:16px 0 6px 0;'>"
            "<p style='font-size:27px;font-weight:900;color:#0d1b2a;margin:0 0 4px 0;'>"
            "Military Capacity and Disaster Recovery</p>"
            "<p style='font-size:14px;color:#555;margin:0;font-style:italic;'>"
            "Evaluated as a candidate predictor and excluded from the final model "
            "(Section 4.6 · DoD BSR FY2019–2025 · NTAD-MB USDOT BTS · 15 high-exposure states)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── KPI strip — 2 insight cards ──────────────────────────────────────
        _mk1, _mk2 = st.columns(2)
        with _mk1:
            st.markdown(
                "<div style='background:#f8f9fa;border-left:4px solid #455a64;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#455a64;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Does military capacity predict recovery speed?</p>"
                f"<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                f"r = {_r_rec:+.2f}  (p = {_p_rec:.3f})</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                "No statistically significant relationship. Having more military bases and "
                "infrastructure does not mean a state recovers faster after a disaster. "
                "Tennessee ranks <b>#1 in military capacity</b> but only <b>#12 in recovery speed</b>.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _mk2:
            st.markdown(
                "<div style='background:#fdf3f3;border-left:4px solid #c0392b;"
                "border-radius:6px;padding:14px 20px;'>"
                "<p style='font-size:11px;font-weight:600;color:#922b21;"
                "text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;'>"
                "Why was it excluded from the model?</p>"
                f"<p style='font-size:36px;font-weight:900;color:#0d1b2a;margin:0;'>"
                f"r = {_r_fisc:+.2f}  with fiscal capacity</p>"
                "<p style='font-size:12px;color:#444;margin:5px 0 0 0;line-height:1.7;'>"
                f"Strong inverse collinearity (p = {_p_fisc:.3f}). Military and fiscal capacity "
                "measure overlapping institutional resources. Including both would artificially "
                "inflate the model. <b>Fiscal capacity was retained</b> as the stronger, "
                "more direct predictor of recovery speed.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        # ── Chart row 1: Scatter + Quadrant ──────────────────────────────────
        # ── Chart 1: Military Capacity vs Recovery Time (full width) ────────
        st.markdown(
                "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
                "Military Capacity vs Recovery Time</p>",
                unsafe_allow_html=True,
            )
        st.caption("Each dot = 1 state. Color = performance category. OLS line shows a weak, non-significant trend.")
        import numpy as _np
        _x_line = _np.linspace(_mp["military_capacity_index"].min()-3,
                               _mp["military_capacity_index"].max()+3, 100)
        from scipy.stats import linregress as _lr
        _sl, _ic, *_ = _lr(_mp["military_capacity_index"], _mp["recovery_months_predicted"])
        _y_line = _sl * _x_line + _ic

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_x_line, y=_y_line, mode="lines",
            line=dict(color="#aaa", dash="dash", width=1.5),
            name=f"OLS  r={_r_rec:+.2f}  p={_p_rec:.3f}",
            hoverinfo="skip",
        ))
        for cat, grp in _mp.groupby("category"):
            fig_sc.add_trace(go.Scatter(
                x=grp["military_capacity_index"],
                y=grp["recovery_months_predicted"],
                mode="markers+text",
                text=grp["abbr"],
                textposition="top center",
                textfont=dict(size=10, color=CAT_COLOR[cat]),
                marker=dict(size=14, color=CAT_COLOR[cat],
                            line=dict(color="white", width=1.5)),
                name=cat,
                customdata=grp[["state","fiscal_capacity_per_capita",
                                "rank_mil","rank_rec","rank_gap","category"]].values,
                hovertemplate=(
                    "<b>%{text}  %{customdata[0]}</b><br>"
                    "Category: <b>%{customdata[5]}</b><br>"
                    "<br>"
                    "Military Capacity Index: %{x:.1f}  |  Rank: #%{customdata[2]} of 15<br>"
                    "Recovery time: <b>%{y:.1f} months</b>  |  Rank: #%{customdata[3]} of 15<br>"
                    "Rank gap (Mil minus Recovery): <b>%{customdata[4]:+d}</b>"
                    "  (negative = high military, slow recovery)<br>"
                    "Fiscal capacity: $%{customdata[1]:,.0f} / person<br>"
                    "<extra></extra>"
                ),
            ))
        fig_sc.add_annotation(
            x=0.97, y=0.97, xref="paper", yref="paper",
            text=f"r = {_r_rec:+.2f}  (p = {_p_rec:.3f})<br><i>Not significant</i>",
            showarrow=False, align="right",
            bgcolor="white", bordercolor="#bbb", borderwidth=1,
            font=dict(size=10, color="#555"),
        )
        fig_sc.update_layout(
            height=440, plot_bgcolor="#fafafa", paper_bgcolor="white",
            xaxis_title="Military Capacity Index (0–100)",
            yaxis_title="Predicted Recovery Time (months)",
            legend=dict(orientation="h", y=-0.18, x=0, font_size=11),
            margin=dict(t=20, b=70, l=55, r=20),
            font=dict(size=11),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Chart 2: Military vs Fiscal Capacity (full width) ────────────────
        st.markdown(
                "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:18px 0 2px 0;'>"
                "Military vs Fiscal Capacity: Inverse Relationship</p>",
                unsafe_allow_html=True,
            )
        st.caption("States with more military infrastructure tend to have lower state revenue per capita. They act as substitutes, not complements.")
        _x2 = _np.linspace(_mp["military_capacity_index"].min()-3,
                           _mp["military_capacity_index"].max()+3, 100)
        _sl2, _ic2, *_ = _lr(_mp["military_capacity_index"], _mp["fiscal_capacity_per_capita"])
        _y2 = _sl2 * _x2 + _ic2

        fig_qd = go.Figure()
        fig_qd.add_trace(go.Scatter(
            x=_x2, y=_y2, mode="lines",
            line=dict(color="#c0392b", dash="dash", width=1.5),
            name=f"OLS  r={_r_fisc:+.2f}  p={_p_fisc:.3f}",
            hoverinfo="skip",
        ))
        fig_qd.add_trace(go.Scatter(
            x=_mp["military_capacity_index"],
            y=_mp["fiscal_capacity_per_capita"],
            mode="markers+text",
            text=_mp["abbr"],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=14,
                color=_mp["recovery_months_predicted"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="Recovery<br>Months", thickness=12, len=0.7),
                line=dict(color="white", width=1.5),
            ),
            customdata=_mp[["state","recovery_months_predicted",
                            "rank_mil","rank_rec","category","rank_gap"]].values,
            hovertemplate=(
                "<b>%{text}  %{customdata[0]}</b><br>"
                "Category: <b>%{customdata[4]}</b><br>"
                "<br>"
                "Military Capacity Index: %{x:.1f}  |  Rank: #%{customdata[2]} of 15<br>"
                "Fiscal capacity: <b>$%{y:,.0f} / person</b>"
                "  (higher = more state revenue to fund recovery)<br>"
                "Recovery time: %{customdata[1]:.1f} months  |  Rank: #%{customdata[3]} of 15<br>"
                "Rank gap (Mil minus Recovery): <b>%{customdata[5]:+d}</b>"
                "  (negative = strong military but slow recovery)<br>"
                "<extra></extra>"
            ),
            name="State",
        ))
        fig_qd.add_annotation(
            x=0.03, y=0.97, xref="paper", yref="paper",
            text=f"r = {_r_fisc:+.2f}  (p = {_p_fisc:.3f})<br><i>Significant inverse</i>",
            showarrow=False, align="left",
            bgcolor="white", bordercolor="#c0392b", borderwidth=1,
            font=dict(size=10, color="#c0392b"),
        )
        fig_qd.update_layout(
            height=440, plot_bgcolor="#fafafa", paper_bgcolor="white",
            xaxis_title="Military Capacity Index (0–100)",
            yaxis_title="Fiscal Capacity per Capita (USD)",
            legend=dict(orientation="h", y=-0.18, x=0, font_size=11),
            margin=dict(t=20, b=70, l=75, r=20),
            font=dict(size=11),
        )
        st.plotly_chart(fig_qd, use_container_width=True)

        st.divider()

        # ── Chart row 2: Rank Gap diverging bar ──────────────────────────────
        st.markdown(
            "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
            "Military Rank vs Recovery Rank: Who Is Over or Underperforming?</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Gap = Military Rank minus Recovery Rank. "
            "Negative (red): state has strong military capacity but slow recovery, underperforming expectations. "
            "Positive (green): state recovers faster than its military rank predicts, overperforming expectations."
        )
        _rg = _mp.sort_values("rank_gap").copy()
        _bar_colors = [CAT_COLOR[c] for c in _rg["category"]]

        fig_gap = go.Figure(go.Bar(
            x=_rg["rank_gap"],
            y=_rg["abbr"],
            orientation="h",
            marker_color=_bar_colors,
            text=[f"{g:+d}" for g in _rg["rank_gap"]],
            textposition="outside",
            textfont=dict(size=10, color=_bar_colors),
            customdata=_rg[["state","rank_mil","rank_rec","recovery_months_predicted",
                              "military_capacity_index","category",
                              "fiscal_capacity_per_capita"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: <b>%{customdata[5]}</b><br>"
                "<br>"
                "Military rank: #%{customdata[1]} of 15"
                "  |  Recovery rank: #%{customdata[2]} of 15<br>"
                "Rank gap: <b>%{x:+d}</b><br>"
                "  Negative = strong military capacity, but slow economic recovery<br>"
                "  Positive = fast recovery relative to military rank<br>"
                "<br>"
                "Recovery time: %{customdata[3]:.1f} months<br>"
                "Military Capacity Index: %{customdata[4]:.1f} (0–100 composite)<br>"
                "Fiscal capacity: $%{customdata[6]:,.0f} / person<br>"
                "<extra></extra>"
            ),
        ))
        fig_gap.add_vline(x=0, line_width=2, line_color="#37474f")
        fig_gap.add_vrect(x0=-15, x1=0, fillcolor="#c0392b", opacity=0.04, layer="below")
        fig_gap.add_vrect(x0=0, x1=15, fillcolor="#27ae60", opacity=0.04, layer="below")
        fig_gap.add_annotation(x=-7, y=-0.7, text="◄  High mil / slow recovery",
            showarrow=False, font=dict(size=10, color="#c0392b"), xref="x", yref="y")
        fig_gap.add_annotation(x=7,  y=-0.7, text="Fast recovery / low mil  ►",
            showarrow=False, font=dict(size=10, color="#27ae60"), xref="x", yref="y")
        fig_gap.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Rank Gap (Military Rank − Recovery Rank)",
                       zeroline=False, gridcolor="#f0f0f0"),
            yaxis=dict(title="", tickfont=dict(size=11, color="#1a1a2e")),
            margin=dict(t=20, b=40, l=50, r=60),
            font=dict(size=11),
            showlegend=False,
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        st.divider()

        # ── Ranked state table ────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
            "All 15 States Ranked: Military Capacity vs Recovery Performance</p>",
            unsafe_allow_html=True,
        )
        _tbl = _mp.sort_values("rank_gap")[
            ["abbr","state","military_capacity_index","rank_mil",
             "recovery_months_predicted","rank_rec","rank_gap","category"]
        ].copy()
        _tbl.columns = ["Abbr","State","Military Index","Mil Rank",
                        "Recovery (mo)","Rec Rank","Gap","Category"]
        _tbl["Military Index"] = _tbl["Military Index"].round(1)
        _tbl["Recovery (mo)"]  = _tbl["Recovery (mo)"].round(1)

        def _style_row(row):
            c = "#fff3f3" if row["Category"]=="Underperforming" \
                else "#f0faf0" if row["Category"]=="Overperforming" else "#fffde7"
            return [f"background:{c}" for _ in row]

        st.dataframe(
            _tbl.style.apply(_style_row, axis=1).format({"Gap": "{:+d}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption("Sources: DoD Base Structure Report FY2019–2025; NTAD-MB (USDOT BTS); BLS LAUS; authors' DRPA model.")

        # ── Key findings ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:18px 24px;margin:20px 0 0 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 10px 0;'>Key Findings: Section 4.6</p>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:10px 14px;'>"
            "<p style='font-size:12px;font-weight:700;color:#F4D03F;margin:0 0 4px 0;'>Not a Recovery Predictor</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.6;'>"
            "Military capacity shows no statistically significant relationship with recovery speed "
            "(r = 0.23, p = 0.411). Having more bases does not mean faster recovery.</p></div>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:10px 14px;'>"
            "<p style='font-size:12px;font-weight:700;color:#F4D03F;margin:0 0 4px 0;'>Substitutes, Not Complements</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.6;'>"
            "Military and fiscal capacity are strong substitutes (r = −0.59, p = 0.020). "
            "States with high military presence tend to have fewer fiscal resources.</p></div>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:10px 14px;'>"
            "<p style='font-size:12px;font-weight:700;color:#F4D03F;margin:0 0 4px 0;'>Tennessee Paradox</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.6;'>"
            "Tennessee ranks #1 in military capacity but only #12 in recovery speed, "
            "a rank gap of 11, the largest mismatch in the group.</p></div>"
            "<div style='background:#ffffff0d;border-radius:6px;padding:10px 14px;'>"
            "<p style='font-size:12px;font-weight:700;color:#F4D03F;margin:0 0 4px 0;'>Fiscal Capacity Wins</p>"
            "<p style='font-size:12px;color:#ccc;margin:0;line-height:1.6;'>"
            "Nebraska (lowest military, fastest recovery) and Kansas confirm that fiscal "
            "and structural conditions drive outcomes, not military presence alone.</p></div>"
            "</div>"
            "<p style='font-size:11px;color:#888;margin:12px 0 0 0;font-style:italic;'>"
            "Sources: DoD BSR (ref. 36) · NTAD-MB USDOT BTS (ref. 33) · BLS LAUS · U.S. Census · authors' DRPA model</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── TAB 8: Policy Insights ────────────────────────────────────────────────
    with tab8:

        # ── Header ────────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 2px 0;'>"
            "Policy Insights: From Federal Mandate to Local Action</p>"
            "<p style='font-size:13px;color:#555;margin:0 0 16px 0;'>"
            "How national, state, local, and insurance-sector policies shape disaster recovery speed "
            "· Sources: FEMA · U.S. Census · State Statutes · Insurance Commissioners · BLS · NOAA</p>",
            unsafe_allow_html=True,
        )

        # ── Context banner ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:8px;padding:18px 24px;margin:0 0 20px 0;'>"
            "<p style='font-size:11px;font-weight:700;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.2px;margin:0 0 8px 0;'>The Policy-Recovery Connection</p>"
            "<p style='font-size:13px;color:#fff;margin:0 0 6px 0;line-height:1.8;'>"
            "Recovery speed is not determined by disaster severity alone; it is shaped by the layers "
            "of policy infrastructure a state has in place <b style='color:#F4D03F;'>before, during, and after</b> a disaster. "
            "Our DRPA model identifies fiscal capacity, structural conditions, and preparedness "
            "as the strongest predictors of recovery time, all of which are directly governed by policy."
            "</p>"
            "<p style='font-size:12px;color:#aaa;margin:0;line-height:1.8;'>"
            "This tab maps four policy tiers: Federal, State, Local, and Insurance/Private, "
            "and connects each layer to the recovery patterns identified across all 50 states."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Chart 1: Policy Timeline (full width) ─────────────────────────────
        st.markdown(
            "<p style='font-size:15px;font-weight:700;color:#0d1b2a;margin:0 0 2px 0;'>"
            "Key Policy Milestones and Major Disaster Events</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Federal policy evolution from 1988 to 2025. "
            "Each major reform was triggered by a gap exposed during a preceding disaster."
        )

        _milestones = [
            (1988, "Stafford Act",              "Foundation", "#2980B9"),
            (2006, "Post-Katrina Reform",        "Reform",     "#c0392b"),
            (2011, "Joplin + MS Floods",         "Event",      "#E67E22"),
            (2012, "Hurricane Sandy",            "Event",      "#E67E22"),
            (2017, "Harvey · Irma · Maria",      "Event",      "#E67E22"),
            (2018, "Disaster Recovery Reform",   "Reform",     "#c0392b"),
            (2020, "COVID-19 + Laura",           "Event",      "#E67E22"),
            (2022, "BRIC Expansion",             "Program",    "#1e8449"),
            (2024, "Risk Rating 2.0",            "Program",    "#1e8449"),
            (2025, "BRIC FY25 Competition",      "Emerging",   "#F4D03F"),
        ]
        _mil_y = [1.0 if i % 2 == 0 else -1.0 for i in range(len(_milestones))]

        fig_tl = go.Figure()
        fig_tl.add_shape(
            type="line", x0=1985, x1=2027, y0=0, y1=0,
            line=dict(color="#cccccc", width=2),
        )
        for i, (yr, lbl, typ, col) in enumerate(_milestones):
            fig_tl.add_shape(
                type="line", x0=yr, x1=yr, y0=0, y1=_mil_y[i],
                line=dict(color=col, width=1.5, dash="dot"),
            )
            fig_tl.add_trace(go.Scatter(
                x=[yr], y=[_mil_y[i]],
                mode="markers+text",
                marker=dict(color=col, size=11, line=dict(color="white", width=1.5)),
                text=[f"<b>{yr}</b><br>{lbl}"],
                textposition="top center" if _mil_y[i] > 0 else "bottom center",
                textfont=dict(size=9, color=col),
                hovertemplate=f"<b>{lbl}</b><br>Year: {yr}<br>Type: {typ}<extra></extra>",
                showlegend=False,
            ))

        for _typ, _tcol in [("Foundation","#2980B9"),("Reform","#c0392b"),
                             ("Event","#E67E22"),("Program","#1e8449"),("Emerging","#F4D03F")]:
            fig_tl.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=_tcol, size=9),
                name=_typ, showlegend=True,
            ))

        fig_tl.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(
                range=[1984, 2028],
                tickvals=list(range(1988, 2027, 4)),
                tickfont=dict(size=9), gridcolor="#f0f0f0", showgrid=True,
            ),
            yaxis=dict(visible=False, range=[-2.3, 2.3]),
            margin=dict(t=10, b=20, l=10, r=10),
            font=dict(size=9),
            legend=dict(
                orientation="h", y=-0.06, x=0.5, xanchor="center",
                font=dict(size=9), bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_tl, use_container_width=True, key="policy_timeline")

        # ── Chart 2: Recovery bar (full width, below timeline) ────────────────
        st.markdown(
            "<p style='font-size:22px;font-weight:800;color:#0d1b2a;margin:10px 0 2px 0;'>"
            "Recovery Time: Top 15 High-Exposure States</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Predicted recovery time colored by policy coverage depth. "
            "States with multi-layer policy infrastructure vs. federal-primary states."
        )

        _HE_STATES = [
            "Texas","Florida","Louisiana","California","Oklahoma","Tennessee",
            "Mississippi","Missouri","Georgia","Kentucky",
            "North Carolina","Virginia","South Carolina","Arkansas","Iowa",
        ]
        _MULTILAYER = {"Florida", "California"}
        _STATE_PLUS  = {"Texas", "Louisiana", "Georgia", "North Carolina", "Virginia"}

        _he = df[df["state"].isin(_HE_STATES)].copy() if not df.empty else pd.DataFrame()
        if not _he.empty and "recovery_months_predicted" in _he.columns:
            _he = _he.sort_values("recovery_months_predicted", ascending=True)

            def _pol_cat(s):
                if s in _MULTILAYER: return "Multi-Layer (Fed + State + Local + Insurance)"
                if s in _STATE_PLUS:  return "Federal + State/Local"
                return "Federal Primary"

            _he["pol_cat"] = _he["state"].apply(_pol_cat)
            # Green = most protected  |  Yellow = partial  |  Red = least layers
            _cat_colors_map = {
                "Multi-Layer (Fed + State + Local + Insurance)": "#117a65",
                "Federal + State/Local":                         "#c9a800",
                "Federal Primary":                               "#b03a2e",
            }
            _bar_cols_he   = [_cat_colors_map[c] for c in _he["pol_cat"]]
            _nat_avg_pol   = df["recovery_months_predicted"].mean() if not df.empty else 15.0
            _x_max_he      = _he["recovery_months_predicted"].max()

            fig_he = go.Figure(go.Bar(
                x=_he["recovery_months_predicted"],
                y=_he["state"],
                orientation="h",
                marker_color=_bar_cols_he,
                marker_line=dict(color="white", width=0.6),
                text=[f"  {v:.1f} mo" for v in _he["recovery_months_predicted"]],
                textposition="outside",
                textfont=dict(size=11, color="#333", family="Arial Black"),
                cliponaxis=False,
                customdata=np.stack((
                    _he["pol_cat"].values,
                    _he["drpi_risk_tier"].values,
                    _he["fiscal_capacity_per_capita"].fillna(0).values,
                    _he["avg_disaster_exposure_12m"].fillna(0).values,
                ), axis=-1),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Policy coverage: %{customdata[0]}<br>"
                    "Predicted recovery: <b>%{x:.1f} months</b><br>"
                    "DRPA risk tier: %{customdata[1]}<br>"
                    "Fiscal capacity: $%{customdata[2]:,.0f} / person<br>"
                    "Avg FEMA declarations / yr: %{customdata[3]:.0f}"
                    "<extra></extra>"
                ),
            ))
            fig_he.add_vline(
                x=_nat_avg_pol, line_dash="dash", line_color="#aaa", line_width=1.5,
                annotation_text=f"National avg  {_nat_avg_pol:.1f} mo",
                annotation_font_size=9, annotation_position="top right",
                annotation_font_color="#888",
            )
            fig_he.update_layout(
                height=450, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(
                    title="Predicted recovery (months)",
                    gridcolor="#f5f5f5", tickfont=dict(size=10),
                    range=[0, _x_max_he * 1.28],
                ),
                yaxis=dict(tickfont=dict(size=12, color="#1a1a2e")),
                margin=dict(t=10, b=50, l=10, r=50),
                font=dict(size=11), showlegend=False,
            )
            st.plotly_chart(fig_he, use_container_width=True, key="policy_recovery_bar")

            # Legend with colored squares
            st.markdown(
                "<div style='display:flex;gap:24px;align-items:center;"
                "background:#fafafa;border:1px solid #e8e8e8;border-radius:6px;"
                "padding:10px 18px;margin:4px 0 0 0;flex-wrap:wrap;'>"
                "<span style='font-size:13px;color:#117a65;font-weight:700;'>"
                "&#9632; Multi-Layer</span>"
                "<span style='font-size:12px;color:#555;'>Fed + State + Local + Insurance</span>"
                "<span style='font-size:18px;color:#ddd;'>|</span>"
                "<span style='font-size:13px;color:#c9a800;font-weight:700;'>"
                "&#9632; Federal + State/Local</span>"
                "<span style='font-size:12px;color:#555;'>Targeted state coverage</span>"
                "<span style='font-size:18px;color:#ddd;'>|</span>"
                "<span style='font-size:13px;color:#b03a2e;font-weight:700;'>"
                "&#9632; Federal Primary</span>"
                "<span style='font-size:12px;color:#555;'>National programs only</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Policy Tier Tabs ──────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:26px;font-weight:800;color:#0d1b2a;margin:0 0 4px 0;'>"
            "Policy Architecture: Four Tiers</p>"
            "<p style='font-size:13px;color:#888;margin:0 0 14px 0;'>"
            "Click a tab to explore each tier in detail.</p>",
            unsafe_allow_html=True,
        )

        _ptab1, _ptab2, _ptab3, _ptab4 = st.tabs([
            "Federal Policies",
            "State Policies",
            "Local Policies",
            "Private / Insurance Policies",
        ])

        # ── TAB: FEDERAL ──────────────────────────────────────────────────────
        with _ptab1:
            st.markdown(
                "<div style='background:#e8f4fd;border-top:4px solid #2980B9;border-radius:8px;"
                "padding:18px 22px;margin:8px 0 0 0;'>"
                "<p style='font-size:14px;font-weight:800;color:#1a4a7a;margin:0 0 6px 0;'>"
                "FEDERAL POLICIES &nbsp;·&nbsp; 9 Programs</p>"
                "<p style='font-size:12px;color:#2c3e50;margin:0 0 16px 0;line-height:1.8;'>"
                "Created and implemented by the federal government, these programs shape the largest "
                "disaster-response systems, covering nationwide funding, recovery coordination, and mitigation. "
                "Support reaches a broader scale but can be slower to arrive at local communities due to "
                "declaration requirements and multi-agency approval processes."
                "</p>"
                "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>Stafford Act (1988)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Main legal foundation for federal disaster aid, determining when states and communities "
                "are allowed to access major recovery assistance.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>Post-Katrina Reform Act (2006)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Reorganized FEMA after Hurricane Katrina. Strengthened federal preparedness and "
                "coordination capacity during major natural disasters.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>National Disaster Recovery Framework (NDRF)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Provides the overall structure for long-term recovery, moving states and communities "
                "from immediate response into organized rebuilding.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>Recovery Support Functions (RSFs)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Breaks recovery into major sectors including housing, infrastructure, health, and economy, "
                "enabling coordinated multi-agency response.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>EMAC</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Emergency Management Assistance Compact that allows states to send personnel, equipment, "
                "and mutual support to overwhelmed states, speeding response capacity.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>HMGP · BRIC · FMA</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Mitigation grant programs funding pre- and post-disaster resilience projects that "
                "reduce future damage and shorten recovery timelines.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>NFIP + Risk Rating 2.0</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Federally backed flood insurance linking coverage to floodplain management standards. "
                "Risk Rating 2.0 prices flood exposure at the individual property level.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>Community Rating System (CRS)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Rewards communities that exceed minimum floodplain standards with reduced flood "
                "insurance premiums for all residents in that community.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #2980B9;'>"
                "<p style='font-size:12px;font-weight:800;color:#1a4a7a;margin:0 0 4px 0;'>Disaster Recovery Reform Act (2018)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Shifted federal policy toward mitigation and resilience, reinforcing that faster "
                "recovery depends on planning before disasters happen, not just response after.</p></div>"

                "</div>"
                "<p style='font-size:11px;color:#888;margin:14px 0 0 0;font-style:italic;'>"
                "Data connection: Stafford Act declarations drive avg_disaster_exposure_12m in the DRPA model. "
                "HMGP/BRIC investments build fiscal_capacity_per_capita over time."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── TAB: STATE ────────────────────────────────────────────────────────
        with _ptab2:
            st.markdown(
                "<div style='background:#fef9e7;border-top:4px solid #E67E22;border-radius:8px;"
                "padding:18px 22px;margin:8px 0 0 0;'>"
                "<p style='font-size:14px;font-weight:800;color:#a04000;margin:0 0 6px 0;'>"
                "STATE POLICIES &nbsp;·&nbsp; 6 Programs</p>"
                "<p style='font-size:12px;color:#555;margin:0 0 16px 0;line-height:1.8;'>"
                "State-level policies respond to each state's individual risks, budget capacity, and recovery "
                "priorities. They shape how quickly states prepare for disasters, distribute aid, manage "
                "insurance markets, and support rebuilding that goes beyond what federal programs can provide."
                "</p>"
                "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px;'>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>Florida Emergency Management Act (Ch. 252, FL Statutes)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Establishes Florida's emergency management system and gives state and local governments "
                "the authority to coordinate disaster response and recovery operations.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>My Safe Florida Home Program</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Provides free inspections and grants to strengthen homes against hurricanes — "
                "directly reduces structural damage and improves household-level recovery speed after storms.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>Florida Hurricane Catastrophe Fund (FHCF)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Provides reimbursement support to insurers after catastrophic hurricane losses, "
                "helping stabilize Florida's property insurance market after major disasters.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>California FAIR Plan</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Acts as a backup coverage option for residents and businesses that cannot get "
                "insurance in the regular market, especially critical in high-risk wildfire areas.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>California Sustainable Insurance Strategy</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Tries to expand insurance availability in wildfire-distressed areas while reducing "
                "over-reliance on the FAIR plan, improving broader market access and stability.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #E67E22;'>"
                "<p style='font-size:12px;font-weight:800;color:#a04000;margin:0 0 4px 0;'>Safer from Wildfires (CA)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Links wildfire mitigation actions taken by homeowners and communities to insurance "
                "incentives, encouraging prevention and hardening before losses occur.</p></div>"

                "</div>"
                "<p style='font-size:11px;color:#888;margin:14px 0 0 0;font-style:italic;'>"
                "Data connection: Florida and California are the only 'Multi-Layer' states in the recovery bar chart above. "
                "Their state-level insurance stabilization programs contribute to the service_share_pct and fiscal_capacity_per_capita features."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── TAB: LOCAL ────────────────────────────────────────────────────────
        with _ptab3:
            st.markdown(
                "<div style='background:#f0faf0;border-top:4px solid #1e8449;border-radius:8px;"
                "padding:18px 22px;margin:8px 0 0 0;'>"
                "<p style='font-size:14px;font-weight:800;color:#145a32;margin:0 0 6px 0;'>"
                "LOCAL POLICIES &nbsp;·&nbsp; 6 Programs</p>"
                "<p style='font-size:12px;color:#555;margin:0 0 16px 0;line-height:1.8;'>"
                "County and city-level policies are the most directly connected to what residents experience "
                "on the ground, including zoning decisions, floodplain management, evacuation planning, "
                "rebuilding rules, and community-level resilience projects that shape day-to-day recovery."
                "</p>"
                "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px;'>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>FEMA-Approved Local Hazard Mitigation Plans</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Helps communities identify local risks and prioritize mitigation projects that reduce "
                "future losses, directly supporting faster and more stable long-term recoveries.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>Community Rating System (CRS) Participation</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Allows local governments to lower residents' flood insurance costs by adopting stronger "
                "floodplain management practices and resilience measures beyond federal minimums.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>Miami-Dade Sea Level Rise Strategy</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Focuses on long-term adaptation planning for rising sea levels and coastal flooding — "
                "helping one of the highest-risk regions in the country reduce future recovery strain.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>Harris County Voluntary Buyout Program</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Reduces repeated flood losses by permanently relocating residents out of high-risk "
                "floodplain areas instead of repeatedly rebuilding in the same location.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>New Orleans Hazard Mitigation Strategy</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Prioritizes long-term local actions that reduce losses from hazards and strengthen "
                "recovery resilience across infrastructure and communities post-Katrina.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #1e8449;'>"
                "<p style='font-size:12px;font-weight:800;color:#145a32;margin:0 0 4px 0;'>Los Angeles Wildfire Rebuilding Executive Orders (2025)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Accelerates post-wildfire rebuilding while also embedding resilience standards to "
                "reduce exposure and damage from future fire-related disasters.</p></div>"

                "</div>"
                "<p style='font-size:11px;color:#888;margin:14px 0 0 0;font-style:italic;'>"
                "Data connection: CRS participation and Hazard Mitigation Plans directly reduce avg_damage_per_capita "
                "and avg_disaster_exposure_12m over time, representing two key DRPA model features."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── TAB: INSURANCE / PRIVATE ──────────────────────────────────────────
        with _ptab4:
            st.markdown(
                "<div style='background:#fdf3f3;border-top:4px solid #c0392b;border-radius:8px;"
                "padding:18px 22px;margin:8px 0 0 0;'>"
                "<p style='font-size:14px;font-weight:800;color:#922b21;margin:0 0 6px 0;'>"
                "PRIVATE / INSURANCE POLICIES &nbsp;·&nbsp; 6 Programs</p>"
                "<p style='font-size:12px;color:#555;margin:0 0 16px 0;line-height:1.8;'>"
                "Contained within the private sector and insurance systems, these policies are shaped "
                "by both state and federal regulation. They determine how quickly individuals, homeowners, "
                "and businesses can recover financially after a disaster by governing coverage availability, "
                "premiums, claims processing, and access to protection in high-risk areas."
                "</p>"
                "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px;'>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>Citizens Property Insurance Corporation (FL)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Florida's insurer of last resort, ensuring homeowners can maintain coverage when the "
                "private market pulls back or exits after repeated major losses.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>Private Homeowners and Commercial Property Insurance</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Determines how quickly households and businesses can repair damage, reopen operations, "
                "and financially stabilize after a natural disaster strikes.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>State Insurer-of-Last-Resort Programs</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Provides backup insurance access in high-risk markets when private carriers reduce "
                "availability or exit the market altogether, preventing coverage gaps in disaster-prone areas.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>State Catastrophe Backstops / Reinsurance</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Help insurers absorb extreme losses after major events, reducing the probability of "
                "wider insurance market instability following large-scale natural disasters.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>Mitigation-Linked Insurance Discounts</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Encourages households to harden homes before disasters by tying resilience upgrades "
                "to possible premium savings or improved access to insurability.</p></div>"

                "<div style='background:#fff;border-radius:6px;padding:14px 16px;border-left:4px solid #c0392b;'>"
                "<p style='font-size:12px;font-weight:800;color:#922b21;margin:0 0 4px 0;'>Property Insurance Clearinghouse (FL)</p>"
                "<p style='font-size:12px;color:#444;margin:0;line-height:1.7;'>"
                "Helps move consumers from Citizens Insurance into private-market options when available — "
                "improving overall market flexibility and reducing state liability after disasters.</p></div>"

                "</div>"
                "<p style='font-size:11px;color:#888;margin:14px 0 0 0;font-style:italic;'>"
                "Data connection: Insurance market stability influences fiscal_capacity_per_capita and compound_vulnerability_index "
                "by determining whether households and businesses can self-fund early recovery without government assistance."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Emerging Policies ─────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:22px;font-weight:800;color:#0d1b2a;margin:0 0 6px 0;'>"
            "Emerging Policies: What Comes Next</p>"
            "<p style='font-size:13px;color:#888;margin:0 0 14px 0;'>"
            "Active proposals and pilot programs across federal, state, and insurance sectors.</p>",
            unsafe_allow_html=True,
        )
        _em_cols = st.columns(5)
        _emerging = [
            ("FY 2025 BRIC Competition",
             "#2980B9", "Federal",
             "Pre-disaster mitigation grants — new round of awards funding community resilience projects nationwide."),
            ("FL 2026 My Safe Florida Home Proposals",
             "#E67E22", "State",
             "Expanding the home-hardening grant program with increased funding caps for hurricane mitigation."),
            ("FL My Safe Florida Condo Pilot",
             "#E67E22", "State",
             "Extends hurricane mitigation inspections and grants to condominium associations statewide."),
            ("California Safe Homes Act",
             "#1e8449", "State",
             "Legislation linking home wildfire-hardening standards to insurance availability and premium rates."),
            ("CA Sustainable Insurance Strategy Rollout",
             "#c0392b", "Insurance",
             "Phased rollout expanding private market access in wildfire-distressed zip codes across California."),
        ]
        for _ecol_obj, (_ename, _ecol, _ecat, _edesc) in zip(_em_cols, _emerging):
            with _ecol_obj:
                st.markdown(
                    f"<div style='background:{_ecol}0e;border:1px solid {_ecol}55;"
                    f"border-top:4px solid {_ecol};border-radius:8px;"
                    f"padding:16px 16px 14px 16px;height:180px;'>"
                    f"<p style='font-size:10px;font-weight:800;color:{_ecol};text-transform:uppercase;"
                    f"letter-spacing:1px;margin:0 0 6px 0;'>{_ecat}</p>"
                    f"<p style='font-size:13px;font-weight:700;color:#0d1b2a;"
                    f"margin:0 0 8px 0;line-height:1.4;'>{_ename}</p>"
                    f"<p style='font-size:11px;color:#555;margin:0;line-height:1.6;'>{_edesc}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Key Findings / Policy Implications ───────────────────────────────
        st.markdown(
            "<div style='background:#0d1b2a;border-radius:10px;padding:28px 30px;margin:24px 0 0 0;'>"
            "<p style='font-size:13px;font-weight:800;color:#F4D03F;text-transform:uppercase;"
            "letter-spacing:1.4px;margin:0 0 20px 0;'>Policy Implications from the DRPA Model</p>"
            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;'>"

            "<div style='background:#ffffff10;border-radius:8px;border-top:3px solid #F4D03F;"
            "padding:18px 20px;'>"
            "<p style='font-size:14px;font-weight:800;color:#F4D03F;margin:0 0 10px 0;line-height:1.4;'>"
            "Pre-Disaster Investment Pays Off</p>"
            "<p style='font-size:13px;color:#ddd;margin:0;line-height:1.8;'>"
            "States with strong fiscal capacity recover faster. Policies that fund mitigation and "
            "resilience before disasters reduce the Compound Vulnerability Index scores that our "
            "model identifies as the top driver of slow recovery, adding an average of "
            "<b style='color:#F4D03F;'>+2.96 months</b> to predicted recovery time.</p></div>"

            "<div style='background:#ffffff10;border-radius:8px;border-top:3px solid #F4D03F;"
            "padding:18px 20px;'>"
            "<p style='font-size:14px;font-weight:800;color:#F4D03F;margin:0 0 10px 0;line-height:1.4;'>"
            "Service-Heavy States Need Longer Aid Windows</p>"
            "<p style='font-size:13px;color:#ddd;margin:0;line-height:1.8;'>"
            "Service-dominated states (NY, NJ, FL, CA) average <b style='color:#F4D03F;'>16.2 months</b> "
            "to recover compared to <b style='color:#F4D03F;'>11.5 months</b> for industrial states, "
            "a 4.7-month gap. Policy aid windows should reflect economic structure, "
            "not just disaster severity.</p></div>"

            "<div style='background:#ffffff10;border-radius:8px;border-top:3px solid #F4D03F;"
            "padding:18px 20px;'>"
            "<p style='font-size:14px;font-weight:800;color:#F4D03F;margin:0 0 10px 0;line-height:1.4;'>"
            "Aid Volume Does Not Equal Recovery Speed</p>"
            "<p style='font-size:13px;color:#ddd;margin:0;line-height:1.8;'>"
            "Top aid recipients (LA, FL, TX) are among the slowest to recover "
            "(r = +0.62, no significant effect). <b style='color:#F4D03F;'>Nebraska</b> received "
            "the least aid and recovered fastest at just 4 months. More aid without structural "
            "reform does not accelerate recovery.</p></div>"

            "</div>"
            "<p style='font-size:11px;color:#666;margin:20px 0 0 0;font-style:italic;'>"
            "Sources: FEMA · BLS LAUS · NOAA Storm Events · BEA SAGDP2 · U.S. Census · "
            "Disaster Recovery Reform Act (2018) · policies.md · authors' DRPA model (DTSC 4302)</p>"
            "</div>",
            unsafe_allow_html=True,
        )

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
        f"after a major disaster, <b>{_vs_avg}</b>. "
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
    st.markdown(
        "<p style='font-size:11px;font-weight:700;color:#888;text-transform:uppercase;"
        "letter-spacing:1px;margin:14px 0 8px 0;'>Step 4: Supporting Data</p>",
        unsafe_allow_html=True,
    )

    # Context vs national averages
    _avg_exp  = df["avg_disaster_exposure_12m"].mean()
    _avg_dmg  = df["avg_damage_per_capita"].mean()
    _avg_fisc = df["fiscal_capacity_per_capita"].mean()
    _avg_cvi  = df["compound_vulnerability_index"].mean()

    _exp_val  = float(row.get("avg_disaster_exposure_12m", 0))
    _dmg_val  = float(row.get("avg_damage_per_capita", 0))
    _fisc_val = float(row.get("fiscal_capacity_per_capita", 0)) if pd.notna(row.get("fiscal_capacity_per_capita")) else 0
    _cvi_val  = float(row.get("compound_vulnerability_index", 0))

    def _vs_nat(val, avg, invert=False):
        """Return color + label vs national average."""
        diff = val - avg
        pct  = abs(diff / avg * 100) if avg else 0
        if invert:
            col  = "#27ae60" if diff > 0 else "#c0392b"
            lbl  = ("above" if diff > 0 else "below") + f" national avg (better)"
        else:
            col  = "#c0392b" if diff > 0 else "#27ae60"
            lbl  = ("above" if diff > 0 else "below") + f" national avg"
        return col, f"{pct:.0f}% {lbl}"

    _ec, _el = _vs_nat(_exp_val,  _avg_exp)
    _dc, _dl = _vs_nat(_dmg_val,  _avg_dmg)
    _fc, _fl = _vs_nat(_fisc_val, _avg_fisc, invert=True)
    _cc, _cl = _vs_nat(_cvi_val,  _avg_cvi)

    _cvi_tier = ("High vulnerability"    if _cvi_val > 0.60
                 else "Moderate vulnerability" if _cvi_val > 0.35
                 else "Low vulnerability")

    _d4c1, _d4c2, _d4c3, _d4c4 = st.columns(4)

    def _data_card(col_ctx, border_color, label, big_value, sub_note, context_color, context_label):
        with col_ctx:
            st.markdown(
                f"<div style='border:1px solid #e0e0e0;border-top:3px solid {border_color};"
                f"border-radius:6px;padding:12px 14px;background:#fff;height:130px;'>"
                f"<p style='font-size:10px;font-weight:700;color:#888;text-transform:uppercase;"
                f"letter-spacing:0.9px;margin:0 0 4px 0;'>{label}</p>"
                f"<p style='font-size:22px;font-weight:900;color:#0d1b2a;margin:0 0 2px 0;"
                f"white-space:nowrap;overflow:hidden;text-overflow:clip;'>{big_value}</p>"
                f"<p style='font-size:11px;color:#555;margin:0 0 4px 0;line-height:1.5;'>{sub_note}</p>"
                f"<p style='font-size:10px;font-weight:600;color:{context_color};margin:0;'>"
                f"{context_label}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _data_card(
        _d4c1, _ec,
        "FEMA Disaster Exposure",
        f"{_exp_val:.0f} decl / yr",
        "Avg federally declared disasters per year. Higher = more frequent shocks.",
        _ec, _el,
    )
    _data_card(
        _d4c2, _dc,
        "Avg Economic Damage / Capita",
        f"${_dmg_val:,.0f} / person",
        "Annual avg storm damage per resident (NOAA). Reflects physical shock severity.",
        _dc, _dl,
    )
    _data_card(
        _d4c3, _fc,
        "Fiscal Capacity",
        f"${_fisc_val:,.0f} / person",
        "State revenue per capita (Census). Higher = more funding for disaster response.",
        _fc, _fl,
    )
    _data_card(
        _d4c4, _cc,
        "Compound Vulnerability (CVI)",
        f"{_cvi_val:.3f}",
        f"{_cvi_tier}. Composite of labor, disaster, damage, fiscal, structural and education risk (0 to 1).",
        _cc, _cl,
    )

    st.divider()

    col1, col2 = st.columns(2)

    # ── CVI Radar Chart ───────────────────────────────────────────────────────
    with col1:
        st.markdown("**Compound Vulnerability Index: 6 Dimensions**")
        st.caption("Higher CVI = higher vulnerability and slower recovery. Combines exposure, fiscal strength, structural risk, and education disruption.")
        cvi_dims = ["cvi_labor","cvi_disaster","cvi_damage","cvi_fiscal","cvi_structure","cvi_education"]
        cvi_labels = ["Labor Volatility","Disaster Exposure","Damage Burden","Fiscal Weakness","Structural Risk","Education Disruption"]
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
        st.markdown("**What Drives This State's Recovery Prediction?**")
        if not shap_df.empty and "state" in shap_df.columns:
            s_row = shap_df[shap_df["state"]==selected]
            if not s_row.empty:
                s_row = s_row.iloc[0]
                shap_cols  = [c for c in shap_df.columns if c.startswith("shap_")]
                feat_names = [c.replace("shap_","") for c in shap_cols]
                shap_vals  = [float(s_row[c]) for c in shap_cols]
                base_val   = float(s_row.get("base_value", df["recovery_months_predicted"].mean()))

                # Human-readable feature labels
                FEATURE_LABELS = {
                    "avg_disaster_exposure_12m":   "Disaster Exposure Frequency",
                    "avg_damage_per_capita":        "Average Damage Per Capita",
                    "fiscal_capacity_per_capita":   "Fiscal Capacity Per Capita",
                    "unemployment_shock_magnitude": "Unemployment Shock Magnitude",
                    "service_share_pct":            "Service Sector Share",
                    "compound_vulnerability_index": "Compound Vulnerability Index",
                }

                # Sort by absolute impact
                order = np.argsort(np.abs(shap_vals))[::-1]
                sorted_feats  = [FEATURE_LABELS.get(feat_names[i], feat_names[i]) for i in order]
                sorted_shap   = [shap_vals[i]  for i in order]
                colors = ["#C0392B" if v>0 else "#27AE60" for v in sorted_shap]

                def fmt_label(v):
                    months = abs(v)
                    days = round(months * 30)
                    sign = "+" if v > 0 else "-"
                    if months >= 1:
                        return f"{v:+.2f} ({sign}{months:.1f} mo)"
                    else:
                        return f"{v:+.2f} ({sign}{days} days)"

                fig_wf = go.Figure(go.Bar(
                    x=sorted_shap, y=sorted_feats,
                    orientation="h", marker_color=colors,
                    text=[fmt_label(v) for v in sorted_shap],
                    textposition="auto",
                    insidetextanchor="middle",
                ))
                fig_wf.add_vline(x=0, line_width=1.5, line_color="black")
                fig_wf.update_layout(
                    title=f"Feature Impact on Predicted Recovery<br><sup>Base prediction = {base_val:.1f} months</sup>",
                    xaxis_title="Impact on Predicted Recovery (Months)",
                    height=360, paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(zeroline=True,
                               range=[min(min(sorted_shap)*1.4, -0.5),
                                      max(max(sorted_shap)*1.4,  0.5)]),
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
            "CVI: Education Disruption": f"{row.get('cvi_education',0):.3f}",
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
