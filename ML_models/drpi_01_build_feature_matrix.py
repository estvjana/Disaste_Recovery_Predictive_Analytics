#!/usr/local/bin/python3.12
"""
=============================================================================
DRPI — Disaster Recovery Predictive models
Step 01: Build Per-State Feature Matrix  (v2)
=============================================================================
NEW in v2:
  - Adds fiscal_capacity_per_capita from Census State Finance data
  - Adds compound_vulnerability_index (multi-sector composite per paper)
  - Fixes duplicate states; limits to 50 US states
  - Adds profile_avg_recovery_months (paper benchmarks per economic type)

Output: state_feature_matrix.csv
Run:    python3.12 ML_models/drpi_01_build_feature_matrix.py
=============================================================================
"""
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLS_FILE    = os.path.join(ROOT,"data","BLS","laus_with_fema_disaster_exposure_2006_2025.csv")
DAMAGE_FILE = os.path.join(ROOT,"state_year_damage_per_capita.csv")
RECOVERY    = os.path.join(ROOT,"recovery_speed_results.csv")
BEA_DIR     = os.path.join(ROOT,"data","BEA","BEA_StateGDP_1997_2024")
CENSUS_DIR  = os.path.join(ROOT,"data","Census","Census_StateFinances_2006_2021")
POP_FILE    = os.path.join(ROOT,"data","Census","state_population_acs_2010_2024.csv")
OUTPUT      = os.path.join(ROOT,"state_feature_matrix.csv")

YEAR_START = 2010;  YEAR_END = 2024
BEA_YEARS  = ["2019","2020","2021","2022","2023"]

ABBR_TO_STATE = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}
STATE_TO_ABBR = {v:k for k,v in ABBR_TO_STATE.items()}

print("="*65)
print("DRPI Step 01 — Feature Matrix v2 (+ Fiscal Capacity + CVI)")
print("="*65)

# ── 1. BLS / FEMA ─────────────────────────────────────────────────────────────
print("\n[1/5] BLS + FEMA features...")
bls = pd.read_csv(BLS_FILE)
bls["state"] = bls["state"].str.strip().str.title()
bls = bls[(bls["year"] >= YEAR_START) & (bls["year"] <= YEAR_END)]
bls_nc = bls[~bls["year"].isin([2020,2021])]

fema = (
    bls.groupby("state").agg(
        avg_unemployment_rate     =("unemployment_rate",    "mean"),
        unemployment_volatility   =("unemployment_rate",    "std"),
        avg_disaster_exposure_12m =("disaster_exposure_12m","mean"),
        max_disaster_exposure_12m =("disaster_exposure_12m","max"),
        avg_disaster_count_month  =("disaster_count_month", "mean"),
        total_disaster_months     =("disaster_count_month", "sum"),
    ).reset_index()
)
vol_nc = (
    bls_nc.groupby("state")["unemployment_rate"].std()
    .reset_index().rename(columns={"unemployment_rate":"unemployment_volatility_excl_covid"})
)
fema = fema.merge(vol_nc, on="state", how="left")
print(f"  ✓ {len(fema)} states")

# ── 2. NOAA Damage ────────────────────────────────────────────────────────────
print("\n[2/5] NOAA damage per capita...")
dmg = pd.read_csv(DAMAGE_FILE)
dmg["state"] = dmg["state"].str.strip().str.title()
dmg = dmg[(dmg["year"] >= YEAR_START) & (dmg["year"] <= YEAR_END)]
dmg = dmg.drop_duplicates(subset=["state","year"])
damage = (
    dmg.groupby("state").agg(
        avg_damage_per_capita   =("damage_per_capita","mean"),
        max_damage_per_capita   =("damage_per_capita","max"),
        total_damage_per_capita =("damage_per_capita","sum"),
        years_with_damage       =("damage_per_capita", lambda x:(x>0).sum()),
    ).reset_index()
)
print(f"  ✓ {len(damage)} states")

# ── 3. Recovery Speed ─────────────────────────────────────────────────────────
print("\n[3/5] COVID recovery speed...")
rec = pd.read_csv(RECOVERY)
rec["state"] = rec["state"].str.strip().str.title()
rec["abbr"]  = rec["state"].map(STATE_TO_ABBR)
rec["unemployment_shock_magnitude"] = rec["shock_unemployment"] - rec["baseline_unemployment"]
rec_feat = rec[["state","abbr","baseline_unemployment","shock_unemployment",
                "recovery_months","unemployment_shock_magnitude"]].copy()
print(f"  ✓ {len(rec_feat)} states")

# ── 4. BEA Industry Composition ───────────────────────────────────────────────
print("\n[4/5] BEA industry composition...")

def extract_bea(abbr):
    fp = os.path.join(BEA_DIR, f"SAGDP2_{abbr}_1997_2024.csv")
    if not os.path.exists(fp): return None
    try:
        df = pd.read_csv(fp, encoding="latin1", low_memory=False)
        df["Description"] = df["Description"].str.strip()
        def gv(desc):
            row = df[df["Description"]==desc]
            if row.empty: return 0.0
            vals = []
            for y in BEA_YEARS:
                if y not in row.columns: continue
                try:
                    v = float(str(row[y].values[0]).replace(",",""))
                    if not np.isnan(v): vals.append(v)
                except: pass
            return float(np.mean(vals)) if vals else 0.0
        total = gv("All industry total")
        if total == 0: return None
        return {
            "abbr":abbr,
            "service_share_pct":   round(100*sum(gv(d) for d in [
                "Finance, insurance, real estate, rental, and leasing",
                "Professional and business services",
                "Educational services, health care, and social assistance",
                "Retail trade","Information","Wholesale trade",
                "Other services (except government and government enterprises)"])/total,2),
            "industry_share_pct":  round(100*sum(gv(d) for d in [
                "Manufacturing","Construction","Utilities"])/total,2),
            "natres_share_pct":    round(100*sum(gv(d) for d in [
                "Agriculture, forestry, fishing and hunting",
                "Mining, quarrying, and oil and gas extraction"])/total,2),
            "tourism_share_pct":   round(100*gv(
                "Arts, entertainment, recreation, accommodation, and food services")/total,2),
            "govt_share_pct":      round(100*gv(
                "Government and government enterprises")/total,2),
        }
    except: return None

bea_rows = [extract_bea(a) for a in ABBR_TO_STATE]
bea_df   = pd.DataFrame([r for r in bea_rows if r])
bea_df["state"] = bea_df["abbr"].map(ABBR_TO_STATE)

def classify_profile(row):
    if row["natres_share_pct"]  >= 4.0:  return "Resource-Intensive"
    if row["industry_share_pct"]>= 20.0: return "Industry-Intensive"
    if row["govt_share_pct"]    >= 15.0: return "Government-Intensive"
    return "Service-Dominated"

bea_df["economic_profile"] = bea_df.apply(classify_profile, axis=1)
# Paper benchmarks: Industry=11.5mo, Service=16.2mo, Resource=14.4mo, Govt=18.0mo
PROFILE_AVG = {"Industry-Intensive":11.5,"Service-Dominated":16.2,
               "Resource-Intensive":14.4,"Government-Intensive":18.0}
bea_df["profile_benchmark_months"] = bea_df["economic_profile"].map(PROFILE_AVG)
print(f"  ✓ {len(bea_df)} states")

# ── 5. Fiscal Capacity ────────────────────────────────────────────────────────
print("\n[5/5] Fiscal capacity from Census State Finances...")
try:
    census_file = max(glob.glob(os.path.join(CENSUS_DIR,"GOVSSTATEFINTIMESERIES*.xlsx")))
    fc_raw = pd.read_excel(census_file, sheet_name="Data")
    fc_raw["Geographic Area Name"] = fc_raw["Geographic Area Name"].str.strip()

    # SF0001 = Total State Revenue
    rev = fc_raw[
        (fc_raw["Aggregate Description"]=="SF0001") &
        (fc_raw["Type of Government"]==2) &
        (fc_raw["Geographic Area Name"]!="United States")
    ][["Geographic Area Name","Amount"]].copy()
    rev.columns = ["state","total_revenue_str"]
    rev["total_revenue_thousands"] = (
        rev["total_revenue_str"].astype(str)
        .str.replace(",","").str.strip()
        .apply(lambda x: float(x) if x.replace(".","").isdigit() else np.nan)
    )
    rev = rev.dropna(subset=["total_revenue_thousands"])

    pop = pd.read_csv(POP_FILE)
    pop["state"] = pop["state"].str.strip().str.title()
    pop_latest = pop.sort_values("year").groupby("state")["population"].last().reset_index()

    fiscal = rev.merge(pop_latest, on="state", how="left")
    fiscal["fiscal_capacity_per_capita"] = (
        (fiscal["total_revenue_thousands"] * 1000) / fiscal["population"]
    ).round(0)
    fiscal = fiscal[["state","fiscal_capacity_per_capita"]].dropna()
    print(f"  ✓ {len(fiscal)} states — median ${fiscal['fiscal_capacity_per_capita'].median():,.0f}/person")
except Exception as e:
    print(f"  ⚠ Fiscal load failed ({e}) — using median imputation")
    med = 6500.0
    fiscal = pd.DataFrame({"state":list(ABBR_TO_STATE.values()),
                           "fiscal_capacity_per_capita":[med]*50})

# ── ASSEMBLE ─────────────────────────────────────────────────────────────────
print("\n── Assembling & computing Compound Vulnerability Index...")
matrix = (
    fema
    .merge(damage,   on="state", how="outer")
    .merge(rec_feat.drop(columns=["abbr"], errors="ignore"), on="state", how="left")
    .merge(bea_df[["state","service_share_pct","industry_share_pct",
                   "natres_share_pct","tourism_share_pct","govt_share_pct",
                   "economic_profile","profile_benchmark_months"]], on="state", how="left")
    .merge(fiscal,   on="state", how="left")
)
# Derive abbr cleanly from state name
matrix["abbr"] = matrix["state"].map(STATE_TO_ABBR)

# Limit to 50 states
matrix = matrix[matrix["state"].isin(ABBR_TO_STATE.values())].copy()

# ── Compound Vulnerability Index (CVI) ────────────────────────────────────────
# Five dimensions from paper — each normalised 0-1, weighted average
def mm(s):
    mn,mx = s.min(), s.max()
    return ((s-mn)/(mx-mn+1e-9)).clip(0,1)

v_labor    = mm(matrix["unemployment_volatility_excl_covid"].fillna(matrix["unemployment_volatility_excl_covid"].median()))
v_disaster = mm(matrix["avg_disaster_exposure_12m"].fillna(0))
v_damage   = mm(matrix["avg_damage_per_capita"].fillna(0))
v_fiscal   = 1 - mm(matrix["fiscal_capacity_per_capita"].fillna(matrix["fiscal_capacity_per_capita"].median()))
v_struct   = mm(matrix["service_share_pct"].fillna(60))   # higher svc = slower recovery

matrix["cvi_labor"]    = v_labor.round(4)
matrix["cvi_disaster"] = v_disaster.round(4)
matrix["cvi_damage"]   = v_damage.round(4)
matrix["cvi_fiscal"]   = v_fiscal.round(4)
matrix["cvi_structure"]= v_struct.round(4)
matrix["compound_vulnerability_index"] = (
    v_labor*0.25 + v_disaster*0.25 + v_damage*0.20 + v_fiscal*0.15 + v_struct*0.15
).round(4)

num = matrix.select_dtypes("number").columns
matrix[num] = matrix[num].round(4)
matrix = matrix.sort_values("state").reset_index(drop=True)
matrix.to_csv(OUTPUT, index=False)

print(f"\n{'='*65}")
print(f"✓ Feature matrix → {OUTPUT}")
print(f"  Rows: {len(matrix)}  |  Features: {len(matrix.columns)}")
print(f"\n  Columns: {matrix.columns.tolist()}")
print(f"\n  CVI sample (top 5 most vulnerable):")
print(matrix.nlargest(5,"compound_vulnerability_index")
      [["state","compound_vulnerability_index","fiscal_capacity_per_capita",
        "avg_disaster_exposure_12m","economic_profile"]].to_string(index=False))
print("="*65)
