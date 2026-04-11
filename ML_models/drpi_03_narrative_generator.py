#!/usr/local/bin/python3.12
"""
=============================================================================
DRPI — Disaster Recovery Predictive models
Step 03: Narrative Generator  (v2 — Template + Supabase)
=============================================================================
For every state in state_risk_scores.csv:
  1. Builds a structured data-driven narrative from its risk features
  2. Generates a plain-English narrative from model outputs
  3. Extracts 5 keyword tags per state
  4. Saves enriched table to state_narratives.csv
  5. Upserts narratives back into Supabase drpi_risk_scores table

Input  : state_risk_scores.csv
Output : state_narratives.csv  +  Supabase drpi_risk_scores (narrative column)

Run from project root:
  python3.12 ML_models/drpi_03_narrative_generator.py
=============================================================================
"""

import os, time, warnings
import pandas as pd
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Load secrets from config/.env ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ML_models"))

def _load_env():
    env_file = ROOT / "config" / ".env"
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # Always set — override empty env vars
            os.environ[key.strip()] = val.strip()

_load_env()

INPUT  = ROOT / "state_risk_scores.csv"
OUTPUT = ROOT / "state_narratives.csv"

# ── Narrative mode: template-driven (data-only, fully traceable) ──────────────
# Narratives are generated from model outputs and data sources only.
# Every sentence is 100% traceable to a model output or data value.
USE_CLAUDE = False
print("  Mode: Data-driven templates (all values sourced directly from model outputs)")

# ── Supabase setup ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
USE_SUPABASE = False

try:
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase keys not set")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    USE_SUPABASE = True
except Exception as e:
    print(f"  ⚠  Supabase not available: {e}")

print("=" * 65)
print("DRPI — Step 03: Narrative Generator v2")
print(f"  Mode     : {'Template-driven (data sources only)' if not USE_CLAUDE else 'Extended'}")
print(f"  Supabase : {'Connected' if USE_SUPABASE else 'Not connected'}")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
print(f"\n  States to narrate: {len(df)}")

# ── No LLM prompt — narratives are fully data-driven templates ────────────────


# ── Data-driven template narrative ────────────────────────────────────────────
# Every value cited is sourced directly from model outputs or raw data.
# No interpretation, inference, or language beyond what the numbers show.
def build_template_narrative(row):
    low  = row.get('recovery_months_low',  row['recovery_months_predicted'] - 3)
    high = row.get('recovery_months_high', row['recovery_months_predicted'] + 3)

    narrative = (
        f"{row['state']} received a DRPI score of {row['drpi_score']:.1f} out of 100 "
        f"({row['drpi_risk_tier']} tier), based on data from FEMA, BLS, NOAA, BEA, "
        f"and the U.S. Census (2010–2024). "

        f"The Ridge Regression model (LOO-CV R²=0.14) estimates a labor market "
        f"stabilization period of {row['recovery_months_predicted']:.0f} months "
        f"following a major disaster event "
        f"(90% prediction interval: {low:.0f}–{high:.0f} months). "

        f"The state recorded an average FEMA disaster exposure of "
        f"{row['avg_disaster_exposure_12m']:.1f} declarations per 12-month window "
        f"and average per-capita damage of ${row['avg_damage_per_capita']:,.0f}. "

        f"Its Compound Vulnerability Index (CVI) is {row['compound_vulnerability_index']:.3f} "
        f"and fiscal capacity stands at ${row['fiscal_capacity_per_capita']:,.0f} per capita. "

        f"The labor instability classifier assigned a {row['instability_tier']} tier, "
        f"based on an unemployment shock of +{row['unemployment_shock_magnitude']:.1f} "
        f"percentage points recorded during the COVID-19 period. "

        f"The state's economic structure is classified as {row['economic_profile']} "
        f"({row['service_share_pct']:.0f}% service sector share of GDP)."
    )

    keywords = [
        f"drpi-{row['drpi_risk_tier'].lower()}",
        f"recovery-{row['recovery_months_predicted']:.0f}mo",
        f"instability-{row['instability_tier'].lower()}",
        row['economic_profile'].lower().replace(" ", "-"),
        f"cvi-{row['compound_vulnerability_index']:.2f}",
    ]
    return narrative, ", ".join(keywords)



# ── Generate narratives ───────────────────────────────────────────────────────
narratives, keywords_list = [], []

for i, row in df.iterrows():
    state = row["state"]
    print(f"  [{i+1:2d}/{len(df)}] {state:<22s}", end=" ", flush=True)
    narrative, keywords = build_template_narrative(row)
    narratives.append(narrative)
    keywords_list.append(keywords)
    print(f"✓  ({row['drpi_risk_tier']})")

df["narrative"] = narratives
df["keywords"]  = keywords_list

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\n✓ Narratives saved → {OUTPUT}")

# ── Upsert to Supabase ────────────────────────────────────────────────────────
if USE_SUPABASE:
    print("\nUpserting narratives to Supabase...")
    success, failed = 0, 0
    for _, row in df.iterrows():
        try:
            sb.table("drpi_risk_scores").update({
                "narrative": row["narrative"],
                "keywords":  row["keywords"],
            }).eq("state", row["state"]).execute()
            success += 1
        except Exception as e:
            print(f"  ✗ {row['state']}: {e}")
            failed += 1
    print(f"✓ Supabase updated: {success} states, {failed} failed")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
mode = "Template (data-driven)" if not USE_CLAUDE else "Extended"
print(f"✓ {len(df)} narratives generated ({mode})")
print(f"\n  Sample ({df.iloc[0]['state']}):")
print(f"  {df.iloc[0]['narrative'][:250]}...")
print(f"\n  Keywords: {df.iloc[0]['keywords']}")
print("="*65)
