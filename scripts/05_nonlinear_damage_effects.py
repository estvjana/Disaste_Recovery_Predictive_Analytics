import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ----------------------------
# SETTINGS
# ----------------------------
OUTDIR = Path("visual png")
OUTDIR.mkdir(exist_ok=True)

DAMAGE_FILE = "top15_states_2010_2024.csv"
RECOVERY_FILE = "recovery_speed_top15_2010_2024.csv"

# ----------------------------
# LOAD
# ----------------------------
damage = pd.read_csv(DAMAGE_FILE)
recovery = pd.read_csv(RECOVERY_FILE)

# --- find STATE column ---
def find_state_col(df):
    for c in df.columns:
        if "state" in c.lower():
            return c
    return df.columns[0]

# --- find DAMAGE column ---
def find_damage_col(df):
    # prefer columns with "damage" + "per" in name
    candidates = [c for c in df.columns if ("damage" in c.lower() and "per" in c.lower())]
    if candidates:
        return candidates[0]
    # otherwise take first numeric column that isn't state/year
    numeric = df.select_dtypes(include="number").columns.tolist()
    numeric = [c for c in numeric if "year" not in c.lower()]
    return numeric[0]

# --- find RECOVERY MONTHS column ---
def find_recovery_col(df):
    # prefer names like months_to_recover / recovery_months
    candidates = [c for c in df.columns if ("month" in c.lower() and ("recover" in c.lower() or "recovery" in c.lower()))]
    if candidates:
        return candidates[0]
    # otherwise take first numeric column that isn't shock_year/year
    numeric = df.select_dtypes(include="number").columns.tolist()
    numeric = [c for c in numeric if "year" not in c.lower() and "shock" not in c.lower()]
    return numeric[0]

damage_state = find_state_col(damage)
damage_val = find_damage_col(damage)

recovery_state = find_state_col(recovery)
recovery_val = find_recovery_col(recovery)

# standardize
damage = damage.rename(columns={damage_state: "state", damage_val: "damage_per_capita"})
recovery = recovery.rename(columns={recovery_state: "state", recovery_val: "months_to_recover"})

damage["state"] = damage["state"].astype(str).str.upper().str.strip()
recovery["state"] = recovery["state"].astype(str).str.upper().str.strip()

# keep only what we need (THIS prevents 2-column y problems)
damage = damage[["state", "damage_per_capita"]].copy()
recovery = recovery[["state", "months_to_recover"]].copy()

# numeric + clean
damage["damage_per_capita"] = pd.to_numeric(damage["damage_per_capita"], errors="coerce")
recovery["months_to_recover"] = pd.to_numeric(recovery["months_to_recover"], errors="coerce")

df = damage.merge(recovery, on="state", how="inner").dropna()
df = df[df["damage_per_capita"] > 0].copy()

print("Merged df shape:", df.shape)
print(df.head())

# ----------------------------
# NOW your regression will work
# ----------------------------
x = df["damage_per_capita"].to_numpy()
y = df["months_to_recover"].to_numpy()

lx = np.log(x)

slope1, intercept1, r1, p1, _ = stats.linregress(lx, y)
print("OK linear regression:", r1, p1)
