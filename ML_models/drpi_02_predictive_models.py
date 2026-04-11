#!/usr/local/bin/python3.12
"""
=============================================================================
DRPI — Disaster Recovery Predictive Models
Step 02: Predictive Models  (v2 — fixed overfitting + SHAP)
=============================================================================
FIXES in v2:
  - Uses Ridge regression + Leave-One-Out CV (correct for n=50)
  - Reduces to 6 core features (prevents overfitting with small n)
  - Adds fiscal_capacity_per_capita + compound_vulnerability_index
  - SHAP values for explainability per state
  - Composite DRPI score rebuilt with CVI

Output: state_risk_scores.csv, state_shap_values.csv
Run:    python3.12 ML_models/drpi_02_predictive_models.py
=============================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
import shap

warnings.filterwarnings("ignore")

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT,"state_feature_matrix.csv")
OUTPUT = os.path.join(ROOT,"state_risk_scores.csv")
SHAP_OUT = os.path.join(ROOT,"state_shap_values.csv")

print("="*65)
print("DRPI Step 02 — Predictive Models v2 (Ridge + SHAP + LOO-CV)")
print("="*65)

df = pd.read_csv(INPUT)
df = df.dropna(subset=["recovery_months"]).copy()
print(f"\n  States with full data: {len(df)}")

# ── FEATURE SELECTION (6 core — avoids overfitting on n=50) ──────────────────
# Selected based on paper findings: economic structure, fiscal cap,
# disaster exposure, compound vulnerability, baseline labor
CORE_FEATURES = [
    "avg_disaster_exposure_12m",      # disaster frequency (FEMA)
    "avg_damage_per_capita",          # economic damage severity (NOAA)
    "fiscal_capacity_per_capita",     # state ability to fund recovery (Census)
    "unemployment_shock_magnitude",   # labor vulnerability (BLS)
    "service_share_pct",              # economic structure (BEA) — paper: svc = slower
    "compound_vulnerability_index",   # multi-sector composite
]

X_raw = df[CORE_FEATURES].copy()
for c in CORE_FEATURES:
    X_raw[c] = X_raw[c].fillna(X_raw[c].median())

y_months = df["recovery_months"].values

# ── MODEL A: Ridge Regression with LOO-CV ─────────────────────────────────────
print("\n── Model A: Recovery Time — Ridge Regression (LOO-CV) ──")

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# RidgeCV selects best alpha automatically
ridge = RidgeCV(alphas=[0.1,1,5,10,20,50,100,200], cv=LeaveOneOut(), scoring="r2")
ridge.fit(X_scaled, y_months)

# LOO cross-validation R² and MAE
loo = LeaveOneOut()
y_pred_loo = np.zeros(len(y_months))
for train_idx, test_idx in loo.split(X_scaled):
    m = Ridge(alpha=ridge.alpha_)
    m.fit(X_scaled[train_idx], y_months[train_idx])
    y_pred_loo[test_idx] = m.predict(X_scaled[test_idx])

ss_res = np.sum((y_months - y_pred_loo)**2)
ss_tot = np.sum((y_months - y_months.mean())**2)
loo_r2 = 1 - ss_res/ss_tot
loo_mae = np.mean(np.abs(y_months - y_pred_loo))

df["recovery_months_predicted"] = ridge.predict(X_scaled).clip(1,60).round(1)

# Prediction interval via residual std
residual_std = np.std(y_months - df["recovery_months_predicted"])
df["recovery_months_low"]  = (df["recovery_months_predicted"] - 1.645*residual_std).clip(1,60).round(1)
df["recovery_months_high"] = (df["recovery_months_predicted"] + 1.645*residual_std).clip(1,60).round(1)

print(f"  Best alpha : {ridge.alpha_}")
print(f"  LOO R²     : {loo_r2:.3f}")
print(f"  LOO MAE    : {loo_mae:.2f} months")
print(f"  Predicted  : {df['recovery_months_predicted'].min():.0f}–{df['recovery_months_predicted'].max():.0f} months")

# Feature importance (coefficients)
coef = pd.Series(np.abs(ridge.coef_), index=CORE_FEATURES).sort_values(ascending=False)
print(f"  Top drivers: {', '.join(coef.head(4).index.tolist())}")

# ── SHAP VALUES ───────────────────────────────────────────────────────────────
print("\n── Computing SHAP values for explainability...")
# Use GBR for SHAP (tree-based, interpretable with small n)
gbr_shap = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                      learning_rate=0.08, random_state=42)
gbr_shap.fit(X_raw, y_months)

explainer  = shap.TreeExplainer(gbr_shap)
shap_vals  = explainer.shap_values(X_raw)
shap_df    = pd.DataFrame(shap_vals, columns=[f"shap_{c}" for c in CORE_FEATURES])
shap_df["state"] = df["state"].values
shap_df["base_value"] = float(explainer.expected_value)

# Most influential feature per state
shap_abs = np.abs(shap_vals)
shap_df["top_driver"] = [CORE_FEATURES[i] for i in shap_abs.argmax(axis=1)]

shap_full = pd.concat([df[["state","abbr"]].reset_index(drop=True),
                       shap_df.reset_index(drop=True)], axis=1)
shap_full.to_csv(SHAP_OUT, index=False)
print(f"  ✓ SHAP values saved → {SHAP_OUT}")

# Mean |SHAP| feature importance
mean_shap = pd.Series(shap_abs.mean(axis=0), index=CORE_FEATURES).sort_values(ascending=False)
print(f"  SHAP importance:")
for f,v in mean_shap.items():
    print(f"    {f:<40s} {v:.3f}")

# ── MODEL B: Labor Instability Classifier ─────────────────────────────────────
print("\n── Model B: Labor Instability Classifier (RF + LOO) ──")

# Build target from shock magnitude + volatility
df["instability_raw"] = (
    0.5 * (df["unemployment_shock_magnitude"] - df["unemployment_shock_magnitude"].min()) /
          (df["unemployment_shock_magnitude"].max() - df["unemployment_shock_magnitude"].min() + 1e-9) +
    0.5 * (df["unemployment_volatility_excl_covid"] - df["unemployment_volatility_excl_covid"].min()) /
          (df["unemployment_volatility_excl_covid"].max() - df["unemployment_volatility_excl_covid"].min() + 1e-9)
)
q1,q2,q3 = df["instability_raw"].quantile([0.25,0.50,0.75])
df["instability_tier_actual"] = df["instability_raw"].apply(
    lambda v: "LOW" if v<=q1 else "MODERATE" if v<=q2 else "HIGH" if v<=q3 else "CRITICAL"
)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_tier = le.fit_transform(df["instability_tier_actual"])

clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                              min_samples_leaf=3, random_state=42)
clf.fit(X_raw, y_tier)

# LOO accuracy
y_pred_clf = np.zeros(len(y_tier), dtype=int)
for train_idx, test_idx in loo.split(X_raw):
    c = RandomForestClassifier(n_estimators=200, max_depth=4,
                               min_samples_leaf=3, random_state=42)
    c.fit(X_raw.iloc[train_idx], y_tier[train_idx])
    y_pred_clf[test_idx] = c.predict(X_raw.iloc[test_idx])

loo_acc = (y_pred_clf == y_tier).mean()
print(f"  LOO Accuracy: {loo_acc:.3f}")

proba = clf.predict_proba(X_raw)
tier_labels = le.classes_
df["instability_tier"] = le.inverse_transform(clf.predict(X_raw))
for lbl in ["LOW","MODERATE","HIGH","CRITICAL"]:
    idx = np.where(tier_labels==lbl)[0]
    df[f"prob_{lbl}"] = proba[:,idx[0]].round(3) if len(idx) else 0.0

# ── COMPOSITE DRPI RISK SCORE (v2) ────────────────────────────────────────────
print("\n── DRPI Composite Score v2 ──")

def mm(s): return ((s-s.min())/(s.max()-s.min()+1e-9)).clip(0,1)

df["drpi_score"] = (
    0.30 * mm(df["recovery_months_predicted"]) * 100 +
    0.25 * mm(df["avg_disaster_exposure_12m"]) * 100 +
    0.20 * mm(df["compound_vulnerability_index"]) * 100 +
    0.15 * mm(df["avg_damage_per_capita"])      * 100 +
    0.10 * mm(df["unemployment_volatility_excl_covid"]) * 100
).round(1)

df["drpi_risk_tier"] = df["drpi_score"].apply(
    lambda s: "CRITICAL" if s>=75 else "HIGH" if s>=50 else "MODERATE" if s>=25 else "LOW"
)

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_cols = [
    "state","abbr",
    "recovery_months_predicted","recovery_months_low","recovery_months_high",
    "instability_tier","prob_LOW","prob_MODERATE","prob_HIGH","prob_CRITICAL",
    "drpi_score","drpi_risk_tier",
    "avg_disaster_exposure_12m","avg_damage_per_capita",
    "fiscal_capacity_per_capita","compound_vulnerability_index",
    "cvi_labor","cvi_disaster","cvi_damage","cvi_fiscal","cvi_structure",
    "avg_unemployment_rate","unemployment_volatility_excl_covid",
    "baseline_unemployment","unemployment_shock_magnitude",
    "service_share_pct","industry_share_pct","natres_share_pct","govt_share_pct",
    "economic_profile","profile_benchmark_months","recovery_months",
]
out_cols = [c for c in out_cols if c in df.columns]
results = df[out_cols].sort_values("drpi_score",ascending=False).reset_index(drop=True)
results.to_csv(OUTPUT, index=False)

tier_counts = results["drpi_risk_tier"].value_counts()
print(f"\n{'='*65}")
print(f"✓ Risk scores → {OUTPUT}")
print(f"✓ SHAP values → {SHAP_OUT}")
print(f"\n  Risk Tier Distribution:")
for t in ["CRITICAL","HIGH","MODERATE","LOW"]:
    n = tier_counts.get(t,0)
    print(f"    {t:10s} {'█'*n}  ({n} states)")
print(f"\n  Top 10 Highest-Risk States:")
print(results[["state","drpi_score","drpi_risk_tier",
               "recovery_months_predicted","instability_tier","economic_profile"]]
      .head(10).to_string(index=False))
print("="*65)
