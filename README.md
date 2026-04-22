# Multi-Sector Disaster Recovery Predictive Analytics
**UNC Charlotte · DTSC 4302**

A data-driven dashboard that predicts how long U.S. states take to recover across labor markets, fiscal systems, and economic sectors after a natural disaster.

---

## How to Run

### 1. Clone the repo
```bash
git clone <repo-url>
cd cvs.noaa.gov.dataset
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment (optional, only needed for narrative generator)
```bash
cp config/.env.example config/.env
# Edit config/.env and fill in your keys if needed
```

### 4. Run the ML pipeline (builds the model outputs)
```bash
python3 ML_models/drpi_01_build_feature_matrix.py
python3 ML_models/drpi_02_predictive_models.py
python3 ML_models/drpi_03_narrative_generator.py
```

### 5. Launch the dashboard
```bash
python3 -m streamlit run ML_models/drpi_04_dashboard.py
```

---

## Project Structure

```
ML_models/
  drpi_01_build_feature_matrix.py   ← Step 1: builds state feature matrix
  drpi_02_predictive_models.py      ← Step 2: Ridge regression + SHAP + CVI
  drpi_03_narrative_generator.py    ← Step 3: generates state narratives
  drpi_04_dashboard.py              ← Step 4: Streamlit dashboard (main app)
  drpi_config.py                    ← shared config loader
  drpi_gov_data.py                  ← political economy tab data
  drpi_tab_data.py                  ← policy insights tab data

data/
  BLS/        ← BLS LAUS unemployment data (2006–2025)
  FEMA/       ← FEMA disaster declarations
  BEA/        ← BEA State GDP by industry
  Census/     ← U.S. Census population & state finances
  Education/  ← COVID School Data Hub + NCES attendance
  NOAA/       ← NOAA storm events (large files, see note below)

scripts/      ← standalone figure generation scripts (paper figures)
config/       ← environment variables (.env.example provided)
```

---

## Data Sources

| Dataset | Source | Used For |
|---|---|---|
| FEMA Disaster Declarations | FEMA OpenFEMA | Disaster exposure |
| NOAA Storm Events | NOAA NCEI | Damage per capita |
| BLS LAUS | Bureau of Labor Statistics | Unemployment volatility |
| BEA SAGDP2 | Bureau of Economic Analysis | Economic structure |
| U.S. Census State Finances | U.S. Census Bureau | Fiscal capacity |
| COVID School Data Hub | covidschooldatahub.com | Education disruption |
| NCES Table 203.80 | nces.ed.gov | Avg daily attendance |

---

## Note on Large Files

NOAA storm event CSVs and BEA GDP state files are excluded from this repo due to size.
Download them directly from:
- NOAA: https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- BEA: https://apps.bea.gov/regional/downloadzip.cfm

Place them in `data/NOAA/` and `data/BEA/BEA_StateGDP_1997_2024/` respectively.

---

## Dashboard Navigation

| Page | Description |
|---|---|
| Overview Dashboard | National risk map, KPI cards, and risk tier rankings across all 50 states |
| Risk Map | Explore by indicator: composite risk, recovery time, CVI, fiscal capacity, disaster exposure |
| Research Analytics | Seven tabs: economic structure, fiscal capacity, service economy, political economy, education disruption, recovery predictors, policy insights |
| State Deep-Dive | Select any state: predicted recovery time, labor instability tier, SHAP explanation, CVI radar |


## Research Team
| Name | GitHub |
|---|---|
| Ana Julia Abreu Estevez | [@estvjana](https://github.com/estvjana) |
| Wendy Ceja-Huerta | [@wcejahuerta](https://github.com/wcejahuerta) |
| Maria Eduarda C. F. de Resende Silva | [@mecfrs](https://github.com/mecfrs) |
| Jake Fabrizio | [@Jakefab245](https://github.com/Jakefab245) |
| Carolina Rangel Lara | [@caro-98](https://github.com/caro-98) |
| Riya Vadadoria | [@riya0106](https://github.com/riya0106) |
