# Project Context for LLMs (Companion to `band_model.pdf`)

This file documents implementation details that are not fully explicit in `band_model.pdf`, so another LLM can understand and reproduce the project end-to-end.

---

## 1) What this project is

`ohlc-excursion-model` is a session-structured market modeling project for NQ futures.  
It has two layers:

1. **Structural layer**: compute adaptive excursion bands around a reference open.
2. **Predictive layer**: use band context + microstructure + macro/event features to predict direction (mainly AM / `Target_1`), with walk-forward validation.

The reusable pipeline lives in `src\ohlc_dss_model\...`.  
The thesis model scripts live in `notebooks\approaches\band_model\...`.

---

## 2) Core terminology (code-grounded)

- **Bar**: one time bucket OHLCV row (1m or 30m).
- **Session day (`Session`)**: trading day key, aligned to 17:00 ET close rule.
- **Intraday session**:
  - `Pre_Target_1`: 18:00–03:00 ET
  - `Pre_Target_2`: 03:00–08:30 ET
  - `Target_1`: 08:30–12:00 ET
  - `Target_2`: 13:00–17:00 ET
- **`O_Ref`**: reference open (`O_Pre_Target_1` in current pipeline).
- **Excursion**: move distance from `O_Ref`.
- **Normalized move**: move divided by sigma-price (`Sigma * O_Ref`).
- **AE/FE bands**:
  - AE = typical/adverse excursion band (narrower)
  - FE = favorable/extreme excursion band (wider)
- **Band state (1..7)**: zone of `C_Pre_Target_2` relative to band boundaries.

---

## 3) Repo map and ownership

- `src\ohlc_dss_model\data`
  - ingestion, timezone conversion, session tagging, completeness filtering, pipeline loaders
- `src\ohlc_dss_model\features`
  - aggregation, volatility, excursion bands, band states, VWAP/OFI/VPIN, macro/events, momentum/range/regime, pivot features
- `src\ohlc_dss_model\utils`
  - timezone helper, plotting, csv→parquet utility
- `tests\excursion_bands_test.py`
  - no-leak expectation for last session without NY data
- `notebooks\approaches\band_model`
  - thesis-oriented model training/eval/inference/audit/analysis scripts

---

## 4) Exact dataflow used by model scripts

### 4.1 Raw load and sessionization

`load_raw_data(...)` in `data\pipeline_loaders.py`:

1. `load_parquet`
2. `convert_to_timezone` (UTC → America/New_York)
3. `session_tagging` (17:00 ET boundary)
4. `intraday_session_tagging` (maps rows to 4 windows)
5. `remove_incomplete_days` (must contain all 4 session labels)
6. select canonical columns: `DateTime, Session, Intraday_Session, Open, High, Low, Close, Volume`

### 4.2 Session aggregation + bands

`load_aggregated_data(raw_30m)`:

1. `aggregate_sessions` → pivoted OHLC columns per session bucket  
   (e.g., `O_Pre_Target_1`, `H_Target_2`, ...)
2. `filter_valid_sessions` (exchange calendar)
3. set `O_Ref = O_Pre_Target_1`
4. `yang_zhang(... FULL_DAY_SPEC, mode="historical")` → `Sigma_Historical`
5. `assign_direction(...)`
6. `calculate_excursion_bands(...)`

### 4.3 1m microstructure join

In `notebooks\approaches\band_model\model.py`:

1. load 1m raw with same sessionization process
2. `compute_session_vwap(... keep_cumulative=True)`
3. `compute_ofi_vpin(... keep_cumulative=True)`
4. snapshot 08:25–08:35 ET grouped by day:
   - `vwap_0830`, `ofi_cum`, `ofi_abs`, `vol_cum`, `close_0830`, `ofi_5min_sum`
   - derived: `vpin`, `ofi_per_vol`
5. left-join snapshot into aggregated frame by session date

### 4.4 Optional macro/events

If `FRED_API_KEY` exists:

- `get_macro_features(...)` builds/loads cached parquet tables under `data\processed`:
  - `event_table.parquet`
  - `fred_macro_table.parquet`
  - `individual_event_flags.parquet`
- then joins macro and event flags to session rows.

---

## 5) Excursion band implementation (actual formulas in code)

Main file: `features\excursion_bands.py`

### 5.1 Direction helper metrics

- `Z_Body = abs(log(C_Target_2 / O_Ref)) / Sigma_Historical`
- `Z_Sigma = Sigma_Historical / rolling_mean(Sigma_Historical, n).shift(1)`
- `Tau = clip(tau_0 * Z_Sigma^(-0.5), tau_min, tau_max)`
- `Direction`:
  - bullish if `Z_Body > Tau` and `C_Target_2 > O_Ref`
  - bearish if `Z_Body > Tau` and `C_Target_2 < O_Ref`
  - else neutral

Defaults from `config.py`:

- `n=7`, `tau_0=0.4`, `tau_min=0.26`, `tau_max=1.75`, `k=0.1`

### 5.2 Excursion means and band widths

- Builds full-day low/high: `L_Day`, `H_Day` from pre + target blocks.
- Computes `_epsilon_ae`, `_epsilon_fe` by direction logic.
- Normalizes by sigma, rolling means (`_mu_ae`, `_mu_fe`) with shift(1), rescales by prior sigma.
- `delta_t = k * Sigma_Historical.shift(1) * O_Ref`

### 5.3 Band levels created

- Centers:
  - `Band_AE_Pos_Center = O_Ref + _mu_ae_scaled`
  - `Band_AE_Neg_Center = O_Ref - _mu_ae_scaled`
  - `Band_FE_Pos_Center = O_Ref + _mu_fe_scaled`
  - `Band_FE_Neg_Center = O_Ref - _mu_fe_scaled`
- Each center gets `Upper/Lower = Center ± delta_t`

The final frame keeps boundary columns and drops temporary intermediates.

---

## 6) Band state mapping (1..7)

File: `features\band_state.py`

Using `C_Pre_Target_2`, zone mapping is:

- 6: `> Band_FE_Pos_Upper`
- 4: `>= Band_FE_Pos_Lower`
- 2: `>= Band_AE_Pos_Upper`
- 1: inside AE middle region (around AE negative/positive middle condition chain)
- 3: `>= Band_FE_Neg_Upper`
- 5: `>= Band_FE_Neg_Lower`
- 7: below FE negative lower

Output column: `band_state_ps2`.

---

## 7) Feature families used by thesis model code

Main feature engineering occurs in `notebooks\approaches\band_model\model.py::build_features`.

### 7.1 Price/band geometry

- Distances from pre-close and VWAP to each band (`pd_*`, `vd_*`)
- Relative VWAP-vs-price tension against each boundary (`vp_rel_*`)
- AE position normalization (`vwap_in_ae`, `price_in_ae`)

### 7.2 OFI/VPIN and interactions

- Directional OFI sign (`ofi_dir`)
- zone-conditioned interactions:
  - `ofi_x_*`, `vpin_x_*`, `ofipv_x_*`
- OFI momentum features: `ofi_rel_5d`, `ofi_rel_20d`, `ofi_pv_chg_5d`

### 7.3 Volume and dynamics

- `vol_rel_5d`, `vol_rel_20d`, `vol_x_*`, `vol_per_delta`
- band dynamics:
  - previous state, state change flags, relative delta/sigma features, normalized delta

### 7.4 Macro-derived engineered signals

On top of raw macro columns:

- VIX z-score/range position
- VIX × zone interactions
- rate acceleration / spread changes
- stress/event composite terms (e.g., `spread_x_vix`, `fomc_anticipation`, `event_count_week`)

---

## 8) Labeling and training protocol in code

### 8.1 Per-session label construction

In `model.py::build_matrix`:

- For both `Target_1` and `Target_2`:
  - `z_pos = max(0, H - O) / sigma_price`
  - `z_neg = max(0, O - L) / sigma_price`
  - direction label `zd = 1 if z_pos > z_neg else 0`
  - magnitude label `zm = max(z_pos, z_neg)`
  - ambiguity flag `amb = 1 if zm < 0.3 and abs(z_pos-z_neg) < 0.3 else 0`

Adds lagged targets (`pzd`, `pzm`) and weekday/session-type one-hot terms.

### 8.2 Training style

- chronological walk-forward by year (`fit`)
- each fold:
  - train on all years `< test_year`
  - evaluate on `== test_year`
- ensemble seeds `(42, 123, 456)`
- two models per seed:
  - classifier (`XGBClassifier`) for direction
  - regressor (`XGBRegressor`) for log magnitude

Prediction returns:

- probability up
- class prediction
- confidence (`max(p, 1-p)`)
- predicted log magnitude and magnitude

---

## 9) Leakage and chronology controls currently present

- Heavy use of `shift(1)` in volatility/band inputs.
- Session-level predictors derived from pre-target windows or prior info.
- Target OHLC columns excluded from model feature matrix (`COLUMN_SKIP`).
- Walk-forward year splits (not random shuffle).
- Test `tests\excursion_bands_test.py` checks no NY leakage in a specific edge case.

---

## 10) Reproduction requirements from code

### 10.1 Input files expected

- `data\raw\nq_30m.parquet`
- `data\raw\nq_1m.parquet`
- optional macro/event cache files in `data\processed`

### 10.2 Environment/deps

From `flake.nix` + runtime imports:

- python 3.11
- polars, pyarrow, numpy, scipy, matplotlib, scikit-learn, xgboost
- exchange-calendars, fredapi, holidays, requests, python-dotenv
- optional notebook stack (jupyter/ipykernel)

### 10.3 Main runnable scripts

- Train/evaluate main model:
  - `notebooks\approaches\band_model\model.py`
- Single-date inference demo:
  - `notebooks\approaches\band_model\inference.py`
- Statistical charts:
  - `notebooks\approaches\band_model\band_analysis.py`
- Feature/leakage audit:
  - `notebooks\approaches\band_model\final_audit.py`
- Lookback sweep:
  - `notebooks\approaches\band_model\test_lookback.py`

---

## 11) Important implementation notes not obvious in PDF

1. **Code trains on both `Target_1` and `Target_2` rows**, with `is_t1` feature; thesis emphasis is mostly `Target_1`.
2. **`O_Ref` in pipeline is `O_Pre_Target_1`** (explicitly set in loader).
3. **Macro loaders are incremental cache updaters**, not always full recompute.
4. **Some operational details are code-specific** (null fills, anti-join deduping on cache extension, session-date casting/join behavior).
5. **There is a pivot/transformer sub-pipeline** (`pivot*.py`, `pivot_transformer_input.py`) that is adjacent research, not required for the core band XGBoost path.

---

## 12) Minimal “mental execution trace”

Given one date:

1. Build sessionized bars (1m and 30m).
2. Aggregate 30m into session OHLC.
3. Compute sigma and excursion bands.
4. Locate pre-close in band state 1..7.
5. Snapshot 1m VWAP/OFI/VPIN near 08:30.
6. Join macro/event/momentum context.
7. Transform to feature row(s), infer direction probability and confidence.

That is the operational meaning of the thesis claim: **bands give structural location; interactions with flow/context produce predictive edge**.
