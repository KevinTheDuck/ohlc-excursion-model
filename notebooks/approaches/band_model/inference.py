import sys
from pathlib import Path
from datetime import date as dt_date
import numpy as np
import polars as pl

_PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BandDirectionModel, COLUMN_SKIP

TARGET_DATE = dt_date(2024, 7, 3)

print("Loading model...")
model = BandDirectionModel()
model.fit(burn_in=dt_date(2011, 4, 5), test_year_start=2016)

print(f"\nPredicting {TARGET_DATE}...")
agg = model.load_data()
agg = model.build_features(agg)

target_row = agg.filter(pl.col("Session").cast(pl.Date) == TARGET_DATE)
if target_row.height == 0:
    print(f"No data for {TARGET_DATE}")
    sys.exit(1)

sr = target_row.row(0, named=True)
sp = (sr.get("Sigma_Historical", 0.01) or 0.01) * (sr.get("O_Ref", 1) or 1)
o = sr.get("O_Target_1", 0)

if not o:
    print(f"No Target_1 data for {TARGET_DATE}")
    sys.exit(1)

zp = max(0, sr.get("H_Target_1", 0) - o) / (sp + 1e-9)
zn = max(0, o - sr.get("L_Target_1", 0)) / (sp + 1e-9)
zm = max(zp, zn)
zd = "UP" if zp > zn else "DOWN"

# Build single-session feature vector
row_dict = {
    "s": TARGET_DATE.year,
    "zd": 0,
    "amb": 0,
    "zm": 0.0,
    "wd": TARGET_DATE.weekday(),
    "it": 0,
}
for c in agg.columns:
    if c in COLUMN_SKIP or agg[c].dtype not in (
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
    ):
        continue
    v = sr.get(c)
    row_dict[c] = float(v) if v is not None else 0.0

df1 = pl.DataFrame([row_dict])
df1 = df1.with_columns([pl.lit(0.0).alias("pzd"), pl.lit(0.0).alias("pzm")])
for c in df1.columns:
    if df1[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        df1 = df1.with_columns(pl.col(c).fill_null(0.0))

Xl = []
for c in model.features_:
    if c.startswith("wd_"):
        w = int(c[3])
        Xl.append(
            np.array([[1.0 if TARGET_DATE.weekday() == w else 0.0]], dtype=np.float32)
        )
    elif c == "is_t1":
        Xl.append(np.ones((1, 1), dtype=np.float32))
    elif c in df1.columns:
        x = df1[c].to_numpy().astype(np.float32)
        Xl.append(np.nan_to_num(x, nan=0.0).reshape(1, -1))
    else:
        Xl.append(np.zeros((1, 1), dtype=np.float32))
X_pred = np.hstack(Xl).astype(np.float32)

proba, pred, conf, pred_lm, pred_m = model.predict(X_pred, year=2026)

direction = "UP" if pred[0] == 1 else "DOWN"
conf_pct = conf[0] * 100
pred_z = pred_m[0]
actual_z = zm
correct = "✓ CORRECT" if direction == zd else "✗ WRONG"

print(f"\n{'=' * 60}")
print(f"  INFERENCE — {TARGET_DATE} (Target_1 / AM)")
print(f"{'=' * 60}")
print(
    f"  Band State: {int(sr.get('band_state_ps2', 0))}  |  Weekday: {TARGET_DATE.strftime('%A')}"
)
print(
    f"  O_Ref: {sr.get('O_Ref', 0):.2f}  |  Pre-Close: {sr.get('C_Pre_Target_2', 0):.2f}"
)
print(f"  VPIN: {sr.get('vpin', 0):.4f}  |  VIX: {sr.get('vix_t1', 0):.2f}")

print(f"\n  ── Prediction ──")
print(f"  Direction:    {direction}  (P={proba[0]:.4f}, conf={conf_pct:.1f}%)")
print(f"  Pred z_max:   {pred_z:.4f} σ  (≈ {pred_z * sp:.1f} pts)")
print(f"  Expected move: {'BIG' if pred_z > 0.003 else 'SMALL'}")

print(f"\n  ── Actual ──")
print(f"  Direction:    {zd}")
print(f"  z_max:        {actual_z:.4f} σ  (≈ {actual_z * sp:.1f} pts)")
print(f"  z_pos / z_neg: {zp:.4f} / {zn:.4f}")
print(
    f"  O/H/L: {o:.2f} / {sr.get('H_Target_1', 0):.2f} / {sr.get('L_Target_1', 0):.2f}"
)
print(f"  Outcome:      {correct}  |  Mag Δ = {abs(pred_z - actual_z):.4f} σ")
