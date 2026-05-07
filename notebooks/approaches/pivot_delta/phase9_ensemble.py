import sys
from pathlib import Path
from datetime import date as dt_date
import gc, warnings
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score

_PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ / "src"))

from ohlc_dss_model.data import load_raw_data
from ohlc_dss_model.data.pipeline_loaders import load_aggregated_data
from ohlc_dss_model.features.band_state import get_band_state_on_ps2
from ohlc_dss_model.features.momentum import calculate_momentum_features
from ohlc_dss_model.features.vwap import compute_session_vwap, get_vwap_position
from ohlc_dss_model.features.ofi_vpin import compute_ofi_vpin
import os
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
BURN_IN = dt_date(2011, 4, 5)

print("=" * 70)
print("  PHASE 9: Volume×Band Ensemble + Calibration")
print("=" * 70)

print("[1/4] Session data...")
raw_30m = load_raw_data(str(_PROJ / "data" / "raw" / "nq_30m.parquet"))
raw_1m = load_raw_data(str(_PROJ / "data" / "raw" / "nq_1m.parquet"))

agg = load_aggregated_data(raw_30m)

if os.getenv("FRED_API_KEY"):
    from ohlc_dss_model.features.macro_loaders import get_macro_features
    from ohlc_dss_model.config import config

    agg = get_macro_features(
        config.data.processed_folder_path, agg, os.getenv("FRED_API_KEY")
    )

agg = get_band_state_on_ps2(agg)
agg = calculate_momentum_features(agg)

print("[2/4] 1m VWAP/OFI snapshot...")
bars_1m_vwap, _ = compute_session_vwap(raw_1m, keep_cumulative=True)
bars_1m_ofi, _ = compute_ofi_vpin(bars_1m_vwap, keep_cumulative=True)

# ALSO get VWAP at each band level for volume profile
# Get cumulative values at 08:30
snap = (
    bars_1m_ofi.with_columns(
        pl.col("DateTime").dt.time().alias("time"),
        pl.col("Session").cast(pl.Date).alias("sd"),
    )
    .filter(pl.col("time") >= pl.time(8, 25), pl.col("time") <= pl.time(8, 35))
    .group_by("sd")
    .agg(
        [
            pl.col("vwap_cumulative").last().alias("vwap_0830"),
            pl.col("_cum_ofi").last().alias("ofi_cum"),
            pl.col("_cum_abs_ofi").last().alias("ofi_abs"),
            pl.col("_cum_vol").last().alias("vol_cum"),
            pl.col("Close").last().alias("close_0830"),
            pl.col("ofi_bar").sum().alias("ofi_5min_sum"),
        ]
    )
)
snap = snap.with_columns(
    [
        (pl.col("ofi_cum") / (pl.col("ofi_abs") + 1e-9)).alias("vpin"),
        (pl.col("ofi_cum") / (pl.col("vol_cum") + 1e-9)).alias("ofi_per_vol"),
    ]
)

del raw_1m, bars_1m_vwap, bars_1m_ofi
gc.collect()

agg = agg.join(snap, left_on=pl.col("Session").cast(pl.Date), right_on="sd", how="left")

try:
    agg = get_vwap_position(agg)
except:
    pass

# Fill nulls
for c in agg.columns:
    if agg[c].dtype in (pl.Float64, pl.Float32):
        agg = agg.with_columns(pl.col(c).fill_null(0.0))

print(f"  Sessions: {agg.height} | Columns: {len(agg.columns)}")

print("[3/4] Engineering volume×band features...")

sp = pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref") + 1e-9
agg = agg.with_columns(sp.alias("sigma_price"))

sp_c = pl.col("sigma_price")
vwap = pl.col("vwap_0830")
c_pre = pl.col("C_Pre_Target_2")
bs = pl.col("band_state_ps2")
ofi = pl.col("ofi_cum")
vol = pl.col("vol_cum")
oref = pl.col("O_Ref")
delta = pl.col("_delta_t")

BM = {
    "AEPU": "Band_AE_Pos_Upper",
    "AEPL": "Band_AE_Pos_Lower",
    "AENU": "Band_AE_Neg_Upper",
    "AENL": "Band_AE_Neg_Lower",
    "FEPU": "Band_FE_Pos_Upper",
    "FEPL": "Band_FE_Pos_Lower",
    "FENU": "Band_FE_Neg_Upper",
    "FENL": "Band_FE_Neg_Lower",
}

exprs = []
for sn, col in BM.items():
    b = pl.col(col)
    # Price distance to band
    exprs.append(((c_pre - b) / (sp_c + 1e-9)).alias(f"pd_{sn}"))
    # VWAP distance to band
    exprs.append(((vwap - b) / (sp_c + 1e-9)).alias(f"vd_{sn}"))
    # VWAP minus price (who's ahead?)
    exprs.append(((vwap - c_pre) / (b - c_pre + 1e-9)).alias(f"vp_rel_{sn}"))

agg = agg.with_columns(
    pl.when(vwap > pl.col("Band_FE_Pos_Upper"))
    .then(pl.lit(6))
    .when(vwap >= pl.col("Band_FE_Pos_Lower"))
    .then(pl.lit(4))
    .when(vwap >= pl.col("Band_AE_Pos_Upper"))
    .then(pl.lit(2))
    .when(vwap > pl.col("Band_AE_Neg_Upper"))
    .then(pl.lit(1))
    .when(vwap >= pl.col("Band_AE_Neg_Lower"))
    .then(pl.lit(1))
    .when(vwap >= pl.col("Band_FE_Neg_Upper"))
    .then(pl.lit(3))
    .when(vwap >= pl.col("Band_FE_Neg_Lower"))
    .then(pl.lit(5))
    .otherwise(pl.lit(7))
    .alias("vwap_zone")
)

ae_low = pl.col("Band_AE_Neg_Lower")
ae_up = pl.col("Band_AE_Pos_Upper")
exprs.append(((vwap - ae_low) / (ae_up - ae_low + 1e-9)).alias("vwap_in_ae"))
exprs.append(((c_pre - ae_low) / (ae_up - ae_low + 1e-9)).alias("price_in_ae"))

exprs.extend(
    [
        ((vwap - c_pre) / (sp_c + 1e-9)).alias("vwap_price_gap"),
        (vwap > c_pre).cast(pl.Int32).alias("vwap_above"),
        (pl.col("vwap_zone") == bs).cast(pl.Int32).alias("vwap_same_zone"),
        ((vwap - oref) / (sp_c + 1e-9)).alias("vwap_vs_oref"),
        ((oref - c_pre) / (sp_c + 1e-9)).alias("cpre_vs_oref"),
    ]
)

ofi_dir = (
    pl.when(ofi > 0).then(pl.lit(1.0)).when(ofi < 0).then(pl.lit(-1.0)).otherwise(0.0)
)
vpin_val = pl.col("vpin")

for z, cond in [
    ("z1", bs == 1),
    ("bull", bs.is_in([2, 4])),
    ("bear", bs.is_in([3, 5])),
    ("ext", bs.is_in([6, 7])),
]:
    zc = cond.cast(pl.Int32)
    exprs.extend(
        [
            (ofi_dir * zc).alias(f"ofi_x_{z}"),
            (vpin_val * zc).alias(f"vpin_x_{z}"),
            (pl.col("ofi_per_vol") * zc).alias(f"ofipervol_x_{z}"),
        ]
    )

exprs.extend(
    [
        (ofi / (ofi.shift(1).rolling_mean(5).over(pl.lit(1)) + 1e-9)).alias(
            "ofi_rel_5d"
        ),
        (ofi / (ofi.shift(1).rolling_mean(20).over(pl.lit(1)) + 1e-9)).alias(
            "ofi_rel_20d"
        ),
        (pl.col("ofi_per_vol") - pl.col("ofi_per_vol").shift(1).rolling_mean(5)).alias(
            "ofi_pv_chg_5d"
        ),
    ]
)

exprs.extend(
    [
        (vol / (vol.shift(1).rolling_mean(5).over(pl.lit(1)) + 1e-9)).alias(
            "vol_rel_5d"
        ),
        (vol / (vol.shift(1).rolling_mean(20).over(pl.lit(1)) + 1e-9)).alias(
            "vol_rel_20d"
        ),
        (vol * (bs == 1).cast(pl.Int32)).alias("vol_x_z1"),
        (vol * bs.is_in([2, 4]).cast(pl.Int32)).alias("vol_x_bull"),
        (vol * bs.is_in([3, 5]).cast(pl.Int32)).alias("vol_x_bear"),
        (vol / (delta + 1e-9)).alias("vol_per_delta"),
    ]
)

exprs.extend(
    [
        bs.shift(1).alias("bs_prev"),
        ((bs != bs.shift(1)).cast(pl.Int32)).alias("bs_changed"),
        (delta / (delta.shift(1).rolling_mean(20) + 1e-9)).alias("delta_rel_20d"),
        (sp_c / (sp_c.shift(1).rolling_mean(20) + 1e-9)).alias("sigma_rel_20d"),
        (pl.col("_delta_t") / (sp_c + 1e-9)).alias("dt_norm"),
    ]
)

agg = agg.with_columns(exprs)

# Fill nulls from rolling features
for c in agg.columns:
    if agg[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        agg = agg.with_columns(pl.col(c).fill_null(0.0))

print("[4/4] Building dataset...")

rows = []
for sr in agg.iter_rows(named=True):
    sess = sr["Session"]
    if sess < BURN_IN:
        continue
    _sp = sr.get("sigma_price", 0) or 1e-9
    wd = sess.weekday()
    for tcol, tn in [("Target_1", 0), ("Target_2", 1)]:
        o = sr.get(f"O_{tcol}", 0)
        h = sr.get(f"H_{tcol}", 0)
        l = sr.get(f"L_{tcol}", 0)
        if not o or o == 0:
            continue
        zp = max(0, h - o) / (_sp + 1e-9)
        zn = max(0, o - l) / (_sp + 1e-9)
        zm = max(zp, zn)
        zd = 1 if zp > zn else 0
        amb = 1 if zm < 0.3 and abs(zp - zn) < 0.3 else 0
        row = {
            "s": sess.year,
            "zd": zd,
            "amb": amb,
            "zm": float(zm),
            "wd": wd,
            "it": tn,
        }
        skip = {
            "Session",
            "Sigma_Historical",
            "Z_Body",
            "Z_Sigma",
            "Tau",
            "Direction",
            "sd",
            "sd_right",
            "SessionDate",
            "sigma_price",
        }
        skip.update(
            {
                f"{o}_{t}"
                for o in "OHLC"
                for t in ["Target_1", "Target_2", "Pre_Target_1", "Pre_Target_2"]
            }
        )
        skip.update({c for c in agg.columns if c.startswith("Band_")})
        for c in agg.columns:
            if c in skip or agg[c].dtype not in (
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
            ):
                continue
            v = sr.get(c)
            row[c] = float(v) if v is not None else 0.0
        rows.append(row)

df = pl.DataFrame(rows)
df = df.with_columns(
    [pl.col("zd").shift(2).alias("pzd"), pl.col("zm").shift(2).alias("pzm")]
)
for c in df.columns:
    if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        df = df.with_columns(pl.col(c).fill_null(0.0))

# Build X
y = df["zd"].to_numpy().astype(np.float32)
amb = df["amb"].to_numpy().astype(np.int64)
years = df["s"].to_numpy()
skip_x = {"s", "zd", "amb", "zm", "wd", "it"}
# Exclude raw non-stationary features that poison tree models
raw_price_cols = {
    "close_0830",
    "vwap_0830",
    "ofi_cum",
    "ofi_abs",
    "vol_cum",
    "ofi_5min_sum",
    "sigma_price",
}
X_list = []
fnames = []
for c in df.columns:
    if c in skip_x or c in raw_price_cols:
        continue
    if df[c].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        continue
    x = df[c].to_numpy().astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    if np.std(x) < 1e-9:
        continue
    X_list.append(x.reshape(-1, 1))
    fnames.append(c)
for w in range(5):
    X_list.append((df["wd"].to_numpy() == w).astype(np.float32).reshape(-1, 1))
    fnames.append(f"wd_{w}")
X_list.append((df["it"].to_numpy() == 0).astype(np.float32).reshape(-1, 1))
fnames.append("is_t1")
X = np.hstack(X_list).astype(np.float32)

print(f"  Dataset: {df.height:,} | Features: {len(fnames)} | X={X.shape}")


test_years = sorted(set(years[years >= 2016]))
HP = {"max_depth": 6, "lr": 0.03, "ra": 1.0, "rl": 1.0, "ss": 0.7, "cs": 0.5}
seeds = [42, 123, 456]

all_p = []
all_a = []
all_c = []
all_it = []
all_am = []
fold_accs = []

for ty in test_years:
    tr = years < ty
    te = years == ty
    tr_c = tr & (amb == 0)
    if tr_c.sum() < 100 or te.sum() < 20:
        continue
    wv = max(1.0, (y[tr_c] == 0).sum() / max((y[tr_c] == 1).sum(), 1))

    # Train ensemble
    probas = []
    for seed in seeds:
        m = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=HP["max_depth"],
            learning_rate=HP["lr"],
            subsample=HP["ss"],
            colsample_bytree=HP["cs"],
            reg_alpha=HP["ra"],
            reg_lambda=HP["rl"],
            scale_pos_weight=wv,
            random_state=seed,
            verbosity=0,
        )
        m.fit(X[tr_c], y[tr_c], verbose=False)
        probas.append(m.predict_proba(X[te])[:, 1])

    # Ensemble probabilities
    proba = np.mean(probas, axis=0)
    pred = (proba >= 0.5).astype(float)
    conf = np.maximum(proba, 1 - proba)
    acc = accuracy_score(y[te], pred)
    bl = max((y[te] == 1).mean(), (y[te] == 0).mean())
    fold_accs.append(acc)

    all_p.extend(pred)
    all_a.extend(y[te])
    all_c.extend(conf)
    all_am.extend(amb[te])
    all_it.extend(df["it"].to_numpy()[te])

ap = np.array(all_p)
aa = np.array(all_a)
ac = np.array(all_c)
am = np.array(all_am)
ai = np.array(all_it)
acc = accuracy_score(aa, ap)
bl = max((aa == 1).mean(), (aa == 0).mean())
na = am == 0

print(f"\n{'=' * 70}")
print(f"RESULTS — Volume×Band Ensemble + Platt Calibration")
print(f"{'=' * 70}")
print(f"  Features: {len(fnames)} | WFO Folds: {len(fold_accs)}")
print(f"  Overall: {acc * 100:.2f}% (BL={bl * 100:.2f}% Δ={(acc - bl) * 100:+.2f}%)")
print(f"  Clean:  {accuracy_score(aa[na], ap[na]) * 100:.1f}% ({na.sum():,})")

print(f"\n  Full Confidence Curve:")
print(f"  {'Conf≥':>8} {'Acc%':>7} {'BL%':>7} {'Δ%':>7} {'N':>7} {'%cov':>6}")
print(f"  {'-' * 50}")
for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    m = ac >= th
    if m.sum() > 5:
        a = accuracy_score(aa[m], ap[m])
        bt = max((aa[m] == 1).mean(), (aa[m] == 0).mean())
        sig = " ✓" if a > bt + 0.03 else ""
        print(
            f"  {th:>8.2f} {a * 100:>6.1f}% {bt * 100:>6.1f}% {(a - bt) * 100:>+6.1f}% {m.sum():>6,} {m.sum() / len(ac) * 100:>5.0f}%{sig}"
        )

# ── Per-Target Breakdown ──
print(f"\n{'=' * 70}")
print("PER-TARGET BREAKDOWN (T1=AM 08:30-12:00, T2=PM 13:00-17:00)")
print(f"{'=' * 70}")

for tname, tlabel, tm in [
    ("Target_1", "AM (08:30-12:00)", ai == 0),
    ("Target_2", "PM (13:00-17:00)", ai == 1),
]:
    na_tm = tm.sum()
    if na_tm < 20:
        continue

    acc_t = accuracy_score(aa[tm], ap[tm])
    bl_t = max((aa[tm] == 1).mean(), (aa[tm] == 0).mean())
    na_clean_t = (tm & (am == 0)).sum()
    acc_clean_t = (
        accuracy_score(aa[tm & (am == 0)], ap[tm & (am == 0)]) if na_clean_t > 10 else 0
    )

    print(f"\n  ── {tname} ({tlabel}) ──")
    print(
        f"  Samples: {na_tm:,}  |  Clean: {na_clean_t:,}  |  "
        f"z_dir=1: {(aa[tm] == 1).sum():,} ({(aa[tm] == 1).mean() * 100:.1f}%)"
    )
    print(
        f"  Overall Acc: {acc_t * 100:.2f}%  |  Baseline: {bl_t * 100:.2f}%  "
        f"|  Δ: {(acc_t - bl_t) * 100:+.2f}%"
    )
    if na_clean_t > 10:
        print(f"  Clean Acc:  {acc_clean_t * 100:.1f}%")

    print(f"  {'Conf≥':>8} {'Acc%':>7} {'BL%':>7} {'Δ%':>7} {'N':>7} {'%cov':>6}")
    print(f"  {'-' * 45}")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        m = (ac >= th) & tm
        if m.sum() > 5:
            a_c = accuracy_score(aa[m], ap[m])
            bt_c = max((aa[m] == 1).mean(), (aa[m] == 0).mean())
            sig = " ✓" if a_c > bt_c + 0.03 else ""
            pct = m.sum() / na_tm * 100
            print(
                f"  {th:>8.2f} {a_c * 100:>6.1f}% {bt_c * 100:>6.1f}% {(a_c - bt_c) * 100:>+6.1f}% {m.sum():>6,} {pct:>5.0f}%{sig}"
            )
print(f"\n  Top 10 Band×Volume Features:")
imp = 0.0
# Average importance from ensemble
for seed in seeds:
    m = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=HP["max_depth"],
        learning_rate=HP["lr"],
        subsample=HP["ss"],
        colsample_bytree=HP["cs"],
        reg_alpha=HP["ra"],
        reg_lambda=HP["rl"],
        scale_pos_weight=1.0,
        random_state=seed,
        verbosity=0,
    )
    m.fit(X[years >= test_years[-1]], y[years >= test_years[-1]], verbose=False)
    imp += m.feature_importances_
imp /= len(seeds)
top = np.argsort(imp)[-10:][::-1]
for i in top:
    print(f"    {fnames[i]:<40} {imp[i]:.4f}")

print("\nDone.")
