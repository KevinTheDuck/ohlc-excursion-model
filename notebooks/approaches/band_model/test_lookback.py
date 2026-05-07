import sys
from pathlib import Path
from datetime import date as dt_date
import warnings, numpy as np, polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score

_PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ / "src"))

from ohlc_dss_model.data import load_raw_data
from ohlc_dss_model.data.pipeline_loaders import load_aggregated_data
from ohlc_dss_model.features.band_state import get_band_state_on_ps2
from ohlc_dss_model.features.momentum import calculate_momentum_features
from ohlc_dss_model.features.session_aggregation import aggregate_sessions
from ohlc_dss_model.features.volatility import yang_zhang
from ohlc_dss_model.features.excursion_bands import calculate_excursion_bands
from ohlc_dss_model.features.estimator_spec import FULL_DAY_SPEC, PRE_NY_SPEC

warnings.filterwarnings("ignore")
BURN_IN = dt_date(2011, 4, 5)

print("=" * 60)
print("  BAND LOOKBACK SWEEP — n = [7, 15, 30, 60]")
print("=" * 60)

for n_lookback in [7, 15, 30, 60]:
    print(f"\n── n={n_lookback} ──")

    raw = load_raw_data(str(_PROJ / "data" / "raw" / "nq_30m.parquet"))
    raw = raw.filter(pl.col("Session") >= BURN_IN)

    agg = aggregate_sessions(raw)
    agg = agg.with_columns(pl.col("C_Target_2").alias("_prior_close"))
    agg = yang_zhang(agg, FULL_DAY_SPEC, "historical", n=n_lookback)
    agg = yang_zhang(agg, PRE_NY_SPEC, "session", n=n_lookback)
    agg = calculate_excursion_bands(
        agg.with_columns(
            [
                pl.col("O_Pre_Target_1").alias("O_Ref"),
                pl.coalesce([pl.col("Sigma_Historical"), pl.col("Sigma_Today")]).alias(
                    "Sigma_Historical"
                ),
            ]
        ),
        n=n_lookback,
    )

    if True:
        fred = pl.read_parquet(
            str(_PROJ / "data" / "processed" / "fred_macro_table.parquet")
        )
        events = pl.read_parquet(
            str(_PROJ / "data" / "processed" / "individual_event_flags.parquet")
        )
        events = events.with_columns(pl.col("Session").cast(pl.Date))
        agg = agg.join(fred, on="Session", how="left")
        agg = agg.join(
            events,
            left_on=pl.col("Session").cast(pl.Date),
            right_on="Session",
            how="left",
            suffix="_event",
        )

    agg = get_band_state_on_ps2(agg)
    agg = calculate_momentum_features(agg)
    for c in agg.columns:
        if agg[c].dtype in (pl.Float64, pl.Float32):
            agg = agg.with_columns(pl.col(c).fill_null(0.0))
        elif agg[c].dtype == pl.Boolean:
            agg = agg.with_columns(pl.col(c).cast(pl.Float32).fill_null(0.0))

    rows = []
    for sr in agg.iter_rows(named=True):
        sess = sr["Session"]
        sig = sr.get("Sigma_Historical", 0) or 0.01
        sp = max(sig * (sr.get("O_Ref", 1) or 1), 1e-9)
        bs = int(sr.get("band_state_ps2", 0) or 0)
        wd = sess.weekday()

        for tcol, tn in [("Target_1", 0), ("Target_2", 1)]:
            o = sr.get(f"O_{tcol}", 0)
            h = sr.get(f"H_{tcol}", 0)
            l = sr.get(f"L_{tcol}", 0)
            if not o:
                continue
            zp = max(0, h - o) / (sp + 1e-9)
            zn = max(0, o - l) / (sp + 1e-9)
            zd = 1 if zp > zn else 0
            amb = 1 if max(zp, zn) < 0.3 and abs(zp - zn) < 0.3 else 0

            row = {
                "s": sess.year,
                "zd": zd,
                "amb": amb,
                "wd": wd,
                "bs": bs,
                "sp": sp,
                "dt": sr.get("_delta_t", 0) or 0,
            }

            for mc in ["nq_5d_return", "nq_20d_return", "nq_dist_ma_20_norm"]:
                row[mc] = float(sr.get(mc) or 0)
            rows.append(row)

    df = pl.DataFrame(rows)
    df = df.with_columns([pl.col("zd").shift(2).alias("pzd")])
    for c in df.columns:
        if df[c].dtype in (pl.Float64, pl.Float32):
            df = df.with_columns(pl.col(c).fill_null(0.0))

    X_list = []
    for z in range(1, 8):
        X_list.append((df["bs"].to_numpy() == z).astype(np.float32).reshape(-1, 1))
    for c in ["sp", "dt", "nq_5d_return", "nq_20d_return", "nq_dist_ma_20_norm", "pzd"]:
        X_list.append(
            np.nan_to_num(df[c].to_numpy().astype(np.float32), nan=0.0).reshape(-1, 1)
        )
    for w in range(5):
        X_list.append((df["wd"].to_numpy() == w).astype(np.float32).reshape(-1, 1))
    X_list.append((df["it"].to_numpy() == 0).astype(np.float32).reshape(-1, 1))
    X = np.hstack(X_list).astype(np.float32)

    y = df["zd"].to_numpy().astype(np.float32)
    amb = df["amb"].to_numpy().astype(np.int64)
    yrs = df["s"].to_numpy()

    tr = (yrs < 2024) & (amb == 0)
    te = yrs >= 2024
    w = max(1.0, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=w,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X[tr], y[tr], verbose=False)
    pred = clf.predict_proba(X[te])[:, 1]
    pred_b = (pred >= 0.5).astype(float)
    conf = np.maximum(pred, 1 - pred)

    acc = accuracy_score(y[te], pred_b)
    bl = max((y[te] == 1).mean(), (y[te] == 0).mean())
    c60 = (
        accuracy_score(y[te][conf >= 0.60], pred_b[conf >= 0.60])
        if (conf >= 0.60).sum() > 10
        else 0
    )
    c70 = (
        accuracy_score(y[te][conf >= 0.70], pred_b[conf >= 0.70])
        if (conf >= 0.70).sum() > 10
        else 0
    )
    c80 = (
        accuracy_score(y[te][conf >= 0.80], pred_b[conf >= 0.80])
        if (conf >= 0.80).sum() > 10
        else 0
    )

    print(
        f"  Acc={acc * 100:.1f}% BL={bl * 100:.1f}% C60={c60 * 100:.1f}% C70={c70 * 100:.1f}% C80={c80 * 100:.1f}%"
    )

print("\nDone.")
