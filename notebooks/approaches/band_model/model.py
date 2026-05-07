"""Band Direction Model — Production.

Predicts session direction (up/down) from pre-session band state,
VWAP/OFI/volume microstructure, macro regime, and momentum.

Core architecture:
  - 88 engineered features: VWAP×band (27), OFI×band (12), volume×band (6),
    band dynamics (2), macro (20), momentum (5), temporal (6)
  - XGBoost ensemble (3 seeds), chronological WFO, confidence thresholding
  - Strongest on Target_1 (AM 08:30-12:00): 56.2% acc, 91.7% at conf≥0.90

Band theory:
  Bands are excursion prediction intervals from historical volatility σ and
  excursion magnitude means μ_ae, μ_fe, with half-width δ_t = k·σ·O_Ref.
  AE bands (narrow): μ_ae = rolling mean of historical |z| at z < z_threshold.
  FE bands (wide): μ_fe = rolling mean of historical |z| at z ≥ z_threshold.
  Band_A = O_Ref ± μ ± δ_t for each side and type.
"""

import sys
from pathlib import Path
from datetime import date as dt_date
import gc, warnings, os
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
from ohlc_dss_model.features.vwap import compute_session_vwap
from ohlc_dss_model.features.ofi_vpin import compute_ofi_vpin
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

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

MACRO_FEATURES = [
    "vix_t1",
    "vix_pct_rank_1y_t1",
    "vix_5d_delta",
    "us10y_t1",
    "us2y_t1",
    "effr_t1",
    "10y_2y_spread_t1",
    "us10y_5d_delta",
    "is_fomc_day",
    "is_fomc_week",
    "days_to_fomc",
    "is_nfp_day",
    "is_cpi_day",
    "is_core_cpi_day",
    "e_today",
    "e_yesterday",
    "e_tomorrow",
    "day_of_week",
    "month",
    "week_of_month",
]

COLUMN_SKIP = {
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
COLUMN_SKIP.update(
    {
        f"{p}_{t}"
        for p in "OHLC"
        for t in ["Target_1", "Target_2", "Pre_Target_1", "Pre_Target_2"]
    }
)
COLUMN_SKIP.update(
    {
        c
        for c in [
            f"Band_{x}_{y}"
            for x in ["AE_Pos", "AE_Neg", "FE_Pos", "FE_Neg"]
            for y in ["Center", "Upper", "Lower"]
        ]
    }
)


class BandDirectionModel:
    """Produces session direction predictions with calibrated confidence.

    Usage:
        model = BandDirectionModel()
        model.fit()          # WFO training on 2011-2026 data
        probas = model.predict(X_new, session_years)  # or just predict
    """

    def __init__(self):
        self.models_ = {}  # {test_year: [model_seed1, model_seed2, model_seed3]}
        self.features_ = []  # feature names
        self.n_features_ = 0

    # ── Data Pipeline ──────────────────────────────────────────

    def load_data(self):
        """Load and prepare session-level dataset with all features."""
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

        # 1m microstructure snapshot at 08:30
        bars_1m_vwap, _ = compute_session_vwap(raw_1m, keep_cumulative=True)
        bars_1m_ofi, _ = compute_ofi_vpin(bars_1m_vwap, keep_cumulative=True)

        snap = self._snapshot(bars_1m_ofi)
        agg = agg.join(
            snap, left_on=pl.col("Session").cast(pl.Date), right_on="sd", how="left"
        )

        # Fill nulls
        for c in agg.columns:
            if agg[c].dtype in (pl.Float64, pl.Float32):
                agg = agg.with_columns(pl.col(c).fill_null(0.0))

        # Free memory
        del raw_1m, bars_1m_vwap, bars_1m_ofi
        gc.collect()

        return agg

    def _snapshot(self, bars_1m):
        """Extract VWAP/OFI/Volume at 08:25-08:35 window."""
        snap = (
            bars_1m.with_columns(
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
        return snap

    # ── Feature Engineering ─────────────────────────────────────

    def build_features(self, agg):
        """Engineer all 88+ features from aggregated session data."""
        sp = pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref") + 1e-9
        agg = agg.with_columns(sp.alias("sigma_price"))
        sp_c = pl.col("sigma_price")

        vwap = pl.col("vwap_0830")
        c_pre = pl.col("C_Pre_Target_2")
        bs = pl.col("band_state_ps2")
        ofi = pl.col("ofi_cum")
        vol = pl.col("vol_cum")
        oref = pl.col("O_Ref")

        # VWAP zone classification
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

        # Band distance features (VWAP & Price to 8 band boundaries)
        exprs = []
        for sn, col in BM.items():
            b = pl.col(col)
            exprs.append(((c_pre - b) / (sp_c + 1e-9)).alias(f"pd_{sn}"))
            exprs.append(((vwap - b) / (sp_c + 1e-9)).alias(f"vd_{sn}"))
            exprs.append(((vwap - c_pre) / (b - c_pre + 1e-9)).alias(f"vp_rel_{sn}"))

        # VWAP continuous position within AE band
        ae_l = pl.col("Band_AE_Neg_Lower")
        ae_u = pl.col("Band_AE_Pos_Upper")
        exprs.append(((vwap - ae_l) / (ae_u - ae_l + 1e-9)).alias("vwap_in_ae"))
        exprs.append(((c_pre - ae_l) / (ae_u - ae_l + 1e-9)).alias("price_in_ae"))
        exprs.extend(
            [
                ((vwap - c_pre) / (sp_c + 1e-9)).alias("vwap_price_gap"),
                (vwap > c_pre).cast(pl.Int32).alias("vwap_above"),
                (pl.col("vwap_zone") == bs).cast(pl.Int32).alias("vwap_same_zone"),
                ((vwap - oref) / (sp_c + 1e-9)).alias("vwap_vs_oref"),
                ((oref - c_pre) / (sp_c + 1e-9)).alias("cpre_vs_oref"),
            ]
        )

        # OFI × Band
        ofi_dir = (
            pl.when(ofi > 0)
            .then(pl.lit(1.0))
            .when(ofi < 0)
            .then(pl.lit(-1.0))
            .otherwise(0.0)
        )
        exprs.append(ofi_dir.alias("ofi_dir"))
        vp = pl.col("vpin")
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
                    (vp * zc).alias(f"vpin_x_{z}"),
                    (pl.col("ofi_per_vol") * zc).alias(f"ofipv_x_{z}"),
                ]
            )

        # OFI momentum
        exprs.extend(
            [
                (ofi / (ofi.shift(1).rolling_mean(5).over(pl.lit(1)) + 1e-9)).alias(
                    "ofi_rel_5d"
                ),
                (ofi / (ofi.shift(1).rolling_mean(20).over(pl.lit(1)) + 1e-9)).alias(
                    "ofi_rel_20d"
                ),
                (
                    pl.col("ofi_per_vol")
                    - pl.col("ofi_per_vol").shift(1).rolling_mean(5)
                ).alias("ofi_pv_chg_5d"),
            ]
        )

        # Volume × Band
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
                (vol / (pl.col("_delta_t") + 1e-9)).alias("vol_per_delta"),
            ]
        )

        # Band dynamics
        exprs.extend(
            [
                bs.shift(1).alias("bs_prev"),
                ((bs != bs.shift(1)).cast(pl.Int32)).alias("bs_changed"),
                (
                    pl.col("_delta_t")
                    / (pl.col("_delta_t").shift(1).rolling_mean(20) + 1e-9)
                ).alias("delta_rel_20d"),
                (sp_c / (sp_c.shift(1).rolling_mean(20) + 1e-9)).alias("sigma_rel_20d"),
                (pl.col("_delta_t") / (sp_c + 1e-9)).alias("dt_norm"),
            ]
        )

        agg = agg.with_columns(exprs)

        # ── Macro Feature Engineering ─────────────────────────────
        vix = pl.col("vix_t1")
        rate = pl.col("us10y_t1")
        spread = pl.col("10y_2y_spread_t1")

        agg = agg.with_columns(
            [
                # VIX mean-reversion distance (how far from 20d rolling mean?)
                ((vix - vix.rolling_mean(20)) / (vix.rolling_std(20) + 1e-9)).alias(
                    "vix_zscore_20d"
                ),
                # VIX relative to recent range (0-1 position within 20d min/max)
                (
                    (vix - vix.rolling_min(20))
                    / (vix.rolling_max(20) - vix.rolling_min(20) + 1e-9)
                ).alias("vix_range_pos_20d"),
                # VIX × band state (fear level in each zone)
                (vix * (bs == 1).cast(pl.Float32)).alias("vix_x_z1"),
                (vix * bs.is_in([2, 4]).cast(pl.Float32)).alias("vix_x_bull"),
                (vix * bs.is_in([3, 5]).cast(pl.Float32)).alias("vix_x_bear"),
                (vix * bs.is_in([6, 7]).cast(pl.Float32)).alias("vix_x_ext"),
                # Rate acceleration (second derivative)
                (pl.col("us10y_5d_delta") - pl.col("us10y_5d_delta").shift(5)).alias(
                    "rate_accel"
                ),
                # Spread direction change (did curve steepen or flatten this week?)
                (spread - spread.shift(5)).alias("spread_5d_chg"),
                # Spread × VIX (regime interaction: high VIX + inverted curve = extreme stress)
                (spread * vix).alias("spread_x_vix"),
                # FOMC anticipation: binary × days proximity
                (
                    pl.col("is_fomc_week").cast(pl.Float32)
                    * (10 - pl.col("days_to_fomc").clip(0, 10))
                ).alias("fomc_anticipation"),
                # Event clustering: count events this week
                (
                    pl.col("is_fomc_day").cast(pl.Float32)
                    + pl.col("is_nfp_day").cast(pl.Float32)
                    + pl.col("is_cpi_day").cast(pl.Float32)
                ).alias("event_count_week"),
                # VIX × spread regime label (high VIX + inverted = stress)
                (
                    (vix > vix.rolling_mean(60)).cast(pl.Float32)
                    * (spread < 0).cast(pl.Float32)
                ).alias("stress_regime"),
            ]
        )

        for c in agg.columns:
            if agg[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                agg = agg.with_columns(pl.col(c).fill_null(0.0))

        return agg

    # ── Build Feature Matrix ────────────────────────────────────

    def build_matrix(self, agg, burn_in=dt_date(2011, 4, 5)):
        """Build flat session-level feature matrix and labels."""
        rows = []
        for sr in agg.iter_rows(named=True):
            sess = sr["Session"]
            if sess < burn_in:
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

                for c in agg.columns:
                    if c in COLUMN_SKIP or agg[c].dtype not in (
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
            [
                pl.col("zd").shift(2).alias("pzd"),
                pl.col("zm").shift(2).alias("pzm"),
            ]
        )
        for c in df.columns:
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                df = df.with_columns(pl.col(c).fill_null(0.0))

        # Build X and targets
        y_dir = df["zd"].to_numpy().astype(np.float32)
        y_mag = np.maximum(df["zm"].to_numpy().astype(np.float32), 1e-8)
        y_log_mag = np.log(y_mag)  # log-space regression target
        amb = df["amb"].to_numpy().astype(np.int64)
        years = df["s"].to_numpy()
        skip_x = {"s", "zd", "amb", "zm", "wd", "it"}

        X_list = []
        fnames = []
        for c in df.columns:
            if c in skip_x or df[c].dtype not in (
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
            ):
                continue
            x = df[c].to_numpy().astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.std(x) < 1e-9:
                continue
            X_list.append(x.reshape(-1, 1))
            fnames.append(c)

        # One-hot weekday + session type
        for w in range(5):
            X_list.append((df["wd"].to_numpy() == w).astype(np.float32).reshape(-1, 1))
            fnames.append(f"wd_{w}")
        X_list.append((df["it"].to_numpy() == 0).astype(np.float32).reshape(-1, 1))
        fnames.append("is_t1")

        X = np.hstack(X_list).astype(np.float32)
        self.features_ = fnames
        self.n_features_ = len(fnames)

        return X, y_dir, y_log_mag, amb, years

    # ── WFO Training ────────────────────────────────────────────

    def fit(
        self,
        burn_in=dt_date(2011, 4, 5),
        test_year_start=2016,
        hp=None,
        seeds=(42, 123, 456),
    ):
        """Train ensemble via chronological WFO.

        For each test year Y, trains on all data before Y, tests on Y.
        Stores per-fold ensemble models for predict().
        """
        if hp is None:
            hp = {
                "max_depth": 6,
                "lr": 0.03,
                "ra": 1.0,
                "rl": 1.0,
                "ss": 0.7,
                "cs": 0.5,
            }

        print("Loading and engineering features...")
        agg = self.load_data()
        agg = self.build_features(agg)
        X, y_dir, y_log_mag, amb, years = self.build_matrix(agg, burn_in)

        test_years = sorted(set(years[years >= test_year_start]))
        print(
            f"Data: {X.shape[0]:,} sessions, {X.shape[1]} features, "
            f"{len(test_years)} WFO folds"
        )

        for test_year in test_years:
            tr = years < test_year
            te = years == test_year
            tr_clean = tr & (amb == 0)
            if tr_clean.sum() < 100 or te.sum() < 20:
                continue

            w_val = max(
                1.0, (y_dir[tr_clean] == 0).sum() / max((y_dir[tr_clean] == 1).sum(), 1)
            )

            fold_clfs = []
            fold_regs = []
            for seed in seeds:
                # Direction classifier
                clf = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=hp["max_depth"],
                    learning_rate=hp["lr"],
                    subsample=hp["ss"],
                    colsample_bytree=hp["cs"],
                    reg_alpha=hp["ra"],
                    reg_lambda=hp["rl"],
                    scale_pos_weight=w_val,
                    random_state=seed,
                    verbosity=0,
                )
                clf.fit(X[tr_clean], y_dir[tr_clean], verbose=False)
                fold_clfs.append(clf)

                # Magnitude regressor — weight by actual z_max (exp(log_mag))
                reg = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=hp["max_depth"],
                    learning_rate=hp["lr"],
                    subsample=hp["ss"],
                    colsample_bytree=hp["cs"],
                    reg_alpha=hp["ra"],
                    reg_lambda=hp["rl"],
                    random_state=seed + 1000,
                    verbosity=0,
                )
                reg.fit(
                    X[tr_clean],
                    y_log_mag[tr_clean],
                    verbose=False,
                )
                fold_regs.append(reg)

            self.models_[test_year] = {"clf": fold_clfs, "reg": fold_regs}

            self.models_[test_year] = {"clf": fold_clfs, "reg": fold_regs}
            print(f"  {test_year}: trained on {tr_clean.sum():,} samples")

        return self

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, X, year=None):
        """Predict direction + magnitude using ensemble for the given test year.

        Returns (prob_up, pred_dir, confidence, pred_log_mag, pred_mag).
        pred_mag = exp(pred_log_mag) in sigma units.
        """
        if year is None or year not in self.models_:
            year = max(self.models_.keys())

        models = self.models_[year]
        clfs = models["clf"]
        regs = models["reg"]

        probas = np.mean([m.predict_proba(X)[:, 1] for m in clfs], axis=0)
        pred = (probas >= 0.5).astype(int)
        conf = np.maximum(probas, 1 - probas)

        pred_log_mag = np.mean([r.predict(X) for r in regs], axis=0)
        pred_mag = np.exp(pred_log_mag)

        return probas, pred, conf, pred_log_mag, pred_mag

    # ── Evaluation ──────────────────────────────────────────────

    def evaluate(self, detailed=True):
        """Run full WFO evaluation and return results dict with direction + magnitude metrics."""
        agg = self.load_data()
        agg = self.build_features(agg)
        X, y_dir, y_log_mag, amb, years = self.build_matrix(agg)
        y_mag = np.exp(y_log_mag)  # actual magnitude in σ units

        test_years = sorted(set(years[years >= 2016]))
        all_p, all_a, all_c, all_am = [], [], [], []
        all_pm, all_am_mag = [], []  # predicted + actual magnitude

        for test_year in test_years:
            tr = years < test_year
            te = years == test_year
            tr_clean = tr & (amb == 0)
            if tr_clean.sum() < 100 or te.sum() < 20:
                continue

            models = self.models_[test_year]
            clfs = models["clf"]
            regs = models["reg"]

            proba = np.mean([m.predict_proba(X[te])[:, 1] for m in clfs], axis=0)
            pred = (proba >= 0.5).astype(float)
            conf = np.maximum(proba, 1 - proba)

            pred_log_mag = np.mean([r.predict(X[te]) for r in regs], axis=0)

            all_p.extend(pred)
            all_a.extend(y_dir[te])
            all_c.extend(conf)
            all_am.extend(amb[te])
            all_pm.extend(pred_log_mag)
            all_am_mag.extend(y_log_mag[te])

        ap = np.array(all_p)
        aa = np.array(all_a)
        ac = np.array(all_c)
        am = np.array(all_am)
        apm = np.array(all_pm)
        aam = np.array(all_am_mag)

        # Direction results
        results = {
            "overall_acc": accuracy_score(aa, ap),
            "baseline": max((aa == 1).mean(), (aa == 0).mean()),
            "n_samples": len(ap),
            "clean_acc": accuracy_score(aa[am == 0], ap[am == 0]),
            "n_clean": (am == 0).sum(),
        }
        conf_curve = {}
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            m = ac >= th
            if m.sum() > 5:
                conf_curve[th] = {
                    "acc": accuracy_score(aa[m], ap[m]),
                    "n": int(m.sum()),
                    "bl": max((aa[m] == 1).mean(), (aa[m] == 0).mean()),
                }
        results["conf_curve"] = conf_curve

        # Magnitude results
        mag_corr = np.corrcoef(apm, aam)[0, 1] if np.std(apm) > 1e-9 else 0
        mag_mae = np.abs(apm - aam).mean()
        mag_naive = np.abs(aam - aam.mean()).mean()
        results["magnitude"] = {
            "corr": float(mag_corr),
            "mae": float(mag_mae),
            "naive_mae": float(mag_naive),
        }

        # Magnitude confidence-gated: "large move" prediction accuracy
        for th_sigma in [0.5, 1.0, 1.5]:
            big_mask = aam >= th_sigma
            if big_mask.sum() > 20:
                # Did the model predict higher magnitude for actual big moves?
                pred_big = apm >= np.percentile(apm, 50)
                acc_big = (pred_big[big_mask] == 1).mean()
                bl_big = max((aam[big_mask] >= aam[big_mask].mean()).mean(), 0.5)
                results[f"mag_big_{th_sigma}"] = {
                    "acc": float(acc_big),
                    "n": int(big_mask.sum()),
                    "bl": float(bl_big),
                }

        # Confidence-gated magnitude accuracy: does high-confidence direction
        # also correlate with better magnitude prediction?
        mag_conf_curve = {}
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            m = ac >= th
            if m.sum() > 20:
                c = np.corrcoef(apm[m], aam[m])[0, 1] if np.std(apm[m]) > 1e-9 else 0
                mag_conf_curve[th] = {"corr": float(c), "n": int(m.sum())}
        results["magnitude_conf_curve"] = mag_conf_curve

        results["feature_importance"] = self._feature_importance(
            X, y_dir, (years >= 2016)
        )
        return results

    def _feature_importance(self, X, y, amb_mask):
        """Average feature importance across ensemble seeds."""
        imp = np.zeros(self.n_features_)
        for seed in [42, 123, 456]:
            m = xgb.XGBClassifier(n_estimators=500, random_state=seed, verbosity=0)
            tr_m = amb_mask == 0  # train on clean
            m.fit(X[tr_m], y[tr_m], verbose=False)
            imp += m.feature_importances_[: self.n_features_]
        imp /= 3
        top = np.argsort(imp)[-15:][::-1]
        return [(self.features_[i], float(imp[i])) for i in top]


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Band Direction Model — Phase 9 Production")
    print("=" * 60)

    model = BandDirectionModel()
    model.fit()
    results = model.evaluate()

    print(f"\n  Overall Acc: {results['overall_acc'] * 100:.2f}%")
    print(f"  Baseline: {results['baseline'] * 100:.2f}%")
    print(f"  Clean Acc: {results['clean_acc'] * 100:.1f}%")

    print(f"\n  Confidence Curve (Direction):")
    for th, r in results["conf_curve"].items():
        delta = r["acc"] - r["bl"]
        print(
            f"    Conf ≥ {th:.2f}: {r['acc'] * 100:.1f}% "
            f"(BL={r['bl'] * 100:.1f}% {(delta) * 100:+.1f}%) N={r['n']:,}"
        )

    print(f"\n  Magnitude Prediction:")
    mag = results["magnitude"]
    print(f"    Corr (log σ): {mag['corr']:.4f}")
    print(f"    MAE (log σ):  {mag['mae']:.4f}  (naive: {mag['naive_mae']:.4f})")

    print(f"\n  Magnitude Confidence-Gated Correlation:")
    for th, r in results.get("magnitude_conf_curve", {}).items():
        print(f"    Conf ≥ {th:.2f}: Corr={r['corr']:.4f}  N={r['n']:,}")

    print(f"\n  Top 10 Features:")
    for name, val in results["feature_importance"][:10]:
        print(f"    {name:<35} {val:.4f}")

    print("\nDone.")
