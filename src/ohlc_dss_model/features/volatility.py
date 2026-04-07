import polars as pl

from ohlc_dss_model.config import Volatility_cfg
from ohlc_dss_model.features.estimator_spec import Spec

def _log_returns(spec: Spec):
    log_overnight = (
        pl.col(spec.open) / pl.col(spec.prev_close).shift(1)
    ).log()

    log_oc = (pl.col(spec.close) / pl.col(spec.open)).log()

    return log_overnight.alias("_log_overnight"), log_oc.alias("_log_oc")


def _rogers_satchell(spec: Spec):
    h = pl.max_horizontal(*spec.high_cols)
    l = pl.min_horizontal(*spec.low_cols)
    o = pl.col(spec.open)
    c = pl.col(spec.close)

    rs = (h / c).log() * (h / o).log() + (l / c).log() * (l / o).log()
    return rs.alias("_rs")

def _yang_zhang_rolling(df: pl.DataFrame, n: int, label: str) -> pl.DataFrame:
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    yz = (
        pl.col("_log_overnight").rolling_var(n)
        + k * pl.col("_log_oc").rolling_var(n)
        + (1 - k) * pl.col("_rs").rolling_mean(n)
    )

    return df.with_columns(yz.sqrt().alias(label))

def _yang_zhang_today(df: pl.DataFrame, label: str) -> pl.DataFrame:
    # No k here since its purely for today so no n needed
    yz = (
        pl.col("_log_overnight")**2
        + pl.col("_log_oc")**2
        + pl.col("_rs")
    )

    return df.with_columns(yz.sqrt().alias(label))

def yang_zhang(df: pl.DataFrame, spec: Spec, mode: str, n: int = Volatility_cfg.n):

    log_overnight, log_oc = _log_returns(spec)
    rs = _rogers_satchell(spec)

    df = df.with_columns([log_overnight, log_oc, rs])

    if mode == "historical":
        df = _yang_zhang_rolling(df, n=n, label=spec.label)
    else:
        df = _yang_zhang_today(df, label=spec.label)

    return df.drop(["_log_overnight", "_log_oc", "_rs"])