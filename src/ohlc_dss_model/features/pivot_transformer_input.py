import numpy as np
import polars as pl

from ohlc_dss_model.config import config

def _get_sigma_historical_shifted(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("Sigma_Historical").shift(1).alias("Sigma_Historical_Shifted")
    )

def _calculate_normalized_ohlc(df: pl.DataFrame) -> pl.DataFrame:
    sigma_price = pl.col("Sigma_Historical_Shifted") * pl.col("O_Ref")
    return df.with_columns([
        ((pl.col("H_Pre_Target_1") - pl.col("O_Ref")) / sigma_price).alias("H_Pre_Target_1_Normalized"),
        ((pl.col("L_Pre_Target_1") - pl.col("O_Ref")) / sigma_price).alias("L_Pre_Target_1_Normalized"),
        ((pl.col("C_Pre_Target_1") - pl.col("O_Ref")) / sigma_price).alias("C_Pre_Target_1_Normalized"),
        ((pl.col("O_Pre_Target_2") - pl.col("O_Ref")) / sigma_price).alias("O_Pre_Target_2_Normalized"),
        ((pl.col("H_Pre_Target_2") - pl.col("O_Ref")) / sigma_price).alias("H_Pre_Target_2_Normalized"),
        ((pl.col("L_Pre_Target_2") - pl.col("O_Ref")) / sigma_price).alias("L_Pre_Target_2_Normalized"),
        ((pl.col("C_Pre_Target_2") - pl.col("O_Ref")) / sigma_price).alias("C_Pre_Target_2_Normalized"),
    ])

def _build_context_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select(["Session", *config.pivot_transformer.context_whitelist])
            .drop_nulls(["Session", *config.pivot_transformer.context_whitelist])
            .sort("Session"))

def _label_construction(df: pl.DataFrame, target_col: str = "Target_1") -> pl.DataFrame:
    eps = 1e-8

    z_pos = ((pl.col(f"H_{target_col}") - pl.col(f"O_{target_col}")).clip(lower_bound=0) / (pl.col("Sigma_Historical") * pl.col(f"O_{target_col}") + eps))
    z_neg = ((pl.col(f"O_{target_col}") - pl.col(f"L_{target_col}")).clip(lower_bound=0) / (pl.col("Sigma_Historical") * pl.col(f"O_{target_col}") + eps))
    label_df = (
        df.select(["Session", f"H_{target_col}", f"O_{target_col}", f"L_{target_col}", "Sigma_Historical"]).with_columns([
            z_pos.alias("z_pos"),
            z_neg.alias("z_neg")
        ]).select(["Session", "z_pos", "z_neg"]).drop_nulls().sort("Session")
    )
    label_df = label_df.with_columns([
        (pl.max_horizontal(pl.col("z_pos"), pl.col("z_neg")).alias("z_max")),
    ])
    label_df = label_df.with_columns([
        ((pl.when((pl.col("z_max") >= 0.3) | ((pl.col("z_pos") - pl.col("z_neg")).abs() >= 0.3)).then(pl.lit(0)).otherwise(pl.lit(1))).alias("ambiguous")),
        ((pl.when(pl.col("z_pos") > pl.col("z_neg")).then(pl.lit(1)).otherwise(pl.lit(-1))).alias("z_dir"))
    ])
    return label_df.select(["Session", "z_max", "ambiguous", "z_dir"])

def _calculate_range_features(df: pl.DataFrame) -> pl.DataFrame:
    sigma_price = pl.col("Sigma_Historical_Shifted") * pl.col("O_Ref")
    eps = 1e-9
    df = df.with_columns([
        ((pl.col("H_Pre_Target_2") - pl.col("L_Pre_Target_2")) / sigma_price).alias("ps2_range_norm"),

        ((pl.col("H_Pre_Target_1") - pl.col("L_Pre_Target_1")) / sigma_price).alias("ps1_range_norm"),

        ((pl.col("C_Pre_Target_2") - pl.col("L_Pre_Target_2")) /
        (pl.col("H_Pre_Target_2") - pl.col("L_Pre_Target_2") + eps))
        .alias("ps2_close_pct_within_ps2_range"),

        ((pl.col("C_Pre_Target_1") - pl.col("L_Pre_Target_1")) /
        (pl.col("H_Pre_Target_1") - pl.col("L_Pre_Target_1") + eps))
        .alias("ps1_close_pct_within_ps1_range"),

        ((pl.col("O_Target_1") - pl.col("C_Pre_Target_2")) / sigma_price)
        .alias("target_open_gap_norm"),

        (pl.col("C_Pre_Target_1") > pl.col("O_Pre_Target_1")).alias("_ps1_bull"),
        (pl.col("C_Pre_Target_2") > pl.col("O_Pre_Target_2")).alias("_ps2_bull"),
    ]).with_columns([
        (pl.col("ps2_range_norm") / (pl.col("ps1_range_norm") + eps))
        .alias("ps2_vs_ps1_range_ratio"),

        (pl.col("_ps1_bull") == pl.col("_ps2_bull")).alias("sessions_agree"),
    ]).drop(["_ps1_bull", "_ps2_bull"])

def _calculate_ma_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("C_Target_2").shift(1).alias("_prior_close"),
    ]).with_columns([
        pl.col("_prior_close").rolling_mean(20).alias("_ma20"),
        pl.col("_prior_close").rolling_mean(200).alias("_ma200"),
        (pl.col("_prior_close") / pl.col("_prior_close").shift(5) - 1.0).alias("nq_5d_return"),
        (pl.col("_prior_close") / pl.col("_prior_close").shift(20) - 1.0).alias("nq_20d_return"),
    ]).with_columns([
        (pl.col("_prior_close") > pl.col("_ma20")).alias("nq_above_20d_ma"),
        (pl.col("_prior_close") > pl.col("_ma200")).alias("nq_above_200d_ma"),
        ((pl.col("_prior_close") - pl.col("_ma20")) /
        (pl.col("Sigma_Historical") * pl.col("O_Ref") + 1e-9))
        .alias("nq_dist_from_20d_ma_norm"),
    ]).drop(["_prior_close", "_ma20", "_ma200"])

def _get_ps2_band_state(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("C_Pre_Target_2") > pl.col("Band_FE_Pos_Upper")).then(pl.lit(6))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Pos_Lower")).then(pl.lit(4))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_AE_Pos_Upper")).then(pl.lit(2))
        .when(pl.col("C_Pre_Target_2") > pl.col("Band_AE_Neg_Upper")).then(pl.lit(1))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_AE_Neg_Lower")).then(pl.lit(1))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Neg_Upper")).then(pl.lit(3))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Neg_Lower")).then(pl.lit(5))
        .otherwise(pl.lit(7))
        .alias("band_state_ps2")
    )

def build_transformer_input(pivots_data: pl.DataFrame, aggregated_data: pl.DataFrame, target_col: str = "Target_1") -> dict:
    _df = _get_sigma_historical_shifted(aggregated_data)
    _df = _calculate_range_features(_df)
    _df = _calculate_ma_features(_df)
    _df = _get_ps2_band_state(_df)
    # _df = _calculate_normalized_ohlc(_df)
    ctx_df = _build_context_dataframe(_df)
    label_df = _label_construction(_df, target_col=target_col)

    required = ["Session", *config.pivot_transformer.pivot_numerical_whitelist, *config.pivot_transformer.pivot_categorical_whitelist]
    pivots_df = pivots_data.select(required)
    pivots_df = pivots_df.with_columns(pl.int_range(pl.len()).over("Session").alias("global_pos"))

    valid_sessions = (
        ctx_df.select("Session")
        .join(label_df.select("Session"), on="Session", how="inner")
        .join(pivots_df.select("Session").unique(), on="Session", how="inner")
        .sort("Session")
    )

    valid_sessions = valid_sessions.filter(pl.col("Session") >= config.pivot_transformer.burn_in_buffer)["Session"].to_list()

    intraday_session_map = {"Pre_Target_1": 1, "Pre_Target_2": 2}
    s_k_map = {-1: 0, 1: 1}

    Fc = len(config.pivot_transformer.context_whitelist)
    Fp = len(config.pivot_transformer.pivot_numerical_whitelist)
    Fcat = len(config.pivot_transformer.pivot_categorical_whitelist)

    max_pivots = config.pivot_transformer.max_pivots

    token_dim = 1 + Fc + Fp + Fcat
    T = 1 + max_pivots

    max_seen_pivots = 0
    truncated_days = 0

    x_tokens, mask, position = [], [], []
    z_max_labels, z_dir_labels, ambiguous_labels = [], [], []
    sessions = []

    for sess in valid_sessions:
        c_row = ctx_df.filter(pl.col("Session") == sess)
        p_day = pivots_df.filter(pl.col("Session") == sess)
        l_row = label_df.filter(pl.col("Session") == sess)

        if c_row.height != 1 or l_row.height != 1:
            continue

        max_seen_pivots = max(max_seen_pivots, p_day.height)
        if p_day.height > max_pivots:
            truncated_days += 1
            p_day = p_day.tail(max_pivots)

        x = np.zeros((T, token_dim), dtype=np.float32)
        m = np.zeros((T,), dtype=np.int64)
        p = np.zeros((T,), dtype=np.int64)

        ctx_vals = c_row.select(config.pivot_transformer.context_whitelist).to_numpy().astype(np.float32)[0]

        x[0, 0] = 1.0

        x[0, 1:1+Fc] = ctx_vals
        m[0] = 1
        p[0] = 0

        i = 1
        for row in p_day.iter_rows(named=True):
            if i >= T:
                break

            p_num = np.array([float(row[col]) for col in config.pivot_transformer.pivot_numerical_whitelist], dtype=np.float32)
            s_k_encoded = s_k_map.get(int(row["s_k"]), 0)
            intraday_session_encoded = intraday_session_map.get(str(row["Intraday_Session"]), 0)

            start = 1 + Fc
            x[i, start:start+Fp] = p_num
            x[i, start+Fp] = float(s_k_encoded)
            x[i, start+Fp+1] = float(intraday_session_encoded)

            m[i] = 1
            p[i] = int(row["global_pos"]) + 1
            i += 1
        x_tokens.append(x)
        mask.append(m)
        position.append(p)
        z_max_labels.append(float(l_row["z_max"][0]))
        z_dir_labels.append(int(l_row["z_dir"][0]))
        ambiguous_labels.append(int(l_row["ambiguous"][0]))
        sessions.append(sess)

    return {
        "X_Tokens": np.stack(x_tokens, axis=0),
        "Attention_Mask": np.stack(mask, axis=0),
        "Token_Position": np.stack(position, axis=0),
        "Z_Max_Labels": np.array(z_max_labels, dtype=np.float32),
        "Z_Dir_Labels": np.array(z_dir_labels, dtype=np.int64),
        "Z_Ambiguous_Labels": np.array(ambiguous_labels, dtype=np.int64),
        "Sessions": sessions,
        "Features_Metadata": {
            "Max_Pivots": max_pivots,
            "Height": T,
            "Token_Dim": token_dim,
            "Truncated_Days": truncated_days,
            "Max_Seen_Pivots": max_seen_pivots,
        }
    }
