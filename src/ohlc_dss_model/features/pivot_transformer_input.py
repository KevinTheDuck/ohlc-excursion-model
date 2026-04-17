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
        ((pl.col("H_Asia") - pl.col("O_Ref")) / sigma_price).alias("H_Asia_Normalized"),
        ((pl.col("L_Asia") - pl.col("O_Ref")) / sigma_price).alias("L_Asia_Normalized"),
        ((pl.col("C_Asia") - pl.col("O_Ref")) / sigma_price).alias("C_Asia_Normalized"),
        ((pl.col("O_London") - pl.col("O_Ref")) / sigma_price).alias("O_London_Normalized"),
        ((pl.col("H_London") - pl.col("O_Ref")) / sigma_price).alias("H_London_Normalized"),
        ((pl.col("L_London") - pl.col("O_Ref")) / sigma_price).alias("L_London_Normalized"),
        ((pl.col("C_London") - pl.col("O_Ref")) / sigma_price).alias("C_London_Normalized"),
    ])

def _build_context_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select(["Session", *config.pivot_transformer.context_whitelist])
            .drop_nulls(["Session", *config.pivot_transformer.context_whitelist])
            .sort("Session"))

def _label_construction(df: pl.DataFrame) -> pl.DataFrame:
    eps = 1e-8

    z_pos = ((pl.col("H_New York") - pl.col("O_New York")).clip(lower_bound=0) / (pl.col("Sigma_Historical") * pl.col("O_New York") + eps))
    z_neg = ((pl.col("O_New York") - pl.col("L_New York")).clip(lower_bound=0) / (pl.col("Sigma_Historical") * pl.col("O_New York") + eps))
    label_df = (
        df.select(["Session", "H_New York", "O_New York", "L_New York", "Sigma_Historical"]).with_columns([
            z_pos.alias("z_pos"),
            z_neg.alias("z_neg")
        ]).select(["Session", "z_pos", "z_neg"]).drop_nulls().sort("Session")
    )
    return label_df

def build_transformer_input(pivots_data: pl.DataFrame, aggregated_data: pl.DataFrame) -> dict:
    _df = _get_sigma_historical_shifted(aggregated_data)
    _df = _calculate_normalized_ohlc(_df)
    ctx_df = _build_context_dataframe(_df)
    label_df = _label_construction(_df)

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

    intraday_session_map = {"Asia": 1, "London": 2}
    s_k_map = {-1: 1, 1: 2}

    Fc = len(config.pivot_transformer.context_whitelist)
    Fp = len(config.pivot_transformer.pivot_numerical_whitelist)
    Fcat = len(config.pivot_transformer.pivot_categorical_whitelist)

    max_pivots = config.pivot_transformer.max_pivots

    token_dim = 1 + Fc + Fp + Fcat
    T = 1 + max_pivots

    max_seen_pivots = 0
    truncated_days = 0

    x_tokens, mask, position = [], [], []
    z_pos_labels, z_neg_labels = [], []
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
        z_pos_labels.append(float(l_row["z_pos"][0]))
        z_neg_labels.append(float(l_row["z_neg"][0]))
        sessions.append(sess)

    return {
        "X_Tokens": np.stack(x_tokens, axis=0),
        "Attention_Mask": np.stack(mask, axis=0),
        "Token_Position": np.stack(position, axis=0),
        "Z_Pos_Labels": np.array(z_pos_labels, dtype=np.float32),
        "Z_Neg_Labels": np.array(z_neg_labels, dtype=np.float32),
        "Sessions": sessions,
        "Features_Metadata": {
            "Max_Pivots": max_pivots,
            "Height": T,
            "Token_Dim": token_dim,
            "Truncated_Days": truncated_days,
            "Max_Seen_Pivots": max_seen_pivots,
        }
    }
