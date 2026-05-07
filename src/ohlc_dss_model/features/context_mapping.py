from dataclasses import dataclass
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class ContextMapping:
    target_stage: int
    ps1_sessions: list[str]
    ps2_sessions: list[str]


CONTEXT_MAPPINGS = {
    1: ContextMapping(1, ["Pre_Target_1"], ["Pre_Target_2"]),
    2: ContextMapping(2, ["Pre_Target_1", "Pre_Target_2"], ["Target_1"]),
}


def _resolve_combined(
    session_features: pl.DataFrame,
    sessions: list[str],
    feat_col: str,
) -> Optional[str]:
    if len(sessions) == 1:
        col = f"{feat_col}_{sessions[0]}"
        return col if col in session_features.columns else None

    combined_name = f"{feat_col}_Pre_Combined"
    if combined_name in session_features.columns:
        return combined_name

    combined = [f"{feat_col}_{s}" for s in sessions]
    available = [c for c in combined if c in session_features.columns]
    if not available:
        return None
    if len(available) == 1:
        return available[0]
    return None


def map_1m_session_features(
    df_target: pl.DataFrame,
    session_features: pl.DataFrame,
    feature_prefixes: list[str],
) -> pl.DataFrame:
    stage = df_target.select(pl.col("target_stage").first()).item()
    mapping = CONTEXT_MAPPINGS.get(stage)
    if mapping is None:
        raise ValueError(f"Unknown target_stage: {stage}")

    rename_map = {}
    needed_cols = {"Session"}

    for feat in feature_prefixes:
        ps1 = _resolve_combined(session_features, mapping.ps1_sessions, feat)
        ps2 = _resolve_combined(session_features, mapping.ps2_sessions, feat)

        if ps1 is None or ps2 is None:
            rename_map[f"{feat}_ps1_missing"] = f"{feat}_ps1"
            rename_map[f"{feat}_ps2_missing"] = f"{feat}_ps2"
            session_features = session_features.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias(f"{feat}_ps1_missing"),
                    pl.lit(None, dtype=pl.Float64).alias(f"{feat}_ps2_missing"),
                ]
            )
            needed_cols.add(f"{feat}_ps1_missing")
            needed_cols.add(f"{feat}_ps2_missing")
        else:
            rename_map[ps1] = f"{feat}_ps1"
            rename_map[ps2] = f"{feat}_ps2"
            needed_cols.add(ps1)
            needed_cols.add(ps2)

    sf_subset = session_features.select(list(needed_cols))
    joined = df_target.join(sf_subset, on="Session", how="left")
    return joined.rename(rename_map)
