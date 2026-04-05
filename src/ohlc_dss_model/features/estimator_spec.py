from dataclasses import dataclass

@dataclass
class Spec:
    open: str
    high_cols: list[str]
    low_cols: list[str]
    close: str
    prev_close: str
    label: str


HISTORICAL_SPEC = Spec(
    open="O_Asia",
    high_cols=["H_Asia", "H_London", "H_New York"],
    low_cols=["L_Asia", "L_London", "L_New York"],
    close="C_New York",
    prev_close="C_New York",
    label="Sigma_Historical",
)

TODAY_SPEC = Spec(
    open="O_Asia",
    high_cols=["H_Asia", "H_London"],
    low_cols=["L_Asia", "L_London"],
    close="C_London",
    prev_close="C_New York",
    label="Sigma_Today",
)