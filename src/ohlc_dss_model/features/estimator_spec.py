from dataclasses import dataclass

@dataclass
class Spec:
    open: str
    high_cols: list[str]
    low_cols: list[str]
    close: str
    prev_close: str
    label: str


FULL_DAY_SPEC = Spec(
    open="O_Pre_Target_1",
    high_cols=["H_Pre_Target_1", "H_Pre_Target_2", "H_Target_1", "H_Target_2"],
    low_cols=["L_Pre_Target_1", "L_Pre_Target_2", "L_Target_1", "L_Target_2"],
    close="C_Pre_Target_2",
    prev_close="C_Pre_Target_2",
    label="Sigma_Historical",
)

PRE_NY_SPEC = Spec(
    open="O_Pre_Target_1",
    high_cols=["H_Pre_Target_1", "H_Pre_Target_2"],
    low_cols=["L_Pre_Target_1", "L_Pre_Target_2"],
    close="C_Pre_Target_2",
    prev_close="C_Target_2",
    label="Sigma_Today",
)