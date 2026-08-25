import pytest

from common import CASES, case_id, run_case


MODEL = "phi4-mini"


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=case_id,
)
def test_phi4mini(case):
    run_case(MODEL, case)