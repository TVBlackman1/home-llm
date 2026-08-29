import pytest

from common import ALL_CASES, case_id, run_case


MODEL = "phi4-mini"


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=case_id,
)
def test_phi4mini(case):
    run_case(MODEL, case)