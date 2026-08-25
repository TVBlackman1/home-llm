import pytest

from common import CASES, case_id, run_case


MODEL = "ministral-3:3b"


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=case_id,
)
def test_mistral(case):
    run_case(MODEL, case)