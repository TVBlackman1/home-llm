import pytest

from common import CASES, case_id, run_case


MODEL = "gemma3:4b"


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=case_id,
)
def test_gemma3(case):
    run_case(MODEL, case)