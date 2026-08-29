import pytest

from common import ALL_CASES, case_id, run_case


MODEL = "gemma3:4b"


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=case_id,
)
def test_gemma3(case):
    run_case(MODEL, case)