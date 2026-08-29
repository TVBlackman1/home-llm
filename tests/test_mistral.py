import pytest

from common import ALL_CASES, case_id, run_case


MODEL = "ministral-3:3b"


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=case_id,
)
def test_mistral(case):
    run_case(MODEL, case)