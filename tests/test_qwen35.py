import pytest

from common import CASES, case_id, run_case


MODEL = "qwen3.5:4b"


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=case_id,
)
def test_qwen35(case):
    run_case(MODEL, case)