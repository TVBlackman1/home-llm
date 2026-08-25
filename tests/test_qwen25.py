import pytest

from common import CASES, case_id, run_case


MODEL = "qwen2.5:3b"


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=case_id,
)
def test_qwen25(case):
    run_case(MODEL, case)