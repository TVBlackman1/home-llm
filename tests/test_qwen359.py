import pytest

from common import ALL_CASES, case_id, run_case


MODEL = "qwen3.5:9b"


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=case_id,
)
def test_qwen359(case):
    run_case(MODEL, case)