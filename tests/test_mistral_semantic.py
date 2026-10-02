import pytest

from common import ALL_CASES, case_id, run_semantic_case
from ollama_runner.toogle_light import animation

MODEL = "ministral-3:3b"


@pytest.fixture(scope="module", autouse=True)
def after_all_tests():
    yield
    animation()


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=case_id,
)
def test_mistral_semantic(case):
    run_semantic_case(MODEL, case)
