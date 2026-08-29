import pytest

from common import EMBED_CASES, case_id, run_embed_case


@pytest.mark.embed
@pytest.mark.parametrize("case", EMBED_CASES, ids=case_id)
def test_embed(case):
    run_embed_case(case)
