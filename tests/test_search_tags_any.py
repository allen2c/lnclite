import pytest


@pytest.mark.asyncio
async def test_search_filters_by_tags_any(seeded_lnclite):
    results = await seeded_lnclite.search(
        "business stakeholders and staging environment",
        tags_any=["manual", "security"],
        limit=5,
    )

    assert {result.document.content.splitlines()[0] for result in results.results} == {
        "# Negative Testing and Error Path Validation",
        "# User Acceptance Testing (UAT) Process",
    }
    assert all(
        set(result.document.tags) & {"manual", "security"} for result in results.results
    )
