import pytest


@pytest.mark.asyncio
async def test_search_filters_by_tags_all(seeded_lnclite):
    results = await seeded_lnclite.search(
        "ci pipeline automated tests",
        tags_all=["qa", "automation"],
        limit=5,
    )

    assert [result.document.content.splitlines()[0] for result in results.results] == [
        "# CI Integration for Test Automation",
    ]
    assert results.results[0].document.tags == ["qa", "automation", "ci"]
