import pytest


@pytest.mark.asyncio
async def test_documents_list_filters_by_tags_any(seeded_lnclite):
    page = await seeded_lnclite.documents.list(
        tags_any=["automation", "manual"],
        limit=10,
    )

    assert {doc.content.splitlines()[0] for doc in page.data} == {
        "# CI Integration for Test Automation",
        "# User Acceptance Testing (UAT) Process",
    }
    assert all(set(doc.tags) & {"automation", "manual"} for doc in page.data)
