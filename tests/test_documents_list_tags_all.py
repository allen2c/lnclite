import pytest


@pytest.mark.asyncio
async def test_documents_list_filters_by_tags_all(seeded_lnclite):
    page = await seeded_lnclite.documents.list(
        tags_all=["qa", "security"],
        limit=10,
    )

    assert [doc.content.splitlines()[0] for doc in page.data] == [
        "# Negative Testing and Error Path Validation",
    ]
    assert page.data[0].tags == ["qa", "negative", "security"]
