"""Vector return behavior tests."""

import pytest


@pytest.mark.asyncio
async def test_documents_get_hides_vector_by_default(seeded_lnclite):
    page = await seeded_lnclite.documents.list(limit=1)
    doc = await seeded_lnclite.documents.get(page.data[0].id)

    assert doc is not None
    assert doc.vector is None


@pytest.mark.asyncio
async def test_documents_get_can_include_vector(seeded_lnclite):
    page = await seeded_lnclite.documents.list(limit=1)
    doc = await seeded_lnclite.documents.get(page.data[0].id, include_vector=True)

    assert doc is not None
    assert doc.vector is not None
    assert len(doc.vector) == 1536


@pytest.mark.asyncio
async def test_documents_list_hides_vectors_by_default(seeded_lnclite):
    page = await seeded_lnclite.documents.list(limit=2)

    assert page.data
    assert all(document.vector is None for document in page.data)


@pytest.mark.asyncio
async def test_documents_list_can_include_vectors(seeded_lnclite):
    page = await seeded_lnclite.documents.list(limit=2, include_vector=True)

    assert page.data
    assert all(document.vector is not None for document in page.data)


@pytest.mark.asyncio
async def test_search_hides_vectors_by_default(seeded_lnclite):
    results = await seeded_lnclite.search("test automation")

    assert results.results
    assert all(result.document.vector is None for result in results.results)


@pytest.mark.asyncio
async def test_search_can_include_vectors(seeded_lnclite):
    results = await seeded_lnclite.search("test automation", include_vector=True)

    assert results.results
    assert all(result.document.vector is not None for result in results.results)
