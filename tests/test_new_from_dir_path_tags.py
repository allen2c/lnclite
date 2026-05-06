import pytest
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import Lnclite, get_openai_embeddings_model


@pytest.mark.asyncio
async def test_new_from_dir_stores_relative_path_as_tag(tmp_path):
    source_dir = tmp_path / "source"
    notes_dir = source_dir / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "release.md").write_text(
        "Release notes for vector search.",
        encoding="utf-8",
    )

    client = await Lnclite.new_from_dir(
        dir_path=source_dir,
        lancedb_path=tmp_path / "lancedb",
        dataset_name="Test dataset",
        dataset_description="Directory ingest test dataset",
        openai_embeddings_model=get_openai_embeddings_model(
            openai_client=AsyncOpenAI(),
        ),
        model_settings=ModelSettings(dimensions=1536),
    )

    page = await client.documents.list(tags_all=["path:notes/release.md"])

    assert len(page.data) == 1
    assert page.data[0].content == "Release notes for vector search."
    assert page.data[0].tags == ["path:notes/release.md"]
