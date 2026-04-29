import functools
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Final, List, Text, Type

import diskcache
import lancedb
import tiktoken
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
)
from pydantic import Field, model_validator

__version__: Final[Text] = "0.1.0"

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_TABLE = "manifest"
DEFAULT_DOCUMENT_TABLE = "documents"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def gen_id() -> str:
    from lnclite.utils.snowflake import generate_id

    return generate_id()


def get_document_model(dim: int) -> Type[LanceModel]:
    class Document(LanceModel):
        id: Text = Field(default_factory=gen_id)
        content: Text = Field(description="The content of the document.")  # noqa: E501
        md5: Text = ""
        vector: Vector(dim)
        tags: List[Text] = Field(default_factory=list)
        metadata: Dict[Text, Text] = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_values(self) -> "Document":
            self.content = self.content.strip()
            if not self.content:
                raise ValueError("Content cannot be empty")
            self.md5 = hashlib.md5(self.content.encode()).hexdigest()
            return self

    return Document


def get_default_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


def get_default_openai_embeddings_model_name() -> str:
    return DEFAULT_OPENAI_MODEL


def get_default_dimensions() -> int:
    return 1536


def get_default_embeddings_cache() -> diskcache.Cache:
    return diskcache.Cache(".cache/embeddings")


def get_default_openai_embeddings_model() -> "AsyncOpenAIEmbeddingsModel":
    return AsyncOpenAIEmbeddingsModel(
        model=get_default_openai_embeddings_model(),
        openai_client=get_default_openai_client(),
        cache=get_default_embeddings_cache(),
        encoding=tiktoken.encoding_for_model(get_default_openai_embeddings_model()),
    )


def get_default_model_settings() -> "ModelSettings":
    return ModelSettings(dimensions=get_default_dimensions())


class ManifestModel(LanceModel):
    id: Text = Field(default_factory=gen_id)
    name: Text = Field(description="The name of the database.")
    description: Text = Field(description="The description of the database.")
    model: Text = Field(description="The embedding model name.")
    dimensions: int = Field(description="The dimensions of the embeddings.")
    last_updated: int = Field(default_factory=lambda: int(time.time()))


class Lnclite:
    def __init__(
        self,
        lancedb_path: Path | str | None = None,
        *,
        manifest_table: Text = DEFAULT_MANIFEST_TABLE,
        document_table: Text = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: "AsyncOpenAIEmbeddingsModel",
        model_settings: "ModelSettings",
    ):
        self.lancedb_path = Path(lancedb_path)
        self._connection: lancedb.AsyncConnection | None = None

        self.manifest_table = manifest_table
        self.document_table = document_table

        self.openai_embeddings_model = openai_embeddings_model
        self.model_settings = model_settings

        self.document_model = get_document_model(self.model_settings.dimensions)
        self.manifest_model = ManifestModel

    @classmethod
    async def build_from_dir(
        cls,
        dir_path: Path | str,
        lancedb_path: Path | str,
        *,
        dataset_name: Text,
        dataset_description: Text,
        manifest_table: Text = DEFAULT_MANIFEST_TABLE,
        document_table: Text = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: "AsyncOpenAIEmbeddingsModel",
        model_settings: "ModelSettings",
    ) -> "Lnclite":
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory {dir_path} not found")

        lancedb_path = Path(lancedb_path)
        if lancedb_path.is_dir():
            raise ValueError(f"Lancedb path {lancedb_path} already exists ")

        lnclite = cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
        )
        await lnclite.upsert(
            name=dataset_name,
            description=dataset_description,
            model=openai_embeddings_model.model,
            dimensions=model_settings.dimensions,
        )
        return lnclite

    async def get_connection(self) -> lancedb.AsyncConnection:
        if self._connection is None:
            self._connection = await lancedb.connect_async(self.lancedb_path)
            logger.info(f"Lancedb connected to {self.lancedb_path}")
        return self._connection

    @functools.cached_property
    def manifest(self) -> "Manifest":
        return Manifest(self)

    @functools.cached_property
    def documents(self) -> "Documents":
        return Documents(self)


class Manifest:
    def __init__(self, client: "Lnclite"):
        self.client = client

    @functools.cache
    async def get_table(self) -> lancedb.AsyncTable:
        return (await self.client.get_connection()).open_table(
            self.client.manifest_table
        )

    async def get(self) -> ManifestModel | None:
        manifest_table = await self.get_table()
        _query_builder = await manifest_table.search()
        manifests = await _query_builder.limit(1).to_pydantic(self.manifest_model)
        if len(manifests) > 0:
            return manifests[0]
        return None

    async def retrieve(self) -> ManifestModel:
        might_manifest = await self.get()
        if might_manifest is not None:
            return might_manifest
        raise LncliteNotFoundError("Manifest not found")

    async def upsert(
        self,
        *,
        name: Text,
        description: Text,
        model: Text,
        dimensions: int,
    ):
        table = await self.get_table()
        might_manifest = await self.get()

        if might_manifest is None:
            manifest = ManifestModel(
                name=name, description=description, model=model, dimensions=dimensions
            )
            await table.add([manifest])

        else:
            manifest = might_manifest
            manifest.name = name
            manifest.description = description
            manifest.model = model
            manifest.dimensions = dimensions
            manifest.last_updated = int(time.time())
            await table.update(where=f"id = {manifest.id}", values=manifest)

        return manifest


class Documents:
    def __init__(self, client: "Lnclite"):
        self.client = client

    @functools.cache
    async def get_table(self) -> lancedb.AsyncTable:
        return (await self.client.get_connection()).open_table(
            self.client.document_table
        )

    async def count(self) -> int:
        document_table = await self.get_table()
        return await document_table.count_rows()


class LncliteNotFoundError(Exception):
    pass
