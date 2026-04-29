import functools
import hashlib
import logging
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Text,
    Type,
)

import diskcache
import lancedb
import tiktoken
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
)
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from lnclite.file_ingestor import FileReader

__version__: Final[Text] = "0.1.0"

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_TABLE = "manifest"
DEFAULT_DOCUMENT_TABLE = "documents"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TOKENS = 4096


def gen_id() -> str:
    from lnclite.utils.snowflake import generate_id

    return generate_id()


def get_document_lancedb_model(dim: int) -> Type[LanceModel]:
    class DocumentLancedbModel(LanceModel):
        id: Text = Field(default_factory=gen_id)
        content: Text = Field(description="The content of the document.")  # noqa: E501
        md5: Text = ""
        vector: Vector(dim)
        tags: List[Text] = Field(default_factory=list)
        metadata: Dict[Text, Text] = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_values(self) -> "DocumentLancedbModel":
            self.content = self.content.strip()
            if not self.content:
                raise ValueError("Content cannot be empty")
            self.md5 = hashlib.md5(self.content.encode()).hexdigest()
            return self

    return DocumentLancedbModel


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


class ManifestLancedbModel(LanceModel):
    id: Text = Field(default_factory=gen_id)
    name: Text = Field(description="The name of the database.")
    description: Text = Field(description="The description of the database.")
    model: Text = Field(description="The embedding model name.")
    dimensions: int = Field(description="The dimensions of the embeddings.")
    last_updated: int = Field(default_factory=lambda: int(time.time()))


class ManifestModel(LanceModel):
    id: Text
    name: Text
    description: Text
    model: Text
    dimensions: int
    last_updated: int


class DocumentCreate(BaseModel):
    content: Text
    tags: List[Text] = Field(default_factory=list)
    metadata: Dict[Text, Text] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "DocumentCreate":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Content cannot be empty")
        return self


class Document(BaseModel):
    id: Text
    content: Text
    md5: Text
    vector: Optional[str]
    tags: List[Text]
    metadata: Dict[Text, Text]


class Lnclite:
    def __init__(
        self,
        lancedb_path: Path | str | None = None,
        *,
        manifest_table: Text = DEFAULT_MANIFEST_TABLE,
        document_table: Text = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: "AsyncOpenAIEmbeddingsModel",
        model_settings: "ModelSettings",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.lancedb_path = Path(lancedb_path)
        self._connection: lancedb.AsyncConnection | None = None

        self.manifest_table = manifest_table
        self.document_table = document_table

        self.openai_embeddings_model = openai_embeddings_model
        self.model_settings = model_settings
        self.max_tokens = max_tokens

        self._document_lancedb_model: Type[LanceModel] = get_document_lancedb_model(
            self.model_settings.dimensions
        )
        self._manifest_lancedb_model = ManifestLancedbModel

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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extension_readers: Optional[Dict[str, "FileReader"]] = None,
    ) -> "Lnclite":
        from lnclite.file_ingestor import FileIngestor

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
            max_tokens=max_tokens,
        )
        await lnclite.manifest.upsert(
            name=dataset_name,
            description=dataset_description,
            model=openai_embeddings_model.model,
            dimensions=model_settings.dimensions,
        )

        file_ingestor = FileIngestor()
        if extension_readers is not None:
            for extension, reader in extension_readers.items():
                file_ingestor.register_reader(extension, reader)

        async for file in file_ingestor.ingest_async(dir_path):
            await lnclite.documents.add(
                content=file["content"],
                metadata={
                    "path": file["path"],
                },
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


class Manifest(Generic):
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
            return ManifestModel.model_validate_json(manifests[0].model_dump_json())
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
    ) -> ManifestModel:
        table = await self.get_table()
        might_manifest = await self.get()

        if might_manifest is None:
            manifest = self.client._manifest_lancedb_model(
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

        return ManifestModel.model_validate_json(manifest.model_dump_json())


class Documents(Generic):
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

    async def create(self, document_create: DocumentCreate) -> Document:
        document_table = await self.get_table()

        emb_res = await self.client.openai_embeddings_model.get_embeddings(
            document_create.content, model_settings=self.client.model_settings
        )

        document = self.client._document_lancedb_model(
            content=document_create.content,
            tags=document_create.tags,
            metadata=document_create.metadata,
            vector=emb_res.to_python()[0],
        )

        await document_table.add([document])

        output = Document.model_validate_json(
            document.model_dump_json(exclude_none=True)
        )
        output.vector = emb_res.output[0]  # In base64 format

        return output

    async def batch_create(
        self, document_creates: List[DocumentCreate]
    ) -> List[Document]:
        document_table = await self.get_table()

        emb_res = await self.client.openai_embeddings_model.get_embeddings(
            [d.content for d in document_creates],
            model_settings=self.client.model_settings,
        )

        documents = [
            self.client._document_lancedb_model(
                content=d.content, tags=d.tags, metadata=d.metadata, vector=v
            )
            for d, v in zip(document_creates, emb_res.to_python())
        ]

        await document_table.add(documents)

        output: List[Document] = []
        for document in documents:
            output.append(
                Document.model_validate_json(
                    document.model_dump_json(exclude_none=True)
                )
            )
            output.vector = emb_res.output[0]  # In base64 format

        return output


class LncliteNotFoundError(Exception):
    pass
