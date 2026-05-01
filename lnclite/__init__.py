import functools
import hashlib
import logging
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Text,
    Type,
)

import diskcache
import lancedb
import numpy as np
import tiktoken
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
)
from openai_embeddings_model.normalize import normalize
from paginatic import TokenPaginatic
from paginatic.helpers import decode_and_verify, encode_and_sign
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


@functools.cache
def get_document_lancedb_model(dim: int) -> Type[LanceModel]:
    class DocumentLancedbModel(LanceModel):
        id: int = Field(default_factory=gen_id)
        content: Text = Field(description="The content of the document.")  # noqa: E501
        md5: Text = ""
        vector: Vector(dim)
        tags: List[Text] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_values(self) -> "DocumentLancedbModel":
            self.content = self.content.strip()
            if not self.content:
                raise ValueError("Content cannot be empty")
            self.md5 = hashlib.md5(self.content.encode()).hexdigest()
            return self

    return DocumentLancedbModel


@functools.cache
def get_default_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


@functools.cache
def get_default_openai_embeddings_model_name() -> str:
    return DEFAULT_OPENAI_MODEL


@functools.cache
def get_default_dimensions() -> int:
    return 1536


@functools.cache
def get_default_embeddings_cache() -> diskcache.Cache:
    return diskcache.Cache(".cache/embeddings")


@functools.cache
def get_default_openai_embeddings_model() -> "AsyncOpenAIEmbeddingsModel":
    return AsyncOpenAIEmbeddingsModel(
        model=get_default_openai_embeddings_model_name(),
        openai_client=get_default_openai_client(),
        cache=get_default_embeddings_cache(),
        encoding=tiktoken.encoding_for_model(
            get_default_openai_embeddings_model_name()
        ),
    )


@functools.cache
def get_default_model_settings() -> "ModelSettings":
    return ModelSettings(dimensions=get_default_dimensions())


class ManifestLancedbModel(LanceModel):
    id: int = Field(default_factory=gen_id)
    name: Text = Field(description="The name of the database.")
    description: Text = Field(description="The description of the database.")
    model: Text = Field(description="The embedding model name.")
    dimensions: int = Field(description="The dimensions of the embeddings.")
    last_updated: int = Field(default_factory=lambda: int(time.time()))


class ManifestModel(LanceModel):
    id: int
    name: Text
    description: Text
    model: Text
    dimensions: int
    last_updated: int


class DocumentCreate(BaseModel):
    content: Text
    tags: List[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self) -> "DocumentCreate":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Content cannot be empty")
        return self


class Document(BaseModel):
    id: int
    content: Text
    md5: Text
    vector: Optional[List[float]]
    tags: List[Text]


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
        token_secret_key: Text = "__lnclite__",
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

        self._secret_key = token_secret_key

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
        batch_size: int = 100,
    ) -> "Lnclite":
        from lnclite.file_ingestor import FileIngestor

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory {dir_path} not found")

        lancedb_path = Path(lancedb_path)
        if lancedb_path.is_dir():
            # If not a empty directory, raise an error
            for _ in lancedb_path.iterdir():
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

        batch: List[DocumentCreate] = []
        async for file in file_ingestor.ingest_async(dir_path):
            _file_content = file["content"].strip()
            _file_path = str(file["path"])

            if not _file_content:
                logger.warning(f"Skipping {_file_path} due to empty content")
                continue

            batch.append(
                DocumentCreate(content=_file_content, metadata={"path": _file_path})
            )

            if len(batch) >= batch_size:
                await lnclite.documents.batch_create(batch)
                logger.info(f"Created {len(batch)} documents")
                batch = []

        if batch:
            await lnclite.documents.batch_create(batch)
            logger.info(f"Created {len(batch)} documents")

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

    async def embed(self, texts: List[Text]) -> np.ndarray:
        emb_res = await self.openai_embeddings_model.get_embeddings(
            texts, model_settings=self.model_settings
        )
        return normalize(emb_res.to_numpy())  # (n, d)

    async def search(self, query: Text, *, limit: int = 5) -> "SearchResults":
        document_table = await self.documents.get_table()

        query_vector = (await self.embed([query]))[0]

        search_query = await document_table.search(query_vector)
        search_results: List[Dict] = (
            await search_query.distance_type("dot").limit(limit).to_list()
        )

        results: List[SearchResult] = []
        for result in search_results:
            _doc = Document.model_validate(result)
            _distance = result["_distance"]
            results.append(SearchResult(document=_doc, distance=_distance))

        return SearchResults(results=results)


class Manifest:
    def __init__(self, client: "Lnclite"):
        self.client = client
        self._table: lancedb.AsyncTable | None = None

    async def get_table(self) -> lancedb.AsyncTable:
        if self._table is not None:
            return self._table

        conn = await self.client.get_connection()
        if self.client.manifest_table in (await conn.table_names()):
            self._table = await conn.open_table(self.client.manifest_table)
        else:
            self._table = await conn.create_table(
                self.client.manifest_table, schema=self.client._manifest_lancedb_model
            )
        return self._table

    async def get(self) -> ManifestModel | None:
        manifest_table = await self.get_table()
        _query_builder = manifest_table.query()
        manifests = await _query_builder.limit(1).to_pydantic(
            self.client._manifest_lancedb_model
        )
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


class Documents:
    def __init__(self, client: "Lnclite"):
        self.client = client
        self._table: lancedb.AsyncTable | None = None

    async def get_table(self) -> lancedb.AsyncTable:
        if self._table is not None:
            return self._table

        conn = await self.client.get_connection()
        if self.client.document_table in (await conn.table_names()):
            self._table = await conn.open_table(self.client.document_table)
        else:
            self._table = await conn.create_table(
                self.client.document_table, schema=self.client._document_lancedb_model
            )
        return self._table

    async def count(self) -> int:
        document_table = await self.get_table()
        return await document_table.count_rows()

    async def get(self, id: int) -> Document | None:
        document_table = await self.get_table()
        documents = await (
            document_table.query()
            .where(f"id = {id}")
            .limit(1)
            .to_pydantic(self.client._document_lancedb_model)
        )
        if len(documents) > 0:
            return Document.model_validate_json(documents[0].model_dump_json())
        return None

    async def retrieve(self, id: int) -> Document:
        might_document = await self.get(id)
        if might_document is not None:
            return might_document
        raise LncliteNotFoundError(f"Document with id {id} not found")

    async def create(self, document_create: DocumentCreate) -> Document:
        return (await self.batch_create([document_create]))[0]

    async def batch_create(
        self, document_creates: List[DocumentCreate]
    ) -> List[Document]:
        document_table = await self.get_table()

        normalized_vectors = await self.client.embed(
            [d.content for d in document_creates]
        )

        documents = [
            self.client._document_lancedb_model(
                content=d.content, tags=d.tags, vector=v
            )
            for d, v in zip(document_creates, normalized_vectors)
        ]

        await document_table.add(documents)

        output: List[Document] = []
        for document, v in zip(documents, normalized_vectors):
            _doc = Document.model_validate_json(
                document.model_dump_json(exclude_none=True)
            )
            _doc.vector = v.tolist()
            output.append(_doc)

        return output

    async def list(
        self,
        *,
        limit: int = 10,
        order: Literal["asc", "desc", 1, -1] = "asc",
        next_page_token: Optional[Text] = None,
    ) -> TokenPaginatic[Document]:
        valid_order: Literal["ASC", "DESC"] = (
            "ASC" if order in ("asc", 1) else "DESC" if order in ("desc", -1) else None
        )
        if valid_order is None:
            raise ValueError(f"Invalid order: {order}")
        valid_op = ">" if valid_order == "ASC" else "<"

        if limit < 1:
            raise ValueError(f"Limit must be greater than 0, got {limit}")
        valid_limit = int(limit)

        after_id: Optional[int] = None
        if next_page_token is not None:
            decoded_token = decode_and_verify(next_page_token, self.client.__secret_key)
            after_id = decoded_token.get("after")

        document_table = await self.client.documents.get_table()

        query_builder = document_table.query()

        where_clause = (
            f"id {valid_op} {after_id}" if after_id is not None else "id > 0"
        ) + f" ORDER BY id {valid_order}"
        logger.info(f"Query: {where_clause}")

        documents = await (
            query_builder.where(where_clause)
            .limit(valid_limit + 1)
            .to_pydantic(self.client._document_lancedb_model)
        )

        has_more = len(documents) > valid_limit
        documents = documents[:valid_limit]

        _next_token = (
            encode_and_sign({"after": documents[-1].id}, self.client._secret_key)
            if has_more
            else None
        )

        return TokenPaginatic(
            object="list",
            data=[Document.model_validate_json(d.model_dump_json()) for d in documents],
            next_page_token=_next_token,
        )


class SearchResult(BaseModel):
    document: Document
    distance: float


class SearchResults(BaseModel):
    results: List[SearchResult]


class LncliteNotFoundError(Exception):
    pass
