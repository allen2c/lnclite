"""Small async LanceDB document store with OpenAI embeddings."""

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
import lancedb.index
import numpy as np
import tiktoken
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI
from openai_embeddings_model import MAX_BATCH_SIZE as DEFAULT_EMBEDDINGS_MAX_BATCH_SIZE
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
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_DIMENSIONS = 1536

VectorIndexPreference = Literal["storage", "balanced", "accuracy", "latency"]
ListOrder = Literal["asc", "desc", 1, -1]
SqlOrder = Literal["ASC", "DESC"]


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
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


@functools.cache
def get_embeddings_cache() -> diskcache.Cache:
    return diskcache.Cache(".cache/embeddings")


@functools.cache
def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(
            f"Encoding for model {model} not found, using default encoding 'gpt-5"
        )
        return tiktoken.encoding_for_model("gpt-5")


def get_openai_embeddings_model(
    model: str = DEFAULT_OPENAI_MODEL,
    openai_client: AsyncOpenAI | None = None,
    cache: diskcache.Cache | None = None,
    encoding: tiktoken.Encoding | None = None,
    max_batch_size: int = DEFAULT_EMBEDDINGS_MAX_BATCH_SIZE,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> "AsyncOpenAIEmbeddingsModel":
    return AsyncOpenAIEmbeddingsModel(
        model=model,
        openai_client=openai_client or get_openai_client(),
        cache=cache or get_embeddings_cache(),
        encoding=encoding or get_encoding_for_model(model),
        max_batch_size=max_batch_size,
        max_input_tokens=max_input_tokens,
    )


def get_model_settings(dimensions: int = DEFAULT_DIMENSIONS) -> "ModelSettings":
    return ModelSettings(dimensions=dimensions)


def quote_sql_string(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def tag_filter_any(tags: list[str]) -> str:
    values = ", ".join(quote_sql_string(tag) for tag in tags)
    return f"array_has_any(tags, [{values}])"


def tag_filter_all(tags: list[str]) -> str:
    values = ", ".join(quote_sql_string(tag) for tag in tags)
    return f"array_has_all(tags, [{values}])"


def recommended_vector_index_config(
    row_count: int,
    dim: int,
    *,
    prefer: VectorIndexPreference = "balanced",
):
    """Return a LanceDB vector index config for dot-search.

    Assumes document vectors and query vectors are normalized.
    Returns None when brute-force is better or when there are not enough rows.
    """

    # Too small. Brute-force is exact and fast.
    # Also avoids PQ training errors.
    if row_count < 256:
        return None

    # Still small. Brute-force is usually fine.
    # If you really want an index, IvfFlat is safer than PQ.
    if row_count < 10_000:
        if prefer in {"accuracy", "latency"}:
            return lancedb.index.IvfFlat(
                distance_type="dot",
                num_partitions=32,
            )
        return None

    if row_count < 50_000:
        return lancedb.index.IvfFlat(
            distance_type="dot",
            num_partitions=128,
        )

    if row_count < 100_000:
        if prefer == "storage":
            return lancedb.index.IvfPq(
                distance_type="dot",
                num_partitions=256,
                num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
                num_bits=8,
            )

        return lancedb.index.IvfFlat(
            distance_type="dot",
            num_partitions=256,
        )

    if row_count < 500_000:
        return lancedb.index.IvfPq(
            distance_type="dot",
            num_partitions=1024 if prefer == "storage" else 2048,
            num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
            num_bits=8,
        )

    if row_count < 1_000_000:
        return lancedb.index.IvfPq(
            distance_type="dot",
            num_partitions=2048 if prefer == "storage" else 4096,
            num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
            num_bits=8,
        )

    if prefer == "latency":
        return lancedb.index.HnswPq(
            distance_type="dot",
            m=20,
            ef_construction=300,
            num_sub_vectors=recommended_num_sub_vectors(dim, "balanced"),
            num_bits=8,
        )

    return lancedb.index.IvfPq(
        distance_type="dot",
        num_partitions=4096 if prefer in {"storage", "balanced"} else 8192,
        num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
        num_bits=8,
    )


def recommended_num_sub_vectors(
    dim: int, prefer: VectorIndexPreference = "balanced"
) -> int:
    """Return a PQ subvector count.

    Higher = more accurate, larger index.
    Lower = more compressed, lower recall.
    """

    # Prefer subvector sizes around 8~16 dimensions.
    if prefer == "storage":
        target_sub_dim = 16
    elif prefer == "accuracy":
        target_sub_dim = 8
    else:
        target_sub_dim = 12

    candidates = [x for x in range(1, dim + 1) if dim % x == 0]
    return min(candidates, key=lambda x: abs((dim / x) - target_sub_dim))


async def get_default_dimensions(
    openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
) -> int:
    emb_result = await openai_embeddings_model.get_embeddings(
        input="Hello, world!",
        model_settings=ModelSettings(),
    )
    return emb_result.to_numpy().shape[1]


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
        token_secret_key: Text = "__lnclite__",
        vector_search_prefer: VectorIndexPreference = "balanced",
        verbose: bool = False,
    ):
        self.lancedb_path = Path(lancedb_path)
        self._connection: lancedb.AsyncConnection | None = None

        self.manifest_table = manifest_table
        self.document_table = document_table

        self.openai_embeddings_model = openai_embeddings_model
        self.model_settings = model_settings
        self.max_tokens = self.openai_embeddings_model._max_input_tokens

        self._secret_key = token_secret_key
        self.vector_search_prefer = vector_search_prefer
        self.verbose = verbose

        if self.model_settings.dimensions is None:
            raise ValueError("Model settings dimensions is not set")
        self._document_lancedb_model: Type[LanceModel] = get_document_lancedb_model(
            self.model_settings.dimensions
        )
        self._manifest_lancedb_model = ManifestLancedbModel

    @classmethod
    async def new(
        cls,
        lancedb_path: Path | str,
        *,
        manifest_table: Text = DEFAULT_MANIFEST_TABLE,
        document_table: Text = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: "AsyncOpenAIEmbeddingsModel",
        model_settings: "ModelSettings",
        token_secret_key: Text = "__lnclite__",
        vector_search_prefer: VectorIndexPreference = "balanced",
        verbose: bool = False,
    ) -> "Lnclite":
        lancedb_path = Path(lancedb_path)
        if lancedb_path.is_dir():
            # If not a empty directory, raise an error
            for _ in lancedb_path.iterdir():
                raise ValueError(f"Lancedb path {lancedb_path} already exists ")
        else:
            lancedb_path.mkdir(parents=True, exist_ok=True)

        if model_settings.dimensions is None:
            model_settings.dimensions = await get_default_dimensions(
                openai_embeddings_model
            )

        return cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
            token_secret_key=token_secret_key,
            vector_search_prefer=vector_search_prefer,
            verbose=verbose,
        )

    @classmethod
    async def new_from_dir(
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

        if model_settings.dimensions is None:
            model_settings.dimensions = await get_default_dimensions(
                openai_embeddings_model
            )

        lnclite = cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
        )

        # Create manifest
        await lnclite.manifest.upsert(
            name=dataset_name,
            description=dataset_description,
            model=openai_embeddings_model.model,
            dimensions=model_settings.dimensions,
        )

        # Create documents
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

        await lnclite.documents.create_index()

        return lnclite

    @classmethod
    async def load(
        cls,
        lancedb_path: Path | str,
        *,
        manifest_table: Text = DEFAULT_MANIFEST_TABLE,
        document_table: Text = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: "AsyncOpenAIEmbeddingsModel",
        model_settings: "ModelSettings",
        vector_search_prefer: VectorIndexPreference = "balanced",
        refresh_index: bool = False,
        verbose: bool = False,
    ) -> "Lnclite":
        lancedb_path = Path(lancedb_path)
        if not lancedb_path.is_dir():
            raise FileNotFoundError(f"Lancedb path {lancedb_path} not found")

        if model_settings.dimensions is None:
            model_settings.dimensions = await get_default_dimensions(
                openai_embeddings_model
            )

        lnclite = cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
            vector_search_prefer=vector_search_prefer,
            verbose=verbose,
        )

        # Validate manifest
        manifest = await lnclite.manifest.get()
        if manifest is None:
            raise LncliteNotFoundError("Manifest not found")
        if manifest.model != openai_embeddings_model.model:
            raise ValueError(
                f"OpenAI embeddings model mismatch: {manifest.model} != {openai_embeddings_model.model}"  # noqa: E501
            )
        if manifest.dimensions != model_settings.dimensions:
            raise ValueError(
                f"Model settings dimensions mismatch: {manifest.dimensions} != {model_settings.dimensions}"  # noqa: E501
            )

        if refresh_index:
            await lnclite.documents.create_index()

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

    async def create_index(self) -> None:
        await self.documents.create_index()

    async def embed(self, texts: List[Text]) -> np.ndarray:
        emb_res = await self.openai_embeddings_model.get_embeddings(
            texts, model_settings=self.model_settings
        )
        return normalize(emb_res.to_numpy())  # (n, d)

    async def search(
        self,
        query: Text,
        *,
        tags_any: Optional[List[Text]] = None,
        tags_all: Optional[List[Text]] = None,
        limit: int = 5,
        verbose: bool = False,
    ) -> "SearchResults":
        document_table = await self.documents.get_table()

        query_vector = (await self.embed([query]))[0]

        search_query = await document_table.search(query_vector)
        tags_filter = _tags_filter(tags_any=tags_any, tags_all=tags_all)
        if tags_filter is not None:
            search_query = search_query.where(tags_filter)

        if verbose or self.verbose:
            logger.info(f"Query plan: {await search_query.explain_plan()}")

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
        if await _table_exists(conn, self.client.manifest_table):
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
        if await _table_exists(conn, self.client.document_table):
            self._table = await conn.open_table(self.client.document_table)
        else:
            self._table = await conn.create_table(
                self.client.document_table, schema=self.client._document_lancedb_model
            )
        return self._table

    async def create_index(self) -> None:
        table = await self.get_table()

        await table.create_index("tags", config=lancedb.index.LabelList())

        row_count = await self.count()

        vs_config = recommended_vector_index_config(
            row_count,
            self.client.model_settings.dimensions,
            prefer=self.client.vector_search_prefer,
        )
        if vs_config is None:
            logger.info(
                "Skipping vector index: row_count=%s is too small; brute-force search is exact and fast",  # noqa: E501
                row_count,
            )
        else:
            await table.create_index("vector", config=vs_config)
            logger.info(f"Created vector index with config: {vs_config}")

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
        tags_any: Optional[List[Text]] = None,
        tags_all: Optional[List[Text]] = None,
        limit: int = 10,
        order: ListOrder = "asc",
        next_page_token: Optional[Text] = None,
        verbose: bool = False,
    ) -> TokenPaginatic[Document]:
        if limit < 1:
            raise ValueError(f"Limit must be greater than 0, got {limit}")

        sql_order = _to_sql_order(order)
        id_operator = ">" if sql_order == "ASC" else "<"
        after_id: Optional[int] = None
        if next_page_token is not None:
            decoded_token = decode_and_verify(next_page_token, self.client._secret_key)
            after_id = decoded_token.get("after")

        document_table = await self.client.documents.get_table()

        # Prepare query
        query_builder = document_table.query()

        query_builder = query_builder.where(
            _documents_list_where_clause(
                id_operator=id_operator,
                sql_order=sql_order,
                after_id=after_id,
                tags_any=tags_any,
                tags_all=tags_all,
            )
        ).limit(limit + 1)
        if verbose or self.client.verbose:
            logger.info(f"Query plan: {await query_builder.explain_plan()}")

        # Execute query
        documents = await query_builder.to_pydantic(self.client._document_lancedb_model)

        has_more = len(documents) > limit
        documents = documents[:limit]

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


def _documents_list_where_clause(
    *,
    id_operator: Literal[">", "<"],
    sql_order: SqlOrder,
    after_id: Optional[int] = None,
    tags_any: Optional[List[Text]] = None,
    tags_all: Optional[List[Text]] = None,
) -> str:
    id_filter = f"id {id_operator} {after_id}" if after_id is not None else "id > 0"
    filters = [id_filter]

    if tags_filter := _tags_filter(tags_any=tags_any, tags_all=tags_all):
        filters.append(tags_filter)

    where_clause = " AND ".join(f"({filter_})" for filter_ in filters)
    return f"{where_clause} ORDER BY id {sql_order}"


def _tags_filter(
    *,
    tags_any: Optional[List[Text]] = None,
    tags_all: Optional[List[Text]] = None,
) -> Optional[str]:
    filters: List[str] = []
    if tags_any:
        filters.append(tag_filter_any(tags_any))
    if tags_all:
        filters.append(tag_filter_all(tags_all))
    if not filters:
        return None
    return " AND ".join(f"({filter_})" for filter_ in filters)


def _to_sql_order(order: ListOrder) -> SqlOrder:
    if order in ("asc", 1):
        return "ASC"
    if order in ("desc", -1):
        return "DESC"
    raise ValueError(f"Invalid order: {order}")


async def _table_exists(conn: lancedb.AsyncConnection, table_name: Text) -> bool:
    table_list = await conn.list_tables()
    return table_name in table_list.tables
