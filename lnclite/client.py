"""Async LanceDB-backed lnclite client."""

import functools
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import numpy as np
from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings
from openai_embeddings_model.normalize import normalize

from lnclite.constants import (
    DEFAULT_DOCUMENT_TABLE,
    DEFAULT_MANIFEST_TABLE,
    VectorIndexPreference,
)
from lnclite.documents import Documents, document_from_lance_row
from lnclite.filters import tags_filter
from lnclite.manifest import Manifest, ManifestLancedbModel
from lnclite.models import (
    DocumentCreate,
    LncliteNotFoundError,
    SearchResult,
    SearchResults,
)

if TYPE_CHECKING:
    from lnclite.file_ingestor import FileReader

logger = logging.getLogger(__name__)


class Lnclite:
    def __init__(
        self,
        lancedb_path: Path | str,
        *,
        manifest_table: str = DEFAULT_MANIFEST_TABLE,
        document_table: str = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
        model_settings: ModelSettings,
        token_secret_key: str = "__lnclite__",
        vector_search_prefer: VectorIndexPreference = "balanced",
        verbose: bool = False,
        index_cache_size: int | None = None,
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
        self.index_cache_size = index_cache_size

        if self.model_settings.dimensions is None:
            raise ValueError("Model settings dimensions is not set")

        from lnclite.documents import get_document_lancedb_model

        self._document_lancedb_model = get_document_lancedb_model(
            self.model_settings.dimensions
        )
        self._manifest_lancedb_model = ManifestLancedbModel

    @classmethod
    async def new(
        cls,
        lancedb_path: Path | str,
        *,
        name: str = "lnclite",
        description: str = "",
        manifest_table: str = DEFAULT_MANIFEST_TABLE,
        document_table: str = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
        model_settings: ModelSettings,
        token_secret_key: str = "__lnclite__",
        vector_search_prefer: VectorIndexPreference = "balanced",
        verbose: bool = False,
        index_cache_size: int | None = None,
    ) -> "Lnclite":
        lancedb_path = Path(lancedb_path)
        if lancedb_path.is_dir():
            for _ in lancedb_path.iterdir():
                raise ValueError(f"Lancedb path {lancedb_path} already exists ")
        else:
            lancedb_path.mkdir(parents=True, exist_ok=True)

        if model_settings.dimensions is None:
            model_settings.dimensions = await get_default_dimensions(
                openai_embeddings_model
            )

        client = cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
            token_secret_key=token_secret_key,
            vector_search_prefer=vector_search_prefer,
            verbose=verbose,
            index_cache_size=index_cache_size,
        )
        await client.manifest.upsert(
            name=name,
            description=description,
            model=openai_embeddings_model.model,
            dimensions=model_settings.dimensions,
        )
        return client

    @classmethod
    async def new_from_dir(
        cls,
        dir_path: Path | str,
        lancedb_path: Path | str,
        *,
        dataset_name: str,
        dataset_description: str,
        manifest_table: str = DEFAULT_MANIFEST_TABLE,
        document_table: str = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
        model_settings: ModelSettings,
        extension_readers: dict[str, "FileReader"] | None = None,
        batch_size: int = 100,
    ) -> "Lnclite":
        from lnclite.file_ingestor import FileIngestor

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory {dir_path} not found")

        client = await cls.new(
            lancedb_path=lancedb_path,
            name=dataset_name,
            description=dataset_description,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
        )

        file_ingestor = FileIngestor()
        if extension_readers is not None:
            for extension, reader in extension_readers.items():
                file_ingestor.register_reader(extension, reader)

        batch: list[DocumentCreate] = []
        async for file in file_ingestor.ingest_async(dir_path):
            file_content = file["content"].strip()
            file_path = str(file["path"])

            if not file_content:
                logger.warning("Skipping %s due to empty content", file_path)
                continue

            relative_path = Path(file_path).relative_to(dir_path).as_posix()
            batch.append(
                DocumentCreate(
                    content=file_content,
                    tags=[f"path:{relative_path}"],
                )
            )

            if len(batch) >= batch_size:
                await client.documents.batch_create(batch)
                logger.info("Created %s documents", len(batch))
                batch = []

        if batch:
            await client.documents.batch_create(batch)
            logger.info("Created %s documents", len(batch))

        await client.documents.create_index()
        return client

    @classmethod
    async def load(
        cls,
        lancedb_path: Path | str,
        *,
        manifest_table: str = DEFAULT_MANIFEST_TABLE,
        document_table: str = DEFAULT_DOCUMENT_TABLE,
        openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
        model_settings: ModelSettings,
        token_secret_key: str = "__lnclite__",
        vector_search_prefer: VectorIndexPreference = "balanced",
        refresh_index: bool = False,
        verbose: bool = False,
        index_cache_size: int | None = None,
    ) -> "Lnclite":
        lancedb_path = Path(lancedb_path)
        if not lancedb_path.is_dir():
            raise FileNotFoundError(f"Lancedb path {lancedb_path} not found")

        if model_settings.dimensions is None:
            model_settings.dimensions = await get_default_dimensions(
                openai_embeddings_model
            )

        client = cls(
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_embeddings_model=openai_embeddings_model,
            model_settings=model_settings,
            token_secret_key=token_secret_key,
            vector_search_prefer=vector_search_prefer,
            verbose=verbose,
            index_cache_size=index_cache_size,
        )

        manifest = await client.manifest.get()
        if manifest is None:
            raise LncliteNotFoundError("Manifest not found")
        if manifest.model != openai_embeddings_model.model:
            raise ValueError(
                "OpenAI embeddings model mismatch: "
                f"{manifest.model} != {openai_embeddings_model.model}"
            )
        if manifest.dimensions != model_settings.dimensions:
            raise ValueError(
                "Model settings dimensions mismatch: "
                f"{manifest.dimensions} != {model_settings.dimensions}"
            )

        if refresh_index:
            await client.documents.create_index()

        return client

    async def get_connection(self) -> lancedb.AsyncConnection:
        if self._connection is None:
            self._connection = await lancedb.connect_async(self.lancedb_path)
            logger.info("Lancedb connected to %s", self.lancedb_path)
        return self._connection

    @functools.cached_property
    def manifest(self) -> Manifest:
        return Manifest(self)

    @functools.cached_property
    def documents(self) -> Documents:
        return Documents(self)

    async def create_index(self) -> None:
        await self.documents.create_index()

    async def embed(self, texts: list[str]) -> np.ndarray:
        emb_res = await self.openai_embeddings_model.get_embeddings(
            texts,
            model_settings=self.model_settings,
        )
        return normalize(emb_res.to_numpy())

    async def search(
        self,
        query: str,
        *,
        tags_any: list[str] | None = None,
        tags_all: list[str] | None = None,
        limit: int = 5,
        include_vector: bool = False,
        verbose: bool = False,
    ) -> SearchResults:
        document_table = await self.documents.get_table()
        query_vector = (await self.embed([query]))[0]

        search_query = await document_table.search(query_vector)
        filter_ = tags_filter(tags_any=tags_any, tags_all=tags_all)
        if filter_ is not None:
            search_query = search_query.where(filter_)

        if verbose or self.verbose:
            logger.info("Query plan: %s", await search_query.explain_plan())

        search_results = await search_query.distance_type("dot").limit(limit).to_list()

        results: list[SearchResult] = []
        for result in search_results:
            document = document_from_lance_row(
                result,
                include_vector=include_vector,
            )
            results.append(
                SearchResult(
                    document=document,
                    distance=result["_distance"],
                )
            )

        return SearchResults(results=results)

    async def close(self) -> None:
        if "documents" in self.__dict__ and self.documents._table is not None:
            await _maybe_await(self.documents._table.close())
            self.documents._table = None
        if "manifest" in self.__dict__ and self.manifest._table is not None:
            await _maybe_await(self.manifest._table.close())
            self.manifest._table = None
        if self._connection is not None:
            await _maybe_await(self._connection.close())
            self._connection = None

    async def table_exists(self, table_name: str) -> bool:
        conn = await self.get_connection()
        table_list = await conn.list_tables()
        return table_name in table_list.tables


async def get_default_dimensions(
    openai_embeddings_model: AsyncOpenAIEmbeddingsModel,
) -> int:
    emb_result = await openai_embeddings_model.get_embeddings(
        input="Hello, world!",
        model_settings=ModelSettings(),
    )
    return emb_result.to_numpy().shape[1]


async def _maybe_await(value) -> None:
    if inspect.isawaitable(value):
        await value
