"""Documents collection API for lnclite stores."""

import asyncio
import contextlib
import functools
import hashlib
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import lancedb
import lancedb.index
import numpy as np
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from openai_embeddings_model.normalize import normalize
from paginatic import TokenPaginatic
from paginatic.helpers import decode_and_verify, encode_and_sign
from pydantic import Field, model_validator

from lnclite.constants import ListOrder
from lnclite.filters import documents_list_where_clause, to_sql_order
from lnclite.indexing import build_document_index_plan, recommended_vector_index_config
from lnclite.models import Document, DocumentCreate, LncliteNotFoundError

if TYPE_CHECKING:
    from lnclite.client import Lnclite

logger = logging.getLogger(__name__)


@functools.cache
def get_document_lancedb_model(dim: int) -> type[LanceModel]:
    class DocumentLancedbModel(LanceModel):
        id: int = Field(default_factory=lambda: _generate_id())
        content: str = Field(description="The content of the document.")
        md5: str = ""
        vector: Vector(dim)
        tags: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_values(self) -> "DocumentLancedbModel":
            self.content = self.content.strip()
            if not self.content:
                raise ValueError("Content cannot be empty")
            self.md5 = hashlib.md5(self.content.encode()).hexdigest()
            return self

    return DocumentLancedbModel


def document_from_lance_model(
    document: LanceModel,
    *,
    include_vector: bool,
) -> Document:
    output = Document.model_validate_json(document.model_dump_json())
    if not include_vector:
        output.vector = None
    return output


def document_from_lance_row(row: dict, *, include_vector: bool) -> Document:
    output = Document.model_validate(row)
    if not include_vector:
        output.vector = None
    return output


class Documents:
    def __init__(self, client: "Lnclite"):
        self.client = client
        self._table: lancedb.AsyncTable | None = None
        self._table_lock = asyncio.Lock()

    async def get_table(self) -> lancedb.AsyncTable:
        if self._table is not None:
            return self._table

        async with self._table_lock:
            if self._table is not None:
                return self._table

            conn = await self.client.get_connection()
            if await self.client.table_exists(self.client.document_table):
                self._table = await conn.open_table(
                    self.client.document_table,
                    index_cache_size=self.client.index_cache_size,
                )
            else:
                self._table = await conn.create_table(
                    self.client.document_table,
                    schema=self.client._document_lancedb_model,
                )
        return self._table

    async def index_plan(self):
        return build_document_index_plan(
            row_count=await self.count(),
            dimensions=self.client.model_settings.dimensions,
            vector_search_prefer=self.client.vector_search_prefer,
        )

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
                "Skipping vector index: row_count=%s is too small; "
                "brute-force search is exact and fast",
                row_count,
            )
        else:
            await table.create_index("vector", config=vs_config)
            logger.info("Created vector index with config: %s", vs_config)

    async def count(self) -> int:
        document_table = await self.get_table()
        return await document_table.count_rows()

    async def get(self, id: int, *, include_vector: bool = False) -> Document | None:
        document_table = await self.get_table()
        documents = await (
            document_table.query()
            .where(f"id = {id}")
            .limit(1)
            .to_pydantic(self.client._document_lancedb_model)
        )
        if documents:
            return document_from_lance_model(
                documents[0],
                include_vector=include_vector,
            )
        return None

    async def retrieve(
        self,
        id: int,
        *,
        include_vector: bool = False,
    ) -> Document:
        document = await self.get(id, include_vector=include_vector)
        if document is not None:
            return document
        raise LncliteNotFoundError(f"Document with id {id} not found")

    async def create(self, document_create: DocumentCreate) -> Document:
        return (await self.batch_create([document_create]))[0]

    async def batch_create(
        self,
        document_creates: list[DocumentCreate],
        *,
        verbose: bool = False,
    ) -> list[Document]:
        documents = await self._batch_add(
            document_creates,
            return_documents=True,
            verbose=verbose,
            operation="batch_create",
        )
        return documents

    async def batch_insert(
        self,
        document_creates: list[DocumentCreate],
        *,
        verbose: bool = False,
    ) -> int:
        await self._batch_add(
            document_creates,
            return_documents=False,
            verbose=verbose,
            operation="batch_insert",
        )
        return len(document_creates)

    async def batch_insert_embedded(
        self,
        document_creates: list[DocumentCreate],
        vectors: np.ndarray | Sequence[Sequence[float]],
        *,
        normalize_vectors: bool = True,
        verbose: bool = False,
    ) -> int:
        if not document_creates:
            return 0

        timings: dict[str, float] = {}
        with _timed(timings, "total"):
            with _timed(timings, "get_table"):
                document_table = await self.get_table()
            with _timed(timings, "normalize_vectors"):
                vector_array = _coerce_vectors(
                    vectors,
                    expected_count=len(document_creates),
                    expected_dimensions=self.client.model_settings.dimensions,
                    normalize_vectors=normalize_vectors,
                )
            with _timed(timings, "row_construction"):
                rows = _documents_to_arrow(
                    document_creates,
                    vector_array,
                    dimensions=self.client.model_settings.dimensions,
                )
            with _timed(timings, "table_add"):
                await document_table.add(rows)

        _log_timings(
            "batch_insert_embedded",
            timings,
            enabled=verbose or self.client.verbose,
        )
        return len(document_creates)

    async def _batch_add(
        self,
        document_creates: list[DocumentCreate],
        *,
        return_documents: bool,
        verbose: bool,
        operation: str,
    ) -> list[Document]:
        if not document_creates:
            return []

        output: list[Document] = []
        timings: dict[str, float] = {}
        with _timed(timings, "total"):
            with _timed(timings, "get_table"):
                document_table = await self.get_table()
            with _timed(timings, "embed"):
                vectors = await self.client.embed([d.content for d in document_creates])
            with _timed(timings, "row_construction"):
                vectors = _coerce_vectors(
                    vectors,
                    expected_count=len(document_creates),
                    expected_dimensions=self.client.model_settings.dimensions,
                    normalize_vectors=False,
                )
                rows, row_models = _document_rows_to_arrow_and_models(
                    document_creates,
                    vectors,
                    dimensions=self.client.model_settings.dimensions,
                )
            with _timed(timings, "table_add"):
                await document_table.add(rows)
            if return_documents:
                with _timed(timings, "return_construction"):
                    output = [
                        Document(
                            id=row.id,
                            content=row.content,
                            md5=row.md5,
                            vector=vector.tolist(),
                            tags=row.tags,
                        )
                        for row, vector in zip(row_models, vectors)
                    ]

        _log_timings(operation, timings, enabled=verbose or self.client.verbose)
        return output

    async def list(
        self,
        *,
        tags_any: list[str] | None = None,
        tags_all: list[str] | None = None,
        limit: int = 10,
        order: ListOrder = "asc",
        next_page_token: str | None = None,
        include_vector: bool = False,
        verbose: bool = False,
    ) -> TokenPaginatic[Document]:
        if limit < 1:
            raise ValueError(f"Limit must be greater than 0, got {limit}")

        sql_order = to_sql_order(order)
        id_operator = ">" if sql_order == "ASC" else "<"
        after_id: int | None = None
        if next_page_token is not None:
            decoded_token = decode_and_verify(next_page_token, self.client._secret_key)
            after_id = decoded_token.get("after")

        document_table = await self.client.documents.get_table()
        query_builder = document_table.query()
        query_builder = query_builder.where(
            documents_list_where_clause(
                id_operator=id_operator,
                sql_order=sql_order,
                after_id=after_id,
                tags_any=tags_any,
                tags_all=tags_all,
            )
        ).limit(limit + 1)

        if verbose or self.client.verbose:
            logger.info("Query plan: %s", await query_builder.explain_plan())

        documents = await query_builder.to_pydantic(self.client._document_lancedb_model)
        has_more = len(documents) > limit
        documents = documents[:limit]
        next_token = (
            encode_and_sign({"after": documents[-1].id}, self.client._secret_key)
            if has_more and documents
            else None
        )

        return TokenPaginatic(
            object="list",
            data=[
                document_from_lance_model(
                    document,
                    include_vector=include_vector,
                )
                for document in documents
            ],
            next_page_token=next_token,
        )


def _generate_id() -> int:
    from lnclite.utils.snowflake import generate_id

    return generate_id()


@contextlib.contextmanager
def _timed(timings: dict[str, float], phase: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = time.perf_counter() - start


def _coerce_vectors(
    vectors: np.ndarray | Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
    normalize_vectors: bool,
) -> np.ndarray:
    vector_array = np.asarray(vectors, dtype=np.float32)
    if vector_array.ndim != 2:
        raise ValueError(f"Expected 2-dimensional vectors, got {vector_array.ndim}")
    if vector_array.shape[0] != expected_count:
        raise ValueError(
            f"Expected {expected_count} vectors, got {vector_array.shape[0]}"
        )
    if vector_array.shape[1] != expected_dimensions:
        raise ValueError(
            f"Expected vectors with {expected_dimensions} dimensions, "
            f"got {vector_array.shape[1]}"
        )
    if normalize_vectors:
        vector_array = normalize(vector_array)
    return vector_array


def _documents_to_arrow(
    document_creates: list[DocumentCreate],
    vectors: np.ndarray,
    *,
    dimensions: int,
) -> pa.Table:
    row_models = _document_row_models(document_creates)
    return _document_row_models_to_arrow(row_models, vectors, dimensions=dimensions)


def _document_rows_to_arrow_and_models(
    document_creates: list[DocumentCreate],
    vectors: np.ndarray,
    *,
    dimensions: int,
) -> tuple[pa.Table, list["_DocumentRow"]]:
    row_models = _document_row_models(document_creates)
    return (
        _document_row_models_to_arrow(row_models, vectors, dimensions=dimensions),
        row_models,
    )


def _document_row_models(
    document_creates: list[DocumentCreate],
) -> list["_DocumentRow"]:
    return [
        _DocumentRow(
            id=_generate_id(),
            content=document_create.content,
            md5=hashlib.md5(document_create.content.encode()).hexdigest(),
            tags=document_create.tags,
        )
        for document_create in document_creates
    ]


def _document_row_models_to_arrow(
    row_models: list["_DocumentRow"],
    vectors: np.ndarray,
    *,
    dimensions: int,
) -> pa.Table:
    flattened_vectors = pa.array(vectors.reshape(-1), type=pa.float32())
    fixed_vectors = pa.FixedSizeListArray.from_arrays(flattened_vectors, dimensions)
    return pa.table(
        {
            "id": pa.array([row.id for row in row_models], type=pa.int64()),
            "content": pa.array([row.content for row in row_models], type=pa.string()),
            "md5": pa.array([row.md5 for row in row_models], type=pa.string()),
            "vector": fixed_vectors,
            "tags": pa.array(
                [row.tags for row in row_models], type=pa.list_(pa.string())
            ),
        }
    )


def _log_timings(
    operation: str,
    timings: dict[str, float],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    logger.info(
        "%s timings: %s",
        operation,
        " ".join(f"{phase}={elapsed:.6f}s" for phase, elapsed in timings.items()),
    )


@dataclass(frozen=True)
class _DocumentRow:
    id: int
    content: str
    md5: str
    tags: list[str]
