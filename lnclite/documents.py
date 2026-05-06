"""Documents collection API for lnclite stores."""

import functools
import hashlib
import logging
from typing import TYPE_CHECKING

import lancedb
import lancedb.index
from lancedb.pydantic import LanceModel, Vector
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

    async def get_table(self) -> lancedb.AsyncTable:
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
    ) -> list[Document]:
        document_table = await self.get_table()
        normalized_vectors = await self.client.embed(
            [d.content for d in document_creates]
        )

        documents = [
            self.client._document_lancedb_model(
                content=d.content,
                tags=d.tags,
                vector=v,
            )
            for d, v in zip(document_creates, normalized_vectors)
        ]

        await document_table.add(documents)

        output: list[Document] = []
        for document, vector in zip(documents, normalized_vectors):
            created = Document.model_validate_json(
                document.model_dump_json(exclude_none=True)
            )
            created.vector = vector.tolist()
            output.append(created)

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
