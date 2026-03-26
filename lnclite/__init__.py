import functools
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, List, Optional, Self, Text, Type

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

if TYPE_CHECKING:
    from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings


__version__: Final[Text] = "0.1.0"

logger = logging.getLogger(__name__)


def gen_id() -> str:
    from lnclite.utils.snowflake import generate_id

    return generate_id()


def get_document_model(dim: int) -> Type[LanceModel]:
    class Document(LanceModel):
        id: Text = Field(default_factory=gen_id)
        path: Text = Field(
            description="The path to the document relative to the root of the files directory."  # noqa: E501
        )
        vector: Vector(dim)
        tags: List[Text] = Field(default_factory=list)
        metadata: Dict[Text, Text] = Field(default_factory=dict)

    return Document


class ManifestModel(LanceModel):
    id: Text = Field(default_factory=gen_id)
    model: Text = Field(description="The model used to generate the manifest.")
    dimensions: int = Field(description="The dimensions of the embeddings.")
    last_fingerprint: Optional[Text] = Field(default=None)
    last_updated: Optional[int] = Field(default=None)


class Lnclite:
    DEFAULT_MANIFEST_TABLE = "manifest"
    DEFAULT_DOCUMENT_TABLE = "documents"

    @classmethod
    async def new(
        cls,
        files_dir: Path | str = ".",
        *,
        lancedb_path: Path | str | None = None,
        manifest_table: Optional[Text] = None,
        document_table: Optional[Text] = None,
        openai_model: Optional["AsyncOpenAIEmbeddingsModel"] = None,
        model_settings: Optional["ModelSettings"] = None,
        batch_size: int = 100,
    ) -> "Lnclite":
        lnclite = cls(
            files_dir=files_dir,
            lancedb_path=lancedb_path,
            manifest_table=manifest_table,
            document_table=document_table,
            openai_model=openai_model,
            model_settings=model_settings,
            batch_size=batch_size,
        )

        db = await lnclite.get_connection()
        existing_tables = await db.table_names()

        # Check if tables already exist
        for table in [lnclite.manifest_table, lnclite.document_table]:
            if table in existing_tables:
                raise ValueError(f"Table {table} already exists")

        # Create manifest table
        manifest_tb = await db.create_table(
            lnclite.manifest_table, schema=lnclite.manifest_model, mode="overwrite"
        )

        from lnclite.utils.get_folder_fingerprint import get_folder_fingerprint

        fingerprint = get_folder_fingerprint(files_dir)
        await manifest_tb.add(
            lnclite.manifest_model.model_validate(
                {
                    "model": openai_model.model,
                    "dimensions": model_settings.dimensions,
                    "last_fingerprint": fingerprint,
                    "last_updated": int(time.time()),
                }
            )
        )

        # Create document table
        from lnclite.utils.calculate_index_params import calculate_index_params

        document_tb = await db.create_table(
            lnclite.document_table, schema=lnclite.document_model, mode="overwrite"
        )
        num_partitions, num_sub_vectors = calculate_index_params(
            await lnclite.total_documents(use_file_paths=True),
            model_settings.dimensions,
        )
        await document_tb.create_index(
            column_name="vector",
            metric="dot",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            replace=True,
        )

        return lnclite

    def __init__(
        self,
        files_dir: Path | str = ".",
        *,
        lancedb_path: Path | str | None = None,
        manifest_table: Optional[Text] = None,
        document_table: Optional[Text] = None,
        openai_model: Optional["AsyncOpenAIEmbeddingsModel"] = None,
        model_settings: Optional["ModelSettings"] = None,
        batch_size: int = 100,
    ):
        import diskcache
        import tiktoken
        from openai import AsyncOpenAI
        from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

        self.files_dir = Path(files_dir)
        if not self.files_dir.is_dir():
            raise FileNotFoundError(f"Files directory {self.files_dir} not found")

        self.lancedb_path = lancedb_path or self.files_dir.parent.joinpath(
            "." + self.files_dir.name + ".index"
        )
        self.connection: lancedb.AsyncConnection | None = None

        self.manifest_table = manifest_table or self.DEFAULT_MANIFEST_TABLE
        self.document_table = document_table or self.DEFAULT_DOCUMENT_TABLE

        self.last_fingerprint = None

        self.openai_model = openai_model or AsyncOpenAIEmbeddingsModel(
            model="text-embedding-3-small",
            openai_client=AsyncOpenAI(),
            cache=diskcache.Cache(".cache/embeddings"),
            encoding=tiktoken.encoding_for_model("text-embedding-3-small"),
        )
        self.model_settings = model_settings or ModelSettings(dimensions=1536)

        self.document_model = get_document_model(self.model_settings.dimensions)
        self.manifest_model = ManifestModel

    async def get_connection(self) -> lancedb.AsyncConnection:
        if self.connection is None:
            self.connection = await lancedb.connect_async(self.lancedb_path)
            logger.info(f"Lancedb connected to {self.lancedb_path}")
        return self.connection

    @functools.cache
    async def get_table(self, table_name: Text) -> lancedb.AsyncTable:
        return (await self.get_connection()).open_table(table_name)

    async def retrieve_manifest(self) -> ManifestModel | None:
        manifest_table = await self.get_table(self.manifest_table)
        _query_builder = await manifest_table.search()
        manifests = await _query_builder.limit(1).to_pydantic(self.manifest_model)
        if len(manifests) > 0:
            return manifests[0]
        return None

    async def search(self):
        pass

    async def sync(self) -> Self:
        if await self.is_synced():
            logger.debug(f"Already synced {self.files_dir}.")
            return

        logger.debug(f"Syncing {self.files_dir}.")

        manifest_table = await self.get_table(self.manifest_table)
        document_table = await self.get_table(self.document_table)

        def _batch_file_paths(root: Path, batch_size: int):
            _batch: list[Path] = []
            for root, dirs, files in os.walk(self.files_dir):
                for file in files:
                    file_path = Path(root).joinpath(file)
                    _batch.append(relative_path)
                    if len(_batch) >= batch_size:
                        yield _batch
                        _batch = []
            if _batch:
                yield _batch

        all_file_paths: list[Path] = []
        for batch_file_paths in _batch_file_paths(self.files_dir, self.batch_size):
            all_file_paths.extend(batch_file_paths)

        for root, dirs, files in os.walk(self.files_dir):
            for file in files:
                file_path = Path(root).joinpath(file)
                relative_path = file_path.relative_to(self.files_dir)
                document = self.document_model(
                    path=relative_path,
                )
                document_table.insert(document)

    async def is_synced(self) -> bool:
        from lnclite.utils.get_folder_fingerprint import get_folder_fingerprint

        might_manifest = await self.retrieve_manifest()

        if might_manifest is None:
            logger.debug(f"No manifest found for {self.files_dir}.")
            return False

        manifest = might_manifest

        current_fingerprint = get_folder_fingerprint(self.files_dir)

        # Already synced
        if manifest.last_fingerprint == current_fingerprint:
            self.last_fingerprint = current_fingerprint
            return True

        else:
            logger.debug(
                f"Manifest found for {self.files_dir} but fingerprint mismatch. "
                + f"Expected {manifest.last_fingerprint} but got {current_fingerprint}."
            )
            return False

    async def total_documents(self, use_file_paths: bool = False) -> int:
        if use_file_paths:
            count = 0
            for _, _, files in os.walk(self.files_dir):
                for _ in files:
                    count += 1
            return count
        else:
            document_table = await self.get_table(self.document_table)
            return await document_table.count_rows()
