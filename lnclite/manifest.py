"""Manifest table API for lnclite stores."""

import time
from typing import TYPE_CHECKING

import lancedb
from lancedb.pydantic import LanceModel
from pydantic import Field

from lnclite.models import LncliteNotFoundError, ManifestModel

if TYPE_CHECKING:
    from lnclite.client import Lnclite


class ManifestLancedbModel(LanceModel):
    id: int = Field(default_factory=lambda: _generate_id())
    name: str = Field(description="The name of the database.")
    description: str = Field(description="The description of the database.")
    model: str = Field(description="The embedding model name.")
    dimensions: int = Field(description="The dimensions of the embeddings.")
    last_updated: int = Field(default_factory=lambda: int(time.time()))


class Manifest:
    def __init__(self, client: "Lnclite"):
        self.client = client
        self._table: lancedb.AsyncTable | None = None

    async def get_table(self) -> lancedb.AsyncTable:
        if self._table is not None:
            return self._table

        conn = await self.client.get_connection()
        if await self.client.table_exists(self.client.manifest_table):
            self._table = await conn.open_table(self.client.manifest_table)
        else:
            self._table = await conn.create_table(
                self.client.manifest_table,
                schema=self.client._manifest_lancedb_model,
            )
        return self._table

    async def get(self) -> ManifestModel | None:
        table = await self.get_table()
        manifests = (
            await table.query()
            .limit(1)
            .to_pydantic(self.client._manifest_lancedb_model)
        )
        if manifests:
            return ManifestModel.model_validate_json(manifests[0].model_dump_json())
        return None

    async def retrieve(self) -> ManifestModel:
        manifest = await self.get()
        if manifest is not None:
            return manifest
        raise LncliteNotFoundError("Manifest not found")

    async def upsert(
        self,
        *,
        name: str,
        description: str,
        model: str,
        dimensions: int,
    ) -> ManifestModel:
        table = await self.get_table()
        existing = await self.get()

        if existing is None:
            manifest = self.client._manifest_lancedb_model(
                name=name,
                description=description,
                model=model,
                dimensions=dimensions,
            )
            await table.add([manifest])
            return ManifestModel.model_validate_json(manifest.model_dump_json())

        last_updated = int(time.time())
        await table.update(
            where=f"id = {existing.id}",
            updates={
                "name": name,
                "description": description,
                "model": model,
                "dimensions": dimensions,
                "last_updated": last_updated,
            },
        )
        return ManifestModel(
            id=existing.id,
            name=name,
            description=description,
            model=model,
            dimensions=dimensions,
            last_updated=last_updated,
        )


def _generate_id() -> int:
    from lnclite.utils.snowflake import generate_id

    return generate_id()
