import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, List, Optional, Text, Type

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
        vector: Vector(dim)
        tags: List[Text] = Field(default_factory=list)
        metadata: Dict[Text, Text] = Field(default_factory=dict)

    return Document


class ManifestModel(LanceModel):
    id: Text = Field(default_factory=gen_id)
    last_fingerprint: Optional[Text] = Field(default=None)
    last_updated: Optional[int] = Field(default=None)


class Lnclite:
    def __init__(
        self,
        files_dir: Path | str = ".",
        openai_model: Optional["AsyncOpenAIEmbeddingsModel"] = None,
        model_settings: Optional["ModelSettings"] = None,
    ):
        import diskcache
        import tiktoken
        from openai import AsyncOpenAI
        from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

        self.files_dir = Path(files_dir).resolve()
        if not self.files_dir.is_dir():
            raise FileNotFoundError(f"Files directory {self.files_dir} not found")

        self.lancedb_path = self.files_dir.parent.joinpath(
            "." + self.files_dir.name + ".index"
        )
        self.db: lancedb.AsyncConnection | None = None

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

    async def connect(self):
        if self.db is None:
            self.db = await lancedb.connect_async(self.lancedb_path)
            logger.info(f"Lancedb connected to {self.lancedb_path}")
        return self.db

    async def search(self):
        pass

    async def sync(self):
        pass

    async def is_synced(self) -> bool:
        if self.last_fingerprint is None:
            from lnclite.utils.get_folder_fingerprint import get_folder_fingerprint

            self.last_fingerprint = get_folder_fingerprint(self.files_dir)

        return self.last_fingerprint == get_folder_fingerprint(self.files_dir)
