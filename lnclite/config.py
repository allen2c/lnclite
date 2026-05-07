"""Configuration models for lnclite clients."""

from dataclasses import dataclass
from pathlib import Path

from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

from lnclite.constants import (
    DEFAULT_DOCUMENT_TABLE,
    DEFAULT_MANIFEST_TABLE,
    VectorIndexPreference,
)


@dataclass(frozen=True)
class LncliteConfig:
    lancedb_path: Path | str
    openai_embeddings_model: AsyncOpenAIEmbeddingsModel
    model_settings: ModelSettings
    manifest_table: str = DEFAULT_MANIFEST_TABLE
    document_table: str = DEFAULT_DOCUMENT_TABLE
    token_secret_key: str = "__lnclite__"
    vector_search_prefer: VectorIndexPreference = "balanced"
    verbose: bool = False
    index_cache_size: int | None = None
